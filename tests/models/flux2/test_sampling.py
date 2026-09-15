"""Klein precision, target-only updates, CFG, and early input admission."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import save_weights
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.models.flux2 import loading
from kinomlx.models.flux2.catalog import Flux2Variant, profile
from kinomlx.models.flux2.generation import generate_image
from kinomlx.models.flux2.loading import _target
from kinomlx.models.flux2.preparation import prepare_weight
from kinomlx.models.flux2.sampling import SamplingRecipe, denoise, euler_step, schedule
from kinomlx.models.flux2.shapes import transformer_shapes
from kinomlx.models.flux2.transformer import Flux2Transformer, QKNorm, rotary_positions, rotate
from tests._reporting import RecordingReporter

from .test_resources import _component


def test_all_transformer_weights_have_unique_graph_owners():
    model = Flux2Transformer()
    targets = [_target(model, key, shape) for key, shape in transformer_shapes().items()]
    assert len(targets) == 233
    assert len({(id(t.owner), t.attribute) for t in targets}) == 233


def test_preparation_rule_casts_and_bakes_only_selected_transformer_layouts():
    value = (mx.arange(12).reshape(3, 4) / 7).astype(mx.bfloat16)
    down = prepare_weight(
        "transformer", "transformer_blocks.2.ff.linear_out.weight", value, mx.float16
    )
    assert down.shape == (4, 3)
    assert down.dtype == mx.float16
    assert mx.array_equal(down, value.astype(mx.float16).T).item()
    plain = prepare_weight("transformer", "x_embedder.weight", value, mx.float16)
    assert plain.shape == (3, 4)
    assert mx.array_equal(plain, value.astype(mx.float16)).item()
    # Other components only cast; the selection is a transformer rule.
    text = prepare_weight(
        "text_encoder", "transformer_blocks.2.ff.linear_out.weight", value, mx.bfloat16
    )
    assert text.shape == (3, 4)
    assert mx.array_equal(text, value).item()


def test_source_binding_is_lazy_and_layout_selection_follows_the_component(tmp_path, monkeypatch):
    keys = {"transformer_blocks.0.ff.linear_out.weight": (3, 4), "x_embedder.weight": (2, 3)}
    values = {
        key: (mx.arange(shape[0] * shape[1]).reshape(shape) / 7).astype(mx.bfloat16)
        for key, shape in keys.items()
    }
    source = tmp_path / "model.safetensors"
    save_weights(source, values)
    monkeypatch.setattr(loading, "transformer_shapes", lambda: dict(keys))

    def eager(*args):
        raise AssertionError("source binding must not evaluate weights")

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", eager)
        model = loading.load_transformer((source,), dtype=mx.float16)
    down = _target(model, "transformer_blocks.0.ff.linear_out.weight", (4, 3)).owner
    assert isinstance(down, PretransposedLinear)
    assert down.weight.shape == (4, 3)
    assert down.weight.dtype == mx.float16
    assert mx.array_equal(
        down.weight, values["transformer_blocks.0.ff.linear_out.weight"].astype(mx.float16).T
    ).item()
    assert not isinstance(model.x_embedder, PretransposedLinear)
    assert mx.array_equal(
        model.x_embedder.weight, values["x_embedder.weight"].astype(mx.float16)
    ).item()
    with pytest.raises(ValueError, match="only FP16/BF16"):
        loading.load_transformer((source,), dtype=mx.float32)
    save_weights(source, {key: value.astype(mx.float16) for key, value in values.items()})
    with pytest.raises(ValueError, match="dtype F16"):
        loading.load_transformer((source,), dtype=mx.float16)
    calls = []
    monkeypatch.setattr(loading, "load_transformer", lambda *a, **k: calls.append(a))
    component = _component(tmp_path)
    loading.load_transformer_component(component, dtype=mx.float16)
    assert calls == [((component.bindings[0].source_file,),)]


@pytest.mark.parametrize("kind", ["noise", "negative_embeddings"])
def test_image_api_rejects_fp16_input_overflow_before_loading_components(kind):
    def component(name):
        assert name == "transformer", "invalid input must fail before loading text or VAE"
        return SimpleNamespace(dtype_policy="float16")

    resources = SimpleNamespace(
        memory_policy=MLXMemoryPolicy(), profile=profile("base"), component=component
    )
    shape = (1, 16, 128) if kind == "noise" else (1, 512, 12288)
    value = mx.full(shape, 100000, dtype=mx.float32)
    with pytest.raises(ValueError, match="must be finite"):
        generate_image(resources, "example", width=64, height=64, **{kind: value})


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_qk_norm_multiplies_learned_scale_before_output_rounding(dtype):
    value = mx.random.normal((1, 7, 32, 128), key=mx.random.key(42)).astype(dtype)
    norm = QKNorm()
    norm.weight = mx.linspace(0.1, 5, 128).astype(dtype)
    wide = value.astype(mx.float32)
    expected = (
        wide
        * mx.rsqrt(mx.mean(wide * wide, axis=-1, keepdims=True) + 1e-6)
        * norm.weight.astype(mx.float32)
    ).astype(dtype)
    delta = norm(value).astype(mx.float32) - expected.astype(mx.float32)
    assert (
        mx.sqrt(mx.mean(delta * delta)) / mx.sqrt(mx.mean(expected.astype(mx.float32) ** 2))
    ).item() < 0.0001
    rounded = (wide * mx.rsqrt(mx.mean(wide * wide, axis=-1, keepdims=True) + 1e-6)).astype(
        dtype
    ) * norm.weight
    assert not mx.array_equal(expected, rounded).item()


def test_rotary_axes_preserve_fractional_positions_and_vector_norm():
    value = mx.random.normal((1, 3, 2, 128), key=mx.random.key(8))
    ids = mx.array([[[0, 0, 0, 0], [10, 0.5, 3, 0], [20, 2, 1.5, 0]]])
    actual = rotate(value, rotary_positions(ids))
    assert mx.array_equal(actual[:, 0], value[:, 0]).item()
    assert (
        mx.max(mx.abs(mx.sum(actual * actual, axis=-1) - mx.sum(value * value, axis=-1))).item()
        < 0.0001
    )


class _Velocity:
    def __init__(self):
        self.inputs = []

    def __call__(self, image, text, timestep, image_ids, text_ids):
        self.inputs.append(image)
        return mx.ones_like(image) * text[:, :1, :1]


def _inputs():
    return (mx.zeros((1, 4, 128)), mx.ones((1, 2, 12288)), mx.zeros((1, 4, 4)), mx.zeros((1, 2, 4)))


@pytest.mark.parametrize("mode", ["sequential", "batched"])
def test_cfg_updates_only_targets_and_keeps_references_fixed(mode):
    model = _Velocity()
    noise, text, ids, tids = _inputs()
    reference = mx.full((1, 3, 128), 17.0)
    result = denoise(
        model,
        noise,
        text,
        ids,
        tids,
        recipe=SamplingRecipe(Flux2Variant.BASE, 4, 4.0, mode),
        references=reference,
        reference_ids=mx.zeros((1, 3, 4)),
        negative=-text,
    )
    assert mx.max(mx.abs(result + 7)).item() < 1e-6
    for seen in model.inputs:
        assert mx.all(seen[:, 4:] == 17).item()
    assert result.shape == noise.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recipe": SamplingRecipe(Flux2Variant.KV, guidance=4)},
        {"recipe": SamplingRecipe(Flux2Variant.BASE)},
        {"recipe": SamplingRecipe(guidance=4)},
        {"negative": mx.zeros((1, 2, 12288))},
        {"references": mx.zeros((1, 1, 128))},
    ],
)
def test_invalid_recipes_fail_before_transformer_execution(kwargs):
    model = _Velocity()
    with pytest.raises(ValueError, match="KV|CFG|steps|References|Negative"):
        denoise(model, *_inputs(), **kwargs)
    assert not model.inputs


def test_progress_starts_before_first_velocity_and_closes_on_failure():
    reporter = RecordingReporter()

    def fail(*args):
        assert reporter.events[0][0] == "start"
        raise RuntimeError("velocity failed")

    with pytest.raises(RuntimeError, match="velocity failed"):
        denoise(fail, *_inputs(), reporter=reporter)
    assert reporter.events[-1][0] == "end"


def test_schedule_and_step_precision():
    sigmas = schedule(384, 4)
    assert sigmas[0].item() == 1
    assert sigmas[-1].item() == 0
    assert mx.all(sigmas[1:] < sigmas[:-1]).item()
    source = mx.array([0.5], dtype=mx.bfloat16)
    velocity = mx.array([1.25], dtype=mx.bfloat16)
    actual = euler_step(source, velocity, mx.array(-0.123, dtype=mx.float32))
    increment = mx.array(-0.123, dtype=mx.bfloat16) * velocity
    assert mx.array_equal(
        actual, (source.astype(mx.float32) + increment.astype(mx.float32)).astype(mx.bfloat16)
    ).item()


@pytest.mark.parametrize(
    ("tokens", "steps", "midpoint"),
    [
        (1024, 4, 0.8839818317017591),
        (4096, 4, 0.908143922521333),
        (4300, 10, 0.9067067588045681),
        (4301, 10, 0.7657912730789151),
        (4096, 200, 0.7595102922715895),
        (16384, 4, 0.9619472529114758),
    ],
)
def test_empirical_schedule_pins_bfl_coefficients_and_branch_boundary(tokens, steps, midpoint):
    # Independently evaluated BFL empirical mu and SNR shift at unshifted t=0.5.
    assert schedule(tokens, steps)[steps // 2].item() == pytest.approx(midpoint, abs=1e-7)


def test_pretranspose_policy_selects_only_the_40_repeated_down_projections():
    from kinomlx.models.flux2.layout import pretransposed_weight

    selected = {
        key: shape for key, shape in transformer_shapes().items() if pretransposed_weight(key)
    }
    assert len(selected) == 40
    assert sum(key.startswith("transformer_blocks.") for key in selected) == 16
    for shape in selected.values():
        assert len(shape) == 2
        assert shape[0] < shape[1]


@pytest.mark.parametrize("failure", [None, "weight", "latent"])
def test_denoise_checks_adapted_weights_once_and_stops_on_nonfinite(failure):
    from kinomlx.adapters.validation import AdapterWeightCheck, AdapterWeightError
    from kinomlx.modeling.loading import WeightTarget

    class Model:
        def __init__(self):
            self.weight = mx.array(float("inf") if failure == "weight" else 0.25)
            self.adapter_weight_check = AdapterWeightCheck()
            self.adapter_weight_check.add("weight", WeightTarget(self, "weight", ()))
            self.calls = 0

        def __call__(self, image, *args):
            self.calls += 1
            if failure == "latent":
                return mx.full(image.shape, float("nan"))
            return mx.ones_like(image) * self.weight

    model = Model()
    reporter = RecordingReporter()
    if failure:
        with pytest.raises(AdapterWeightError if failure == "weight" else FloatingPointError):
            denoise(model, *_inputs(), recipe=SamplingRecipe(steps=3), reporter=reporter)
        assert model.calls == 1
    else:
        denoise(model, *_inputs(), recipe=SamplingRecipe(steps=3), reporter=reporter)
        assert model.calls == 3
        assert not model.adapter_weight_check.targets
    checks = [event for event in reporter.events if event[:2] == ("start", "check adapter weights")]
    assert len(checks) == 1
    assert reporter.events[-1][:2] == ("end", "denoise")
