"""Small real-file Klein admission, fresh fusion, bias, and exclusion controls."""

import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.adapters.live import live_adapters
from kinomlx.adapters.schedule import AdapterSchedule
from kinomlx.adapters.selection import AdapterSelection
from kinomlx.adapters.source import inspect_adapter_source
from kinomlx.io.safetensors import save_weights
from kinomlx.modeling.layers import linear_shell
from kinomlx.models.flux2.adapters import loading


@pytest.fixture
def graph(monkeypatch, tmp_path):
    shapes = {"x_embedder.weight": (6, 4), "norm_out.linear.weight": (12, 4)}
    monkeypatch.setattr(loading, "transformer_shapes", lambda: dict(shapes))
    return shapes, tmp_path


def model_and_folder(graph, transposed):
    shapes, folder = graph
    layers = {}
    for name, shape in shapes.items():
        layer = linear_shell(bias=False, pretransposed=transposed)
        value = mx.arange(shape[0] * shape[1]).reshape(shape).astype(mx.float16) / 100
        layer.weight = mx.contiguous(value.T) if transposed else value
        layers[name] = layer
    model = SimpleNamespace(
        x_embedder=layers["x_embedder.weight"],
        norm_out=SimpleNamespace(linear=layers["norm_out.linear.weight"]),
    )
    return model, folder


def adapter(folder, name="adapter.safetensors", *, bias=False, bad=False):
    path = folder / name
    weights = {
        "transformer.x_embedder.lora_A.weight": mx.full((2, 4), 0.1),
        "transformer.x_embedder.lora_B.weight": mx.full((6 if not bad else 5, 2), 0.2),
    }
    if bias:
        weights["transformer.x_embedder.lora_B.bias"] = mx.full((6,), 0.3)
    save_weights(path, weights)
    return AdapterSelection(path, 0.7)


@pytest.mark.parametrize("transposed", [False, True])
def test_fresh_fusion_bias_and_stack_order(graph, transposed):
    model, folder = model_and_folder(graph, transposed)
    original = model.x_embedder.weight
    first = adapter(folder, bias=True)
    second = replace(adapter(folder, "second.safetensors"), strength=-0.2)
    plan = loading.prepare_flux2_adapters((first, second))
    receipts = loading.apply_flux2_adapters(
        model,
        plan,
        variant="base",
    )
    result = model.x_embedder.weight.T if transposed else model.x_embedder.weight
    base = original.T if transposed else original
    assert mx.allclose(result, (base.astype(mx.float32) + 0.02).astype(mx.float16)).item()
    assert mx.allclose(model.x_embedder.bias, mx.full((6,), 0.21, dtype=mx.float16)).item()
    assert all(receipt.coverage == 1.0 for receipt in receipts)
    other, folder = model_and_folder(graph, transposed)
    loading.apply_flux2_adapters(
        other,
        tuple(reversed(plan)),
        variant="base",
    )
    assert mx.array_equal(other.x_embedder.weight, model.x_embedder.weight).item()
    fresh, _ = model_and_folder(graph, transposed)
    assert mx.array_equal(fresh.x_embedder.weight, original).item()
    assert getattr(fresh.x_embedder, "bias", None) is None


def test_a_constant_step_array_fuses_exactly_like_one_strength(graph):
    # "0.7,0.7,0.7,0.7" is the obvious spelling of "every step" once the flag is
    # known, and it asks for nothing a constant cannot express.
    scheduled_model, folder = model_and_folder(graph, False)
    plain = adapter(folder)
    constant = replace(plain, strength=1.0, schedule=AdapterSchedule((0.7,) * 4))
    (receipt,) = loading.apply_flux2_adapters(
        scheduled_model, loading.prepare_flux2_adapters((constant,)), variant="base"
    )
    plain_model, _ = model_and_folder(graph, False)
    loading.apply_flux2_adapters(
        plain_model, loading.prepare_flux2_adapters((plain,)), variant="base"
    )

    assert live_adapters(scheduled_model) is None
    assert mx.array_equal(scheduled_model.x_embedder.weight, plain_model.x_embedder.weight).item()
    assert receipt.step_strengths == (0.7, 0.7, 0.7, 0.7)
    assert receipt.to_metadata()["application"] == "fused"


def test_source_mutation_rejected_before_fusion(graph):
    model, folder = model_and_folder(graph, False)
    selected = adapter(folder)
    plan = loading.prepare_flux2_adapters((selected,))
    save_weights(selected.path, {"other": mx.zeros((2,))})
    with pytest.raises(ValueError, match="changed"):
        loading.apply_flux2_adapters(
            model,
            plan,
            variant="base",
        )


def test_exclusions_and_zero_strength_are_noops(graph):
    model, folder = model_and_folder(graph, False)
    original = model.x_embedder.weight
    selected = adapter(folder)
    for item in (replace(selected, strength=0), replace(selected, exclude=("input",))):
        receipts = loading.apply_flux2_adapters(
            model,
            loading.prepare_flux2_adapters((item,)),
            variant="base",
        )
        assert not receipts[0].effective
        assert model.x_embedder.weight is original
    with pytest.raises(ValueError, match="shape"):
        loading.prepare_flux2_adapters((adapter(folder, "bad.safetensors", bad=True),))


@pytest.mark.parametrize("component_prefix", ["", "transformer."])
def test_embedded_diffusers_config_preserves_rank_and_alpha(tmp_path, component_prefix):
    selected = adapter(tmp_path)
    weights = {
        "transformer.x_embedder.lora_A.weight": mx.full((2, 4), 0.1),
        "transformer.x_embedder.lora_B.weight": mx.full((6, 2), 0.2),
    }
    config = {"peft_type": "LORA", "r": 2, "lora_alpha": 3}
    save_weights(
        selected.path,
        weights,
        {
            "lora_adapter_metadata": json.dumps(
                {component_prefix + key: value for key, value in config.items()}
            )
        },
    )
    source = inspect_adapter_source(selected, model="Klein", categories=frozenset())
    assert source.config == config
    op = loading.parse_configured_adapter(weights, source.config).operations[0]
    assert op.alpha == 3
    save_weights(selected.path, weights, {"lora_adapter_metadata": "[]"})
    with pytest.raises(ValueError, match="declare peft_type"):
        inspect_adapter_source(selected, model="Klein", categories=frozenset())


def test_lazy_fusion_does_not_reduce_base_weights(graph, monkeypatch):
    model, folder = model_and_folder(graph, False)
    selected = adapter(folder)
    prepared = loading.prepare_flux2_adapters((selected,))
    original_max = mx.max

    def bounded_max(value, *args, **kwargs):
        assert value.shape != (6, 4), "ordinary safe fusion reduced a full base weight"
        return original_max(value, *args, **kwargs)

    monkeypatch.setattr(mx, "max", bounded_max)
    receipts = loading.apply_flux2_adapters(
        model,
        prepared,
        variant="distilled",
    )
    assert receipts[0].effective


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("kind", ["lora", "lokr", "stack"])
def test_source_and_normal_layout_fusion_preserve_rounding(tmp_path, monkeypatch, dtype, kind):
    from kinomlx.adapters.validation import check_adapter_weights
    from kinomlx.models.flux2 import loading as components
    from kinomlx.reporting import NullReporter

    key = "transformer_blocks.0.ff.linear_out.weight"
    shapes = {key: (6, 4)}
    monkeypatch.setattr(components, "transformer_shapes", lambda: shapes)
    monkeypatch.setattr(loading, "transformer_shapes", lambda: shapes)
    source = tmp_path / "source.safetensors"
    base = (mx.arange(24).reshape(6, 4) / 7).astype(mx.bfloat16)
    save_weights(source, {key: base})
    selections = []
    for fmt in ("lora", "lokr") if kind == "stack" else (kind,):
        prefix = key.removesuffix(".weight")
        factors = (
            {"lora_A.weight": mx.full((2, 4), 0.1), "lora_B.weight": mx.full((6, 2), 0.2)}
            if fmt == "lora"
            else {"lokr_w1": mx.full((2, 2), 0.1), "lokr_w2": mx.full((3, 2), 0.2)}
        )
        path = tmp_path / (fmt + ".safetensors")
        save_weights(path, {prefix + "." + k: v for k, v in factors.items()})
        selections.append(AdapterSelection(path, 0.7))
    plan = loading.prepare_flux2_adapters(tuple(selections))
    source_model = components.load_transformer((source,), dtype=dtype)
    # Compare source-loaded pretransposed weights with normal-layout weights.
    normal_model = components.Flux2Transformer()
    reference = components._target(normal_model, key, (6, 4))
    setattr(reference.owner, reference.attribute, base.astype(dtype))
    for model in (source_model, normal_model):
        loading.apply_flux2_adapters(model, plan, variant="distilled")
        layer = components._target(model, key, (6, 4)).owner
        mx.eval(layer(mx.ones((1, 4), dtype=dtype)))
        check_adapter_weights(model, context="test", reporter=NullReporter())
        assert not model.adapter_weight_check.targets
    assert mx.array_equal(
        components._target(source_model, key, ()).owner.weight.T,
        components._target(normal_model, key, ()).owner.weight,
    ).item()
