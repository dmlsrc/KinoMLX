"""Reference attention semantics, owned storage, and invocation-local lifetime."""

import gc

import mlx.core as mx
import pytest

from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.kv import LayerKV, ReferenceKV, blend_modulation, reference_attention
from kinomlx.models.flux2.sampling import SamplingRecipe, denoise
from tests._reporting import RecordingReporter


def test_reference_attention_matches_dense_mask_and_cached_recomputation():
    q, k, v = [mx.random.normal((1, 9, 2, 8), key=mx.random.key(i)) for i in range(3)]
    state = ReferenceKV(3, 1)
    actual = reference_attention(q, k, v, text_tokens=2, state=state, layer=0)
    scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) / 8**0.5
    positions = mx.arange(9)
    reference = (positions >= 2) & (positions < 5)
    allowed = ~reference[:, None] | reference[None, :]
    expected = (
        mx.softmax(mx.where(allowed, scores, -mx.inf), axis=-1) @ v.transpose(0, 2, 1, 3)
    ).transpose(0, 2, 1, 3)
    mx.eval(actual, *state.tensors())
    assert mx.max(mx.abs(actual - expected)).item() < 1e-5
    state.extracting = False
    active = mx.array([0, 1, 5, 6, 7, 8])
    cached = reference_attention(
        mx.take(q, active, axis=1),
        mx.take(k, active, axis=1),
        mx.take(v, active, axis=1),
        text_tokens=2,
        state=state,
        layer=0,
    )
    assert mx.max(mx.abs(cached - mx.take(actual, active, axis=1))).item() < 1e-5
    # Reference outputs cannot depend on current text or target tokens.
    changed = mx.where(reference[None, :, None, None], v, v + 100)
    other = reference_attention(q, k, changed, text_tokens=2, state=ReferenceKV(3, 1), layer=0)
    assert mx.max(mx.abs(other[:, 2:5] - actual[:, 2:5])).item() == 0


def test_reference_modulation_only_replaces_reference_positions():
    current, fixed = mx.array([[1.0, 2.0]]), mx.array([[9.0, 10.0]])
    result = blend_modulation(current, fixed, 2, 3, 7)
    assert result.tolist() == [
        [[1.0, 2.0], [1.0, 2.0], [9.0, 10.0], [9.0, 10.0], [9.0, 10.0], [1.0, 2.0], [1.0, 2.0]]
    ]


def test_one_reference_row_does_not_retain_full_projection_storage():
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    source = mx.random.normal((1, 8192, 4, 16))
    mx.eval(source)
    layer = LayerKV()
    layer.store(source, source, 10, 1)
    mx.eval(*layer.tensors())
    del source
    gc.collect()
    mx.clear_cache()
    assert mx.get_active_memory() - before < 64 * 1024
    assert sum(t.nbytes for t in layer.tensors()) == 512


def test_extracting_many_layers_releases_each_full_projection():
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    state = ReferenceKV(1, 16)
    for index, layer in enumerate(state.layers):
        source = mx.random.normal((1, 8192, 4, 16), key=mx.random.key(index))
        mx.eval(source)
        layer.store(source, source, 10, 1)
        del source
    gc.collect()
    # Unsettled gather graphs would retain 16 full 2 MiB projections here,
    # although the reference cache itself contains only 8 KiB of values.
    assert mx.get_active_memory() - before < 128 * 1024
    state.clear()


class _Model:
    def __init__(self, fail=None, fail_at=2):
        self.caches = []
        self.calls = []
        self.fail = fail
        self.fail_at = fail_at
        self.scale = 1.0

    def new_reference_kv(self, count):
        state = ReferenceKV(count, 1)
        self.caches.append(state)
        return state

    def __call__(self, image, text, timestep, ids, tids, *, reference_kv=None):
        self.calls.append((image.shape[1], reference_kv.extracting if reference_kv else None))
        if reference_kv is None:
            return mx.ones_like(image) * self.scale
        if reference_kv.extracting:
            reference = image[:, : reference_kv.reference_tokens, :1, None] * self.scale
            reference_kv.layers[0].store(reference, reference, 0, reference.shape[1])
            image = image[:, reference_kv.reference_tokens :]
        if self.fail and len(self.calls) == self.fail_at:
            raise self.fail("denoise interrupted")
        # A previously fused adapter changes extraction, with no adapter logic
        # in cache ownership. Subsequent calls must observe the new weights.
        return mx.ones_like(image) * mx.mean(reference_kv.layers[0].tensors()[0])


def _run(model, *, steps=4, refs=True, recompute=False, on_step=None, reference_value=2.0):
    reporter = RecordingReporter()
    result = denoise(
        model,
        mx.zeros((1, 4, 128)),
        mx.ones((1, 2, 12288)),
        mx.zeros((1, 4, 4)),
        mx.zeros((1, 2, 4)),
        references=mx.full((1, 3, 128), reference_value) if refs else None,
        reference_ids=mx.zeros((1, 3, 4)) if refs else None,
        recipe=SamplingRecipe(Flux2Variant.KV, steps=steps, recompute_reference_kv=recompute),
        reporter=reporter,
        on_step=on_step,
    )
    return result, reporter.events


@pytest.mark.parametrize("steps", [1, 4])
def test_extraction_reuse_counts_and_repeated_generation_isolation(steps):
    model = _Model()
    first, events = _run(model, steps=steps)
    assert model.calls == [(7, True)] + [(4, False)] * (steps - 1)
    assert sum(e[0] == "start" and e[1] == "kv_extract" for e in events) == 1
    assert sum(e[0] == "start" and e[1] == "kv_reuse" for e in events) == steps - 1
    again, _ = _run(model, steps=steps)
    assert mx.array_equal(first, again).item()
    model.scale = 3.0
    changed, _ = _run(model, steps=steps)
    assert not mx.array_equal(first, changed).item()
    assert len({id(c) for c in model.caches}) == 3
    assert all(c.closed and c.layers[0].key is None for c in model.caches)
    changed_reference, _ = _run(model, steps=steps, reference_value=5.0)
    assert not mx.array_equal(changed, changed_reference).item()


def test_recomputation_and_no_reference_paths():
    cached, _ = _run(_Model())
    model = _Model()
    recomputed, _ = _run(model, recompute=True)
    assert mx.array_equal(cached, recomputed).item()
    assert model.calls == [(7, True)] * 4
    no_ref = _Model()
    _run(no_ref, refs=False)
    assert not no_ref.caches
    assert no_ref.calls == [(4, None)] * 4


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("step", [1, 2])
def test_failure_and_cancellation_clear_materialized_cache(error, step):
    model = _Model(fail=error, fail_at=step)
    with pytest.raises(error):
        _run(model)
    assert model.caches[0].closed
    assert model.caches[0].layers[0].key is None


def test_callback_failure_also_releases_cache():
    model = _Model()

    def fail(*args):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError):
        _run(model, on_step=fail)
    assert model.caches[0].closed
    assert model.caches[0].layers[0].value is None


def test_a_kv_edit_refusal_names_a_control_the_reader_can_reach():
    """F-23: the remedy has to exist where the reader is, and no flag enables it."""
    from kinomlx.adapters.live import LiveAdapters, LiveAdapterTerm
    from kinomlx.adapters.schedule import AdapterSchedule
    from kinomlx.model_registry import config_registry

    model = _Model()
    model.live_adapters = LiveAdapters(
        (
            LiveAdapterTerm(
                order_key="style.safetensors",
                kind="lora",
                target="block.0.attn.to_q.weight",
                plan=AdapterSchedule((1.0, 0.0, 1.0, 0.0)),
                apply=lambda _value, result, _strength: result,
                retained_bytes=0,
            ),
        )
    )
    with pytest.raises(ValueError, match="reuses reference K/V") as error:
        _run(model)
    message = str(error.value)
    assert "Fuse the adapter at one constant strength" in message
    assert "enable recompute_reference_kv (--recompute-reference-kv)" in message
    assert "Enable reference K/V recomputation" not in message
    # The control the message names is a registered Klein input on every surface.
    schema = config_registry().model("flux2")
    field = schema.field(("generate", "recompute_reference_kv"))
    assert field.cli is not None
    assert field.cli_dest == "recompute_reference_kv"
    from kinomlx.models.flux2.arguments import build_parser

    flags = {option for action in build_parser()._actions for option in action.option_strings}
    assert "--recompute-reference-kv" in flags
