"""Prepared conditioning lifetime, CFG, and targeted fusion precision controls."""

from functools import partial

import mlx.core as mx
import pytest

from kinomlx.models.flux2 import sampling, transformer
from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.kv import ReferenceKV
from tests._reporting import RecordingReporter


class PreparedProbe:
    def __init__(self, reporter):
        self.reporter = reporter
        self.preparations = []
        self.calls = []
        self.caches = []

    def prepare_velocities(self, text, image_ids, text_ids):
        assert self.reporter.events[-1][0:2] == ("start", "denoise")
        self.preparations.append((text, image_ids))
        return tuple(partial(self.predict, text, ids) for ids in image_ids)

    def new_reference_kv(self, count):
        cache = ReferenceKV(count, 1)
        self.caches.append(cache)
        return cache

    def predict(self, text, ids, image, timestep, cache):
        assert ids.shape[1] == image.shape[1]
        self.calls.append((image.shape[1], text.shape[0]))
        if cache is not None and cache.extracting:
            count = cache.reference_tokens
            key = image[:, :count, :1, None]
            cache.layers[0].store(key, key, 0, count)
            image = image[:, count:]
        return mx.ones_like(image) * text[:, :1, :1]

    def __call__(self, *args, **kwargs):
        raise AssertionError("prepared model must use its generation callables")


@pytest.mark.parametrize("mode", ["sequential", "batched"])
def test_prepared_cfg_and_reference_order_are_generation_owned(mode):
    reporter = RecordingReporter()
    model = PreparedProbe(reporter)
    for value in (1.0, 2.0):
        result = sampling.denoise(
            model,
            mx.zeros((1, 4, 128)),
            mx.full((1, 2, 12288), value),
            mx.zeros((1, 4, 4)),
            mx.zeros((1, 2, 4)),
            negative=mx.full((1, 2, 12288), -value),
            references=mx.ones((1, 3, 128)),
            reference_ids=mx.full((1, 3, 4), 7),
            recipe=sampling.SamplingRecipe(Flux2Variant.BASE, 4, 4, mode),
            reporter=reporter,
        )
        assert mx.max(mx.abs(result + 7 * value)).item() < 1e-5
    assert len(model.preparations) == (4 if mode == "sequential" else 2)
    for _, layouts in model.preparations:
        assert len(layouts) == 1
        assert mx.all(layouts[0][:, 4:] == 7).item()
    expected_calls = [(7, 1)] * 16 if mode == "sequential" else [(7, 2)] * 8
    assert model.calls == expected_calls


@pytest.mark.parametrize("cancel", [False, True])
def test_kv_prepares_text_once_for_both_layouts_and_cleans_up(cancel):
    reporter = RecordingReporter()
    model = PreparedProbe(reporter)

    def on_step(index, current, velocity):
        if cancel:
            raise KeyboardInterrupt

    def run():
        return sampling.denoise(
            model,
            mx.zeros((1, 4, 128)),
            mx.ones((1, 2, 12288)),
            mx.zeros((1, 4, 4)),
            mx.zeros((1, 2, 4)),
            references=mx.ones((1, 3, 128)),
            reference_ids=mx.full((1, 3, 4), 7),
            recipe=sampling.SamplingRecipe(Flux2Variant.KV),
            reporter=reporter,
            on_step=on_step,
        )

    if cancel:
        with pytest.raises(KeyboardInterrupt):
            run()
    else:
        run()
        assert model.calls == [(7, 1)] + [(4, 1)] * 3
    assert len(model.preparations) == 1
    _, layouts = model.preparations[0]
    assert [ids.shape[1] for ids in layouts] == [7, 4]
    assert mx.all(layouts[0][:, :3] == 7).item()
    assert all(cache.closed and cache.layers[0].key is None for cache in model.caches)
    assert reporter.events[-1][0:2] == ("end", "denoise")


def test_transformer_prepares_one_projection_shared_by_both_layouts():
    class Probe(transformer.Flux2Transformer):
        def _forward(self, text, positions, image, timestep, reference_kv):
            return text + mx.sum(positions[0])

    model = Probe()
    model.context_embedder.weight = mx.ones((3, 4))
    text = mx.ones((1, 2, 4))
    text_ids = mx.zeros((1, 2, 4))
    ids = (mx.zeros((1, 3, 4)), mx.ones((1, 1, 4)))
    prepared = model.prepare_velocities(text, ids, text_ids)
    # Mutating the model after preparation must not retroactively change this
    # generation's evaluated projection; the next generation sees new weights.
    model.context_embedder.weight = mx.full((3, 4), 2.0)
    again = model.prepare_velocities(text, ids, text_ids)
    for before, after in zip(prepared, again, strict=True):
        assert mx.all(after(None, None, None) - before(None, None, None) == 4).item()
    assert prepared[0].args[0] is prepared[1].args[0]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_targeted_fusions_preserve_varied_inputs(dtype):
    value = mx.random.normal((1, 17, 32, 128), key=mx.random.key(42)).astype(dtype)
    positions = transformer.rotary_positions(mx.arange(68).reshape(1, 17, 4) / 7)
    source = transformer.rotate(value, positions)
    actual = transformer._compiled_rotate(value, positions)
    delta = actual.astype(mx.float32) - source.astype(mx.float32)
    assert mx.max(mx.abs(delta)).item() < (1e-6 if dtype == mx.float32 else 0.001)
    shift = mx.random.normal((1, 1, 4096), key=mx.random.key(43)).astype(dtype)
    scale = mx.random.normal((1, 1, 4096), key=mx.random.key(44)).astype(dtype)
    value = value.reshape(1, 17, 4096)
    source = transformer.normalized(value, shift, scale)
    actual = transformer._compiled_normalized(value, shift, scale)
    assert mx.array_equal(source, actual).item()
