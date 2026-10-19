"""The vocoder's one-pass alias-free SnakeBeta against MLX's operations for the same layer."""

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.audio_vae.vocoder_layers import Activation1d
from kinomlx.models.ltx2.kernels import alias_free_snake


def _layer(channels: int, seed: int) -> Activation1d:
    keys = mx.random.split(mx.random.key(seed), 4)
    layer = Activation1d(channels)
    layer.upsample.filter = mx.random.normal((1, 1, 12), key=keys[0]) * 0.3
    layer.upsample.checkpoint_filter_loaded = True
    layer.downsample.lowpass.filter = mx.random.normal((1, 1, 12), key=keys[1]) * 0.3
    layer.downsample.lowpass.checkpoint_filter_loaded = True
    layer.act.alpha = mx.random.normal((channels,), key=keys[2]) * 0.5
    layer.act.beta = mx.random.normal((channels,), key=keys[3]) * 0.5
    return layer


def _graph(layer: Activation1d, x: mx.array) -> mx.array:
    alias_free_snake.set_enabled(False)
    try:
        return layer(x)
    finally:
        alias_free_snake.set_enabled(True)


@pytest.mark.parametrize(
    ("batch", "channels", "length"),
    # One sample, under one threadgroup's outputs, ragged over several, and batched.
    [(1, 3, 1), (1, 4, 37), (1, 5, 1537), (2, 3, 1024)],
)
def test_kernel_matches_the_graph_to_fp32_rounding(batch: int, channels: int, length: int) -> None:
    layer = _layer(channels, seed=length + channels)
    x = mx.random.normal((batch, channels, length), key=mx.random.key(7 * length)) * 2.0
    assert layer._fused(x)
    fused = layer(x)
    expected = _graph(layer, x)
    mx.eval(fused, expected)
    assert fused.shape == expected.shape == x.shape
    assert fused.dtype == mx.float32
    scale = max(float(mx.max(mx.abs(expected)).item()), 1.0)
    assert float(mx.max(mx.abs(fused - expected)).item()) <= 1e-5 * scale


def test_activation_routes_through_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = alias_free_snake.activate

    def recording(*args: mx.array) -> mx.array:
        calls.append(args[0].shape)
        return original(*args)

    monkeypatch.setattr(alias_free_snake, "activate", recording)
    layer = _layer(2, seed=3)
    x = mx.zeros((1, 2, 40))
    mx.eval(layer(x))
    assert calls == [(1, 2, 40)]
    mx.eval(_graph(layer, x))
    assert len(calls) == 1


def test_kernel_leaves_other_inputs_to_the_graph() -> None:
    layer = _layer(2, seed=5)
    assert not layer._fused(mx.zeros((1, 2, 40), dtype=mx.float16))
    assert not layer._fused(mx.zeros((2, 40)))
    assert not layer._fused(mx.zeros((1, 2, 0)))
    unloaded = Activation1d(2)
    assert not unloaded._fused(mx.zeros((1, 2, 40)))
    with pytest.raises(RuntimeError, match="has not been loaded"):
        unloaded(mx.zeros((1, 2, 40)))
    with pytest.raises(ValueError, match="does not apply"):
        alias_free_snake.activate(
            mx.zeros((1, 2, 40), dtype=mx.float16),
            layer.upsample.filter,
            layer.downsample.lowpass.filter,
            layer.act.alpha,
            layer.act.beta,
        )
