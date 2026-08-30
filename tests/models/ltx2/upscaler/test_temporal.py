from __future__ import annotations

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from kinomlx.models.ltx2.upscaler.temporal import (
    PixelShuffle1d,
    TemporalUpscaler,
    TemporalUpscalerConfig,
    load_temporal_upscaler_weights,
)


def _small_model() -> TemporalUpscaler:
    return TemporalUpscaler(
        TemporalUpscalerConfig(
            in_channels=2,
            mid_channels=8,
            num_blocks_per_stage=1,
        ),
        num_groups=4,
        compute_dtype=mx.float32,
    )


def _checkpoint_weights(model: TemporalUpscaler) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for key, value in tree_flatten(model.parameters()):
        checkpoint_key = key.replace("upsampler.conv.", "upsampler.0.")
        if key.endswith(".weight") and value.ndim == 5:
            value = value.transpose(0, 4, 1, 2, 3)
        weights[checkpoint_key] = value
    return weights


def test_temporal_pixel_shuffle_has_pytorch_channel_order() -> None:
    packed = mx.arange(8, dtype=mx.float32).reshape(1, 1, 1, 1, 8)
    result = PixelShuffle1d(2)(packed)
    assert tuple(result.shape) == (1, 2, 1, 1, 4)
    assert result[0, :, 0, 0, :].tolist() == [
        [0.0, 2.0, 4.0, 6.0],
        [1.0, 3.0, 5.0, 7.0],
    ]


def test_temporal_upscaler_outputs_two_f_minus_one() -> None:
    result = _small_model()(mx.random.normal((1, 2, 3, 2, 2)))
    mx.eval(result)
    assert tuple(result.shape) == (1, 2, 5, 2, 2)
    assert mx.all(mx.isfinite(result)).item()


def test_temporal_loader_accepts_pytorch_layout_and_rejects_partial(tmp_path) -> None:
    source = _small_model()
    weights = _checkpoint_weights(source)
    path = tmp_path / "temporal.safetensors"
    mx.save_safetensors(
        str(path),
        weights,
        metadata={
            "config": json.dumps(
                {
                    "in_channels": 2,
                    "mid_channels": 8,
                    "num_blocks_per_stage": 1,
                    "dims": 3,
                    "spatial_upsample": False,
                    "temporal_upsample": True,
                    "spatial_scale": 1.0,
                    "rational_resampler": True,
                }
            )
        },
    )
    target = _small_model()
    assert load_temporal_upscaler_weights(target, path) == len(weights)
    x = mx.random.normal((1, 2, 2, 2, 2))
    expected = source(x)
    actual = target(x)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected).item()

    partial = tmp_path / "partial.safetensors"
    mx.save_safetensors(str(partial), {"initial_conv.bias": mx.zeros((8,))})
    with pytest.raises(ValueError, match="missing"):
        load_temporal_upscaler_weights(_small_model(), partial)
