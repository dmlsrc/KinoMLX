from __future__ import annotations

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import kinomlx.models.ltx2.upscaler.spatial as spatial
from kinomlx.models.ltx2.upscaler import (
    PixelShuffle2d,
    SpatialUpscaler,
    SpatialUpscalerConfig,
    load_spatial_upscaler_weights,
    upsample_video,
)
from kinomlx.models.ltx2.video_vae.ops import PerChannelStatistics
from kinomlx.reporting import RecordingReporter, Reporter


def _small_model() -> SpatialUpscaler:
    return SpatialUpscaler(
        SpatialUpscalerConfig(
            in_channels=2,
            mid_channels=8,
            num_blocks_per_stage=1,
        ),
        num_groups=4,
        compute_dtype=mx.float32,
    )


def _checkpoint_weights(model: SpatialUpscaler) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for key, value in tree_flatten(model.parameters()):
        checkpoint_key = key.replace("upsampler.conv.", "upsampler.0.")
        if key.endswith(".weight") and value.ndim == 5:
            value = value.transpose(0, 4, 1, 2, 3)
        elif key == "upsampler.conv.weight":
            value = value.transpose(0, 3, 1, 2)
        weights[checkpoint_key] = value
    return weights


def test_pixel_shuffle_has_pytorch_channel_order() -> None:
    x = mx.arange(4, dtype=mx.float32).reshape(1, 1, 1, 4)
    result = PixelShuffle2d(2)(x)
    assert tuple(result.shape) == (1, 2, 2, 1)
    assert result[0, :, :, 0].tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_spatial_upscaler_doubles_only_spatial_axes() -> None:
    model = _small_model()
    x = mx.random.normal((1, 2, 2, 3, 4))
    reporter = RecordingReporter()
    result = model(x, reporter=reporter)
    mx.eval(result)
    assert tuple(result.shape) == (1, 2, 2, 6, 8)
    assert result.dtype == mx.float32
    assert mx.all(mx.isfinite(result)).item()
    assert reporter.events[0] == (
        "start",
        "upscale video latent",
        {"total": 5, "unit": "block"},
    )
    assert reporter.events[-1] == ("end", "upscale video latent", {})


def test_low_precision_forward_uses_fp32_norm_and_precise_activation(monkeypatch) -> None:
    model = SpatialUpscaler(
        SpatialUpscalerConfig(
            in_channels=2,
            mid_channels=8,
            num_blocks_per_stage=1,
        ),
        num_groups=4,
        compute_dtype=mx.bfloat16,
    )
    model.set_dtype(mx.bfloat16)
    normalizers = [model.initial_norm]
    for block in [*model.res_blocks, *model.post_upsample_res_blocks]:
        normalizers.extend((block.norm1, block.norm2))
    for layer in normalizers:
        layer.weight = layer.weight.astype(mx.float32)
        layer.bias = layer.bias.astype(mx.float32)

    norm_dtypes: list[mx.Dtype] = []
    silu_dtypes: list[mx.Dtype] = []
    original_group_norm = spatial.group_norm
    original_silu = spatial.silu

    def recording_group_norm(value, layer):
        norm_dtypes.append(value.dtype)
        return original_group_norm(value, layer)

    def recording_silu(value):
        silu_dtypes.append(value.dtype)
        return original_silu(value)

    monkeypatch.setattr(spatial, "group_norm", recording_group_norm)
    monkeypatch.setattr(spatial, "silu", recording_silu)

    result = model(mx.random.normal((1, 2, 1, 2, 2)).astype(mx.bfloat16))
    mx.eval(result)

    assert result.dtype == mx.bfloat16
    assert norm_dtypes == [mx.bfloat16] * 5
    assert silu_dtypes == [mx.bfloat16] * 5


def test_upsample_video_brackets_raw_model_with_lease_statistics() -> None:
    class Upscaler:
        def __init__(self) -> None:
            self.per_channel_statistics = PerChannelStatistics(2)
            self.per_channel_statistics.mean_of_means = mx.array([3.0, 3.0])
            self.per_channel_statistics.std_of_means = mx.array([2.0, 2.0])

        def __call__(
            self,
            x: mx.array,
            *,
            reporter: Reporter | None = None,
        ) -> mx.array:
            del reporter
            return x * 2

    result = upsample_video(
        mx.ones((1, 2, 1, 1, 1)),
        Upscaler(),  # type: ignore[arg-type]
    )
    mx.eval(result)
    assert mx.allclose(result, mx.full(result.shape, 3.5)).item()


def test_weight_loader_accepts_raw_pytorch_layout(tmp_path) -> None:
    source = _small_model()
    weights = _checkpoint_weights(source)
    consumed = len(weights)
    weights["community.wrapper.unused"] = mx.zeros((1,))
    path = tmp_path / "upscaler.safetensors"
    mx.save_safetensors(
        str(path),
        weights,
        metadata={
            "config": json.dumps(
                {
                    "_class_name": "CommunityWrappedUpsampler",
                    "in_channels": 2,
                    "mid_channels": 8,
                    "num_blocks_per_stage": 1,
                    "dims": 3,
                    "spatial_upsample": True,
                    "temporal_upsample": False,
                    "spatial_scale": 2.0,
                }
            )
        },
    )
    target = _small_model()
    reporter = RecordingReporter()
    assert load_spatial_upscaler_weights(target, path, reporter=reporter) == consumed

    x = mx.random.normal((1, 2, 1, 2, 2))
    expected = source(x)
    actual = target(x)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected).item()
    assert reporter.events[0] == (
        "start",
        "load spatial upscaler",
        {"total": consumed, "unit": "tensor"},
    )
    assert reporter.events[-1] == ("end", "load spatial upscaler", {})


def test_weight_loader_keeps_convolutions_low_precision_and_promotes_norms(
    tmp_path,
) -> None:
    source = _small_model()
    weights = {key: value.astype(mx.bfloat16) for key, value in _checkpoint_weights(source).items()}
    path = tmp_path / "upscaler.safetensors"
    mx.save_safetensors(
        str(path),
        weights,
        metadata={
            "config": json.dumps(
                {
                    "_class_name": "LatentUpsampler",
                    "in_channels": 2,
                    "mid_channels": 8,
                    "num_blocks_per_stage": 1,
                    "dims": 3,
                    "spatial_upsample": True,
                    "temporal_upsample": False,
                    "spatial_scale": 2.0,
                }
            )
        },
    )

    target = _small_model()
    load_spatial_upscaler_weights(target, path)

    assert target.initial_conv.weight.dtype == mx.bfloat16
    assert target.upsampler.conv.weight.dtype == mx.bfloat16
    assert target.final_conv.weight.dtype == mx.bfloat16
    normalizers = [target.initial_norm]
    for block in [*target.res_blocks, *target.post_upsample_res_blocks]:
        normalizers.extend((block.norm1, block.norm2))
    assert all(layer.weight.dtype == mx.float32 for layer in normalizers)
    assert all(layer.bias.dtype == mx.float32 for layer in normalizers)


def test_weight_loader_rejects_partial_checkpoint(tmp_path) -> None:
    path = tmp_path / "partial.safetensors"
    mx.save_safetensors(str(path), {"initial_conv.bias": mx.zeros((8,))})
    reporter = RecordingReporter()
    with pytest.raises(ValueError, match="missing"):
        load_spatial_upscaler_weights(_small_model(), path, reporter=reporter)
    assert reporter.events[-1] == ("end", "load spatial upscaler", {})


def test_weight_loader_preflights_every_shape_before_mutation(tmp_path) -> None:
    source = _small_model()
    weights = _checkpoint_weights(source)
    weights["final_conv.weight"] = mx.zeros((1,))
    path = tmp_path / "wrong-shape.safetensors"
    mx.save_safetensors(str(path), weights)

    target = _small_model()
    original_initial_bias = target.initial_conv.bias
    with pytest.raises(ValueError, match="final_conv.weight.*shape"):
        load_spatial_upscaler_weights(target, path)

    assert target.initial_conv.bias is original_initial_bias
