"""Native MLX spatial latent upscaler for structurally fitting LTX-2 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import load_weights, read_metadata
from kinomlx.kernels import group_norm, silu
from kinomlx.reporting import NullReporter, Reporter

if TYPE_CHECKING:
    from kinomlx.models.ltx2.components import SpatialUpscalerPort


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


@dataclass(frozen=True)
class SpatialUpscalerConfig:
    """Architecture values serialized beside an LTX spatial upscaler."""

    in_channels: int = 128
    mid_channels: int = 1024
    num_blocks_per_stage: int = 4
    dims: int = 3
    spatial_upsample: bool = True
    temporal_upsample: bool = False
    spatial_scale: float = 2.0
    rational_resampler: bool = False

    def __post_init__(self) -> None:
        _strict_int(self.in_channels, field="in_channels")
        _strict_int(self.mid_channels, field="mid_channels")
        _strict_int(self.num_blocks_per_stage, field="num_blocks_per_stage")
        _strict_int(self.dims, field="dims")
        _strict_bool(self.spatial_upsample, field="spatial_upsample")
        _strict_bool(self.temporal_upsample, field="temporal_upsample")
        _strict_bool(self.rational_resampler, field="rational_resampler")
        if self.dims != 3:
            raise ValueError(f"only 3D latent upscalers are supported, got dims={self.dims}")
        if not self.spatial_upsample or self.temporal_upsample:
            raise ValueError("only spatial-only latent upscalers are supported")
        if self.spatial_scale != 2.0:
            raise ValueError(f"only a 2x spatial scale is supported, got {self.spatial_scale}")
        if self.rational_resampler:
            raise ValueError("rational spatial resampling is not implemented")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> SpatialUpscalerConfig:
        """Read and validate the safetensors metadata configuration."""
        metadata = read_metadata(path)
        try:
            raw = json.loads(metadata["config"])
        except KeyError as exc:
            raise ValueError(f"{path}: spatial upscaler metadata has no config") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: spatial upscaler config is invalid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: spatial upscaler config must be an object")
        rational = raw.get(
            "rational_resampler",
            raw.get("use_rational_resampler", False),
        )
        return cls(
            in_channels=_strict_int(raw.get("in_channels", 128), field="in_channels"),
            mid_channels=_strict_int(raw.get("mid_channels", 1024), field="mid_channels"),
            num_blocks_per_stage=_strict_int(
                raw.get("num_blocks_per_stage", 4),
                field="num_blocks_per_stage",
            ),
            dims=_strict_int(raw.get("dims", 3), field="dims"),
            spatial_upsample=_strict_bool(
                raw.get("spatial_upsample", True),
                field="spatial_upsample",
            ),
            temporal_upsample=_strict_bool(
                raw.get("temporal_upsample", False),
                field="temporal_upsample",
            ),
            spatial_scale=_strict_float(
                raw.get("spatial_scale", 2.0),
                field="spatial_scale",
            ),
            rational_resampler=_strict_bool(
                rational,
                field="rational_resampler",
            ),
        )


class PixelShuffle2d(nn.Module):
    """Rearrange NHWC channels into a two-dimensional higher-resolution grid."""

    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        if upscale_factor <= 0:
            raise ValueError("upscale_factor must be positive")
        self.upscale_factor = upscale_factor

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 4:
            raise ValueError(f"pixel shuffle expects NHWC input, got {tuple(x.shape)}")
        batch, height, width, packed_channels = x.shape
        scale = self.upscale_factor
        area = scale * scale
        if packed_channels % area:
            raise ValueError(f"input channels {packed_channels} are not divisible by {area}")
        channels = packed_channels // area
        x = x.reshape(batch, height, width, channels, scale, scale)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        return x.reshape(batch, height * scale, width * scale, channels)


class ResBlock3d(nn.Module):
    """Post-activation residual block in native BFHWC layout."""

    def __init__(self, channels: int, num_groups: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(
            num_groups,
            channels,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups,
            channels,
            pytorch_compatible=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = silu(group_norm(self.conv1(x), self.norm1))
        x = group_norm(self.conv2(x), self.norm2)
        return silu(x + residual)


class SpatialUpsampler2d(nn.Module):
    """Per-frame Conv2d and pixel shuffle used by the 2x checkpoints."""

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels * scale * scale,
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = PixelShuffle2d(scale)

    def __call__(self, x: mx.array) -> mx.array:
        batch, frames, height, width, channels = x.shape
        x = x.reshape(batch * frames, height, width, channels)
        x = self.pixel_shuffle(self.conv(x))
        return x.reshape(batch, frames, x.shape[1], x.shape[2], channels)


class SpatialUpscaler(nn.Module):
    """Double the height and width of a BCFHW LTX video latent."""

    def __init__(
        self,
        config: SpatialUpscalerConfig | None = None,
        *,
        num_groups: int = 32,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config or SpatialUpscalerConfig()
        if self.config.mid_channels % num_groups:
            raise ValueError("mid_channels must be divisible by num_groups")
        self.compute_dtype = compute_dtype
        self.initial_conv = nn.Conv3d(
            self.config.in_channels,
            self.config.mid_channels,
            kernel_size=3,
            padding=1,
        )
        self.initial_norm = nn.GroupNorm(
            num_groups,
            self.config.mid_channels,
            pytorch_compatible=True,
        )
        self.res_blocks = [
            ResBlock3d(self.config.mid_channels, num_groups)
            for _ in range(self.config.num_blocks_per_stage)
        ]
        self.upsampler = SpatialUpsampler2d(self.config.mid_channels)
        self.post_upsample_res_blocks = [
            ResBlock3d(self.config.mid_channels, num_groups)
            for _ in range(self.config.num_blocks_per_stage)
        ]
        self.final_conv = nn.Conv3d(
            self.config.mid_channels,
            self.config.in_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array:
        if x.ndim != 5 or x.shape[1] != self.config.in_channels:
            raise ValueError(
                "spatial upscaler expects BCFHW input with "
                f"{self.config.in_channels} channels, got {tuple(x.shape)}"
            )
        sink = reporter if reporter is not None else NullReporter()
        phase = "upscale video latent"
        sink.phase_start(
            phase,
            total=len(self.res_blocks) + len(self.post_upsample_res_blocks) + 3,
            unit="block",
        )
        try:
            x = x.transpose(0, 2, 3, 4, 1).astype(self.compute_dtype)
            x = silu(group_norm(self.initial_conv(x), self.initial_norm))
            mx.eval(x)
            sink.phase_advance(phase)
            for block in self.res_blocks:
                x = block(x)
                mx.eval(x)
                sink.phase_advance(phase)
            x = self.upsampler(x)
            mx.eval(x)
            sink.phase_advance(phase)
            for block in self.post_upsample_res_blocks:
                x = block(x)
                mx.eval(x)
                sink.phase_advance(phase)
            x = self.final_conv(x)
            mx.eval(x)
            sink.phase_advance(phase)
            return x.transpose(0, 4, 1, 2, 3)
        finally:
            sink.phase_end(phase)


def upsample_video(
    latent: mx.array,
    upscaler: SpatialUpscalerPort,
    *,
    reporter: Reporter | None = None,
) -> mx.array:
    """Upsample using the lease's tiny VAE-statistics payload."""
    statistics = upscaler.per_channel_statistics
    checkpoint_latent = statistics.denormalize(latent)
    upsampled = upscaler(checkpoint_latent, reporter=reporter)
    result = statistics.normalize(upsampled)
    mx.eval(result)
    return result


def _native_conv_weight(
    value: mx.array,
    expected: tuple[int, ...],
    *,
    key: str,
) -> mx.array:
    if tuple(value.shape) == expected:
        return value
    if value.ndim == 5:
        value = value.transpose(0, 2, 3, 4, 1)
    elif value.ndim == 4:
        value = value.transpose(0, 2, 3, 1)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"spatial upscaler tensor {key!r} has shape {tuple(value.shape)}, expected {expected}"
        )
    return mx.contiguous(value)


def _prepare_value(
    target: nn.Module,
    name: str,
    value: mx.array,
    *,
    key: str,
) -> mx.array:
    expected = tuple(getattr(target, name).shape)
    if name == "weight" and value.ndim in {4, 5}:
        value = _native_conv_weight(value, expected, key=key)
    elif tuple(value.shape) != expected:
        raise ValueError(
            f"spatial upscaler tensor {key!r} has shape {tuple(value.shape)}, expected {expected}"
        )
    return value


def _expected_weight_keys(model: SpatialUpscaler) -> set[str]:
    keys = {
        "initial_conv.weight",
        "initial_conv.bias",
        "initial_norm.weight",
        "initial_norm.bias",
        "upsampler.0.weight",
        "upsampler.0.bias",
        "final_conv.weight",
        "final_conv.bias",
    }
    for prefix, blocks in (
        ("res_blocks", model.res_blocks),
        ("post_upsample_res_blocks", model.post_upsample_res_blocks),
    ):
        for index in range(len(blocks)):
            for layer in ("conv1", "norm1", "conv2", "norm2"):
                keys.add(f"{prefix}.{index}.{layer}.weight")
                keys.add(f"{prefix}.{index}.{layer}.bias")
    return keys


def _promote_normalizers(model: SpatialUpscaler) -> None:
    """Keep low-precision convolutions but give GroupNorm FP32 affine parameters."""
    normalizers = [model.initial_norm]
    for block in [*model.res_blocks, *model.post_upsample_res_blocks]:
        normalizers.extend((block.norm1, block.norm2))
    for layer in normalizers:
        if layer.weight is None or layer.bias is None:
            raise ValueError("spatial upscaler GroupNorm must be affine")
        layer.weight = layer.weight.astype(mx.float32)
        layer.bias = layer.bias.astype(mx.float32)


def _target_for_key(model: SpatialUpscaler, key: str) -> tuple[nn.Module, str]:
    parts = key.split(".")
    if parts[0] in {"initial_conv", "initial_norm", "final_conv"}:
        return getattr(model, parts[0]), parts[1]
    if parts[0] == "upsampler":
        return model.upsampler.conv, parts[2]
    blocks = getattr(model, parts[0])
    return getattr(blocks[int(parts[1])], parts[2]), parts[3]


def _resolve_source_key(weights: dict[str, mx.array], logical_key: str) -> str | None:
    if logical_key in weights:
        return logical_key
    matches = sorted(name for name in weights if name.endswith(f".{logical_key}"))
    return None if not matches else matches[0]


def load_spatial_upscaler_weights(
    model: SpatialUpscaler,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Load every consumed upscaler target and ignore unrelated source tensors."""
    expected = _expected_weight_keys(model)
    sink = reporter if reporter is not None else NullReporter()
    phase = "load spatial upscaler"
    sink.phase_start(phase, total=len(expected), unit="tensor")
    weights: dict[str, mx.array] = {}
    try:
        weights = load_weights(path)
        bindings = {
            key: source_key
            for key in expected
            if (source_key := _resolve_source_key(weights, key)) is not None
        }
        missing = sorted(expected - bindings.keys())
        if missing:
            raise ValueError(
                "unsupported spatial upscaler checkpoint: "
                f"missing {len(missing)} consumed tensors (first: {missing[0]})"
            )

        prepared = {}
        for key in sorted(expected):
            target, attribute = _target_for_key(model, key)
            prepared[key] = (
                target,
                attribute,
                _prepare_value(
                    target,
                    attribute,
                    weights[bindings[key]],
                    key=bindings[key],
                ),
            )
        for key in sorted(expected):
            target, attribute, value = prepared[key]
            setattr(target, attribute, value)
            sink.phase_advance(phase)
        _promote_normalizers(model)
        mx.eval(model.parameters())
        return len(expected)
    finally:
        weights.clear()
        sink.phase_end(phase)


def load_spatial_upscaler(
    path: Path | str,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> SpatialUpscaler:
    """Construct and load an upscaler from checkpoint metadata."""
    model = SpatialUpscaler(
        SpatialUpscalerConfig.from_checkpoint(path),
        compute_dtype=compute_dtype,
    )
    load_spatial_upscaler_weights(model, path, reporter=reporter)
    return model


__all__ = [
    "PixelShuffle2d",
    "ResBlock3d",
    "SpatialUpscaler",
    "SpatialUpscalerConfig",
    "load_spatial_upscaler",
    "load_spatial_upscaler_weights",
    "upsample_video",
]
