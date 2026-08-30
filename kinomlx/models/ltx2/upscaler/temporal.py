"""Native temporal x2 latent upscaler component for LTX-2.5 artifacts."""

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

from .spatial import ResBlock3d

if TYPE_CHECKING:
    from kinomlx.models.ltx2.components import TemporalUpscalerPort


@dataclass(frozen=True)
class TemporalUpscalerConfig:
    in_channels: int = 128
    mid_channels: int = 512
    num_blocks_per_stage: int = 4
    dims: int = 3
    spatial_upsample: bool = False
    temporal_upsample: bool = True
    spatial_scale: float = 1.0
    rational_resampler: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("in_channels", self.in_channels),
            ("mid_channels", self.mid_channels),
            ("num_blocks_per_stage", self.num_blocks_per_stage),
            ("dims", self.dims),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dims != 3:
            raise ValueError("temporal upscaling requires 3D convolutions")
        if self.spatial_upsample or not self.temporal_upsample:
            raise ValueError("only temporal-only latent upscalers are supported")
        if self.spatial_scale != 1.0:
            raise ValueError("temporal-only latent upscalers must preserve spatial size")
        if not isinstance(self.rational_resampler, bool):
            raise ValueError("rational_resampler must be a boolean")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> TemporalUpscalerConfig:
        metadata = read_metadata(path)
        try:
            raw = json.loads(metadata["config"])
        except KeyError as exc:
            raise ValueError(f"{path}: temporal upscaler metadata has no config") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: temporal upscaler config is invalid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: temporal upscaler config must be an object")
        rational_resampler = raw.get(
            "rational_resampler",
            raw.get("use_rational_resampler", True),
        )
        if not isinstance(rational_resampler, bool):
            raise ValueError(f"{path}: rational_resampler must be a boolean")
        return cls(
            in_channels=int(raw.get("in_channels", 128)),
            mid_channels=int(raw.get("mid_channels", 512)),
            num_blocks_per_stage=int(raw.get("num_blocks_per_stage", 4)),
            dims=int(raw.get("dims", 3)),
            spatial_upsample=raw.get("spatial_upsample", False),
            temporal_upsample=raw.get("temporal_upsample", True),
            spatial_scale=float(raw.get("spatial_scale", 1.0)),
            rational_resampler=rational_resampler,
        )


class PixelShuffle1d(nn.Module):
    """Rearrange BFHWC channels into a higher-resolution temporal axis."""

    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        if upscale_factor <= 0:
            raise ValueError("upscale_factor must be positive")
        self.upscale_factor = upscale_factor

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim != 5:
            raise ValueError(f"temporal pixel shuffle expects BFHWC input, got {value.shape}")
        batch, frames, height, width, packed_channels = value.shape
        scale = self.upscale_factor
        if packed_channels % scale:
            raise ValueError(f"input channels {packed_channels} are not divisible by {scale}")
        channels = packed_channels // scale
        value = value.reshape(batch, frames, height, width, channels, scale)
        value = value.transpose(0, 1, 5, 2, 3, 4)
        return value.reshape(batch, frames * scale, height, width, channels)


class TemporalUpsampler1d(nn.Module):
    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle1d(scale)

    def __call__(self, value: mx.array) -> mx.array:
        return self.pixel_shuffle(self.conv(value))


class TemporalUpscaler(nn.Module):
    """Double latent time and drop the duplicated causal boundary frame."""

    def __init__(
        self,
        config: TemporalUpscalerConfig | None = None,
        *,
        num_groups: int = 32,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config or TemporalUpscalerConfig()
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
        self.upsampler = TemporalUpsampler1d(self.config.mid_channels)
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

    def __call__(self, value: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        if value.ndim != 5 or value.shape[1] != self.config.in_channels:
            raise ValueError(
                "temporal upscaler expects BCFHW input with "
                f"{self.config.in_channels} channels, got {tuple(value.shape)}"
            )
        sink = reporter if reporter is not None else NullReporter()
        phase = "temporally upscale video latent"
        sink.phase_start(
            phase,
            total=len(self.res_blocks) + len(self.post_upsample_res_blocks) + 3,
            unit="block",
        )
        try:
            value = value.transpose(0, 2, 3, 4, 1).astype(self.compute_dtype)
            value = silu(group_norm(self.initial_conv(value), self.initial_norm))
            mx.eval(value)
            sink.phase_advance(phase)
            for block in self.res_blocks:
                value = block(value)
                mx.eval(value)
                sink.phase_advance(phase)
            value = self.upsampler(value)[:, 1:]
            mx.eval(value)
            sink.phase_advance(phase)
            for block in self.post_upsample_res_blocks:
                value = block(value)
                mx.eval(value)
                sink.phase_advance(phase)
            value = self.final_conv(value)
            mx.eval(value)
            sink.phase_advance(phase)
            return value.transpose(0, 4, 1, 2, 3)
        finally:
            sink.phase_end(phase)


def temporal_upsample_video(
    latent: mx.array,
    upscaler: TemporalUpscalerPort,
    *,
    reporter: Reporter | None = None,
) -> mx.array:
    statistics = upscaler.per_channel_statistics
    checkpoint_latent = statistics.denormalize(latent)
    result = statistics.normalize(upscaler(checkpoint_latent, reporter=reporter))
    mx.eval(result)
    return result


def _expected_weight_keys(model: TemporalUpscaler) -> set[str]:
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


def _target_for_key(model: TemporalUpscaler, key: str) -> tuple[nn.Module, str]:
    parts = key.split(".")
    if parts[0] in {"initial_conv", "initial_norm", "final_conv"}:
        return getattr(model, parts[0]), parts[1]
    if parts[0] == "upsampler":
        return model.upsampler.conv, parts[2]
    blocks = getattr(model, parts[0])
    return getattr(blocks[int(parts[1])], parts[2]), parts[3]


def _source_key(weights: dict[str, mx.array], logical: str) -> str | None:
    if logical in weights:
        return logical
    matches = sorted(name for name in weights if name.endswith(f".{logical}"))
    return None if not matches else matches[0]


def _prepare_value(target: nn.Module, name: str, value: mx.array, *, key: str) -> mx.array:
    expected = tuple(getattr(target, name).shape)
    if name == "weight" and value.ndim == 5:
        value = value.transpose(0, 2, 3, 4, 1)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"temporal upscaler tensor {key!r} has shape {tuple(value.shape)}, expected {expected}"
        )
    return mx.contiguous(value) if value.ndim == 5 else value


def _promote_normalizers(model: TemporalUpscaler) -> None:
    normalizers = [model.initial_norm]
    for block in [*model.res_blocks, *model.post_upsample_res_blocks]:
        normalizers.extend((block.norm1, block.norm2))
    for layer in normalizers:
        if layer.weight is None or layer.bias is None:
            raise ValueError("temporal upscaler GroupNorm must be affine")
        layer.weight = layer.weight.astype(mx.float32)
        layer.bias = layer.bias.astype(mx.float32)


def load_temporal_upscaler_weights(
    model: TemporalUpscaler,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    expected = _expected_weight_keys(model)
    sink = reporter if reporter is not None else NullReporter()
    phase = "load temporal latent upscaler"
    sink.phase_start(phase, total=len(expected), unit="tensor")
    weights: dict[str, mx.array] = {}
    try:
        weights = load_weights(path)
        bindings = {
            key: source for key in expected if (source := _source_key(weights, key)) is not None
        }
        missing = sorted(expected - bindings.keys())
        if missing:
            raise ValueError(
                "unsupported temporal upscaler checkpoint: "
                f"missing {len(missing)} consumed tensors (first: {missing[0]})"
            )
        prepared = []
        for key in sorted(expected):
            target, name = _target_for_key(model, key)
            prepared.append(
                (target, name, _prepare_value(target, name, weights[bindings[key]], key=key))
            )
        for target, name, value in prepared:
            setattr(target, name, value)
            sink.phase_advance(phase)
        _promote_normalizers(model)
        mx.eval(model.parameters())
        return len(prepared)
    finally:
        weights.clear()
        sink.phase_end(phase)


def load_temporal_upscaler(
    path: Path | str,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> TemporalUpscaler:
    model = TemporalUpscaler(
        TemporalUpscalerConfig.from_checkpoint(path),
        compute_dtype=compute_dtype,
    )
    load_temporal_upscaler_weights(model, path, reporter=reporter)
    return model


__all__ = [
    "PixelShuffle1d",
    "TemporalUpscaler",
    "TemporalUpscalerConfig",
    "load_temporal_upscaler",
    "load_temporal_upscaler_weights",
    "temporal_upsample_video",
]
