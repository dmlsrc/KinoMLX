"""Native MLX Conv3d video VAE encoder."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from pathlib import Path

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import load_weights
from kinomlx.kernels import silu
from kinomlx.reporting import NullReporter, Reporter

from .blocks import (
    NativeConv3dBlock,
    NativeResBlockGroup,
    lookup_weight,
    patchify_spatial_bfhwc,
    pixel_norm_bfhwc,
    to_native_conv3d_layout,
)
from .config import LTX23_VIDEO_VAE_CONFIG, VideoVAEConfig, compression_strides
from .ops import PerChannelStatistics

_log = logging.getLogger(__name__)


class NativeConv3dVideoEncoderStatistics:
    """Lightweight holder for encoder latent normalization statistics."""

    def __init__(
        self,
        per_channel_statistics: PerChannelStatistics | None = None,
    ) -> None:
        self.per_channel_statistics = (
            per_channel_statistics
            if per_channel_statistics is not None
            else PerChannelStatistics(latent_channels=128)
        )


class NativeSpaceToDepthDownsample3d(nn.Module):
    """Conv3d followed by space-to-depth in BFHWC layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        *,
        residual: bool = True,
        spatial_padding_mode: str = "manual",
    ) -> None:
        super().__init__()
        if len(stride) != 3 or any(value not in (1, 2) for value in stride):
            raise ValueError(f"unsupported downsample stride {stride}")
        stride_product = math.prod(stride)
        if out_channels % stride_product:
            raise ValueError(
                f"out_channels {out_channels} must be divisible by stride product {stride_product}"
            )
        grouped_channels = in_channels * stride_product
        if grouped_channels % out_channels:
            raise ValueError(f"{grouped_channels} packed channels cannot reduce to {out_channels}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.residual = residual
        self.group_size = grouped_channels // out_channels
        self.conv_out_channels = out_channels // stride_product
        self.conv = NativeConv3dBlock(
            in_channels,
            self.conv_out_channels,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _space_to_depth_bfhwc(self, x: mx.array) -> mx.array:
        batch, frames, height, width, channels = x.shape
        stride_t, stride_h, stride_w = self.stride
        if frames % stride_t or height % stride_h or width % stride_w:
            raise ValueError(f"input {tuple(x.shape)} is not divisible by stride {self.stride}")
        x = x.reshape(
            batch,
            frames // stride_t,
            stride_t,
            height // stride_h,
            stride_h,
            width // stride_w,
            stride_w,
            channels,
        )
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        return x.reshape(
            batch,
            frames // stride_t,
            height // stride_h,
            width // stride_w,
            channels * stride_t * stride_h * stride_w,
        )

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        stride_t, _, _ = self.stride
        if stride_t == 2:
            x = mx.concatenate([x[:, :1], x], axis=1)

        residual = None
        if self.residual:
            packed = self._space_to_depth_bfhwc(x)
            batch, frames, height, width, _ = packed.shape
            packed = packed.reshape(
                batch,
                frames,
                height,
                width,
                self.out_channels,
                self.group_size,
            )
            residual = mx.mean(packed, axis=-1)

        x = self.conv(x, causal=causal)
        x = self._space_to_depth_bfhwc(x)
        if residual is not None:
            x = x + residual
        return x


class NativeConv3dVideoEncoder(nn.Module):
    """Encode normalized BCFHW RGB video into normalized LTX latents."""

    def __init__(
        self,
        config: VideoVAEConfig = LTX23_VIDEO_VAE_CONFIG,
        *,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.latent_channels)

        patch_channels = config.in_channels * config.patch_size**2
        feature_channels = config.encoder_base_channels
        self.conv_in = NativeConv3dBlock(patch_channels, feature_channels)
        self.down_blocks: list[NativeResBlockGroup | NativeSpaceToDepthDownsample3d] = []

        for block_config in config.encoder_blocks:
            if block_config.name == "res_x":
                block: NativeResBlockGroup | NativeSpaceToDepthDownsample3d = NativeResBlockGroup(
                    feature_channels,
                    num_blocks=block_config.num_layers or 0,
                )
            else:
                out_channels = feature_channels * block_config.multiplier
                block = NativeSpaceToDepthDownsample3d(
                    feature_channels,
                    out_channels,
                    stride=compression_strides(block_config.name),
                    residual=True,
                )
                feature_channels = out_channels
            self.down_blocks.append(block)

        self.conv_out = NativeConv3dBlock(
            feature_channels,
            config.latent_channels + 1,
        )

    def __call__(
        self,
        video: mx.array,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array:
        """Encode ``(B, C, F, H, W)`` video in [-1, 1] to BCFHW latent."""
        self._validate_input(video)
        sink = reporter if reporter is not None else NullReporter()
        phase = "VAE encode"
        sink.phase_start(phase, total=len(self.down_blocks) + 3, unit="block")
        try:
            if video.dtype != self.compute_dtype:
                video = video.astype(self.compute_dtype)
            x = video.transpose(0, 2, 3, 4, 1)

            x = patchify_spatial_bfhwc(x, patch_size=self.config.patch_size)
            mx.eval(x)
            sink.phase_advance(phase)

            x = self.conv_in(x, causal=True)
            mx.eval(x)
            sink.phase_advance(phase)

            for block in self.down_blocks:
                x = block(x, causal=True)
                mx.eval(x)
                sink.phase_advance(phase)

            x = self.conv_out(silu(pixel_norm_bfhwc(x)), causal=True)
            mx.eval(x)
            sink.phase_advance(phase)

            x = x.transpose(0, 4, 1, 2, 3)
            means = x[:, : self.config.latent_channels]
            means = self.per_channel_statistics.normalize(means)
            if means.dtype != mx.float32:
                means = means.astype(mx.float32)
            return means
        finally:
            sink.phase_end(phase)

    def _validate_input(self, video: mx.array) -> None:
        if video.ndim != 5 or video.shape[1] != self.config.in_channels:
            raise ValueError(
                "expected BCFHW video with "
                f"{self.config.in_channels} channels, got {tuple(video.shape)}"
            )
        _, _, frames, height, width = video.shape
        scale = self.config.encoder_scale
        if frames < 1 or (frames - 1) % scale.time:
            raise ValueError(f"frame count must be 1 + {scale.time}*k, got {frames}")
        if height <= 0 or width <= 0:
            raise ValueError("video height and width must be positive")
        if height % scale.height or width % scale.width:
            raise ValueError(
                f"video height and width must be divisible by "
                f"{scale.height}x{scale.width}, got {height}x{width}"
            )

    def _iter_convs(self) -> Iterator[tuple[str, NativeConv3dBlock]]:
        """Yield checkpoint key prefixes and their native Conv3d blocks."""
        yield "conv_in.conv", self.conv_in
        for index, block in enumerate(self.down_blocks):
            if isinstance(block, NativeResBlockGroup):
                for residual_index, residual in enumerate(block.res_blocks):
                    prefix = f"down_blocks.{index}.res_blocks.{residual_index}"
                    yield f"{prefix}.conv1.conv", residual.conv1
                    yield f"{prefix}.conv2.conv", residual.conv2
            else:
                yield f"down_blocks.{index}.conv.conv", block.conv
        yield "conv_out.conv", self.conv_out


WeightSource = Mapping[str, mx.array] | Path | str


def _weights_from_source(source: WeightSource) -> Mapping[str, mx.array]:
    if isinstance(source, Mapping):
        return source
    return load_weights(source)


def _require_stat(
    weights: Mapping[str, mx.array],
    name: str,
) -> mx.array | None:
    aliases = {
        "mean_of_means": (
            "vae.per_channel_statistics.mean-of-means",
            "vae.per_channel_statistics.mean",
            "vae_encoder.per_channel_statistics.mean-of-means",
            "vae_encoder.per_channel_statistics.mean",
            "per_channel_statistics.mean-of-means",
            "per_channel_statistics.mean",
        ),
        "std_of_means": (
            "vae.per_channel_statistics.std-of-means",
            "vae.per_channel_statistics.std",
            "vae_encoder.per_channel_statistics.std-of-means",
            "vae_encoder.per_channel_statistics.std",
            "per_channel_statistics.std-of-means",
            "per_channel_statistics.std",
        ),
    }
    return lookup_weight(weights, *aliases[name])


def load_native_vae_encoder_weights(
    encoder: NativeConv3dVideoEncoder,
    source: WeightSource,
) -> int:
    """Load a complete encoder from raw-checkpoint or family-cache tensors."""
    weights = _weights_from_source(source)
    assignments: list[tuple[object, str, mx.array]] = []
    missing: list[str] = []

    for attribute in ("mean_of_means", "std_of_means"):
        value = _require_stat(weights, attribute)
        if value is None:
            missing.append(f"per_channel_statistics.{attribute}")
            continue
        expected = (encoder.config.latent_channels,)
        if tuple(value.shape) != expected:
            raise ValueError(f"{attribute} has shape {tuple(value.shape)}; expected {expected}")
        assignments.append((encoder.per_channel_statistics, attribute, value))

    for local_prefix, conv_block in encoder._iter_convs():
        for suffix in ("weight", "bias"):
            local_key = f"{local_prefix}.{suffix}"
            value = lookup_weight(
                weights,
                f"vae.encoder.{local_key}",
                f"vae_encoder.{local_key}",
                f"encoder.{local_key}",
            )
            if value is None:
                missing.append(local_key)
                continue
            expected = tuple(getattr(conv_block.conv, suffix).shape)
            if suffix == "weight":
                value = to_native_conv3d_layout(value, expected)
            elif tuple(value.shape) != expected:
                raise ValueError(f"{local_key} has shape {tuple(value.shape)}; expected {expected}")
            assignments.append((conv_block.conv, suffix, value))

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", plus {len(missing) - 5} more"
        raise ValueError(f"video VAE encoder weights are incomplete: {preview}{suffix}")

    for target, attribute, value in assignments:
        setattr(target, attribute, value)
    _log.info("Loaded %d native Conv3d VAE encoder tensors", len(assignments))
    return len(assignments)


def load_native_vae_encoder_statistics(
    source: WeightSource,
    *,
    latent_channels: int = 128,
) -> NativeConv3dVideoEncoderStatistics:
    """Load only the latent normalization statistics used between stages."""
    weights = _weights_from_source(source)
    statistics = PerChannelStatistics(latent_channels=latent_channels)
    for attribute in ("mean_of_means", "std_of_means"):
        value = _require_stat(weights, attribute)
        if value is None:
            raise ValueError(f"missing video VAE statistic {attribute}")
        if tuple(value.shape) != (latent_channels,):
            raise ValueError(
                f"{attribute} has shape {tuple(value.shape)}; expected {(latent_channels,)}"
            )
        setattr(statistics, attribute, value)
    mx.eval(statistics.mean_of_means, statistics.std_of_means)
    return NativeConv3dVideoEncoderStatistics(statistics)


__all__ = [
    "NativeConv3dVideoEncoder",
    "NativeConv3dVideoEncoderStatistics",
    "NativeSpaceToDepthDownsample3d",
    "load_native_vae_encoder_statistics",
    "load_native_vae_encoder_weights",
]
