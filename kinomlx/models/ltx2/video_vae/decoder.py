"""Native MLX Conv3d video VAE decoder."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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
    pixel_norm_bfhwc,
    to_native_conv3d_layout,
    unpatchify_spatial_bfhwc,
)
from .config import (
    LTX23_VIDEO_VAE_CONFIG,
    VideoVAEConfig,
    compression_strides,
    is_compression_block,
)
from .ops import PerChannelStatistics

_log = logging.getLogger(__name__)


class NativeDepthToSpaceUpsample3d(nn.Module):
    """Conv3d followed by depth-to-space in BFHWC layout."""

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        *,
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: str = "conv",
    ) -> None:
        super().__init__()
        if len(stride) != 3 or any(value not in (1, 2) for value in stride):
            raise ValueError(f"unsupported upsample stride {stride}")
        if out_channels_reduction_factor <= 0:
            raise ValueError("out_channels_reduction_factor must be positive")
        if in_channels % out_channels_reduction_factor:
            raise ValueError(
                f"in_channels {in_channels} is not divisible by reduction factor "
                f"{out_channels_reduction_factor}"
            )
        stride_product = math.prod(stride)
        if residual and (
            in_channels % stride_product or stride_product % out_channels_reduction_factor
        ):
            raise ValueError(
                "residual depth-to-space requires channels divisible by stride "
                "and stride divisible by the reduction factor"
            )
        self.in_channels = in_channels
        self.stride = stride
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        self.final_out_channels = in_channels // out_channels_reduction_factor
        self.conv = NativeConv3dBlock(
            in_channels,
            stride_product * self.final_out_channels,
            spatial_padding_mode=spatial_padding_mode,
        )

    def _depth_to_space(self, x: mx.array, out_channels: int) -> mx.array:
        batch, frames, height, width, _ = x.shape
        stride_t, stride_h, stride_w = self.stride
        x = x.reshape(
            batch,
            frames,
            height,
            width,
            out_channels,
            stride_t,
            stride_h,
            stride_w,
        )
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(
            batch,
            frames * stride_t,
            height * stride_h,
            width * stride_w,
            out_channels,
        )

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        stride_t, stride_h, stride_w = self.stride
        stride_product = stride_t * stride_h * stride_w

        residual = None
        if self.residual:
            residual_channels = x.shape[-1] // stride_product
            residual = self._depth_to_space(x, residual_channels)
            if stride_t > 1:
                residual = residual[:, 1:]
            repeats = stride_product // self.out_channels_reduction_factor
            residual = mx.tile(residual, (1, 1, 1, 1, repeats))

        x = self.conv(x, causal=causal)
        x = self._depth_to_space(x, self.final_out_channels)
        if stride_t > 1:
            x = x[:, 1:]
        if residual is not None:
            x = x + residual
        return x


def _decoder_bottleneck_channels(config: VideoVAEConfig) -> int:
    multiplier = math.prod(
        block.multiplier for block in config.decoder_blocks if is_compression_block(block.name)
    )
    return config.decoder_base_channels * multiplier


class NativeConv3dVideoDecoder(nn.Module):
    """Decode normalized LTX BCFHW latents into RGB video in [-1, 1]."""

    def __init__(
        self,
        config: VideoVAEConfig = LTX23_VIDEO_VAE_CONFIG,
        *,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.causal = config.causal_decoder
        object.__setattr__(self, "load_receipt", None)
        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.latent_channels)

        feature_channels = _decoder_bottleneck_channels(config)
        self.conv_in = NativeConv3dBlock(
            config.latent_channels,
            feature_channels,
            causal=self.causal,
            spatial_padding_mode="conv",
        )
        self.up_blocks: list[NativeResBlockGroup | NativeDepthToSpaceUpsample3d] = []

        for block_config in reversed(config.decoder_blocks):
            if block_config.name == "res_x":
                block: NativeResBlockGroup | NativeDepthToSpaceUpsample3d = NativeResBlockGroup(
                    feature_channels,
                    num_blocks=block_config.num_layers or 0,
                    spatial_padding_mode="conv",
                )
            else:
                block = NativeDepthToSpaceUpsample3d(
                    feature_channels,
                    stride=compression_strides(block_config.name),
                    residual=block_config.residual,
                    out_channels_reduction_factor=block_config.multiplier,
                    spatial_padding_mode="conv",
                )
                feature_channels //= block_config.multiplier
            self.up_blocks.append(block)

        if feature_channels != config.decoder_base_channels:
            raise ValueError(
                f"decoder blocks end at {feature_channels} channels; expected "
                f"base width {config.decoder_base_channels}"
            )
        self.conv_out = NativeConv3dBlock(
            feature_channels,
            config.out_channels * config.patch_size**2,
            causal=self.causal,
            spatial_padding_mode="conv",
        )

    def __call__(
        self,
        latent: mx.array,
        *,
        timestep: float | None = 0.05,
        causal: bool | None = None,
        reporter: Reporter | None = None,
    ) -> mx.array:
        """Decode a 4D or 5D normalized latent tensor."""
        del timestep
        if latent.ndim == 4:
            latent = latent[None]
        self._validate_input(latent)
        causal = self.causal if causal is None else causal
        sink = reporter if reporter is not None else NullReporter()
        phase = "VAE decode"
        sink.phase_start(phase, total=len(self.up_blocks) + 3, unit="block")
        try:
            if latent.dtype != self.compute_dtype:
                latent = latent.astype(self.compute_dtype)
            x = self.per_channel_statistics.denormalize(latent)
            if x.dtype != self.compute_dtype:
                x = x.astype(self.compute_dtype)
            x = x.transpose(0, 2, 3, 4, 1)

            x = self.conv_in(x, causal=causal)
            mx.eval(x)
            sink.phase_advance(phase)

            for block in self.up_blocks:
                x = block(x, causal=causal)
                mx.eval(x)
                sink.phase_advance(phase)

            x = self.conv_out(silu(pixel_norm_bfhwc(x)), causal=causal)
            mx.eval(x)
            sink.phase_advance(phase)

            x = unpatchify_spatial_bfhwc(
                x,
                patch_size=self.config.patch_size,
            )
            mx.eval(x)
            sink.phase_advance(phase)
            return x.transpose(0, 4, 1, 2, 3)
        finally:
            sink.phase_end(phase)

    def _validate_input(self, latent: mx.array) -> None:
        if latent.ndim != 5 or latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                "expected BCFHW latent with "
                f"{self.config.latent_channels} channels, got {tuple(latent.shape)}"
            )
        if any(dimension <= 0 for dimension in latent.shape):
            raise ValueError(f"latent dimensions must be positive, got {latent.shape}")

    def _iter_convs(self) -> Iterator[tuple[str, NativeConv3dBlock]]:
        """Yield checkpoint key prefixes and their native Conv3d blocks."""
        yield "conv_in.conv", self.conv_in
        for index, block in enumerate(self.up_blocks):
            if isinstance(block, NativeResBlockGroup):
                for residual_index, residual in enumerate(block.res_blocks):
                    prefix = f"up_blocks.{index}.res_blocks.{residual_index}"
                    yield f"{prefix}.conv1.conv", residual.conv1
                    yield f"{prefix}.conv2.conv", residual.conv2
            else:
                yield f"up_blocks.{index}.conv.conv", block.conv
        yield "conv_out.conv", self.conv_out


WeightSource = Mapping[str, mx.array] | Path | str


@dataclass(frozen=True)
class Conv3dDecoderLoadReceipt:
    """Declared binding outcome for one permissive Conv3d checkpoint."""

    loaded_tensors: int
    ignored_decoder_tensors: tuple[str, ...]
    inferred_constructor_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decoder_kind": "native-conv3d",
            "loaded_tensors": self.loaded_tensors,
            "folded_gate_tensors": 0,
            "ignored_decoder_tensors": list(self.ignored_decoder_tensors),
            "inferred_constructor_fields": list(self.inferred_constructor_fields),
        }


def _weights_from_source(source: WeightSource) -> Mapping[str, mx.array]:
    if isinstance(source, Mapping):
        return source
    return load_weights(source)


def _stat_weight(
    weights: Mapping[str, mx.array],
    name: str,
) -> mx.array | None:
    aliases = {
        "mean_of_means": (
            "vae.per_channel_statistics.mean-of-means",
            "vae.per_channel_statistics.mean",
            "vae_decoder.per_channel_statistics.mean-of-means",
            "vae_decoder.per_channel_statistics.mean",
            "per_channel_statistics.mean-of-means",
            "per_channel_statistics.mean",
        ),
        "std_of_means": (
            "vae.per_channel_statistics.std-of-means",
            "vae.per_channel_statistics.std",
            "vae_decoder.per_channel_statistics.std-of-means",
            "vae_decoder.per_channel_statistics.std",
            "per_channel_statistics.std-of-means",
            "per_channel_statistics.std",
        ),
    }
    return lookup_weight(weights, *aliases[name])


def load_native_vae_decoder_weights(
    decoder: NativeConv3dVideoDecoder,
    source: WeightSource,
) -> int:
    """Load a complete decoder from raw-checkpoint or family-cache tensors."""
    weights = _weights_from_source(source)
    assignments: list[tuple[object, str, mx.array]] = []
    missing: list[str] = []

    for attribute in ("mean_of_means", "std_of_means"):
        value = _stat_weight(weights, attribute)
        if value is None:
            missing.append(f"per_channel_statistics.{attribute}")
            continue
        expected = (decoder.config.latent_channels,)
        if tuple(value.shape) != expected:
            raise ValueError(f"{attribute} has shape {tuple(value.shape)}; expected {expected}")
        assignments.append((decoder.per_channel_statistics, attribute, value))

    for local_prefix, conv_block in decoder._iter_convs():
        for suffix in ("weight", "bias"):
            local_key = f"{local_prefix}.{suffix}"
            value = lookup_weight(
                weights,
                f"vae.decoder.{local_key}",
                f"vae_decoder.{local_key}",
                f"decoder.{local_key}",
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
        raise ValueError(f"video VAE decoder weights are incomplete: {preview}{suffix}")

    for target, attribute, value in assignments:
        setattr(target, attribute, value)
    expected_decoder_suffixes = {
        f"decoder.{local_prefix}.{suffix}"
        for local_prefix, _conv_block in decoder._iter_convs()
        for suffix in ("weight", "bias")
    }
    ignored = tuple(
        sorted(
            key
            for key in weights
            if "decoder." in key
            and not any(key.endswith(suffix) for suffix in expected_decoder_suffixes)
        )
    )
    object.__setattr__(
        decoder,
        "load_receipt",
        Conv3dDecoderLoadReceipt(
            loaded_tensors=len(assignments),
            ignored_decoder_tensors=ignored,
            inferred_constructor_fields=decoder.config.inferred_fields,
        ),
    )
    _log.info("Loaded %d native Conv3d VAE decoder tensors", len(assignments))
    return len(assignments)


__all__ = [
    "Conv3dDecoderLoadReceipt",
    "NativeConv3dVideoDecoder",
    "NativeDepthToSpaceUpsample3d",
    "load_native_vae_decoder_weights",
]
