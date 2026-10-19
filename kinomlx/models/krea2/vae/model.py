"""Allocation-light native Qwen Image VAE decoder for Krea 2 stills."""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import silu
from kinomlx.modeling.dtypes import FloatDType, resolve_float_dtype
from kinomlx.modeling.layers import conv2d_shell
from kinomlx.reporting import NullReporter, Reporter

from ..metadata import Krea2VAEConfig
from .tiling import Krea2VAETilingConfig, blend_horizontal, blend_vertical

type Krea2VAEBoundaryObserver = Callable[[str, mx.array], None]

_FLOAT_DTYPES = frozenset((mx.float16, mx.bfloat16, mx.float32))


def resolve_vae_dtype(value: FloatDType | mx.Dtype) -> mx.Dtype:
    """Resolve one supported Krea VAE compute/weight dtype."""
    return resolve_float_dtype(value, owner="Krea 2 VAE")


@dataclass(frozen=True)
class Krea2DecodedRGB:
    """Unclamped floating decoder product in batch/H/W/RGB order."""

    pixels: mx.array

    def __post_init__(self) -> None:
        if self.pixels.ndim != 4 or self.pixels.shape[-1] != 3:
            raise ValueError(
                "Krea decoded RGB must have shape [batch, height, width, 3], got "
                f"{tuple(self.pixels.shape)}"
            )
        if self.pixels.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"Krea decoded RGB must be floating, got {self.pixels.dtype}")


class Krea2RMSNorm(nn.Module):
    """Qwen channel RMS normalization with FP32 norm calculation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("RMSNorm channels must be positive")
        self.channels = channels
        self.gamma = mx.zeros((0,), dtype=mx.float32)

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim != 4 or value.shape[-1] != self.channels:
            raise ValueError(
                f"Krea VAE RMSNorm expects NHWC width {self.channels}, got {tuple(value.shape)}"
            )
        source_dtype = value.dtype
        widened = value.astype(mx.float32)
        length = mx.sqrt(mx.sum(widened * widened, axis=-1, keepdims=True))
        normalized = (widened / mx.maximum(length, 1e-12)).astype(source_dtype)
        scale = mx.array(math.sqrt(self.channels), dtype=source_dtype)
        return normalized * scale * self.gamma


class Krea2StillConv(nn.Module):
    """Current-frame specialization of a Qwen causal Conv3d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int,
        fp32_opmath: bool | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Krea VAE convolution channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Krea VAE kernel size must be a positive odd integer")
        if padding < 0:
            raise ValueError("Krea VAE convolution padding must be non-negative")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fp32_opmath = kernel_size == 1 if fp32_opmath is None else fp32_opmath
        self.conv = conv2d_shell(bias=True, padding=padding)

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.in_channels,
        )

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim != 4 or value.shape[-1] != self.in_channels:
            raise ValueError(
                f"Krea VAE convolution expects NHWC width {self.in_channels}, got "
                f"{tuple(value.shape)}"
            )
        bias = self.conv.bias
        if bias is None:
            raise RuntimeError("Krea VAE convolution lost its required bias")
        if self.fp32_opmath and value.dtype in (mx.float16, mx.bfloat16):
            result = mx.conv2d(
                value.astype(mx.float32),
                self.conv.weight.astype(mx.float32),
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
            result = result + bias.astype(mx.float32)
            return result.astype(value.dtype)
        return self.conv(value)


class Krea2ResidualBlock(nn.Module):
    """Qwen pre-activation residual block specialized to one still frame."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = Krea2RMSNorm(in_channels)
        self.conv1 = Krea2StillConv(in_channels, out_channels, 3, padding=1)
        self.norm2 = Krea2RMSNorm(out_channels)
        self.conv2 = Krea2StillConv(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            Krea2StillConv(in_channels, out_channels, 1, padding=0)
            if in_channels != out_channels
            else None
        )

    def __call__(self, value: mx.array) -> mx.array:
        residual = value if self.conv_shortcut is None else self.conv_shortcut(value)
        value = self.conv1(silu(self.norm1(value)))
        value = self.conv2(silu(self.norm2(value)))
        return value + residual


class Krea2AttentionBlock(nn.Module):
    """Single-head spatial attention at the Qwen decoder bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = Krea2RMSNorm(channels)
        self.to_qkv = Krea2StillConv(channels, channels * 3, 1, padding=0)
        self.proj = Krea2StillConv(channels, channels, 1, padding=0)

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim != 4 or value.shape[-1] != self.channels:
            raise ValueError("Krea VAE attention input has an incompatible channel width")
        identity = value
        batch, height, width, channels = value.shape
        qkv = self.to_qkv(self.norm(value)).reshape(batch, height * width, channels * 3)
        query = qkv[..., :channels][:, None, :, :]
        key = qkv[..., channels : 2 * channels][:, None, :, :]
        projected = qkv[..., 2 * channels :][:, None, :, :]
        attended = mx.fast.scaled_dot_product_attention(
            query.astype(mx.float32),
            key.astype(mx.float32),
            projected.astype(mx.float32),
            scale=channels**-0.5,
        ).astype(value.dtype)
        attended = attended[:, 0].reshape(batch, height, width, channels)
        return self.proj(attended) + identity


class Krea2MidBlock(nn.Module):
    """Residual-attention-residual Qwen bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resnets = [
            Krea2ResidualBlock(channels, channels),
            Krea2ResidualBlock(channels, channels),
        ]
        self.attentions = [Krea2AttentionBlock(channels)]

    def __call__(self, value: mx.array) -> mx.array:
        value = self.resnets[0](value)
        value = self.attentions[0](value)
        return self.resnets[1](value)


class Krea2Upsample(nn.Module):
    """Nearest-exact spatial 2x upsample followed by Qwen's Conv2d."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("Krea VAE upsample channels must be even")
        self.in_channels = channels
        self.out_channels = channels // 2
        self.resample = Krea2StillConv(channels, channels // 2, 3, padding=1)

    def __call__(self, value: mx.array) -> mx.array:
        value = mx.repeat(mx.repeat(value, 2, axis=1), 2, axis=2)
        return self.resample(value)


class Krea2UpBlock(nn.Module):
    """One decoder resolution level with residuals and optional upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_res_blocks: int,
        upsample: bool,
    ) -> None:
        super().__init__()
        current = in_channels
        self.resnets: list[Krea2ResidualBlock] = []
        for _index in range(num_res_blocks + 1):
            self.resnets.append(Krea2ResidualBlock(current, out_channels))
            current = out_channels
        self.upsamplers = [Krea2Upsample(out_channels)] if upsample else []

    def __call__(self, value: mx.array) -> mx.array:
        for residual in self.resnets:
            value = residual(value)
        for upsample in self.upsamplers:
            value = upsample(value)
        return value


class Krea2DecoderCore(nn.Module):
    """Qwen decoder graph after latent denormalization and post-quant projection."""

    def __init__(self, config: Krea2VAEConfig, *, output_channels: int) -> None:
        super().__init__()
        bottleneck = config.base_dim * config.dim_mult[-1]
        self.conv_in = Krea2StillConv(config.z_dim, bottleneck, 3, padding=1)
        self.mid_block = Krea2MidBlock(bottleneck)
        dims = [bottleneck, *(config.base_dim * value for value in reversed(config.dim_mult))]
        self.up_blocks: list[Krea2UpBlock] = []
        for index, (declared_in, out_channels) in enumerate(itertools.pairwise(dims)):
            in_channels = declared_in // 2 if index > 0 else declared_in
            self.up_blocks.append(
                Krea2UpBlock(
                    in_channels,
                    out_channels,
                    num_res_blocks=config.num_res_blocks,
                    upsample=index != len(config.dim_mult) - 1,
                )
            )
        self.norm_out = Krea2RMSNorm(dims[-1])
        self.conv_out = Krea2StillConv(dims[-1], output_channels, 3, padding=1)


class Krea2ImageDecoder(nn.Module):
    """Decode normalized BCHW Krea latents to unclamped floating RGB."""

    def __init__(
        self,
        config: Krea2VAEConfig,
        *,
        compute_dtype: FloatDType | mx.Dtype = mx.bfloat16,
        output_scale: int = 1,
    ) -> None:
        super().__init__()
        _validate_config(config)
        if (
            isinstance(output_scale, bool)
            or not isinstance(output_scale, int)
            or output_scale not in (1, 2)
        ):
            raise ValueError("Krea VAE output_scale must be 1 or 2")
        self.config = config
        self.compute_dtype = resolve_vae_dtype(compute_dtype)
        self.output_scale = output_scale
        self.latent_mean = mx.array(config.latent_mean, dtype=self.compute_dtype)
        self.latent_std = mx.array(config.latent_std, dtype=self.compute_dtype)
        self.post_quant_conv = Krea2StillConv(config.z_dim, config.z_dim, 1, padding=0)
        self.decoder = Krea2DecoderCore(
            config,
            output_channels=config.input_channels * output_scale**2,
        )
        object.__setattr__(self, "load_receipt", None)

    def __call__(
        self,
        latent: mx.array,
        *,
        reporter: Reporter | None = None,
        observer: Krea2VAEBoundaryObserver | None = None,
        tiling: Krea2VAETilingConfig | None = None,
    ) -> Krea2DecodedRGB:
        """Decode one batch of normalized BCHW still latents."""
        self._validate_input(latent)
        value = latent.astype(self.compute_dtype)
        stats_shape = (1, self.config.z_dim, 1, 1)
        value = value * self.latent_std.reshape(stats_shape) + self.latent_mean.reshape(stats_shape)
        _observe_bchw(observer, "normalized_input", value)
        value = value.transpose(0, 2, 3, 1)

        if tiling is not None and self._requires_tiling(value, tiling):
            if observer is not None:
                raise ValueError("Krea VAE boundary observation requires untiled decode")
            decoded = self._decode_tiled(value, tiling, reporter=reporter)
        else:
            decoded = self._decode_full(value, reporter=reporter, observer=observer)
        mx.eval(decoded)
        if not mx.all(mx.isfinite(decoded)).item():
            raise ValueError("Krea VAE decoder produced nonfinite RGB")
        return Krea2DecodedRGB(decoded)

    def _validate_input(self, latent: mx.array) -> None:
        if latent.ndim != 4 or latent.shape[1] != self.config.z_dim:
            raise ValueError(
                f"Krea VAE expects BCHW latent with {self.config.z_dim} channels, got "
                f"{tuple(latent.shape)}"
            )
        if any(dimension <= 0 for dimension in latent.shape):
            raise ValueError(f"Krea VAE latent dimensions must be positive, got {latent.shape}")
        if latent.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"Krea VAE latent must be floating, got {latent.dtype}")

    def _decode_full(
        self,
        value: mx.array,
        *,
        reporter: Reporter | None,
        observer: Krea2VAEBoundaryObserver | None,
    ) -> mx.array:
        """Decode one latent block as a single lazy graph.

        Nothing is evaluated here. The untiled path evaluates once in
        ``__call__`` under an indeterminate progress bar; the tiled path
        evaluates once per tile so tiling keeps bounding peak memory.
        """
        sink = reporter if reporter is not None else NullReporter()
        phase = "decode VAE"
        sink.phase_start(phase)
        try:
            value = self.post_quant_conv(value)
            _observe_bhwc(observer, "post_quant_conv", value)
            value = self.decoder.conv_in(value)
            _observe_bhwc(observer, "decoder_conv_in", value)
            value = self.decoder.mid_block(value)
            _observe_bhwc(observer, "decoder_mid", value)
            for index, block in enumerate(self.decoder.up_blocks):
                value = block(value)
                _observe_bhwc(observer, f"decoder_up_{index}", value)
            value = self.decoder.norm_out(value)
            _observe_bhwc(observer, "decoder_norm_out", value)
            value = self.decoder.conv_out(silu(value))
            if self.output_scale > 1:
                _observe_bhwc(observer, "decoder_conv_out_packed", value)
                value = _pixel_shuffle_rgb(value, self.output_scale)
            _observe_bhwc(observer, "decoder_conv_out_pre_clamp", value)
            return value
        finally:
            sink.phase_end(phase)

    def _requires_tiling(
        self,
        value: mx.array,
        tiling: Krea2VAETilingConfig,
    ) -> bool:
        compression = self.config.spatial_compression
        if tiling.tile_sample_size % compression or tiling.tile_sample_stride % compression:
            raise ValueError("Krea VAE tile size and stride must align to spatial compression")
        tile = tiling.tile_sample_size // compression
        return value.shape[1] > tile or value.shape[2] > tile

    def _decode_tiled(
        self,
        value: mx.array,
        tiling: Krea2VAETilingConfig,
        *,
        reporter: Reporter | None,
    ) -> mx.array:
        compression = self.config.spatial_compression
        tile_size = tiling.tile_sample_size // compression
        tile_stride = tiling.tile_sample_stride // compression
        starts_h = tuple(range(0, value.shape[1], tile_stride))
        starts_w = tuple(range(0, value.shape[2], tile_stride))
        sink = reporter if reporter is not None else NullReporter()
        phase = "decode VAE tiles"
        sink.phase_start(phase, total=len(starts_h) * len(starts_w), unit="tile")
        rows: list[list[mx.array]] = []
        try:
            for top in starts_h:
                row: list[mx.array] = []
                for left in starts_w:
                    tile = value[:, top : top + tile_size, left : left + tile_size, :]
                    decoded_tile = self._decode_full(tile, reporter=None, observer=None)
                    mx.eval(decoded_tile)
                    row.append(decoded_tile)
                    sink.phase_advance(phase)
                rows.append(row)
        finally:
            sink.phase_end(phase)

        output_scale = self.output_scale
        blend = (tiling.tile_sample_size - tiling.tile_sample_stride) * output_scale
        output_stride = tiling.tile_sample_stride * output_scale
        output_rows: list[mx.array] = []
        blended_rows: list[list[mx.array]] = []
        for row_index, row in enumerate(rows):
            blended_row: list[mx.array] = []
            output_tiles: list[mx.array] = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = blend_vertical(blended_rows[row_index - 1][column_index], tile, blend)
                if column_index > 0:
                    tile = blend_horizontal(blended_row[column_index - 1], tile, blend)
                blended_row.append(tile)
                output_tiles.append(tile[:, :output_stride, :output_stride, :])
            blended_rows.append(blended_row)
            output_rows.append(mx.concatenate(output_tiles, axis=2))
        decoded = mx.concatenate(output_rows, axis=1)
        target_height = value.shape[1] * compression * output_scale
        target_width = value.shape[2] * compression * output_scale
        return decoded[:, :target_height, :target_width, :]


def _pixel_shuffle_rgb(value: mx.array, scale: int) -> mx.array:
    channels = 3 * scale**2
    if value.ndim != 4 or value.shape[-1] != channels:
        raise ValueError(f"Krea VAE pixel shuffle expects BHWC{channels}, got {tuple(value.shape)}")
    batch, height, width, _channels = value.shape
    return (
        value.reshape(batch, height, width, 3, scale, scale)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(batch, height * scale, width * scale, 3)
    )


def _validate_config(config: Krea2VAEConfig) -> None:
    if config.base_dim <= 0 or config.z_dim <= 0 or config.input_channels != 3:
        raise ValueError("Krea VAE requires positive dimensions and three output channels")
    if not config.dim_mult or any(value <= 0 for value in config.dim_mult):
        raise ValueError("Krea VAE dim_mult must contain positive values")
    if config.num_res_blocks < 0:
        raise ValueError("Krea VAE num_res_blocks must be non-negative")
    if config.attn_scales:
        raise ValueError("Krea still decoder does not support extra attention scales")
    if len(config.latent_mean) != config.z_dim or len(config.latent_std) != config.z_dim:
        raise ValueError("Krea VAE latent statistics must match z_dim")
    if any(not math.isfinite(value) for value in (*config.latent_mean, *config.latent_std)):
        raise ValueError("Krea VAE latent statistics must be finite")
    if any(value <= 0.0 for value in config.latent_std):
        raise ValueError("Krea VAE latent standard deviations must be positive")


def _observe_bchw(
    observer: Krea2VAEBoundaryObserver | None,
    name: str,
    value: mx.array,
) -> None:
    if observer is not None:
        observer(name, value[:, :, None, :, :])


def _observe_bhwc(
    observer: Krea2VAEBoundaryObserver | None,
    name: str,
    value: mx.array,
) -> None:
    if observer is not None:
        observer(name, value.transpose(0, 3, 1, 2)[:, :, None, :, :])


__all__ = [
    "Krea2AttentionBlock",
    "Krea2DecodedRGB",
    "Krea2ImageDecoder",
    "Krea2RMSNorm",
    "Krea2ResidualBlock",
    "Krea2StillConv",
    "Krea2VAEBoundaryObserver",
    "resolve_vae_dtype",
]
