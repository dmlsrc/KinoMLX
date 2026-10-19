"""Native Qwen Image VAE encoding specialized to the first still frame."""

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import silu
from kinomlx.modeling.dtypes import FloatDType
from kinomlx.reporting import NullReporter, Reporter

from ..metadata import Krea2VAEConfig
from .model import (
    Krea2MidBlock,
    Krea2ResidualBlock,
    Krea2RMSNorm,
    Krea2StillConv,
    Krea2VAEBoundaryObserver,
    _observe_bhwc,
    _validate_config,
    resolve_vae_dtype,
)


class Krea2Downsample(nn.Module):
    """Right/bottom zero padding followed by stride-two spatial convolution.

    Qwen's temporal downsamplers bypass time_conv on the first cached frame.
    A still encoder therefore consumes only their spatial resample weights.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resample = Krea2StillConv(channels, channels, 3, padding=0)
        self.resample.conv.stride = (2, 2)

    def __call__(self, value: mx.array) -> mx.array:
        return self.resample(mx.pad(value, [(0, 0), (0, 1), (0, 1), (0, 0)]))


class Krea2EncoderCore(nn.Module):
    """Spatial encoder producing the mean and log-variance feature channels."""

    def __init__(self, config: Krea2VAEConfig) -> None:
        super().__init__()
        current = config.base_dim
        self.conv_in = Krea2StillConv(config.input_channels, current, 3, padding=1)
        self.down_blocks: list[Krea2ResidualBlock | Krea2Downsample] = []
        for level, multiplier in enumerate(config.dim_mult):
            output = config.base_dim * multiplier
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(Krea2ResidualBlock(current, output))
                current = output
            if level < len(config.dim_mult) - 1:
                self.down_blocks.append(Krea2Downsample(current))
        self.mid_block = Krea2MidBlock(current)
        self.norm_out = Krea2RMSNorm(current)
        self.conv_out = Krea2StillConv(current, 2 * config.z_dim, 3, padding=1)


class Krea2ImageEncoder(nn.Module):
    """Encode floating BHWC RGB in [0, 1] to normalized BCHW posterior mode.

    Encoding is deterministic: image references use the posterior mean without
    sampling its variance. This component owns neither image fitting nor noise.
    """

    def __init__(
        self, config: Krea2VAEConfig, *, compute_dtype: FloatDType | mx.Dtype = mx.bfloat16
    ) -> None:
        super().__init__()
        _validate_config(config)
        if config.num_res_blocks < 1:
            raise ValueError("Krea VAE encoder requires at least one residual per level")
        self.config = config
        self.compute_dtype = resolve_vae_dtype(compute_dtype)
        self.encoder = Krea2EncoderCore(config)
        self.quant_conv = Krea2StillConv(2 * config.z_dim, 2 * config.z_dim, 1, padding=0)
        self.latent_mean = mx.array(config.latent_mean, dtype=self.compute_dtype)
        self.latent_std = mx.array(config.latent_std, dtype=self.compute_dtype)
        object.__setattr__(self, "load_receipt", None)

    def __call__(
        self,
        pixels: mx.array,
        *,
        reporter: Reporter | None = None,
        observer: Krea2VAEBoundaryObserver | None = None,
    ) -> mx.array:
        if pixels.ndim != 4 or pixels.shape[-1] != 3 or min(pixels.shape) <= 0:
            raise ValueError("Krea VAE encoder expects nonempty BHWC RGB")
        if pixels.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise TypeError("Krea VAE encoder expects floating RGB")
        if any(size % self.config.spatial_compression for size in pixels.shape[1:3]):
            raise ValueError("Krea VAE encoder dimensions must align to spatial compression")
        if not mx.all(mx.isfinite(pixels) & (pixels >= 0) & (pixels <= 1)).item():
            raise ValueError("Krea VAE encoder RGB must be finite and in [0, 1]")
        sink = reporter if reporter is not None else NullReporter()
        phase = "encode VAE"
        sink.phase_start(phase)
        try:
            value = pixels.astype(self.compute_dtype) * 2 - 1
            value = self.encoder.conv_in(value)
            _observe_bhwc(observer, "encoder_conv_in", value)
            for index, block in enumerate(self.encoder.down_blocks):
                value = block(value)
                _observe_bhwc(observer, f"encoder_down_{index}", value)
            value = self.encoder.mid_block(value)
            _observe_bhwc(observer, "encoder_mid", value)
            value = self.encoder.conv_out(silu(self.encoder.norm_out(value)))
            value = self.quant_conv(value)
            _observe_bhwc(observer, "encoder_posterior", value)
            mean = value[..., : self.config.z_dim]
            latent = ((mean - self.latent_mean) / self.latent_std).transpose(0, 3, 1, 2)
            mx.eval(latent)
            if not mx.all(mx.isfinite(latent)).item():
                raise ValueError("Krea VAE encoder produced nonfinite latents")
            return latent
        finally:
            sink.phase_end(phase)
