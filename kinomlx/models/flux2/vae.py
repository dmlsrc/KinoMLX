"""Native untiled Klein spatial VAE, using channels-last internal activations."""

from typing import Literal

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import group_norm, silu
from kinomlx.modeling.layers import conv2d_shell, linear_shell


class Resnet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, input_channels, eps=1e-6, pytorch_compatible=True)
        self.norm2 = nn.GroupNorm(32, output_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = conv2d_shell(bias=True, padding=1)
        self.conv2 = conv2d_shell(bias=True, padding=1)
        self.conv_shortcut = conv2d_shell(bias=True) if input_channels != output_channels else None

    def __call__(self, value: mx.array) -> mx.array:
        residual = value if self.conv_shortcut is None else self.conv_shortcut(value)
        value = self.conv1(silu(group_norm(value, self.norm1)))
        return residual + self.conv2(silu(group_norm(value, self.norm2)))


class SpatialAttention(nn.Module):
    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)
        self.to_q = linear_shell(bias=True)
        self.to_k = linear_shell(bias=True)
        self.to_v = linear_shell(bias=True)
        self.to_out = [linear_shell(bias=True)]
        self.scale = channels**-0.5

    def __call__(self, value: mx.array) -> mx.array:
        batch, height, width, channels = value.shape
        normalized = group_norm(value, self.group_norm).reshape(batch, height * width, channels)
        query, key, val = (
            projection(normalized)[:, None] for projection in (self.to_q, self.to_k, self.to_v)
        )
        if height * width <= 4096:
            attended = mx.fast.scaled_dot_product_attention(query, key, val, scale=self.scale)
        else:
            # Head width 512 uses MLX's materialized-score fallback. Evaluate each
            # query chunk before constructing the next to bound its score temporary;
            # every query still attends to all keys, with no spatial tiling seam.
            mx.eval(query, key, val)
            rows = []
            for start in range(0, height * width, 2048):
                row = mx.fast.scaled_dot_product_attention(
                    query[:, :, start : start + 2048], key, val, scale=self.scale
                )
                mx.eval(row)
                rows.append(row)
            attended = mx.concatenate(rows, axis=2)
        return value + self.to_out[0](attended[:, 0]).reshape(value.shape)


class MidBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = [Resnet(512, 512), Resnet(512, 512)]
        self.attentions = [SpatialAttention()]

    def __call__(self, value: mx.array) -> mx.array:
        return self.resnets[1](self.attentions[0](self.resnets[0](value)))


class Resampler(nn.Module):
    def __init__(self, *, down: bool) -> None:
        super().__init__()
        self.down = down
        self.conv = conv2d_shell(bias=True, stride=2 if down else 1, padding=0 if down else 1)

    def __call__(self, value: mx.array) -> mx.array:
        if self.down:
            value = mx.pad(value, [(0, 0), (0, 1), (0, 1), (0, 0)])
        else:
            value = mx.repeat(mx.repeat(value, 2, axis=1), 2, axis=2)
        return self.conv(value)


class ResolutionBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, *, down: bool, resample: bool
    ) -> None:
        super().__init__()
        self.resnets = [Resnet(input_channels, output_channels)] + [
            Resnet(output_channels, output_channels) for _ in range(1 if down else 2)
        ]
        self.downsamplers = [Resampler(down=True)] if down and resample else []
        self.upsamplers = [Resampler(down=False)] if not down and resample else []

    def __call__(self, value: mx.array) -> mx.array:
        for block in self.resnets:
            value = block(value)
        for sampler in (*self.downsamplers, *self.upsamplers):
            value = sampler(value)
        return value


class SpatialCodec(nn.Module):
    def __init__(self, *, encode: bool) -> None:
        super().__init__()
        self.encode = encode
        self.conv_in = conv2d_shell(bias=True, padding=1)
        channels = (128, 256, 512, 512) if encode else (512, 512, 256, 128)
        blocks = [
            ResolutionBlock(channels[max(i - 1, 0)], out, down=encode, resample=i < 3)
            for i, out in enumerate(channels)
        ]
        self.down_blocks = blocks if encode else []
        self.up_blocks = blocks if not encode else []
        self.mid_block = MidBlock()
        self.conv_norm_out = nn.GroupNorm(32, channels[-1], eps=1e-6, pytorch_compatible=True)
        self.conv_out = conv2d_shell(bias=True, padding=1)

    def __call__(self, value: mx.array) -> mx.array:
        value = self.conv_in(value)
        if not self.encode:
            value = self.mid_block(value)
        for block in (*self.down_blocks, *self.up_blocks):
            value = block(value)
        if self.encode:
            value = self.mid_block(value)
        return self.conv_out(silu(group_norm(value, self.conv_norm_out)))


class LatentStatistics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.running_mean = mx.zeros((128,), dtype=mx.bfloat16)
        self.running_var = mx.ones((128,), dtype=mx.bfloat16)


def patchify(latents: mx.array) -> mx.array:
    """BCHW 32-channel latents to BCHW 128-channel 2x2 patches."""
    if (
        latents.ndim != 4
        or latents.shape[1] != 32
        or any(n % 2 or n <= 0 for n in latents.shape[2:])
    ):
        raise ValueError("Klein latents require [batch, 32, even height, even width]")
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .transpose(0, 1, 3, 5, 2, 4)
        .reshape(batch, 128, height // 2, width // 2)
    )


def unpatchify(patches: mx.array) -> mx.array:
    if patches.ndim != 4 or patches.shape[1] != 128 or min(patches.shape) <= 0:
        raise ValueError("Klein patches require nonempty [batch, 128, height, width]")
    batch, _channels, height, width = patches.shape
    return (
        patches.reshape(batch, 32, 2, 2, height, width)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(batch, 32, height * 2, width * 2)
    )


class Flux2VAE(nn.Module):
    """Independently loadable encoder/decoder; encode selects posterior mean."""

    def __init__(self, mode: Literal["encode", "decode", "both"] = "both") -> None:
        super().__init__()
        if mode not in {"encode", "decode", "both"}:
            raise ValueError("Klein VAE mode must be encode, decode, or both")
        self.encoder = SpatialCodec(encode=True) if mode != "decode" else None
        self.decoder = SpatialCodec(encode=False) if mode != "encode" else None
        self.quant_conv = conv2d_shell(bias=True) if self.encoder is not None else None
        self.post_quant_conv = conv2d_shell(bias=True) if self.decoder is not None else None
        self.bn = LatentStatistics()

    def encode(self, pixels: mx.array) -> mx.array:
        """BCHW pixels in [-1, 1] to raw posterior mean, without sampling."""
        if self.encoder is None or self.quant_conv is None:
            raise ValueError("Klein VAE encoder is not loaded")
        if (
            pixels.ndim != 4
            or pixels.shape[1] != 3
            or min(pixels.shape) <= 0
            or any(n % 16 for n in pixels.shape[2:])
        ):
            raise ValueError(
                "Klein VAE pixels require [batch, 3, height, width] with 16-pixel dimensions"
            )
        value = pixels.transpose(0, 2, 3, 1).astype(self.encoder.conv_in.weight.dtype)
        moments = self.quant_conv(self.encoder(value))
        return moments[..., :32].transpose(0, 3, 1, 2)

    def decode(self, latents: mx.array) -> mx.array:
        """Raw BCHW 32-channel latent to unclipped BCHW pixels."""
        if self.decoder is None or self.post_quant_conv is None:
            raise ValueError("Klein VAE decoder is not loaded")
        if latents.ndim != 4 or latents.shape[1] != 32 or min(latents.shape) <= 0:
            raise ValueError("Klein VAE decode requires nonempty [batch, 32, height, width]")
        value = latents.transpose(0, 2, 3, 1).astype(self.post_quant_conv.weight.dtype)
        return self.decoder(self.post_quant_conv(value)).transpose(0, 3, 1, 2)

    def normalize(self, patches: mx.array) -> mx.array:
        """Apply bundled inference statistics using the practical source precision."""
        mean = self.bn.running_mean[None, :, None, None].astype(patches.dtype)
        std = mx.sqrt(self.bn.running_var + 1e-4)[None, :, None, None].astype(patches.dtype)
        return (patches - mean) / std

    def denormalize(self, patches: mx.array) -> mx.array:
        mean = self.bn.running_mean[None, :, None, None].astype(patches.dtype)
        std = mx.sqrt(self.bn.running_var + 1e-4)[None, :, None, None].astype(patches.dtype)
        return patches * std + mean

    def encode_normalized(self, pixels: mx.array) -> mx.array:
        return self.normalize(patchify(self.encode(pixels)))

    def decode_normalized(self, patches: mx.array) -> mx.array:
        return self.decode(unpatchify(self.denormalize(patches)))
