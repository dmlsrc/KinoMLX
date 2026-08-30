"""Shared native Conv3d primitives for the BFHWC video VAE."""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import pixel_norm, silu


def pixel_norm_bfhwc(x: mx.array, eps: float = 1e-8) -> mx.array:
    """Apply reference-stepwise PixelNorm over the channels-last dimension."""
    return pixel_norm(x, axis=-1, eps=eps)


def to_native_conv3d_layout(
    value: mx.array,
    expected_shape: tuple[int, ...],
) -> mx.array:
    """Convert PyTorch OITHW Conv3d weights to MLX OTHWI layout.

    Family-cache tensors are already channels-last, so this operation is
    idempotent when ``value`` has the requested shape.
    """
    if tuple(value.shape) == expected_shape:
        return value
    if value.ndim == 5:
        converted = value.transpose(0, 2, 3, 4, 1)
        if tuple(converted.shape) == expected_shape:
            return converted
    raise ValueError(
        f"cannot load Conv3d weight with shape {tuple(value.shape)}; expected {expected_shape}"
    )


def lookup_weight(
    weights: Mapping[str, mx.array],
    *keys: str,
) -> mx.array | None:
    """Return the first exact or wrapper-suffixed tensor alias."""
    for key in keys:
        if key in weights:
            return weights[key]
    for key in keys:
        matches = sorted(name for name in weights if name.endswith(f".{key}"))
        if matches:
            return weights[matches[0]]
    return None


class NativeConv3dBlock(nn.Module):
    """MLX Conv3d with explicit temporal and spatial padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        causal: bool = False,
        spatial_padding_mode: str = "manual",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Conv3d channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Conv3d kernel_size must be a positive odd integer")
        if padding < 0:
            raise ValueError("Conv3d padding must be non-negative")
        if spatial_padding_mode not in {"manual", "conv"}:
            raise ValueError(
                f"spatial_padding_mode must be 'manual' or 'conv', got {spatial_padding_mode!r}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode
        conv_padding = (0, padding, padding) if spatial_padding_mode == "conv" else 0
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            bias=True,
        )

    def __call__(self, x: mx.array, causal: bool | None = None) -> mx.array:
        """Convolve a ``(batch, frames, height, width, channels)`` tensor."""
        if x.ndim != 5 or x.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected BFHWC input with {self.in_channels} channels, got {tuple(x.shape)}"
            )
        causal = self.causal if causal is None else causal
        kernel_size = self.kernel_size

        if kernel_size > 1:
            if causal:
                first = mx.repeat(x[:, :1], kernel_size - 1, axis=1)
                x = mx.concatenate([first, x], axis=1)
            else:
                pad_size = (kernel_size - 1) // 2
                if pad_size:
                    first = mx.repeat(x[:, :1], pad_size, axis=1)
                    last = mx.repeat(x[:, -1:], pad_size, axis=1)
                    x = mx.concatenate([first, x, last], axis=1)

        if self.padding and self.spatial_padding_mode == "manual":
            pad = self.padding
            x = mx.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)])

        return self.conv(x)


class NativeResBlock3d(nn.Module):
    """Pre-activation VAE residual block in BFHWC layout."""

    def __init__(
        self,
        channels: int,
        spatial_padding_mode: str = "manual",
    ) -> None:
        super().__init__()
        self.conv1 = NativeConv3dBlock(
            channels,
            channels,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = NativeConv3dBlock(
            channels,
            channels,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        residual = x
        x = self.conv1(silu(pixel_norm_bfhwc(x)), causal=causal)
        x = self.conv2(silu(pixel_norm_bfhwc(x)), causal=causal)
        return x + residual


class NativeResBlockGroup(nn.Module):
    """A fixed-width group of VAE residual blocks."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        spatial_padding_mode: str = "manual",
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.res_blocks = [
            NativeResBlock3d(
                channels,
                spatial_padding_mode=spatial_padding_mode,
            )
            for _ in range(num_blocks)
        ]

    def __call__(self, x: mx.array, causal: bool = False) -> mx.array:
        for block in self.res_blocks:
            x = block(x, causal=causal)
        return x


def unpatchify_spatial_bfhwc(
    x: mx.array,
    patch_size: int = 4,
) -> mx.array:
    """Reverse the VAE's spatial patchification in BFHWC layout."""
    if x.ndim != 5:
        raise ValueError(f"expected BFHWC input, got {tuple(x.shape)}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    batch, frames, height, width, packed_channels = x.shape
    patch_area = patch_size * patch_size
    if packed_channels % patch_area:
        raise ValueError(f"packed channels {packed_channels} are not divisible by {patch_area}")
    channels = packed_channels // patch_area
    x = x.reshape(
        batch,
        frames,
        height,
        width,
        channels,
        patch_size,
        patch_size,
    )
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    return x.reshape(
        batch,
        frames,
        height * patch_size,
        width * patch_size,
        channels,
    )


def patchify_spatial_bfhwc(
    x: mx.array,
    patch_size: int = 4,
) -> mx.array:
    """Pack spatial patches into the channel dimension in BFHWC layout."""
    if x.ndim != 5:
        raise ValueError(f"expected BFHWC input, got {tuple(x.shape)}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    batch, frames, height, width, channels = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"height {height} and width {width} must be divisible by patch_size {patch_size}"
        )
    x = x.reshape(
        batch,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)
    return x.reshape(
        batch,
        frames,
        height // patch_size,
        width // patch_size,
        channels * patch_size * patch_size,
    )


__all__ = [
    "NativeConv3dBlock",
    "NativeResBlock3d",
    "NativeResBlockGroup",
    "lookup_weight",
    "patchify_spatial_bfhwc",
    "pixel_norm_bfhwc",
    "to_native_conv3d_layout",
    "unpatchify_spatial_bfhwc",
]
