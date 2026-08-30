"""Video VAE patchification and latent normalization operations."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import mlx.core as mx

import kinomlx._mlx_nn as nn


def _patchify_impl(
    x: mx.array,
    patch_size_hw: int,
    patch_size_t: int,
) -> mx.array:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.ndim == 4:
        batch, channels, height, width = x.shape
        spatial = patch_size_hw
        x = x.reshape(
            batch,
            channels,
            height // spatial,
            spatial,
            width // spatial,
            spatial,
        )
        x = x.transpose(0, 1, 5, 3, 2, 4)
        return x.reshape(
            batch,
            channels * spatial * spatial,
            height // spatial,
            width // spatial,
        )
    batch, channels, frames, height, width = x.shape
    temporal = patch_size_t
    spatial = patch_size_hw
    x = x.reshape(
        batch,
        channels,
        frames // temporal,
        temporal,
        height // spatial,
        spatial,
        width // spatial,
        spatial,
    )
    x = x.transpose(0, 1, 3, 7, 5, 2, 4, 6)
    return x.reshape(
        batch,
        channels * temporal * spatial * spatial,
        frames // temporal,
        height // spatial,
        width // spatial,
    )


@cache
def _compiled_patchify(
    patch_size_hw: int,
    patch_size_t: int,
) -> Callable[[mx.array], mx.array]:
    """Compile one patch geometry lazily, never during module import."""

    def operation(x: mx.array) -> mx.array:
        return _patchify_impl(x, patch_size_hw, patch_size_t)

    return mx.compile(operation)


def patchify(
    x: mx.array,
    patch_size_hw: int,
    patch_size_t: int = 1,
) -> mx.array:
    """Move spatiotemporal patches into the BCFHW channel dimension."""
    if x.ndim not in (4, 5):
        raise ValueError(f"expected 4D or 5D input, got {tuple(x.shape)}")
    if patch_size_hw <= 0 or patch_size_t <= 0:
        raise ValueError("patch sizes must be positive")
    if x.shape[-2] % patch_size_hw or x.shape[-1] % patch_size_hw:
        raise ValueError(
            "spatial dimensions must be divisible by patch_size_hw, got "
            f"{tuple(x.shape[-2:])} and {patch_size_hw}"
        )
    if x.ndim == 5 and x.shape[2] % patch_size_t:
        raise ValueError(
            f"frame count {x.shape[2]} must be divisible by patch_size_t {patch_size_t}"
        )
    return _compiled_patchify(patch_size_hw, patch_size_t)(x)


def _unpatchify_impl(
    x: mx.array,
    patch_size_hw: int,
    patch_size_t: int,
) -> mx.array:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.ndim == 4:
        batch, packed_channels, height, width = x.shape
        spatial = patch_size_hw
        channels = packed_channels // (spatial * spatial)
        x = x.reshape(
            batch,
            channels,
            spatial,
            spatial,
            height,
            width,
        )
        x = x.transpose(0, 1, 4, 3, 5, 2)
        return x.reshape(
            batch,
            channels,
            height * spatial,
            width * spatial,
        )
    batch, packed_channels, frames, height, width = x.shape
    temporal = patch_size_t
    spatial = patch_size_hw
    channels = packed_channels // (temporal * spatial * spatial)
    x = x.reshape(
        batch,
        channels,
        temporal,
        spatial,
        spatial,
        frames,
        height,
        width,
    )
    x = x.transpose(0, 1, 5, 2, 6, 4, 7, 3)
    return x.reshape(
        batch,
        channels,
        frames * temporal,
        height * spatial,
        width * spatial,
    )


@cache
def _compiled_unpatchify(
    patch_size_hw: int,
    patch_size_t: int,
) -> Callable[[mx.array], mx.array]:
    """Compile one unpatch geometry lazily, never during module import."""

    def operation(x: mx.array) -> mx.array:
        return _unpatchify_impl(x, patch_size_hw, patch_size_t)

    return mx.compile(operation)


def unpatchify(
    x: mx.array,
    patch_size_hw: int,
    patch_size_t: int = 1,
) -> mx.array:
    """Move packed BCFHW channels back into spatiotemporal dimensions."""
    if x.ndim not in (4, 5):
        raise ValueError(f"expected 4D or 5D input, got {tuple(x.shape)}")
    if patch_size_hw <= 0 or patch_size_t <= 0:
        raise ValueError("patch sizes must be positive")
    packed_factor = patch_size_hw * patch_size_hw
    if x.ndim == 5:
        packed_factor *= patch_size_t
    if x.shape[1] % packed_factor:
        raise ValueError(f"packed channel count {x.shape[1]} is not divisible by {packed_factor}")
    return _compiled_unpatchify(patch_size_hw, patch_size_t)(x)


class PerChannelStatistics(nn.Module):
    """Checkpoint-owned affine normalization for video latents."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        self.std_of_means = mx.ones((latent_channels,), dtype=mx.float32)
        self.mean_of_means = mx.zeros((latent_channels,), dtype=mx.float32)

    def denormalize(self, x: mx.array) -> mx.array:
        """Map normalized BCFHW latents to checkpoint latent space."""
        std, mean = self._broadcast_stats(x)
        return x * std + mean

    def un_normalize(self, x: mx.array) -> mx.array:
        """Preserve the upstream video-VAE method spelling."""
        return self.denormalize(x)

    def normalize(self, x: mx.array) -> mx.array:
        """Map checkpoint-space BCFHW latents to normalized model space."""
        std, mean = self._broadcast_stats(x)
        return (x - mean) / std

    def _broadcast_stats(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Shape-check ``x`` and return (std, mean) broadcast to BCFHW."""
        channels = int(self.mean_of_means.shape[0])
        if x.ndim != 5 or x.shape[1] != channels:
            raise ValueError(
                f"per-channel statistics require a BCFHW latent with "
                f"C={channels}; got shape {tuple(x.shape)}"
            )
        shape = (1, channels, 1, 1, 1)
        return (
            self.std_of_means.reshape(shape),
            self.mean_of_means.reshape(shape),
        )


__all__ = ["PerChannelStatistics", "patchify", "unpatchify"]
