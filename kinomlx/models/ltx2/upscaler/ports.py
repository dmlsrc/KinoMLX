"""Structural ports shared by LTX-2 component providers and upscaler recipes."""

from typing import Protocol

import mlx.core as mx

from kinomlx.reporting import Reporter


class LatentStatisticsPort(Protocol):
    """Affine normalization surface shared by encoder and upscaler ports."""

    def normalize(self, latent: mx.array) -> mx.array: ...

    def denormalize(self, latent: mx.array) -> mx.array: ...


class SpatialUpscalerPort(Protocol):
    @property
    def per_channel_statistics(self) -> LatentStatisticsPort: ...

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class TemporalUpscalerPort(Protocol):
    @property
    def per_channel_statistics(self) -> LatentStatisticsPort: ...

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


__all__ = ["LatentStatisticsPort", "SpatialUpscalerPort", "TemporalUpscalerPort"]
