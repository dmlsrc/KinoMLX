"""Structural ports shared by LTX-2 component providers and upscaler recipes."""

from typing import Protocol

import mlx.core as mx

from kinomlx.reporting import Reporter


class LTX2LatentStatisticsPort(Protocol):
    """Affine normalization surface shared by encoder and upscaler ports."""

    def normalize(self, latent: mx.array) -> mx.array: ...

    def denormalize(self, latent: mx.array) -> mx.array: ...


class LTX2SpatialUpscalerPort(Protocol):
    @property
    def per_channel_statistics(self) -> LTX2LatentStatisticsPort: ...

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class LTX2TemporalUpscalerPort(Protocol):
    @property
    def per_channel_statistics(self) -> LTX2LatentStatisticsPort: ...

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


__all__ = ["LTX2LatentStatisticsPort", "LTX2SpatialUpscalerPort", "LTX2TemporalUpscalerPort"]
