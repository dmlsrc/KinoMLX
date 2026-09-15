"""Spatial recipe upscaling and public temporal x2 component support."""

from .ports import LatentStatisticsPort, SpatialUpscalerPort, TemporalUpscalerPort
from .spatial import (
    PixelShuffle2d,
    ResBlock3d,
    SpatialUpscaler,
    SpatialUpscalerConfig,
    load_spatial_upscaler,
    load_spatial_upscaler_weights,
    upsample_video,
)
from .temporal import (
    PixelShuffle1d,
    TemporalUpscaler,
    TemporalUpscalerConfig,
    load_temporal_upscaler,
    load_temporal_upscaler_weights,
    temporal_upsample_video,
)

__all__ = [
    "LatentStatisticsPort",
    "PixelShuffle1d",
    "PixelShuffle2d",
    "ResBlock3d",
    "SpatialUpscaler",
    "SpatialUpscalerConfig",
    "SpatialUpscalerPort",
    "TemporalUpscaler",
    "TemporalUpscalerConfig",
    "TemporalUpscalerPort",
    "load_spatial_upscaler",
    "load_spatial_upscaler_weights",
    "load_temporal_upscaler",
    "load_temporal_upscaler_weights",
    "temporal_upsample_video",
    "upsample_video",
]
