"""Behavioral tests for ``kinomlx.types`` (model-agnostic primitives).

LTX-2-specific shape math (pixel <-> latent, audio derivations) is
tested under ``tests/models/ltx2/test_types.py``.
"""

from kinomlx.types import (
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)

# ---------------------------------------------------------------------------
# VideoPixelShape - pure dims, no fps
# ---------------------------------------------------------------------------


def test_video_pixel_shape_carries_only_dims() -> None:
    s = VideoPixelShape(batch=1, frames=121, height=576, width=1024)
    assert s.batch == 1
    assert s.frames == 121
    assert s.height == 576
    assert s.width == 1024
    # Pure dims - no fps field, no channel order, no normalization metadata.
    assert s._fields == ("batch", "frames", "height", "width")


# ---------------------------------------------------------------------------
# VideoLatentShape - tuple round-trip + channel rewrite
# ---------------------------------------------------------------------------


def test_video_latent_shape_to_and_from_tuple() -> None:
    s = VideoLatentShape(batch=1, channels=128, frames=16, height=18, width=32)
    assert s.to_tuple() == (1, 128, 16, 18, 32)
    assert VideoLatentShape.from_tuple(s.to_tuple()) == s


def test_video_latent_with_channels_swaps_channel_dim_only() -> None:
    s = VideoLatentShape(batch=1, channels=128, frames=16, height=18, width=32)
    masked = s.with_channels(1)
    assert masked.channels == 1
    # Other dims untouched.
    assert masked.batch == s.batch
    assert masked.frames == s.frames
    assert masked.height == s.height
    assert masked.width == s.width


# ---------------------------------------------------------------------------
# SpatioTemporalScaleFactors - pure descriptor, no defaults
# ---------------------------------------------------------------------------


def test_scale_factors_have_no_default_factory() -> None:
    # Constructing requires explicit args; there's no .default() because
    # every model's VAE has its own ratios.
    sf = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    assert sf.time == 8
    assert sf.height == 32
    assert sf.width == 32
    assert not hasattr(SpatioTemporalScaleFactors, "default")
