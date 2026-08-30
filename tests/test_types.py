"""Behavioral tests for ``kinomlx.types`` (model-agnostic primitives).

LTX-2-specific shape math (pixel <-> latent, audio derivations) is
tested under ``tests/models/ltx2/test_types.py``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import pytest

from kinomlx.types import (
    LatentState,
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


# ---------------------------------------------------------------------------
# LatentState - construction + uniform_mask default + frozen + footgun
# ---------------------------------------------------------------------------


def _dummy_state(uniform_mask: bool = True) -> LatentState:
    return LatentState(
        latent=mx.zeros((1, 128, 16, 18, 32)),
        denoise_mask=mx.ones((1, 1, 16, 18, 32)),
        positions=mx.zeros((1, 3, 16, 18, 32)),
        clean_latent=mx.zeros((1, 128, 16, 18, 32)),
        uniform_mask=uniform_mask,
    )


def test_latent_state_uniform_mask_defaults_true() -> None:
    s = LatentState(
        latent=mx.zeros((1,)),
        denoise_mask=mx.zeros((1,)),
        positions=mx.zeros((1,)),
        clean_latent=mx.zeros((1,)),
    )
    assert s.uniform_mask is True


def test_latent_state_is_frozen() -> None:
    s = _dummy_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.latent = mx.zeros((1,))  # type: ignore[misc]


def test_latent_state_replace_preserves_uniform_mask_flag() -> None:
    """Pins the footgun documented on ``LatentState``.

    Replacing ``denoise_mask`` alone keeps ``uniform_mask`` as-is -
    callers must pass ``uniform_mask=False`` explicitly when the new
    mask is mixed.
    """
    s = _dummy_state(uniform_mask=True)
    new_mask = mx.zeros_like(s.denoise_mask)
    # Without explicit uniform_mask=False, the flag persists.
    s2 = dataclasses.replace(s, denoise_mask=new_mask)
    assert s2.uniform_mask is True  # footgun stays footgun (documented)
    # Correct usage:
    s3 = dataclasses.replace(s, denoise_mask=new_mask, uniform_mask=False)
    assert s3.uniform_mask is False
