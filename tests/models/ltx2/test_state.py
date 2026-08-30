"""Latent-state construction and reference conditioning semantics."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.conditioning import (
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from kinomlx.models.ltx2.state import (
    create_audio_latent_tools,
    create_video_latent_tools,
    init_audio_latent_state,
    init_video_latent_state,
    post_process_latent,
)
from kinomlx.models.ltx2.types import AudioLatentShape
from kinomlx.types import LatentState, VideoLatentShape


def test_video_state_round_trip_and_clear_conditioning() -> None:
    shape = VideoLatentShape(1, 2, 2, 2, 2)
    tools = create_video_latent_tools(shape, fps=24.0)
    state = init_video_latent_state(tools, dtype=mx.float16)
    assert state.latent.shape == (1, 8, 2)
    assert state.denoise_mask.shape == (1, 8, 1)
    assert state.positions.shape == (1, 3, 8, 2)
    assert state.positions.dtype == mx.float32

    keyframe = mx.ones((1, 2, 1, 2, 2), dtype=mx.float32)
    conditioned = VideoConditionByKeyframeIndex(
        keyframes=keyframe,
        frame_idx=5,
        strength=0.75,
    ).apply_to(state, tools)
    assert conditioned.latent.shape == (1, 12, 2)
    assert mx.array_equal(conditioned.latent[:, 8:], mx.zeros((1, 4, 2))).item()
    assert mx.array_equal(
        conditioned.clean_latent[:, 8:],
        mx.ones((1, 4, 2), dtype=mx.float16),
    ).item()
    assert mx.all(conditioned.denoise_mask[:, 8:] == 0.25).item()
    temporal = conditioned.positions[:, 0, 8:]
    assert mx.allclose(
        temporal[..., 0],
        mx.full(temporal[..., 0].shape, 5.0 / 24.0),
    ).item()
    assert mx.allclose(
        temporal[..., 1],
        mx.full(temporal[..., 1].shape, 6.0 / 24.0),
    ).item()
    assert not conditioned.uniform_mask

    cleared = tools.clear_conditioning(conditioned)
    unpatchified = tools.unpatchify(cleared)
    assert unpatchified.latent.shape == shape.to_tuple()
    assert unpatchified.denoise_mask.shape == shape.with_channels(1).to_tuple()
    assert cleared.uniform_mask


def test_first_frame_conditioning_only_changes_clean_stream_and_mask() -> None:
    shape = VideoLatentShape(1, 2, 2, 2, 2)
    tools = create_video_latent_tools(shape, fps=24.0)
    state = init_video_latent_state(tools, dtype=mx.float32)
    latent_before = state.latent
    conditioning = mx.full((1, 2, 1, 2, 2), 3.0)

    result = VideoConditionByLatentIndex(
        latent=conditioning,
        strength=0.9,
        latent_idx=0,
    ).apply_to(state, tools)

    assert result.latent is latent_before
    assert mx.array_equal(result.latent, mx.zeros_like(result.latent)).item()
    assert mx.array_equal(result.clean_latent[:, :4], mx.full((1, 4, 2), 3.0)).item()
    assert mx.allclose(result.denoise_mask[:, :4], mx.full((1, 4, 1), 0.1)).item()
    assert mx.array_equal(result.denoise_mask[:, 4:], mx.ones((1, 4, 1))).item()
    assert not result.uniform_mask


def test_audio_state_uses_one_mask_feature_and_float32_time_bounds() -> None:
    shape = AudioLatentShape(1, 8, 3, 16)
    tools = create_audio_latent_tools(shape)
    initial = mx.ones(shape.to_tuple(), dtype=mx.bfloat16)
    state = init_audio_latent_state(
        tools,
        dtype=mx.float16,
        initial_latent=initial,
    )
    assert state.latent.shape == (1, 3, 128)
    assert state.latent.dtype == mx.float16
    assert state.denoise_mask.shape == (1, 3, 1)
    assert state.positions.shape == (1, 1, 3, 2)
    assert state.positions.dtype == mx.float32
    assert tools.unpatchify(state).latent.shape == shape.to_tuple()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("mask_value", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_fractional_conditioning_blends_once_after_fp32_arithmetic(
    dtype: mx.Dtype,
    mask_value: float,
) -> None:
    denoised = mx.full((1, 2, 3), -8.0, dtype=dtype)
    clean = mx.full((1, 2, 3), 4.6875, dtype=dtype)
    mask = mx.full((1, 2, 1), mask_value, dtype=mx.float32)
    state = LatentState(
        latent=denoised,
        denoise_mask=mask,
        positions=mx.zeros((1, 1, 2, 2)),
        clean_latent=clean,
        uniform_mask=False,
    )

    actual = post_process_latent(denoised, state)
    expected = (
        denoised.astype(mx.float32) * mask + clean.astype(mx.float32) * (1.0 - mask)
    ).astype(dtype)

    assert actual.dtype == dtype
    assert mx.array_equal(actual, expected).item()
