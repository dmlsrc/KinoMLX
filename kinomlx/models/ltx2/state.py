"""Pure construction and conditioning of LTX-2.3 latent states."""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

from kinomlx.types import LatentState, VideoLatentShape

from .conditioning import (
    AudioLatentTools,
    EncodedCondition,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoLatentTools,
)
from .patchifier import AudioPatchifier, VideoLatentPatchifier
from .types import AudioLatentShape, ImageConditioningConfig


def create_video_latent_tools(
    shape: VideoLatentShape,
    *,
    fps: float,
) -> VideoLatentTools:
    """Create the shape-bound video state helper used by both stages."""
    return VideoLatentTools(
        patchifier=VideoLatentPatchifier(),
        target_shape=shape,
        fps=fps,
    )


def create_audio_latent_tools(shape: AudioLatentShape) -> AudioLatentTools:
    """Create the shape-bound audio state helper used by both stages."""
    return AudioLatentTools(
        patchifier=AudioPatchifier(),
        target_shape=shape,
    )


def init_video_latent_state(
    tools: VideoLatentTools,
    *,
    dtype: mx.Dtype,
    initial_latent: mx.array | None = None,
) -> LatentState:
    """Build an all-generative patchified video state."""
    return tools.create_initial_state(dtype=dtype, initial_latent=initial_latent)


def init_audio_latent_state(
    tools: AudioLatentTools,
    *,
    dtype: mx.Dtype,
    initial_latent: mx.array | None = None,
) -> LatentState:
    """Build an all-generative patchified audio state."""
    return tools.create_initial_state(dtype=dtype, initial_latent=initial_latent)


def image_conditioning_item(
    latent: mx.array,
    config: ImageConditioningConfig,
) -> EncodedCondition:
    """Select first-frame replacement or appended keyframe semantics."""
    if config.frame_index == 0:
        return VideoConditionByLatentIndex(
            latent=latent,
            strength=config.strength,
            latent_idx=0,
        )
    return VideoConditionByKeyframeIndex(
        keyframes=latent,
        frame_idx=config.frame_index,
        strength=config.strength,
    )


def apply_encoded_conditions(
    state: LatentState,
    conditionings: Iterable[EncodedCondition],
    tools: VideoLatentTools,
) -> LatentState:
    """Apply ordered encoded conditions to a patchified state."""
    for conditioning in conditionings:
        state = conditioning.apply_to(state, tools)
    return state


def post_process_latent(
    denoised: mx.array,
    state: LatentState,
) -> mx.array:
    """Composite a denoised prediction over the clean conditioning stream."""
    if state.uniform_mask:
        return denoised
    mask = state.denoise_mask
    if mask.ndim == 2 and denoised.ndim == 3:
        mask = mask[:, :, None]
    denoised_f32 = denoised.astype(mx.float32)
    mask_f32 = mask.astype(mx.float32)
    clean_f32 = state.clean_latent.astype(mx.float32)
    return (denoised_f32 * mask_f32 + clean_f32 * (1.0 - mask_f32)).astype(denoised.dtype)


__all__ = [
    "apply_encoded_conditions",
    "create_audio_latent_tools",
    "create_video_latent_tools",
    "image_conditioning_item",
    "init_audio_latent_state",
    "init_video_latent_state",
    "post_process_latent",
]
