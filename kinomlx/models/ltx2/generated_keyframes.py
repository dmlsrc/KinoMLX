"""LTX-2.5 generated-keyframe slot layout and state transforms."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx.types import LatentState, VideoLatentShape

from .conditioning.tools import VideoLatentTools
from .patchifier import get_pixel_coords
from .types import VIDEO_VAE_SCALE


@dataclass(frozen=True)
class GeneratedKeyframeLayout:
    """Contiguous appended-slot receipt needed for extraction and cleanup."""

    frame_indices: tuple[int, ...]
    tokens_per_slot: int
    first_token: int

    @property
    def count(self) -> int:
        return len(self.frame_indices)

    @property
    def token_count(self) -> int:
        return self.count * self.tokens_per_slot


def generated_keyframe_indices(pixel_frames: int, count: int) -> tuple[int, ...]:
    """Choose evenly spaced, rounded interior pixel-frame indices."""
    if isinstance(pixel_frames, bool) or not isinstance(pixel_frames, int) or pixel_frames < 1:
        raise ValueError("pixel_frames must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("generated keyframe count must be a non-negative integer")
    if count == 0:
        return ()
    available = pixel_frames - 2
    if available < count:
        raise ValueError(
            f"{pixel_frames} frames provide only {max(0, available)} interior keyframe slots"
        )
    denominator = count + 1
    indices = tuple(
        round(index * (pixel_frames - 1) / denominator) for index in range(1, count + 1)
    )
    if (
        any(index <= 0 or index >= pixel_frames - 1 for index in indices)
        or len(set(indices)) != count
    ):
        raise ValueError("generated keyframe request does not map to unique interior frames")
    return indices


def _slot_positions(
    tools: VideoLatentTools,
    frame_indices: tuple[int, ...],
) -> mx.array:
    shape = tools.target_shape
    one_frame = VideoLatentShape(
        shape.batch,
        shape.channels,
        1,
        shape.height,
        shape.width,
    )
    latent_positions = tools.patchifier.get_patch_grid_bounds(one_frame)
    spatial = get_pixel_coords(
        latent_positions,
        VIDEO_VAE_SCALE,
        causal_fix=tools.causal_fix,
    ).astype(mx.float32)[:, 1:]
    groups = []
    for frame_index in frame_indices:
        temporal = mx.array(
            [frame_index / tools.fps, (frame_index + 1) / tools.fps],
            dtype=mx.float32,
        )
        temporal = mx.broadcast_to(
            temporal.reshape(1, 1, 1, 2),
            (shape.batch, 1, spatial.shape[2], 2),
        )
        groups.append(mx.concatenate([temporal, spatial], axis=1))
    return mx.concatenate(groups, axis=2)


def append_generated_keyframe_slots(
    state: LatentState,
    tools: VideoLatentTools,
    *,
    pixel_frames: int,
    count: int,
) -> tuple[LatentState, GeneratedKeyframeLayout, mx.array]:
    """Append all-denoised one-frame slots and build the learned-position mask."""
    indices = generated_keyframe_indices(pixel_frames, count)
    if not indices:
        raise ValueError("append_generated_keyframe_slots requires a positive count")
    expected_main = tools.patchifier.get_token_count(tools.target_shape)
    if state.latent.ndim != 3 or state.latent.shape[0] != tools.target_shape.batch:
        raise ValueError("generated slots require a patchified video latent state")
    if state.latent.shape[1] < expected_main:
        raise ValueError("patchified state is missing target video tokens")
    tokens_per_slot = tools.target_shape.height * tools.target_shape.width
    first_token = int(state.latent.shape[1])
    slot_tokens = len(indices) * tokens_per_slot
    slots = mx.zeros(
        (tools.target_shape.batch, slot_tokens, state.latent.shape[2]),
        dtype=state.latent.dtype,
    )
    slot_mask = mx.ones(
        (tools.target_shape.batch, slot_tokens, state.denoise_mask.shape[-1]),
        dtype=state.denoise_mask.dtype,
    )
    prepared = replace(
        state,
        latent=mx.concatenate([state.latent, slots], axis=1),
        denoise_mask=mx.concatenate([state.denoise_mask, slot_mask], axis=1),
        positions=mx.concatenate([state.positions, _slot_positions(tools, indices)], axis=2),
        clean_latent=mx.concatenate([state.clean_latent, slots], axis=1),
    )
    first_frame = mx.ones(
        (tools.target_shape.batch, tokens_per_slot),
        dtype=mx.float32,
    )
    middle = mx.zeros(
        (tools.target_shape.batch, first_token - tokens_per_slot),
        dtype=mx.float32,
    )
    generated = mx.ones(
        (tools.target_shape.batch, slot_tokens),
        dtype=mx.float32,
    )
    keyframes_mask = mx.concatenate([first_frame, middle, generated], axis=1)
    return (
        prepared,
        GeneratedKeyframeLayout(
            frame_indices=indices,
            tokens_per_slot=tokens_per_slot,
            first_token=first_token,
        ),
        keyframes_mask,
    )


def extract_generated_keyframes(
    patchified_latent: mx.array,
    tools: VideoLatentTools,
    layout: GeneratedKeyframeLayout,
) -> mx.array:
    """Materialize appended slot tokens as one latent frame per selected index."""
    stop = layout.first_token + layout.token_count
    if patchified_latent.ndim != 3 or patchified_latent.shape[1] < stop:
        raise ValueError("denoised latent does not contain the declared generated slots")
    slot_shape = VideoLatentShape(
        tools.target_shape.batch,
        tools.target_shape.channels,
        layout.count,
        tools.target_shape.height,
        tools.target_shape.width,
    )
    return tools.patchifier.unpatchify(
        patchified_latent[:, layout.first_token : stop],
        slot_shape,
    )


__all__ = [
    "GeneratedKeyframeLayout",
    "append_generated_keyframe_slots",
    "extract_generated_keyframes",
    "generated_keyframe_indices",
]
