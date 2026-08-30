"""Append clean keyframe tokens at an arbitrary pixel-frame position."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from kinomlx.types import LatentState, VideoLatentShape

from ..patchifier import get_pixel_coords
from ..types import VIDEO_VAE_SCALE
from .latent import ConditioningError
from .tools import VideoLatentTools


@dataclass(frozen=True)
class VideoConditionByKeyframeIndex:
    """Append one encoded frame as clean K/V tokens at ``frame_idx``."""

    keyframes: mx.array
    frame_idx: int
    strength: float

    def __post_init__(self) -> None:
        if self.frame_idx < 0:
            raise ValueError("keyframe frame_idx must be non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("keyframe strength must be between 0 and 1")

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        if self.keyframes.ndim != 5:
            raise ConditioningError(
                "keyframe latent must have shape (batch, channels, 1, height, width)"
            )
        batch, channels, frames, height, width = self.keyframes.shape
        shape = VideoLatentShape(batch, channels, frames, height, width)
        target = latent_tools.target_shape
        if (
            shape.batch != target.batch
            or shape.channels != target.channels
            or shape.frames != 1
            or shape.height != target.height
            or shape.width != target.width
        ):
            raise ConditioningError(
                f"keyframe latent shape {shape.to_tuple()} is incompatible with {target.to_tuple()}"
            )

        tokens = latent_tools.patchifier.patchify(self.keyframes).astype(latent_state.latent.dtype)
        latent_positions = latent_tools.patchifier.get_patch_grid_bounds(shape)
        positions = get_pixel_coords(
            latent_positions,
            VIDEO_VAE_SCALE,
            causal_fix=(latent_tools.causal_fix if self.frame_idx == 0 else False),
        ).astype(mx.float32)
        # A one-frame keyframe occupies one pixel-frame interval. The VAE
        # latent normally spans eight frames, but retaining that end bound
        # moves middle-index RoPE guidance roughly 3.5 frames past frame_idx.
        temporal_start = positions[:, 0:1, :, :1]
        positions = mx.concatenate(
            [
                mx.concatenate(
                    [temporal_start, temporal_start + 1.0],
                    axis=-1,
                ),
                positions[:, 1:],
            ],
            axis=1,
        )
        positions = mx.concatenate(
            [
                (positions[:, 0:1] + self.frame_idx) / latent_tools.fps,
                positions[:, 1:],
            ],
            axis=1,
        )
        batch, token_count = tokens.shape[:2]
        keep = 1.0 - self.strength
        mask_dtype = latent_state.denoise_mask.dtype
        mask = mx.full((batch, token_count, 1), keep, dtype=mask_dtype)
        return LatentState(
            latent=mx.concatenate(
                [latent_state.latent, mx.zeros_like(tokens)],
                axis=1,
            ),
            denoise_mask=mx.concatenate(
                [latent_state.denoise_mask, mask],
                axis=1,
            ),
            positions=mx.concatenate(
                [latent_state.positions, positions],
                axis=2,
            ),
            clean_latent=mx.concatenate(
                [latent_state.clean_latent, tokens],
                axis=1,
            ),
            uniform_mask=False,
        )


__all__ = ["VideoConditionByKeyframeIndex"]
