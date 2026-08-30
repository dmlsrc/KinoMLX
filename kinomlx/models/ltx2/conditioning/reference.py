"""Append clean full-video latent tokens for explicit IC-LoRA conditioning."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx.types import LatentState, VideoLatentShape

from .latent import ConditioningError
from .tools import VideoLatentTools


@dataclass(frozen=True)
class VideoConditionByReferenceLatent:
    """Append a reference latent, coordinates, clean values, and strength mask."""

    latent: mx.array
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("reference strength must be between 0 and 1")

    @property
    def latent_shape(self) -> VideoLatentShape:
        if self.latent.ndim != 5:
            raise ConditioningError(
                "reference latent must have shape (batch, channels, frames, height, width)"
            )
        batch, channels, frames, height, width = self.latent.shape
        return VideoLatentShape(batch, channels, frames, height, width)

    @property
    def token_count(self) -> int:
        shape = self.latent_shape
        return shape.frames * shape.height * shape.width

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        shape = self.latent_shape
        target = latent_tools.target_shape
        if (
            shape.batch != target.batch
            or shape.channels != target.channels
            or shape.height != target.height
            or shape.width != target.width
            or shape.frames > target.frames
        ):
            raise ConditioningError(
                f"reference latent shape {shape.to_tuple()} is incompatible with {target.to_tuple()}"
            )
        tokens = latent_tools.patchifier.patchify(self.latent).astype(latent_state.latent.dtype)
        positions = latent_tools.positions_for_shape(shape)
        keep = 1.0 - self.strength
        mask = mx.full(
            (tokens.shape[0], tokens.shape[1], 1),
            keep,
            dtype=latent_state.denoise_mask.dtype,
        )
        extended_latent = mx.concatenate([latent_state.latent, tokens], axis=1)
        extended_mask = mx.concatenate([latent_state.denoise_mask, mask], axis=1)
        extended_positions = mx.concatenate([latent_state.positions, positions], axis=2)
        extended_clean = mx.concatenate([latent_state.clean_latent, tokens], axis=1)
        return replace(
            latent_state,
            latent=extended_latent,
            denoise_mask=extended_mask,
            positions=extended_positions,
            clean_latent=extended_clean,
            uniform_mask=False,
        )


__all__ = ["VideoConditionByReferenceLatent"]
