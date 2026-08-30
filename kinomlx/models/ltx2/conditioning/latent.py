"""Replace clean tokens at a latent-frame index without moving positions."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from kinomlx.types import LatentState

from .tools import VideoLatentTools


class ConditioningError(ValueError):
    """A conditioning tensor is incompatible with its target state."""


@dataclass(frozen=True)
class VideoConditionByLatentIndex:
    """Install clean conditioning tokens into an existing latent frame range."""

    latent: mx.array
    strength: float
    latent_idx: int

    def __post_init__(self) -> None:
        if self.latent_idx < 0:
            raise ValueError("conditioning latent_idx must be non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("conditioning strength must be between 0 and 1")

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        if self.latent.ndim != 5:
            raise ConditioningError(
                "conditioning latent must have shape (batch, channels, frames, height, width)"
            )
        cond_batch, cond_channels, _frames, cond_height, cond_width = self.latent.shape
        target = latent_tools.target_shape
        if (cond_batch, cond_channels, cond_height, cond_width) != (
            target.batch,
            target.channels,
            target.height,
            target.width,
        ):
            raise ConditioningError(
                f"conditioning latent shape {tuple(self.latent.shape)} is "
                f"incompatible with {target.to_tuple()}"
            )

        tokens = latent_tools.patchifier.patchify(self.latent).astype(latent_state.latent.dtype)
        max_tokens = latent_tools.patchifier.get_token_count(target)
        if self.latent_idx == 0:
            start_token = 0
        else:
            prefix = target._replace(frames=self.latent_idx)
            start_token = latent_tools.patchifier.get_token_count(prefix)
        stop_token = start_token + tokens.shape[1]
        if stop_token > max_tokens:
            raise ConditioningError(
                f"conditioning token range [{start_token}, {stop_token}) exceeds "
                f"the target's {max_tokens} tokens"
            )

        clean = mx.concatenate(
            [
                latent_state.clean_latent[:, :start_token],
                tokens,
                latent_state.clean_latent[:, stop_token:],
            ],
            axis=1,
        )
        keep = 1.0 - self.strength
        mask_dtype = latent_state.denoise_mask.dtype
        conditioned_mask = mx.full((tokens.shape[0], tokens.shape[1], 1), keep, dtype=mask_dtype)
        mask = mx.concatenate(
            [
                latent_state.denoise_mask[:, :start_token],
                conditioned_mask,
                latent_state.denoise_mask[:, stop_token:],
            ],
            axis=1,
        )
        return LatentState(
            latent=latent_state.latent,
            denoise_mask=mask,
            positions=latent_state.positions,
            clean_latent=clean,
            uniform_mask=False,
        )


__all__ = ["ConditioningError", "VideoConditionByLatentIndex"]
