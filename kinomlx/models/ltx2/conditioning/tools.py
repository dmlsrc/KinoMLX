"""Build, patchify, and clear video/audio latent denoising states."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx.types import LatentState, VideoLatentShape

from ..patchifier import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ..types import VIDEO_VAE_SCALE, AudioLatentShape


@dataclass(frozen=True)
class VideoLatentTools:
    """Shape-bound helpers for a patchified video denoising stream."""

    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    causal_fix: bool = True

    def create_initial_state(
        self,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        initial_latent: mx.array | None = None,
    ) -> LatentState:
        if initial_latent is None:
            initial_latent = mx.zeros(self.target_shape.to_tuple(), dtype=dtype)
        elif tuple(initial_latent.shape) != self.target_shape.to_tuple():
            raise ValueError(
                f"initial video latent shape {tuple(initial_latent.shape)} does not "
                f"match {self.target_shape.to_tuple()}"
            )
        else:
            initial_latent = initial_latent.astype(dtype)

        mask = mx.ones(
            self.target_shape.with_channels(1).to_tuple(),
            dtype=mx.float32,
        )
        positions = self.positions_for_shape(self.target_shape)
        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=mask,
                positions=positions,
                clean_latent=initial_latent,
            )
        )

    def positions_for_shape(self, shape: VideoLatentShape) -> mx.array:
        """Build transformer coordinates for a same-cadence latent shape."""
        latent_positions = self.patchifier.get_patch_grid_bounds(shape)
        positions = get_pixel_coords(
            latent_positions,
            VIDEO_VAE_SCALE,
            causal_fix=self.causal_fix,
        ).astype(mx.float32)
        return mx.concatenate(
            [positions[:, 0:1] / self.fps, positions[:, 1:]],
            axis=1,
        )

    def patchify(self, state: LatentState) -> LatentState:
        return replace(
            state,
            latent=self.patchifier.patchify(state.latent),
            denoise_mask=self.patchifier.patchify(state.denoise_mask),
            clean_latent=self.patchifier.patchify(state.clean_latent),
        )

    def unpatchify(self, state: LatentState) -> LatentState:
        return replace(
            state,
            latent=self.patchifier.unpatchify(state.latent, self.target_shape),
            denoise_mask=self.patchifier.unpatchify(
                state.denoise_mask,
                self.target_shape.with_channels(1),
            ),
            clean_latent=self.patchifier.unpatchify(
                state.clean_latent,
                self.target_shape,
            ),
        )

    def clear_conditioning(self, state: LatentState) -> LatentState:
        token_count = self.patchifier.get_token_count(self.target_shape)
        return LatentState(
            latent=state.latent[:, :token_count],
            denoise_mask=mx.ones_like(state.denoise_mask[:, :token_count]),
            positions=state.positions[:, :, :token_count],
            clean_latent=state.clean_latent[:, :token_count],
            uniform_mask=True,
        )


@dataclass(frozen=True)
class AudioLatentTools:
    """Shape-bound helpers for a patchified audio denoising stream."""

    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        initial_latent: mx.array | None = None,
    ) -> LatentState:
        if initial_latent is None:
            initial_latent = mx.zeros(self.target_shape.to_tuple(), dtype=dtype)
        elif tuple(initial_latent.shape) != self.target_shape.to_tuple():
            raise ValueError(
                f"initial audio latent shape {tuple(initial_latent.shape)} does not "
                f"match {self.target_shape.to_tuple()}"
            )
        else:
            initial_latent = initial_latent.astype(dtype)
        mask_shape = AudioLatentShape(
            self.target_shape.batch,
            1,
            self.target_shape.frames,
            1,
        )
        state = LatentState(
            latent=initial_latent,
            denoise_mask=mx.ones(mask_shape.to_tuple(), dtype=mx.float32),
            positions=self.patchifier.get_patch_grid_bounds(self.target_shape),
            clean_latent=initial_latent,
        )
        return self.patchify(state)

    def patchify(self, state: LatentState) -> LatentState:
        return replace(
            state,
            latent=self.patchifier.patchify(state.latent),
            denoise_mask=self.patchifier.patchify(state.denoise_mask),
            clean_latent=self.patchifier.patchify(state.clean_latent),
        )

    def unpatchify(self, state: LatentState) -> LatentState:
        mask_shape = AudioLatentShape(
            self.target_shape.batch,
            1,
            self.target_shape.frames,
            1,
        )
        return replace(
            state,
            latent=self.patchifier.unpatchify(state.latent, self.target_shape),
            denoise_mask=self.patchifier.unpatchify(
                state.denoise_mask,
                mask_shape,
            ),
            clean_latent=self.patchifier.unpatchify(
                state.clean_latent,
                self.target_shape,
            ),
        )

    def clear_conditioning(self, state: LatentState) -> LatentState:
        token_count = self.patchifier.get_token_count(self.target_shape)
        return LatentState(
            latent=state.latent[:, :token_count],
            denoise_mask=mx.ones_like(state.denoise_mask[:, :token_count]),
            positions=state.positions[:, :, :token_count],
            clean_latent=state.clean_latent[:, :token_count],
            uniform_mask=True,
        )


__all__ = ["AudioLatentTools", "VideoLatentTools"]
