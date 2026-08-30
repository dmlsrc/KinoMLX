"""LTX-2 latent patchification helpers."""

from __future__ import annotations

import math

import mlx.core as mx

from kinomlx.types import SpatioTemporalScaleFactors, VideoLatentShape

from .types import AudioLatentShape


class VideoLatentPatchifier:
    """Convert BCFHW video latents to and from transformer patch tokens."""

    def __init__(self, patch_size: int = 1) -> None:
        if patch_size <= 0:
            raise ValueError("video patch size must be positive")
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self._patch_size

    def _validate_shape(self, shape: VideoLatentShape) -> None:
        if any(value <= 0 for value in shape):
            raise ValueError("video latent dimensions must be positive")
        _, patch_height, patch_width = self._patch_size
        if shape.height % patch_height or shape.width % patch_width:
            raise ValueError("video latent spatial dimensions must divide evenly by the patch size")

    def get_token_count(self, target: VideoLatentShape) -> int:
        self._validate_shape(target)
        return target.frames * target.height * target.width // math.prod(self._patch_size)

    def patchify(self, latents: mx.array) -> mx.array:
        """Convert ``(B, C, F, H, W)`` latents to ``(B, N, D)`` tokens."""
        if latents.ndim != 5:
            raise ValueError(
                "video latents must have shape (batch, channels, frames, height, width)"
            )
        batch, channels, frames, height, width = latents.shape
        shape = VideoLatentShape(batch, channels, frames, height, width)
        self._validate_shape(shape)
        patch_time, patch_height, patch_width = self._patch_size
        grid_frames = frames // patch_time
        grid_height = height // patch_height
        grid_width = width // patch_width
        return (
            latents.reshape(
                batch,
                channels,
                grid_frames,
                patch_time,
                grid_height,
                patch_height,
                grid_width,
                patch_width,
            )
            .transpose(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(
                batch,
                grid_frames * grid_height * grid_width,
                channels * patch_time * patch_height * patch_width,
            )
        )

    def unpatchify(
        self,
        latents: mx.array,
        output_shape: VideoLatentShape,
    ) -> mx.array:
        """Convert ``(B, N, D)`` tokens back to ``(B, C, F, H, W)``."""
        self._validate_shape(output_shape)
        if latents.ndim != 3:
            raise ValueError("video patch tokens must have shape (batch, tokens, features)")
        patch_time, patch_height, patch_width = self._patch_size
        grid_frames = output_shape.frames // patch_time
        grid_height = output_shape.height // patch_height
        grid_width = output_shape.width // patch_width
        expected = (
            output_shape.batch,
            grid_frames * grid_height * grid_width,
            output_shape.channels * patch_time * patch_height * patch_width,
        )
        if tuple(latents.shape) != expected:
            raise ValueError(f"video token shape {tuple(latents.shape)} does not match {expected}")
        return (
            latents.reshape(
                output_shape.batch,
                grid_frames,
                grid_height,
                grid_width,
                output_shape.channels,
                patch_time,
                patch_height,
                patch_width,
            )
            .transpose(0, 4, 1, 5, 2, 6, 3, 7)
            .reshape(output_shape.to_tuple())
        )

    def get_patch_grid_bounds(self, output_shape: VideoLatentShape) -> mx.array:
        """Return ``(B, 3, N, 2)`` latent-grid start/end bounds."""
        self._validate_shape(output_shape)
        starts = mx.meshgrid(
            mx.arange(0, output_shape.frames, self._patch_size[0]),
            mx.arange(0, output_shape.height, self._patch_size[1]),
            mx.arange(0, output_shape.width, self._patch_size[2]),
            indexing="ij",
        )
        patch_starts = mx.stack(list(starts), axis=0)
        patch_delta = mx.array(self._patch_size).reshape(3, 1, 1, 1)
        bounds = mx.stack([patch_starts, patch_starts + patch_delta], axis=-1)
        bounds = bounds.reshape(3, self.get_token_count(output_shape), 2)
        return mx.broadcast_to(
            bounds[None, ...],
            (output_shape.batch, 3, bounds.shape[1], 2),
        )


def get_pixel_coords(
    latent_coords: mx.array,
    scale_factors: SpatioTemporalScaleFactors,
    *,
    causal_fix: bool = False,
) -> mx.array:
    """Map latent patch bounds to pixel-space frame/height/width bounds."""
    if latent_coords.ndim != 4 or latent_coords.shape[1] != 3 or latent_coords.shape[-1] != 2:
        raise ValueError("latent coordinates must have shape (batch, 3, patches, 2)")
    if any(value <= 0 for value in scale_factors):
        raise ValueError("spatiotemporal scale factors must be positive")
    scale = mx.array([scale_factors.time, scale_factors.height, scale_factors.width]).reshape(
        1, 3, 1, 1
    )
    pixel_coords = latent_coords * scale
    if causal_fix:
        temporal = mx.maximum(
            pixel_coords[:, 0, ...] + 1 - scale_factors.time,
            0,
        )
        pixel_coords = mx.concatenate(
            [temporal[:, None, ...], pixel_coords[:, 1:, ...]],
            axis=1,
        )
    return pixel_coords


class AudioPatchifier:
    """Flatten an audio latent's channel and mel axes into token features."""

    def __init__(
        self,
        patch_size: int = 1,
        *,
        is_causal: bool = True,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        shift: int = 0,
    ) -> None:
        if patch_size != 1:
            raise ValueError("only unit audio patches are supported")
        if min(sample_rate, hop_length, audio_latent_downsample_factor) <= 0:
            raise ValueError("audio patchifier rates and factors must be positive")
        self._patch_size = (1, patch_size, patch_size)
        self.shift = shift
        self.is_causal = is_causal
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_latent_downsample_factor = audio_latent_downsample_factor

    @property
    def patch_size(self) -> tuple[int, int, int]:
        """Unit spatiotemporal patch; audio tokens are per latent frame."""
        return self._patch_size

    def get_token_count(self, target: AudioLatentShape) -> int:
        return target.frames

    def patchify(self, audio_latents: mx.array) -> mx.array:
        """Convert ``(B, C, T, F)`` latents to ``(B, T, C*F)``."""
        if audio_latents.ndim != 4:
            raise ValueError("audio latents must have shape (batch, channels, frames, mel_bins)")
        batch, channels, frames, mel_bins = audio_latents.shape
        return audio_latents.transpose(0, 2, 1, 3).reshape(
            batch,
            frames,
            channels * mel_bins,
        )

    def unpatchify(
        self,
        audio_latents: mx.array,
        output_shape: AudioLatentShape,
    ) -> mx.array:
        """Convert ``(B, T, C*F)`` tokens back to ``(B, C, T, F)``."""
        expected = (
            output_shape.batch,
            output_shape.frames,
            output_shape.channels * output_shape.mel_bins,
        )
        if tuple(audio_latents.shape) != expected:
            raise ValueError(
                f"audio token shape {tuple(audio_latents.shape)} does not match {expected}"
            )
        return audio_latents.reshape(
            output_shape.batch,
            output_shape.frames,
            output_shape.channels,
            output_shape.mel_bins,
        ).transpose(0, 2, 1, 3)

    def _latent_time(self, start: int, stop: int) -> mx.array:
        frames = mx.arange(start, stop, dtype=mx.float32)
        mel_frames = frames * self.audio_latent_downsample_factor
        if self.is_causal:
            mel_frames = mx.maximum(
                mel_frames + 1 - self.audio_latent_downsample_factor,
                0,
            )
        return mel_frames * self.hop_length / self.sample_rate

    def get_patch_grid_bounds(self, output_shape: AudioLatentShape) -> mx.array:
        """Return ``(B, 1, T, 2)`` start/end times in seconds."""
        starts = self._latent_time(
            self.shift,
            output_shape.frames + self.shift,
        )
        ends = self._latent_time(
            self.shift + 1,
            output_shape.frames + self.shift + 1,
        )
        bounds = mx.stack([starts, ends], axis=-1)[None, None, :, :]
        return mx.broadcast_to(
            bounds,
            (output_shape.batch, 1, output_shape.frames, 2),
        )


__all__ = ["AudioPatchifier", "VideoLatentPatchifier", "get_pixel_coords"]
