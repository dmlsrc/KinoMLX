"""Noise injection for diffusion latents.

A noiser first moves the generative latent toward Gaussian noise by the
requested noise scale, then composites that result over ``clean_latent``
according to the per-token denoise mask. Where ``mask == 1`` the generative
stream is used; where ``mask == 0`` the clean conditioning value is preserved.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from kinomlx.samplers.noise import (
    NoiseStreamState,
    create_normal_noise_stream,
)
from kinomlx.types import DEFAULT_NOISE_BACKEND, LatentState, NoiseBackend


class GaussianNoiser:
    """Stateful Gaussian noiser backed by the central normal stream."""

    def __init__(
        self,
        seed: int = 0,
        *,
        backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
        position: NoiseStreamState | None = None,
    ) -> None:
        self._noise = create_normal_noise_stream(
            seed,
            backend=backend,
            position=position,
        )

    @property
    def state(self) -> NoiseStreamState:
        """Return the resumable position after all completed draws."""
        return self._noise.state

    def __call__(self, state: LatentState, scale: float = 1.0) -> LatentState:
        """Noise the generative latent, then composite it over ``clean_latent``.

        Args:
            state: Current latent state.  ``latent`` and ``denoise_mask``
                must broadcast-compatible (typically the mask has a
                trailing 1 dim to broadcast across feature channels).
            scale: Multiplier on the mask before blending - typically
                the current sigma value.

        Returns:
            A new ``LatentState`` with ``latent`` replaced; all other fields
            (including ``clean_latent`` and ``uniform_mask``) unchanged.
        """
        noise = self._noise.normal(
            tuple(state.latent.shape),
            state.latent.dtype,
        )

        # The mask may arrive as either (B, T) or (B, T, 1); add the
        # trailing dim so the multiply broadcasts over the latent's
        # feature axis.  Higher-dim masks (e.g. (B, C, F, H, W) from
        # unpatchified contexts) broadcast directly.
        mask = state.denoise_mask
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=-1)
        # Match Lightricks' two-lerp GaussianNoiser semantics. Conditioning
        # values live in clean_latent; the generative latent may deliberately
        # contain zeros at those positions. Compute the blend in float32 so
        # fractional stage-2 masks do not accumulate BF16 rounding error.
        base = state.latent.astype(mx.float32)
        clean = state.clean_latent.astype(mx.float32)
        noised = base + scale * (noise.astype(mx.float32) - base)
        blended = clean + mask.astype(mx.float32) * (noised - clean)
        return dataclasses.replace(
            state,
            latent=blended.astype(state.latent.dtype),
        )

    def advance(self, shape: tuple[int, ...]) -> None:
        """Advance one known tensor draw without materializing its values.

        Restarting a later station must preserve both MLX key splits and the
        Torch-MPS Philox offset established by earlier modality draws.
        """
        self._noise.advance(shape)


class SeededGaussianNoise:
    """Central transition-noise stream with explicit call metadata."""

    def __init__(
        self,
        seed: int,
        *,
        backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    ) -> None:
        self._noise = create_normal_noise_stream(seed, backend=backend)

    @property
    def state(self) -> NoiseStreamState:
        """Return the ordered transition-stream position."""
        return self._noise.state

    def __call__(
        self,
        *,
        stage: int,
        transition: int,
        modality: str,
        shape: tuple[int, ...],
        dtype: mx.Dtype,
    ) -> mx.array:
        """Draw one tensor for the named stage transition and modality."""
        if stage != 1:
            raise ValueError("ancestral transition noise is only defined for stage 1")
        if not 0 <= transition <= 6:
            raise ValueError("ancestral transition index must be between 0 and 6")
        if modality not in {"video", "audio"}:
            raise ValueError(f"unknown ancestral noise modality {modality!r}")
        return self._noise.normal(shape, dtype)


__all__ = ["GaussianNoiser", "SeededGaussianNoise"]
