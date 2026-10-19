"""LTX-2 noise injection over generative and clean conditioning latents.

The noiser first moves the generative latent toward Gaussian noise by the
requested scale, then composites that result over ``clean_latent`` according
to the per-token denoise mask. A mask value of 1 selects the generative stream;
a value of 0 preserves the clean conditioning value.
"""

import dataclasses

import mlx.core as mx

from kinomlx.samplers.noise import NoiseStreamState, create_normal_noise_stream
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend

from .types import LatentState


class GaussianNoiser:
    """Stateful LTX-2 Gaussian noiser backed by the shared normal stream."""

    def __init__(
        self,
        seed: int = 0,
        *,
        backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
        position: NoiseStreamState | None = None,
    ) -> None:
        self._noise = create_normal_noise_stream(seed, backend=backend, position=position)

    @property
    def state(self) -> NoiseStreamState:
        """Return the resumable position after all completed draws."""
        return self._noise.state

    def __call__(self, state: LatentState, scale: float = 1.0) -> LatentState:
        """Noise the generative latent, then composite over the clean latent.

        ``latent`` and ``denoise_mask`` must be broadcast-compatible. A common
        token mask is ``(batch, tokens)`` or ``(batch, tokens, 1)``; an
        unpatchified video mask broadcasts directly over its channel axis.
        The returned state preserves positions, clean latent, and the caller's
        explicit ``uniform_mask`` flag.
        """
        noise = self._noise.normal(tuple(state.latent.shape), state.latent.dtype)

        mask = state.denoise_mask
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=-1)
        # This is the official LTX two-lerp equation: first move the generative
        # latent toward noise, then composite that result over clean_latent.
        # Collapsing the expression loses conditioning whenever the generative
        # latent intentionally contains zeroes at conditioned positions.
        base = state.latent.astype(mx.float32)
        clean = state.clean_latent.astype(mx.float32)
        noised = base + scale * (noise.astype(mx.float32) - base)
        blended = clean + mask.astype(mx.float32) * (noised - clean)
        return dataclasses.replace(state, latent=blended.astype(state.latent.dtype))

    def advance(self, shape: tuple[int, ...]) -> None:
        """Advance one known tensor draw without materializing its values.

        Restarting a later stage must preserve both MLX key splits and the
        Torch-MPS Philox offset established by earlier modality draws.
        """
        self._noise.advance(shape)


__all__ = ["GaussianNoiser"]
