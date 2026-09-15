"""Ordered ancestral-transition noise for the two-stage LTX-2 recipe."""

import mlx.core as mx

from kinomlx.samplers.noise import NoiseStreamState, create_normal_noise_stream
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend


class SeededGaussianNoise:
    """Central LTX-2 transition-noise stream with explicit call metadata."""

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
        """Draw one tensor for the named LTX-2 stage transition and modality."""
        if stage != 1:
            raise ValueError("ancestral transition noise is only defined for stage 1")
        if not 0 <= transition <= 6:
            raise ValueError("ancestral transition index must be between 0 and 6")
        if modality not in {"video", "audio"}:
            raise ValueError(f"unknown ancestral noise modality {modality!r}")
        return self._noise.normal(shape, dtype)


__all__ = ["SeededGaussianNoise"]
