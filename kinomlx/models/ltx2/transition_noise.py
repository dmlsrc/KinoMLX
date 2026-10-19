"""Ordered ancestral-transition noise for one stage of the two-stage LTX-2 recipe."""

import mlx.core as mx

from kinomlx.samplers.noise import NoiseStreamState, create_normal_noise_stream
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend


class SeededGaussianNoise:
    """Central LTX-2 transition-noise stream for one stage, with explicit call metadata.

    Each ancestral stage draws from its own stream, so neither stage reuses the
    initial noise or the other stage's draws. The ordering assertion follows the
    schedule in force rather than the length of the pinned distilled one, because
    that schedule is an override the user sets. A fixed ceiling refused any longer
    schedule partway through the stage, naming an internal counter instead of the
    sigmas that produced it.
    """

    def __init__(
        self,
        seed: int,
        *,
        stage: int,
        transitions: int,
        backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError(f"ancestral transition noise belongs to stage 1 or 2, not {stage}")
        if transitions < 1:
            raise ValueError("an ancestral schedule must drive at least one transition")
        self._stage = stage
        self._transitions = transitions
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
        if stage != self._stage:
            raise ValueError(
                f"the stage-{self._stage} ancestral noise stream was asked for stage {stage}"
            )
        if not 0 <= transition < self._transitions:
            raise ValueError(
                f"ancestral transition index {transition} is outside the stage-{stage} "
                f"schedule, which drives {self._transitions} transition(s)"
            )
        if modality not in {"video", "audio"}:
            raise ValueError(f"unknown ancestral noise modality {modality!r}")
        return self._noise.normal(shape, dtype)


__all__ = ["SeededGaussianNoise"]
