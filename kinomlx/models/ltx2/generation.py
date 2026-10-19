"""LTX-2 generation product shared by recipes, runners, and terminals."""

from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx

from kinomlx.media.frames import CloseableVideoFrameStream
from kinomlx.media.signals import VideoSignalSpec


@dataclass(frozen=True)
class LTX2GenerationOutput:
    """Encoder-ready output from a public recipe.

    ``frames`` is a closeable, single-consumer stream of owned
    ``(height, width, channels)`` frames with an immutable signal specification
    and exact frame count. Audio, when present, is ``(batch, channels, samples)``
    or ``(channels, samples)``.
    """

    frames: CloseableVideoFrameStream
    audio_waveform: mx.array | None = None
    audio_sample_rate: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    diagnostics_provider: Callable[[], dict[str, object]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def signal(self) -> VideoSignalSpec:
        return self.frames.spec

    @property
    def frame_count(self) -> int:
        return self.frames.frame_count

    def runtime_diagnostics(self) -> dict[str, object]:
        """Collect lazy-output receipts after the terminal consumes them."""
        if self.diagnostics_provider is None:
            return {}
        return self.diagnostics_provider()

    def close(self) -> None:
        self.frames.close()

    def __enter__(self) -> LTX2GenerationOutput:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["LTX2GenerationOutput"]
