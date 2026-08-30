"""Stateless recipe host over immutable LTX-2 resources and public ports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Protocol, TypeVar

from kinomlx.artifacts import ArtifactSink, NullArtifactSink
from kinomlx.errors import KinoMLXError
from kinomlx.media.frames import CloseableVideoFrameStream
from kinomlx.media.signals import VideoSignalSpec
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .components import DistilledComponents, NativeLTX2Components
from .resources import LTX2Resources, prepare_resources
from .settings import LTX2Settings
from .text_conditioning import TextConditioner

if TYPE_CHECKING:
    import mlx.core as mx

    from .pipelines.restart import DistilledRestart
    from .types import DistilledRequest


class LTX2Error(KinoMLXError, RuntimeError):
    """Typed operational failure at the LTX-2 host boundary."""


@dataclass(frozen=True)
class GenerationOutput:
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

    def __enter__(self) -> GenerationOutput:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


RequestT = TypeVar("RequestT", contravariant=True)


class Recipe(Protocol[RequestT]):
    """A public recipe callable hosted without recipe-specific branching."""

    def __call__(
        self,
        request: RequestT,
        resources: LTX2Resources,
        *,
        components: DistilledComponents | None = None,
        text_conditioner: TextConditioner | None = None,
        reporter: Reporter | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> GenerationOutput: ...


class LTX2Runner:
    """Convenience host retaining only immutable resources and stateless ports."""

    def __init__(
        self,
        model_settings: LTX2Settings | None = None,
        *,
        infrastructure: Settings | None = None,
        resources: LTX2Resources | None = None,
        components: DistilledComponents | None = None,
        text_conditioner: TextConditioner | None = None,
        reporter: Reporter | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> None:
        self.reporter = reporter if reporter is not None else NullReporter()
        self.artifact_sink = artifact_sink if artifact_sink is not None else NullArtifactSink()
        if resources is None:
            selected_model_settings = (
                model_settings if model_settings is not None else LTX2Settings.from_env()
            )
            try:
                resources = prepare_resources(
                    selected_model_settings,
                    infrastructure=infrastructure,
                    reporter=self.reporter,
                )
            except KinoMLXError:
                raise
            except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                raise LTX2Error(f"cannot prepare LTX-2 resources: {exc}") from exc
        elif model_settings is not None or infrastructure is not None:
            raise ValueError("pass settings or prepared resources, not both")
        self.resources = resources
        self.components = (
            components if components is not None else NativeLTX2Components(reporter=self.reporter)
        )
        # Recipes own their default replay policy. Keeping an absent port as
        # None lets ordinary generation select strict saved-text replay while
        # station restart selects observational identity handling.
        self.text_conditioner = text_conditioner

    def run(self, recipe: Recipe[RequestT], request: RequestT) -> GenerationOutput:
        """Run any LTX-2 recipe that accepts the standard stateless host ports."""
        try:
            output = recipe(
                request,
                self.resources,
                components=self.components,
                text_conditioner=self.text_conditioner,
                reporter=self.reporter,
                artifact_sink=self.artifact_sink,
            )
        except KinoMLXError:
            raise
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
            raise LTX2Error(f"LTX-2 generation failed: {exc}") from exc
        if not isinstance(output, GenerationOutput):
            raise TypeError("an LTX-2 recipe must return GenerationOutput")
        return output

    def restart(
        self,
        request: DistilledRequest,
        restart: DistilledRestart,
    ) -> GenerationOutput:
        """Resume the distilled graph from one typed saved-station product."""
        from .pipelines.restart import restart_distilled

        return self.run(partial(restart_distilled, restart=restart), request)


__all__ = [
    "GenerationOutput",
    "LTX2Error",
    "LTX2Runner",
    "Recipe",
]
