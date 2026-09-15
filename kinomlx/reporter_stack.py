"""The reporter stack every command and converter builds around its work."""

from collections.abc import Mapping
from contextlib import ExitStack

from kinomlx.debug import create_mlx_memory_sampler
from kinomlx.profiling import SignpostReporter
from kinomlx.reporting import Reporter, TimingReporter
from kinomlx.settings import Settings
from kinomlx.ui import RichReporter


def command_reporter(
    stack: ExitStack, settings: Settings, *, phase_labels: Mapping[str, str] | None = None
) -> TimingReporter:
    """Compose the Rich presentation, optional signposts, timing and allocator sampling.

    Quiet disables the live rows without disabling phase accounting, so summaries
    and timings survive every CLI mode.
    """
    presentation = stack.enter_context(
        RichReporter(disable=settings.quiet, phase_labels=phase_labels)
    )
    presentation_reporter: Reporter = presentation
    if settings.profile_signposts:
        presentation_reporter = stack.enter_context(
            SignpostReporter(
                presentation,
                log_path=settings.profile_signpost_log,
                build_dir=settings.app_cache_dir / "_native" / "signpost",
            )
        )
    return TimingReporter(
        presentation_reporter,
        memory_sampler=create_mlx_memory_sampler(),
    )


__all__ = ["command_reporter"]
