"""Shared CLI host services for generic and model-specific converters."""

from __future__ import annotations

import argparse
from contextlib import ExitStack

from kinomlx.reporting import Reporter, TimingReporter
from kinomlx.settings import Settings, settings_from_args


def resolve_conversion_settings(options: argparse.Namespace) -> Settings:
    """Compose and validate infrastructure settings for a converter command."""
    settings = settings_from_args(options, Settings.from_env())
    settings.validate()
    return settings


def conversion_reporter(stack: ExitStack, settings: Settings) -> TimingReporter:
    """Build the normal Rich, signpost, timing, and allocator reporter stack."""
    from kinomlx.debug import create_mlx_memory_sampler
    from kinomlx.ui import RichReporter

    presentation = stack.enter_context(RichReporter(disable=settings.quiet))
    presentation_reporter: Reporter = presentation
    if settings.profile_signposts:
        from kinomlx.profiling import SignpostReporter

        presentation_reporter = stack.enter_context(
            SignpostReporter(
                presentation,
                log_path=settings.profile_signpost_log,
                build_dir=settings.cache_dir / "_native" / "signpost",
            )
        )
    return TimingReporter(
        presentation_reporter,
        memory_sampler=create_mlx_memory_sampler(),
    )


def apply_mlx_cache_limit(settings: Settings) -> None:
    """Apply the resolved allocator cache limit immediately before conversion."""
    if settings.mlx_cache_limit_gb is None:
        return
    import mlx.core as mx

    mx.set_cache_limit(int(settings.mlx_cache_limit_gb * 1024**3))


__all__ = [
    "apply_mlx_cache_limit",
    "conversion_reporter",
    "resolve_conversion_settings",
]
