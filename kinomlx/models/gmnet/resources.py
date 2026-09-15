"""Immutable checkpoint inventory and execution policy for GMNet."""

from dataclasses import dataclass
from pathlib import Path

from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .catalog import GMNetVariantSpec, resolve_variant_weights, variant_spec
from .settings import GMNetSettings


@dataclass(frozen=True)
class GMNetResources:
    """Prepared GMNet assets and policies, never live model weights."""

    weights_path: Path
    spec: GMNetVariantSpec
    memory_policy: MLXMemoryPolicy = MLXMemoryPolicy()


def prepare_resources(
    settings: GMNetSettings,
    *,
    infrastructure: Settings | None = None,
    reporter: Reporter | None = None,
) -> GMNetResources:
    """Resolve a GMNet checkpoint and immutable execution policy."""
    host = infrastructure if infrastructure is not None else Settings.from_env()
    sink = reporter if reporter is not None else NullReporter()
    phase = "prepare resources"
    sink.phase_start(phase, total=1, unit="checkpoint")
    try:
        settings.validate()
        host.validate()
        spec = variant_spec(settings.variant)
        weights_path = resolve_variant_weights(
            spec.variant,
            settings.weights_path,
            app_cache_dir=host.app_cache_dir,
        )
        resources = GMNetResources(
            memory_policy=MLXMemoryPolicy.from_settings(host),
            weights_path=weights_path.expanduser().absolute(),
            spec=spec,
        )
        sink.phase_advance(phase)
        return resources
    finally:
        sink.phase_end(phase)


__all__ = ["GMNetResources", "prepare_resources"]
