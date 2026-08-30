"""Immutable checkpoint inventory and execution policy for GMNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kinomlx.reporting import NullReporter, Reporter
from kinomlx.settings import Settings

from .catalog import GMNetVariantSpec, resolve_variant_weights, variant_spec
from .settings import GMNetSettings


@dataclass(frozen=True)
class GMNetResources:
    """Prepared GMNet assets and policies, never live model weights."""

    weights_path: Path
    spec: GMNetVariantSpec
    mlx_cache_limit_bytes: int | None


def prepare_resources(
    settings: GMNetSettings,
    *,
    infrastructure: Settings | None = None,
    reporter: Reporter | None = None,
) -> GMNetResources:
    """Resolve a GMNet checkpoint and immutable execution policy."""
    host = infrastructure if infrastructure is not None else Settings.from_env()
    sink = reporter if reporter is not None else NullReporter()
    phase = "prepare GMNet resources"
    sink.phase_start(phase, total=1, unit="checkpoint")
    try:
        settings.validate()
        host.validate()
        spec = variant_spec(settings.variant)
        weights_path = resolve_variant_weights(
            spec.variant,
            settings.weights_path,
            cache_dir=host.cache_dir,
        )
        resources = GMNetResources(
            weights_path=weights_path.expanduser().absolute(),
            spec=spec,
            mlx_cache_limit_bytes=(
                None if host.mlx_cache_limit_gb is None else int(host.mlx_cache_limit_gb * 1024**3)
            ),
        )
        sink.phase_advance(phase)
        return resources
    finally:
        sink.phase_end(phase)


__all__ = ["GMNetResources", "prepare_resources"]
