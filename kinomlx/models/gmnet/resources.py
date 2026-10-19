"""Immutable checkpoint inventory and execution policy for GMNet."""

from dataclasses import dataclass
from pathlib import Path

from kinomlx._json import JsonObject
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

    def describe(self) -> JsonObject:
        """Report the resolved checkpoint and its numeric contract.

        One checkpoint with no per-component records has nothing to say about
        bindings, graph identities or tensor accounting, so this report stays
        the resolved path, the variant facts and the memory policy rather than
        inventing structure to resemble a multi-component family.
        """
        return {
            "model": "gmnet",
            "variant": self.spec.variant.value,
            "weights": str(self.weights_path),
            "network_scale": self.spec.network_scale,
            "peak_over_sdr_white": self.spec.peak_over_sdr_white,
            "gain_stops": self.spec.gain_stops,
            "sdr_reference_white_nits": self.spec.sdr_reference_white_nits,
            "memory_policy": {
                "cache_limit_gb": self.memory_policy.cache_limit_gb,
                "wired_limit_gb": self.memory_policy.wired_limit_gb,
            },
        }


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
