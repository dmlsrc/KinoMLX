"""Public GMNet component lease and the default native provider."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from kinomlx.components import ComponentLease
from kinomlx.reporting import NullReporter, Reporter

from .net import GMNet, load_gmnet_weights
from .resources import GMNetResources


class GMNetPort(Protocol):
    """Callable generator surface consumed by the GMNet recipe."""

    def __call__(
        self,
        image: mx.array,
        thumbnail: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]: ...


class GMNetComponents(Protocol):
    """Provider surface for bounded GMNet component ownership."""

    def generator(self, resources: GMNetResources) -> ComponentLease[GMNetPort]: ...


def _cleanup_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def load_generator(
    resources: GMNetResources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[GMNetPort]:
    """Load, evaluate, and lease one GMNet generator."""
    sink = reporter if reporter is not None else NullReporter()
    phase = "load GMNet"
    sink.phase_start(phase, total=1, unit="model")
    model: GMNet | None = None
    try:
        if resources.mlx_cache_limit_bytes is not None:
            mx.set_cache_limit(resources.mlx_cache_limit_bytes)
        model, _metadata = load_gmnet_weights(resources.weights_path)
        sink.phase_advance(phase)
        port: GMNetPort = model
        return ComponentLease(port, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise
    finally:
        sink.phase_end(phase)


@dataclass(frozen=True)
class NativeGMNetComponents:
    """Default provider implementing the public GMNet component surface."""

    reporter: Reporter | None = None

    def generator(self, resources: GMNetResources) -> ComponentLease[GMNetPort]:
        return load_generator(resources, reporter=self.reporter)


__all__ = [
    "GMNetComponents",
    "GMNetPort",
    "NativeGMNetComponents",
    "load_generator",
]
