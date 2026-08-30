"""Model-neutral persistence vocabulary for materialized tensor artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx


@dataclass(frozen=True)
class TensorArtifact:
    """One named tensor bundle offered to an injected persistence sink.

    Artifact names, tensor keys, metadata, and reporting labels are supplied by
    the model recipe that owns their meaning. Layer 0 only transports the
    immutable envelope and never interprets stage numbers or modalities.
    """

    name: str
    tensors: tuple[tuple[str, mx.array], ...]
    metadata: tuple[tuple[str, str], ...] = ()
    reporting_phase: str = "save artifact"


@runtime_checkable
class ArtifactSink(Protocol):
    """Optional persistence port for a materialized tensor artifact."""

    def save(self, artifact: TensorArtifact) -> None:
        """Persist ``artifact`` without retaining its tensor values."""


class NullArtifactSink:
    """No-op artifact sink used by library callers unless explicitly enabled."""

    def save(self, artifact: TensorArtifact) -> None:
        pass


__all__ = ["ArtifactSink", "NullArtifactSink", "TensorArtifact"]
