"""Internal explanations of supplied and derived configuration fields."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .paths import TargetPath


@dataclass(frozen=True)
class SourceLocation:
    """A source identifier and optional one-based assignment position."""

    kind: str
    identifier: str
    position: int | None = None


@dataclass(frozen=True)
class FieldOrigin:
    """A field's origin and the input origins used by its producing rule.

    Explicit presence is independent of normalization: a decoded prompt remains
    supplied intent, while an implied sidecar flag does not become explicit.
    These records are diagnostic metadata and are never serialized as config.
    """

    path: TargetPath
    kind: Literal["default", "explicit", "normalized", "implied", "inherited"]
    explicit: bool
    location: SourceLocation | None = None
    rule: str | None = None
    inputs: tuple[FieldOrigin, ...] = ()


@dataclass(frozen=True)
class PreparedConfig:
    """Sparse prepared values and the explanations of their surviving fields."""

    values: dict[str, object]
    origins: Mapping[TargetPath, FieldOrigin]
    replacements: frozenset[TargetPath] = frozenset()
    layer_explicit: frozenset[TargetPath] | None = None

    @property
    def active_explicit(self) -> frozenset[TargetPath]:
        """Presence in the active layer, independent of inherited source explicitness."""
        return self.explicit if self.layer_explicit is None else self.layer_explicit

    @property
    def explicit(self) -> frozenset[TargetPath]:
        return frozenset(path for path, origin in self.origins.items() if origin.explicit)
