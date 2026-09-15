"""Bounded, declared transformations of one sparse configuration layer."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

FieldPath = tuple[str, ...]
LayerPhase = Literal[
    "source", "before_implications", "after_implications", "effective", "serialization"
]
TRANSFORM_PHASES = (
    "source",
    "before_implications",
    "after_implications",
    "effective",
    "serialization",
)

# A rule can consume a virtual input without leaving a null assignment behind.
DELETE_FIELD = object()


@dataclass(frozen=True)
class LayerTransform:
    """A field patch with declared inputs, outputs, effects, and sequence dependencies.

    The runner supplies only declared inputs. Returned keys are checked against
    declared writes; missing keys mean no assignment, None is a clear marker,
    and DELETE_FIELD removes a consumed input.
    """

    name: str
    reads: frozenset[FieldPath]
    writes: frozenset[FieldPath]
    run: Callable[[Mapping[FieldPath, object]], Mapping[FieldPath, object]]
    phase: LayerPhase = "after_implications"
    after: tuple[str, ...] = ()
    group_paths: tuple[tuple[str, tuple[FieldPath, ...]], ...] = ()
    effects: frozenset[Literal["read_files"]] = frozenset()
