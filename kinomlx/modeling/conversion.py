"""Shared lazy source conversion records, independent of persistence."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import BoundTensorDescriptor


@dataclass(frozen=True)
class SourceTensor:
    """One raw checkpoint tensor selected for conversion.

    ``convert`` is False for auxiliary tensors consumed by another conversion,
    such as FP8 scale companions.
    """

    source_file: Path
    source_key: str
    target_key: str
    convert: bool = True


@dataclass(frozen=True)
class Conversion:
    """The lazy target tensors produced from one source tensor."""

    tensors: tuple[tuple[str, mx.array], ...] = ()


def plan_sources(
    plan: Iterable[BoundTensorDescriptor],
    *,
    convert: Callable[[BoundTensorDescriptor], bool] | None = None,
) -> tuple[SourceTensor, ...]:
    """Derive conversion inputs from validated source bindings."""
    return tuple(
        SourceTensor(
            source_file=entry.source,
            source_key=entry.binding.source_key,
            target_key=entry.target_key,
            convert=True if convert is None else convert(entry),
        )
        for entry in plan
    )
