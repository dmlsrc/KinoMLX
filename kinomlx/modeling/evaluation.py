"""Typed evaluation helpers over MLX arrays."""

from collections.abc import Iterable
from typing import Protocol, cast

import mlx.core as mx


class _AsyncEval(Protocol):
    def __call__(self, *arrays: mx.array) -> None: ...


def async_eval_arrays(arrays: Iterable[mx.array]) -> None:
    """Dispatch evaluation without waiting; MLX ships ``async_eval`` untyped."""
    selected = list(arrays)
    if selected:
        cast(_AsyncEval, mx.async_eval)(*selected)


__all__ = ["async_eval_arrays"]
