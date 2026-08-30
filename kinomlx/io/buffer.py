"""Typed MLX construction from Python buffer-protocol objects."""

from __future__ import annotations

from collections.abc import Buffer

import mlx.core as mx


def mlx_array_from_buffer(
    value: Buffer,
    *,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """Build an owned MLX array directly from a PEP 3118 buffer.

    MLX accepts ``memoryview`` and other Python buffer-protocol objects at
    runtime. Its current type declaration omits that supported input family,
    so the one narrow suppression belongs at this shared boundary rather than
    at every native media caller. No bytes, list, or NumPy staging copy is
    introduced.
    """
    return mx.array(value, dtype=dtype)  # type: ignore[arg-type]


__all__ = ["mlx_array_from_buffer"]
