"""Typed MLX construction from, and exposure as, Python buffer-protocol objects."""

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


def flat_byte_view(value: mx.array) -> mx.array:
    """Reinterpret one array as its flat row-major bytes.

    A contiguous input shares its buffer; anything else is made contiguous
    first. The result is a one-dimensional ``uint8`` array, so ``memoryview``
    exposes it with the plain byte format that hashing and positional file
    writes accept. MLX reports ``bfloat16`` buffers with a custom format that
    those consumers reject, which is why callers go through this view.
    """
    flat = value if value.ndim == 1 else value.reshape(-1)
    return flat.view(mx.uint8)


__all__ = ["flat_byte_view", "mlx_array_from_buffer"]
