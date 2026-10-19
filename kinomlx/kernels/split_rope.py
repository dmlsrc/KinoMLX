"""Split-half rotary position embedding in one pass.

LTX-2 rotates each attention's flattened-head queries and keys, ``(batch, tokens, heads * dim)``
in FP16 or BF16, by FP32 cosine and sine tables of ``dim / 2`` entries per head and token. MLX's
graph for it splits each head into halves, forms four FP32 products, a difference and a sum,
concatenates the halves, restores the layout and casts back: several full passes over FP32
intermediates. This kernel reads each pair of values and their two table entries once and writes
the rotated pair once.

Exactness. Each product is rounded in FP32 in a statement of its own, as MLX's separate
operations round it, before the difference or sum and the one rounding to the activation type,
so the result keeps the graph's bits. ``available`` checks that once per process and dtype on a
small ragged case; otherwise callers keep the graph.

Speed, on an M1 Max with LTX-2.3's tables: rotating the queries and keys of a 30,576-token video
self-attention (4096 wide) took 5.5 ms against 61.8 for the graph, and the 2048-wide video side
of the two audio-video attentions 3.0 ms against 24.5, at up to 363 GB/s.

The tables are read token-major, ``(batch, tokens, heads, dim / 2)``, the layout LTX builds them
in before transposing them to ``(batch, heads, tokens, dim / 2)``, so the transposed tables reach
the kernel without a copy and its table reads run along each token's row like its activation
reads.
"""

from functools import cache
from typing import cast

import mlx.core as mx

from ._typing import MetalKernel

_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_TYPE_NAMES = {mx.float16: "float16", mx.bfloat16: "bfloat16", mx.float32: "float32"}

_SOURCE = r"""
    const uint j = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint h = j / HALF;
    const uint i = j - h * HALF;
    const ulong row = ((ulong)b * TOKENS + t) * (ulong)(HEADS * 2 * HALF) + h * (2 * HALF) + i;
    const ulong entry = ((ulong)(TABLE_BATCHED ? b : 0u) * TOKENS + t) * (ulong)(HEADS * HALF) + j;
    const float c = cos_t[entry];
    const float s = sin_t[entry];
    const float first = (float)x[row];
    const float second = (float)x[row + HALF];
    // One rounding per product, then the difference or sum, then the cast: MLX's operations.
    const float first_c = first * c;
    const float second_s = second * s;
    const float rotated_first = first_c - second_s;
    const float first_s = first * s;
    const float second_c = second * c;
    const float rotated_second = first_s + second_c;
    out[row] = static_cast<T>(rotated_first);
    out[row + HALF] = static_cast<T>(rotated_second);
"""

_enabled = True


def set_enabled(enabled: bool) -> None:
    """Rotate through the kernel wherever it applies (the default), or keep MLX's graph."""
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    """Whether the RoPE kernel is switched on."""
    return _enabled


@cache
def _kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="kinomlx_split_rope",
            input_names=["x", "cos_t", "sin_t"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )


def _graph(value: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    """MLX's operations for the same rotation, the reference ``available`` checks against."""
    batch = value.shape[0]
    _table_batch, heads, tokens, half = cos_freqs.shape
    heads_first = value.reshape(batch, tokens, heads, 2 * half).transpose(0, 2, 1, 3)
    first = heads_first[..., :half]
    second = heads_first[..., half:]
    output = mx.concatenate(
        [first * cos_freqs - second * sin_freqs, first * sin_freqs + second * cos_freqs],
        axis=-1,
    )
    return output.transpose(0, 2, 1, 3).reshape(value.shape).astype(value.dtype)


def _launch(value: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    table_batch, heads, tokens, half = (int(size) for size in cos_freqs.shape)
    batch = int(value.shape[0])
    return _kernel()(
        inputs=[value, cos_freqs.transpose(0, 2, 1, 3), sin_freqs.transpose(0, 2, 1, 3)],
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
        grid=(heads * half, tokens, batch),
        threadgroup=(min(heads * half, 256), 1, 1),
        template=[
            ("T", value.dtype),
            ("HALF", half),
            ("HEADS", heads),
            ("TOKENS", tokens),
            ("TABLE_BATCHED", int(table_batch == batch and batch > 1)),
        ],
    )[0]


@cache
def _available(dtype_name: str) -> bool:
    # Ragged and batched, deterministic values: an odd head half, tables shared by the batch.
    dtype = getattr(mx, dtype_name)
    batch, heads, tokens, half = 2, 3, 7, 5
    count = batch * tokens * heads * 2 * half
    value = (mx.sin(mx.arange(count, dtype=mx.float32) * 0.7310585) * 3.0).astype(dtype)
    value = value.reshape(batch, tokens, heads * 2 * half)
    angles = mx.arange(tokens * heads * half, dtype=mx.float32) * 0.4142135
    angles = angles.reshape(1, tokens, heads, half).transpose(0, 2, 1, 3)
    cos_freqs, sin_freqs = mx.cos(angles), mx.sin(angles)
    try:
        rotated = _launch(value, cos_freqs, sin_freqs)
        mx.eval(rotated)
    except RuntimeError:
        return False
    return bool(mx.array_equal(rotated, _graph(value, cos_freqs, sin_freqs)).item())


def available(dtype: mx.Dtype) -> bool:
    """Whether the kernel builds, runs and gives MLX's bits for this activation dtype here."""
    return dtype in _DTYPES and _available(_TYPE_NAMES[dtype])


def applies(value: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> bool:
    """Whether ``rotate`` handles these flattened-head activations and FP32 head-major tables."""
    if not _enabled or value.ndim != 3 or cos_freqs.ndim != 4:
        return False
    if cos_freqs.shape != sin_freqs.shape:
        return False
    if cos_freqs.dtype != mx.float32 or sin_freqs.dtype != mx.float32:
        return False
    table_batch, heads, tokens, half = (int(size) for size in cos_freqs.shape)
    batch = int(value.shape[0])
    if table_batch not in (1, batch) or tuple(value.shape) != (batch, tokens, heads * 2 * half):
        return False
    return available(value.dtype)


def rotate(value: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    """Rotate each head's halves of ``value`` by the tables, keeping MLX's graph's bits."""
    if not applies(value, cos_freqs, sin_freqs):
        raise ValueError("the RoPE kernel does not apply to these activations and tables")
    return _launch(value, cos_freqs, sin_freqs)


__all__ = ["applies", "available", "enabled", "rotate", "set_enabled"]
