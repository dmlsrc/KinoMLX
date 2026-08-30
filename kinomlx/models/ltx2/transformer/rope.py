"""Split rotary embeddings for LTX-2 video and audio tokens."""

from __future__ import annotations

import math
from enum import Enum
from functools import lru_cache

import mlx.core as mx


class LTXRopeType(Enum):
    """Position-embedding layouts supported by implemented LTX-2 graphs."""

    SPLIT = "split"


def apply_split_rotary_emb(
    value: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
) -> mx.array:
    """Apply split-half RoPE and preserve the activation dtype."""
    if cos_freqs.shape != sin_freqs.shape:
        raise ValueError("RoPE cosine and sine tensors must have matching shapes")
    input_dtype = value.dtype
    reshape_heads = value.ndim != 4 and cos_freqs.ndim == 4
    if reshape_heads:
        batch, heads, tokens, _ = cos_freqs.shape
        if cos_freqs.shape[0] not in (1, value.shape[0]):
            raise ValueError("RoPE batch must be one or match the input batch")
        if value.shape[1] != tokens:
            raise ValueError("RoPE token count must match the input token count")
        value = value.reshape(value.shape[0], tokens, heads, -1).transpose(0, 2, 1, 3)

    half = value.shape[-1] // 2
    if value.shape[-1] % 2:
        raise ValueError("split RoPE requires an even head dimension")
    first = value[..., :half]
    second = value[..., half:]
    output = mx.concatenate(
        [
            first * cos_freqs - second * sin_freqs,
            first * sin_freqs + second * cos_freqs,
        ],
        axis=-1,
    )
    if reshape_heads:
        batch, heads, tokens, head_dim = output.shape
        output = output.transpose(0, 2, 1, 3).reshape(
            batch,
            tokens,
            heads * head_dim,
        )
    return output.astype(input_dtype)


def apply_rotary_emb(
    value: mx.array,
    freqs_cis: tuple[mx.array, mx.array],
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> mx.array:
    """Apply a checkpoint-selected RoPE layout."""
    if rope_type is not LTXRopeType.SPLIT:
        raise ValueError(f"unsupported RoPE type: {rope_type}")
    return apply_split_rotary_emb(value, *freqs_cis)


@lru_cache(maxsize=8)
def _frequency_grid(
    theta: float,
    position_dims: int,
    inner_dim: int,
    double_precision: bool,
) -> tuple[float, ...]:
    """Build the canonical logarithmic grid with host double precision."""
    count = inner_dim // (2 * position_dims)
    if count <= 0:
        raise ValueError("RoPE dimension is too small for its position axes")
    exponents = (0.0,) if count == 1 else tuple(index / (count - 1) for index in range(count))
    # Host libm is the only genuine double-precision route on Apple GPUs.
    # The result is narrowed when it enters the FP32 MLX frequency pipeline.
    # Keep the flag in the cache identity because future checkpoints may select
    # a deliberately different lower-precision construction.
    _ = double_precision
    return tuple(math.pow(theta, exponent) * math.pi / 2 for exponent in exponents)


def _fractional_positions(
    positions: mx.array,
    max_pos: tuple[int, ...],
    *,
    use_middle_indices_grid: bool,
) -> mx.array:
    if positions.shape[1] != len(max_pos):
        raise ValueError("position axes must match max_pos")
    if use_middle_indices_grid:
        if positions.ndim != 4 or positions.shape[-1] != 2:
            raise ValueError("middle-position RoPE expects (B, axes, T, 2) bounds")
        positions = (positions[..., 0] + positions[..., 1]) / 2.0
    elif positions.ndim == 4:
        positions = positions[..., 0]
    elif positions.ndim != 3:
        raise ValueError("RoPE positions must have three or four dimensions")
    scale = mx.array(max_pos, dtype=mx.float32)[None, :, None]
    return (positions.astype(mx.float32) / scale).transpose(0, 2, 1)


def precompute_freqs_cis(
    indices_grid: mx.array,
    *,
    dim: int,
    out_dtype: mx.Dtype,
    theta: float = 10000.0,
    max_pos: tuple[int, ...] | list[int] = (20, 2048, 2048),
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
    use_double_precision: bool = False,
) -> tuple[mx.array, mx.array]:
    """Precompute split RoPE cosine/sine tensors for one token grid."""
    if rope_type is not LTXRopeType.SPLIT:
        raise ValueError(f"unsupported RoPE type: {rope_type}")
    if dim % num_attention_heads:
        raise ValueError("RoPE dimension must divide evenly across attention heads")
    resolved_max_pos = tuple(max_pos)
    fractional = _fractional_positions(
        indices_grid,
        resolved_max_pos,
        use_middle_indices_grid=use_middle_indices_grid,
    )
    grid = mx.array(
        _frequency_grid(
            theta,
            indices_grid.shape[1],
            dim,
            use_double_precision,
        ),
        dtype=mx.float32,
    )
    scaled = fractional * 2.0 - 1.0
    frequencies = scaled[..., None] * grid[None, None, None, :]
    frequencies = frequencies.transpose(0, 1, 3, 2).reshape(
        frequencies.shape[0],
        frequencies.shape[1],
        -1,
    )

    expected = dim // 2
    padding = expected - frequencies.shape[-1]
    if padding < 0:
        raise ValueError("RoPE frequency grid exceeds half the model dimension")
    cos_freqs = mx.cos(frequencies)
    sin_freqs = mx.sin(frequencies)
    if padding:
        cos_freqs = mx.concatenate(
            [mx.ones((*cos_freqs.shape[:-1], padding)), cos_freqs],
            axis=-1,
        )
        sin_freqs = mx.concatenate(
            [mx.zeros((*sin_freqs.shape[:-1], padding)), sin_freqs],
            axis=-1,
        )
    batch, tokens, _ = cos_freqs.shape
    cos_freqs = cos_freqs.reshape(batch, tokens, num_attention_heads, -1)
    sin_freqs = sin_freqs.reshape(batch, tokens, num_attention_heads, -1)
    return (
        cos_freqs.transpose(0, 2, 1, 3).astype(out_dtype),
        sin_freqs.transpose(0, 2, 1, 3).astype(out_dtype),
    )


__all__ = [
    "LTXRopeType",
    "apply_rotary_emb",
    "apply_split_rotary_emb",
    "precompute_freqs_cis",
]
