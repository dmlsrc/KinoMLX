"""Sol-Attn routed sparse attention for LTX video self-attention.

Sol-Attn is the training-free sparse attention of NVlabs' Sana Sol-Engine (Apache-2.0). Tokens
form 64-token blocks. Each query block scores every key block by its mean query against the
block's mean key and routes the blocks above ``mean + tau * std`` of those scores (with a
diagonal key covariance), plus its own block and both neighbors. Routed blocks are computed
exactly. Every other block still contributes, through its mean key scored against each real
query, weighted by the block length, with the block's mean value. Larger ``tau`` routes fewer
blocks: faster, and further from dense attention.

Tokens are routed in an order given per call, normally `morton_order`, which makes a block a
compact cube of the latent grid instead of a strip of rows. One call is three steps: a
preparation kernel that gathers q, k and v into that order, converts them to FP16 and takes
the block means; MLX operations that derive the routing; and the attention kernel, which
writes each output row back to its input position in the input dtype.

The attention kernel is an FP16 build: M1 has no BF16 arithmetic, so BF16 inputs are converted
once while preparing, saturating at FP16's largest finite value (inside FP16's normal range the
conversion is exact). Key and value means are rounded to the input dtype first, so routing is
the same as a BF16 build's.
"""

import math
from functools import cache
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx.kernels._typing import MetalKernel

BLOCK = 64
HEAD_DIM = 128
_QUERY_TILE = 32
_THREADS = 128
# Caps registers so four 128-thread threadgroups fit on a GPU core.
_THREAD_CAP = 512
_WORD_BITS = 32
_LOG2E = 1.4426950408889634
# Added to the log2-scaled score variance before the square root.
_VARIANCE_EPSILON = 1.0e-6
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)
_KERNEL_DTYPE = mx.float16

_METAL = Path(__file__).with_name("metal")


def _read_metal(name: str) -> str:
    return (_METAL / name).read_text(encoding="utf-8")


@cache
def _attention_kernel() -> MetalKernel:
    # MLX's generated source has no parameter for this attribute, so the header redefines
    # `kernel` to carry it and the body undefines it before MLX's explicit instantiation.
    header = (
        _read_metal("fp16_attention_header.metal")
        + f"\n#define kernel kernel, max_total_threads_per_threadgroup({_THREAD_CAP})\n"
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_sol_attention_q32k16",
            input_names=[
                "q",
                "k",
                "v",
                "kmean",
                "vmean",
                "lists",
                "counts",
                "routed_bits",
                "rows",
            ],
            output_names=["out"],
            source=_read_metal("sol_attention_body.metal"),
            header=header,
            ensure_row_contiguous=True,
        ),
    )


@cache
def _prepare_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_sol_attention_prepare",
            input_names=["q", "k", "v", "rows"],
            output_names=["qo", "ko", "vo", "qmean", "kmean", "vmean"],
            source=_read_metal("sol_attention_prepare.metal"),
            header=_read_metal("fp16_attention_header.metal"),
            ensure_row_contiguous=True,
        ),
    )


def block_count(tokens: int) -> int:
    """The number of 64-token blocks covering ``tokens``."""
    return (tokens + BLOCK - 1) // BLOCK


def _spread_bits(index: mx.array, bits: int) -> mx.array:
    """Move bit ``i`` of each index to bit ``3 i``."""
    code = mx.zeros_like(index)
    for bit in range(bits):
        code = code | (((index >> bit) & 1) << (3 * bit))
    return code


def morton_order(frames: int, height: int, width: int, *, tokens: int | None = None) -> mx.array:
    """Token permutation listing an ``(F, H, W)`` raster grid along a 3-D Morton curve.

    Bit ``i`` of a token's frame, row and column index lands at bits ``3 i``, ``3 i + 1``
    and ``3 i + 2`` of its code, so on a grid whose sides are multiples of 4 every run of
    64 tokens is a 4x4x4 cube. Entry ``t`` is the raster index of the ``t``-th token in curve
    order. Tokens past the grid, up to ``tokens``, keep their order after it.
    """
    if min(frames, height, width) <= 0:
        raise ValueError("Morton order needs positive grid dimensions")
    grid = frames * height * width
    total = grid if tokens is None else tokens
    if total < grid:
        raise ValueError(f"{total} tokens cannot hold a {frames}x{height}x{width} grid")
    bits = max(frames, height, width).bit_length()
    if 3 * bits > 63:
        raise ValueError("grid dimensions are too large for a 64-bit Morton code")
    axes = (
        _spread_bits(mx.arange(size, dtype=mx.uint64), bits) << shift
        for size, shift in ((frames, 0), (height, 1), (width, 2))
    )
    frame_code, row_code, column_code = axes
    codes = (
        frame_code[:, None, None] | row_code[None, :, None] | column_code[None, None, :]
    ).reshape(-1)
    order = mx.argsort(codes).astype(mx.int32)
    if total > grid:
        order = mx.concatenate([order, mx.arange(grid, total, dtype=mx.int32)])
    return order


def _check_inputs(q: mx.array, k: mx.array, v: mx.array, heads: int, order: mx.array) -> None:
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"Sol attention needs float16 or bfloat16 q, k and v, got {q.dtype}, {k.dtype}, "
            f"{v.dtype}"
        )
    if q.ndim != 2 or q.shape[1] != heads * HEAD_DIM or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"Sol attention needs (tokens, {heads} * {HEAD_DIM}) q, k and v, got {q.shape}, "
            f"{k.shape}, {v.shape}"
        )
    if order.shape != (q.shape[0],):
        raise ValueError(f"the token order must hold one index per token, got {order.shape}")
    if block_count(q.shape[0]) >= 2**16:
        raise ValueError("Sol attention supports fewer than 65,536 blocks")


def _prepare(
    q: mx.array, k: mx.array, v: mx.array, *, heads: int, order: mx.array
) -> tuple[mx.array, ...]:
    """Gather, convert, and take block means: FP16 q, k, v, FP32 query means, FP16 K/V means."""
    tokens = q.shape[0]
    blocks = block_count(tokens)
    columns = heads * HEAD_DIM // 8
    shapes: list[tuple[int, ...]] = [tuple(q.shape)] * 3 + [(heads, blocks, HEAD_DIM)] * 3
    return tuple(
        _prepare_kernel()(
            inputs=[q, k, v, order],
            template=[
                ("I", q.dtype),
                ("T", _KERNEL_DTYPE),
                ("H", heads),
                ("D", HEAD_DIM),
                ("TOKENS", tokens),
                ("NB", blocks),
            ],
            grid=(columns, blocks, 3),
            threadgroup=(min(256, columns), 1, 1),
            output_shapes=shapes,
            output_dtypes=[_KERNEL_DTYPE] * 3 + [mx.float32, _KERNEL_DTYPE, _KERNEL_DTYPE],
        )
    )


def route_blocks(
    query_mean: mx.array, key_mean: mx.array, *, tau: float
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Route key blocks from block means: mask, ascending block lists, counts and mask words.

    ``query_mean`` is FP32 and ``key_mean`` the rounded key means, both ``(heads, blocks,
    dim)``. Each row's list holds its routed blocks in ascending order, then the rest.
    """
    heads, blocks, dim = query_mean.shape
    keys = key_mean.astype(mx.float32)
    log2_scale = dim**-0.5 * _LOG2E
    scores = (query_mean @ keys.transpose(0, 2, 1)) * log2_scale
    center = keys.mean(axis=1, keepdims=True)
    raw_mean = (query_mean * center).sum(axis=-1)
    key_variance = ((keys - center) ** 2).mean(axis=1, keepdims=True)
    raw_variance = (query_mean**2 * key_variance).sum(axis=-1)
    variance = mx.maximum(raw_variance, 0.0) * (log2_scale * log2_scale)
    limit = raw_mean * log2_scale + tau * mx.sqrt(variance + _VARIANCE_EPSILON)
    index = mx.arange(blocks)
    neighbor = mx.abs(index[:, None] - index[None, :]) <= 1
    routes = neighbor[None] | (scores > limit[..., None])

    counts = routes.sum(axis=-1).astype(mx.int32)
    lists = mx.argsort(mx.where(routes, index, index + blocks), axis=-1).astype(mx.uint16)
    words = (blocks + _WORD_BITS - 1) // _WORD_BITS
    padded = routes
    if words * _WORD_BITS != blocks:
        padded = mx.pad(routes, [(0, 0), (0, 0), (0, words * _WORD_BITS - blocks)])
    weights = mx.left_shift(mx.array(1, dtype=mx.uint32), mx.arange(_WORD_BITS, dtype=mx.uint32))
    routed_bits = (
        (padded.reshape(heads, blocks, words, _WORD_BITS).astype(mx.uint32) * weights)
        .sum(axis=-1)
        .astype(mx.uint32)
    )
    return routes, lists, counts, routed_bits


def sol_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    heads: int,
    tau: float,
    order: mx.array,
) -> mx.array:
    """Sol-Attn of ``(tokens, heads * 128)`` inputs, routed in ``order``, in input order.

    ``order`` is a token permutation such as `morton_order`: blocks are formed and computed in
    that order, and the output comes back in the inputs' order and dtype.
    """
    if not math.isfinite(tau):
        raise ValueError(f"Sol attention tau must be finite, got {tau}")
    _check_inputs(q, k, v, heads, order)
    tokens = q.shape[0]
    blocks = block_count(tokens)
    gathered_q, gathered_k, gathered_v, query_mean, key_mean, value_mean = _prepare(
        q, k, v, heads=heads, order=order
    )
    _, lists, counts, routed_bits = route_blocks(query_mean, key_mean, tau=tau)
    tiles = (tokens + _QUERY_TILE - 1) // _QUERY_TILE
    return _attention_kernel()(
        inputs=[
            gathered_q,
            gathered_k,
            gathered_v,
            key_mean,
            value_mean,
            lists,
            counts,
            routed_bits,
            order,
        ],
        template=[
            ("T", _KERNEL_DTYPE),
            ("H", heads),
            ("D", HEAD_DIM),
            ("TOKENS", tokens),
            ("NB", blocks),
            ("NW", routed_bits.shape[-1]),
            ("LAST_LEN", tokens - (blocks - 1) * BLOCK),
        ],
        grid=(tiles * _THREADS, heads, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(tokens, heads * HEAD_DIM)],
        output_dtypes=[q.dtype],
    )[0]


__all__ = ["BLOCK", "HEAD_DIM", "block_count", "morton_order", "route_blocks", "sol_attention"]
