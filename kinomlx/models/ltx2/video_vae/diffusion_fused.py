"""The FP16 kernel route of the LTX-2.5 diffusion video decoder's blocks.

A block whose hidden states are FP16 and of batch one, with 64-wide heads and a grid that holds
the attention kernels' key box, runs on Metal kernels over its whole token grid at once
(``kernel_fallback`` names the condition that fails otherwise):

- its products on MPP's run-time-row form, with the residual added in the product's epilogue
  (``linear``);
- its RMS norms and AdaLN modulation in one pass (``norm``);
- its Q and K projections with their norms, the query scale and the rotary embedding in one
  pass, and its neighborhood attention on kernels that score only the keys inside each window;
- its SwiGLU as one gate/up product, one SiLU-times-up pass, and the down product with the
  residual, in tiles of ``SWIGLU_TILE_TOKENS`` tokens (``swiglu``).

Where a piece's kernel does not apply, the piece runs the MLX operations it replaces, so a
small or unusual shape costs speed, not correctness. Every piece except the attention computes
the replaced operations' bits; the attention's online softmax sums in another order, by no more
than the masked path differs from itself under another legal tile plan.

The kernels address their arrays from 64-bit bases, so a volume may exceed ``2**31`` elements.
What bounds it is ``mx.fast.metal_kernel``'s signed 32-bit launch grid and the kernels' 32-bit
row counts (``addressable``): the fused Q/K pass launches 128 threads a 32-token tile and head,
so four heads address 134,217,696 tokens, a 68.7 GB feature. ``kernel_fallback`` raises for a
volume past that, and the decode planner proposes no tile past it.

The masked path evaluates each attention tile and each SwiGLU tile to bound its score and
activation working sets. The kernel route has no score matrix; it bounds a block's live set by
where it evaluates: after the context product, after Q, K and V (the normalized rows die
before the attention output is allocated), after the attention (Q, K and V die before the proj
output is), after the proj, and after the SwiGLU, whose tiles' rows are assigned into one
buffer in place, a chain that keeps the tiles' temporaries to the few MLX runs ahead of the
GPU (and the result of which the caller takes). ``mx.eval`` returns on the GPU's event, ahead of
the completion handler that frees the evaluated inputs, so the attention's evaluations are
followed by ``mx.synchronize``; without it the next allocation misses the freed buffers and the
block holds one more. On an M1 Max a 768 x 448, 121-frame FP16 decode took 12.8 s on this route
against 54.0 s on the masked path, both with a 16 GiB allocator cache.
"""

import math

import mlx.core as mx

from kinomlx.kernels import mpp_linear, rms_norm, silu
from kinomlx.models.ltx2.kernels import diffusion_passes, neighborhood_attention
from kinomlx.models.ltx2.vocabularies import DiffusionKernelFallback

Grid = tuple[int, int, int]

# Tokens per SwiGLU tile; at the final stage a tile's gate/up product takes 256 MiB. The tiles
# are assigned into the result in order, and MLX closes a command buffer after a few products
# that size and keeps at most ten in flight, so only a few tiles' temporaries exist at once
# (1.9 GB, about four tiles, measured on a 768x448x65 decode); they and the block's full-volume
# buffers fit a 16 GiB cache together, and 192-frame 1024x576 tiles at 8 GiB fault no more
# with the tiles left to one evaluation than evaluated four at a time.
SWIGLU_TILE_TOKENS = 65_536


def addressable(grid: Grid, *, heads: int, window: Grid) -> bool:
    """Whether the kernel route addresses a block whose hidden states span a ``(T, H, W)`` token
    ``grid`` with ``heads`` 64-wide heads and ``window``: every kernel that runs over the whole
    volume launches within MLX's signed 32-bit grid dimensions and counts its rows in 32 bits
    (``neighborhood_attention.launch_fits``, ``diffusion_passes.rows_addressable``,
    ``mpp_linear.rows_addressable``). The SwiGLU's kernels see ``SWIGLU_TILE_TOKENS`` rows."""
    tokens = math.prod(grid)
    return (
        neighborhood_attention.launch_fits(grid, window)
        and diffusion_passes.rows_addressable(tokens, heads=heads)
        and mpp_linear.rows_addressable(tokens, heads * neighborhood_attention.HEAD_DIM)
    )


def kernel_fallback(
    hidden_states: mx.array,
    *,
    heads: int,
    head_dim: int,
    window: Grid,
) -> DiffusionKernelFallback | None:
    """Why a block of ``(batch, T, H, W, C)`` hidden states keeps the masked path, or ``None``
    when the kernel route computes it. Raises ``ValueError`` for an FP16 volume the route
    cannot address (``addressable``); the decode planner proposes no such tile."""
    if not neighborhood_attention.enabled():
        return DiffusionKernelFallback.DISABLED
    if hidden_states.dtype != mx.float16:
        return DiffusionKernelFallback.DTYPE
    if int(hidden_states.shape[0]) != 1:
        return DiffusionKernelFallback.BATCH
    if head_dim != neighborhood_attention.HEAD_DIM:
        return DiffusionKernelFallback.HEAD_DIM
    grid = (int(hidden_states.shape[1]), int(hidden_states.shape[2]), int(hidden_states.shape[3]))
    if not addressable(grid, heads=heads, window=window):
        raise ValueError(
            f"a {grid[0]} x {grid[1]} x {grid[2]} token volume exceeds what the kernel route "
            "addresses; decode it in temporal or spatial tiles"
        )
    if not neighborhood_attention.supports((1, *grid, heads, head_dim), window):
        return DiffusionKernelFallback.WINDOW
    return None


def linear(
    value: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    residual: mx.array | None = None,
) -> mx.array:
    """``residual + mx.addmm(bias, value, weight.T)`` for ``(rows, k)`` rows and an ``(n, k)``
    weight, on MPP's run-time-row form where it applies; the same bits either way."""
    if mpp_linear.rows_applies(value, weight, bias, residual):
        return mpp_linear.linear_rows(value, weight, bias, residual=residual)
    product = value @ weight.T if bias is None else mx.addmm(bias, value, weight.T)
    return product if residual is None else residual + product


def norm(
    value: mx.array,
    weight: mx.array,
    scale1p: mx.array | None = None,
    shift: mx.array | None = None,
) -> mx.array:
    """The blocks' RMS norm of ``(rows, C)`` rows, then ``* scale1p + shift`` when given."""
    if diffusion_passes.rms_norm_applies(value, weight, scale1p, shift):
        return diffusion_passes.rms_norm(value, weight, scale1p, shift)
    normalized = rms_norm(value, weight, 1e-6)
    if scale1p is None or shift is None:
        return normalized
    return normalized * scale1p + shift


def swiglu(
    value: mx.array,
    gate: mx.array,
    up: mx.array,
    down: mx.array,
    residual: mx.array,
    *,
    tile_tokens: int = SWIGLU_TILE_TOKENS,
) -> mx.array:
    """``residual + down(silu(gate(value)) * up(value))`` over ``(rows, C)`` rows in token tiles,
    with the gate and up products as one product of the stacked weights, evaluated.

    Each tile's rows are assigned into one zero-filled result, which MLX updates in place while
    nothing else holds the result, so the tiles' outputs never sit beside a joined copy, and
    which chains the tiles, so only the few MLX has encoded ahead of the GPU hold temporaries.
    The result is evaluated before it returns, so the rows, the residual and the last tiles'
    temporaries are free before the caller allocates again.
    """
    rows = int(value.shape[0])
    width = int(down.shape[0])
    hidden = int(down.shape[1])
    gate_up = mx.concatenate([gate, up], axis=0)
    result: mx.array | None = None
    for start in range(0, rows, tile_tokens):
        stop = min(start + tile_tokens, rows)
        product = linear(value[start:stop], gate_up)
        if diffusion_passes.silu_mul_applies(product, hidden):
            activated = diffusion_passes.silu_mul(product, hidden)
        else:
            activated = silu(product[:, :hidden]) * product[:, hidden:]
        del product
        tile = linear(activated, down, residual=residual[start:stop])
        del activated
        if result is None:
            result = mx.zeros((rows, width), dtype=tile.dtype)
        result[start:stop] = tile
        del tile
    if result is None:
        raise ValueError("the SwiGLU needs at least one row")
    mx.eval(result)
    return result


__all__ = [
    "SWIGLU_TILE_TOKENS",
    "addressable",
    "kernel_fallback",
    "linear",
    "norm",
    "swiglu",
]
