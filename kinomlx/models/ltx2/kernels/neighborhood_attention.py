"""FP16 exact 3D neighborhood attention for the LTX-2.5 diffusion video decoder.

Every query attends to the ``KT x KH x KW`` keys of its window, which keeps its full size at
the grid edges by shifting inward: on an axis of length ``L`` the window of position ``i`` starts
at ``clamp(i - K // 2, 0, L - K)``. Queries, keys and values are token-major
``(batch, T, H, W, heads, 64)`` FP16 arrays, the layout of the decoder's projections, with the
queries already scaled, so neither side needs a head transpose and no score outside a window is
computed.

Two kernels, each a threadgroup of four SIMD groups with FP32 scores, online softmax and output
accumulation:

- ``runs`` computes a 32-query block of one head over the union of its windows, walked as
  16-key runs along W, two barriers a run. It serves any window whose W extent fits a run.
- ``patches`` gives each SIMD group two frames by a 2 x 2 patch of queries and walks the keys
  as 4 x 4 patches, so the 12 x 12 spatial union of the 11 x 11 windows is nine runs a plane
  with no key that every query of a run misses. It serves the 11 x 11 x 11 window.

A SIMD group none of whose queries sees a run skips its products; every score it would compute
is masked, so the skip is exact, and masked scores weigh exactly zero. On an M1 Max the decoder's
final-stage call (121 x 112 x 192 queries, four heads, 11 x 11 x 11) took 632 ms against
3,078 ms for the masked stock attention it replaces, whose rectangular tiles score 86% of their
cells outside every window; its earlier stages took 17 ms against 246 (3 x 5 x 5) and 3 ms
against 20 (3 x 7 x 7). The outputs differ from the masked path only in summation order, by no
more than that path differs from itself under another legal tile plan.

Each threadgroup forms one 64-bit base address and a 64-bit address for each key run, so the
arrays may hold more than ``2**31`` elements; the launch grid, signed 32-bit in MLX, bounds a
volume at about ``2**29`` tokens of one head and batch (``launch_fits``). Callers keep the
masked path where ``applies`` is false.
"""

import math
from functools import cache
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx.kernels._typing import MetalKernel

HEAD_DIM = 64
_KEY_RUN = 16
_THREADS = 128
# Query blocks (T, H, W) of the runs kernel, measured fastest at the decoder's stage grids.
_RUN_TILES = {(3, 7, 7): (2, 2, 8), (3, 5, 5): (1, 4, 8), (11, 11, 11): (4, 2, 4)}
# The patches kernel stacks four SIMD groups of two frames: an 8 x 2 x 2 query block.
_PATCH_FRAMES = 8
# Threads per threadgroup the compiler may assume. The runs kernel keeps 512; the patches
# kernel builds to fewer registers at 768 with no spills and ran 3.4% faster.
_RUN_THREAD_CAP = 512
_PATCH_THREAD_CAP = 768
# mx.fast.metal_kernel takes signed 32-bit grid dimensions, and the kernels read their
# threadgroup position as int.
_GRID_LIMIT = 2**31 - 1

Grid = tuple[int, int, int]

_METAL = Path(__file__).with_name("metal")
_enabled = True


def set_enabled(enabled: bool) -> None:
    """Use the kernels wherever they apply (the default), or report that they never apply."""
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    """Whether the kernels are switched on."""
    return _enabled


def _read_metal(name: str) -> str:
    return (_METAL / name).read_text(encoding="utf-8")


@cache
def _kernel(body: str, thread_cap: int) -> MetalKernel:
    # MLX's generated source has no parameter for this attribute, so the header redefines
    # `kernel` to carry it and the body undefines it before MLX's explicit instantiation.
    header = (
        _read_metal("fp16_attention_header.metal")
        + f"\n#define kernel kernel, max_total_threads_per_threadgroup({thread_cap})\n"
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=f"ltx_neighborhood_attention_{body}",
            input_names=["q", "k", "v"],
            output_names=["out"],
            source=_read_metal(f"neighborhood_attention_{body}.metal"),
            header=header,
            ensure_row_contiguous=True,
        ),
    )


def run_tile(window: Grid) -> Grid:
    """The runs kernel's 32-query block for ``window``: measured for the decoder's windows,
    otherwise four frames by a row pair (wide W windows) or a row (narrow ones)."""
    if window in _RUN_TILES:
        return _RUN_TILES[window]
    return (4, 2, 4) if window[2] > 9 else (4, 1, 8)


def _patches_fit(window: Grid, grid: Grid) -> bool:
    kt, kh, kw = window
    return (
        window == (11, 11, 11)
        and grid[0] >= _PATCH_FRAMES + kt - 1
        and grid[1] >= kh + 1
        and grid[2] >= kw + 1
    )


def _runs_fit(window: Grid, grid: Grid) -> bool:
    qt, qh, qw = run_tile(window)
    kt, kh, kw = window
    return (
        qw + kw - 1 <= _KEY_RUN
        and grid[0] >= qt + kt - 1
        and grid[1] >= qh + kh - 1
        and grid[2] >= _KEY_RUN
    )


def _launch_tiles(window: Grid, grid: Grid) -> int:
    """Threadgroups the kernel serving ``window`` on ``grid`` launches, 32 queries each."""
    gt, gh, gw = grid
    if _patches_fit(window, grid):
        return -(-gt // _PATCH_FRAMES) * -(-gh // 2) * -(-gw // 2)
    qt, qh, qw = run_tile(window)
    return -(-gt // qt) * -(-gh // qh) * -(-gw // qw)


def launch_fits(grid: Grid, window: Grid) -> bool:
    """Whether the kernel serving ``window`` on a ``(T, H, W)`` ``grid`` launches within MLX's
    signed 32-bit grid dimensions, 128 threads a 32-query block: about ``2**29`` tokens.
    ``supports`` requires it, and the decode planner proposes no tile past it."""
    return _launch_tiles(window, grid) * _THREADS <= _GRID_LIMIT


def supports(shape: tuple[int, ...], window: Grid) -> bool:
    """Whether the kernels compute ``(batch, T, H, W, heads, 64)`` arrays of ``shape`` with
    ``window``: an odd window inside the grid, a grid large enough for one of the kernels' key
    boxes, and a launch grid within MLX's 32-bit dimensions (``launch_fits``)."""
    if len(shape) != 6 or shape[-1] != HEAD_DIM or math.prod(shape) <= 0:
        return False
    grid = (int(shape[1]), int(shape[2]), int(shape[3]))
    if any(size % 2 == 0 or size < 1 for size in window):
        return False
    if any(size > length for size, length in zip(window, grid, strict=True)):
        return False
    if not (_patches_fit(window, grid) or _runs_fit(window, grid)):
        return False
    return launch_fits(grid, window)


def applies(q: mx.array, k: mx.array, v: mx.array, *, window: Grid) -> bool:
    """Whether ``neighborhood_attention`` computes this call: the kernels switched on, matching
    FP16 arrays, and a shape and window ``supports`` admits."""
    if not _enabled or k.shape != q.shape or v.shape != q.shape:
        return False
    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        return False
    return supports(tuple(int(size) for size in q.shape), window)


def neighborhood_attention(q: mx.array, k: mx.array, v: mx.array, *, window: Grid) -> mx.array:
    """Exact neighborhood attention of scaled ``(batch, T, H, W, heads, 64)`` FP16 queries over
    keys and values of the same shape, with the inward-shifted ``window``.

    Call this where ``applies`` holds; the output has the queries' shape and dtype.
    """
    if not applies(q, k, v, window=window):
        raise ValueError(
            f"neighborhood attention does not take {tuple(q.shape)} {q.dtype} arrays with "
            f"window {window}"
        )
    batch, gt, gh, gw, heads, dim = (int(size) for size in q.shape)
    grid = (gt, gh, gw)
    tiles = _launch_tiles(window, grid)
    if _patches_fit(window, grid):
        body, thread_cap = "patches", _PATCH_THREAD_CAP
        geometry = []
    else:
        body, thread_cap = "runs", _RUN_THREAD_CAP
        qt, qh, qw = run_tile(window)
        geometry = [("QT", qt), ("QH", qh), ("QW", qw), ("BK", _KEY_RUN)]
    return _kernel(body, thread_cap)(
        inputs=[q, k, v],
        template=[
            ("T", mx.float16),
            ("HEADS", heads),
            ("D", dim),
            ("GT", gt),
            ("GH", gh),
            ("GW", gw),
            ("KT", window[0]),
            ("KH", window[1]),
            ("KW", window[2]),
            *geometry,
        ],
        grid=(tiles * _THREADS, heads, batch),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


__all__ = [
    "HEAD_DIM",
    "applies",
    "enabled",
    "launch_fits",
    "neighborhood_attention",
    "run_tile",
    "set_enabled",
    "supports",
]
