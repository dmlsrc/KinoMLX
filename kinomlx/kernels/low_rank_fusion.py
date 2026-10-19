"""Single-pass fusion of plain low-rank adapter terms into a weight.

The adapter fuser computes ``float(W)``, then for each plain low-rank term in order
``mx.addmm(fused, up.astype(f32), down.astype(f32), alpha=scale)``, then one cast back, and a
transposing copy for a weight stored ``(in, out)``: a pass to widen the weight, a pass per term
that reads and writes the whole FP32 weight around its product, and passes to narrow and
transpose it. Here one threadgroup fuses one tile of the weight in its stored layout, reading the
weight once and writing it once.

Exactness. MLX runs those FP32 products on an M1 Max as STEEL GEMM tiles whose inner dimension is
staged 16 at a time and summed in 8-wide SIMD-group multiply-accumulates from zero. This kernel
stages the term of largest rank the same way, its factors widened to FP32 as they are staged
(exact for BF16, FP16 and FP32 factors), so every element sums the same exact products in the
same order; the tile shape does not change that order. Every other term, of rank at most
``SMALL_RANK``, is multiplied fragment by fragment in the epilogue, also in 8-wide steps from
zero, and the terms fold in their order as ``fma(product, scale, fused)``, the form MLX's
``addmm`` epilogue compiles to, onto the base value. The result equals the fuser's chain bit for
bit, non-finite values, signed zeros and subnormal products included. Ranks must be multiples of
16: MLX zero-pads a partial block, which can turn a negative-zero sum positive.

Speed, on an M1 Max with LTX-2.3's adapted weights (ranks 16 and 384 on the video stream, 384 on
audio): 7.1 to 7.6 TFLOP/s against 4.4 to 5.5 for the chain, for example 1.77 against 2.54 ms
for a 4096 x 4096 weight and 7.1 against 9.8 to 12.1 ms for a feed-forward weight. MLX's own FP32
``addmm`` alone runs these products at 7.8 to 8.3. With one operation per weight instead of five
or more, MLX's lazy weight reads also overlap the first forward again: a freshly adapted
384x224x121 LTX transformer's first forward took 8.5 s against 11.4.

Choices the measurements made. BF16 factors converted to FP16 scaled by a power of two would let
the products run at MLX's FP16 rate (8.9 against 8.1 TFLOP/s at rank 384), worth about 0.2 s a
load, but need a reduction and a conversion per factor first, which cost an MPP version of this
kernel 0.67 s a load. BF16 widened with conversions while staging cost about 8% of the kernel;
the staging buffers are instead zeroed once and only each FP32 word's high half written.
Keeping an FP32 fused tile live through a second term's staged loop took 128 registers and
spilled, which is why small terms run in the epilogue. Tiles of 64 x 32 on 2 x 2 SIMD groups ran
fastest overall at these shapes.

``available`` checks once per process and layout that a small ragged fusion builds, runs and
equals the chain; otherwise callers keep the chain. Each shape compiles its own kernel once per
machine, and the system shader cache serves later processes.
"""

from collections.abc import Sequence
from functools import cache
from math import prod
from pathlib import Path
from typing import cast

import mlx.core as mx

from ._typing import MetalKernel

# MLX's FP32 GEMM stages the inner dimension 16 at a time and zero-pads a partial block.
RANK_STEP = 16
# Every term but the one of largest rank is multiplied in the epilogue.
SMALL_RANK = 64
MAX_TERMS = 8
_DTYPES = (mx.bfloat16, mx.float16, mx.float32)
_INDEX_LIMIT = 1 << 31
# Output tile, SIMD-group grid, and inner-dimension block.
_BM = 64
_BN = 32
_WM = 2
_WN = 2
_BK = 16
_THREADS = _WM * _WN * 32
_METAL = Path(__file__).with_name("metal") / "low_rank_fusion.metal"

_enabled = True


def set_enabled(enabled: bool) -> None:
    """Fuse through the kernel wherever it applies (the default), or keep the MLX chain."""
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    """Whether the fusion kernel is switched on."""
    return _enabled


def _main_term(ranks: Sequence[int]) -> int:
    """The term staged through the tile loop: the largest rank, the first of equals."""
    return max(range(len(ranks)), key=lambda index: (ranks[index], -index))


def _epilogue_factors(term: int) -> str:
    # The thread's A fragments for tile row r: A[c_row + r][8 s + fn + {0, 1}]. A is up
    # (OUT, R) for a weight stored (out, in) and down^T for one stored (in, out).
    return f"""
    float2 a{term}[R{term} / 8];
    LRF_UNROLL
    for (short s = 0; s < R{term} / 8; s++) {{
      const int k = 8 * s + fn;
      if (full || r < tgp_bm) {{
        if (TA) {{
          a{term}[s] = float2(
              float(down{term}[size_t(k) * IN + c_row + r]),
              float(down{term}[size_t(k + 1) * IN + c_row + r]));
        }} else {{
          a{term}[s] = float2(
              float(up{term}[size_t(c_row + r) * R{term} + k]),
              float(up{term}[size_t(c_row + r) * R{term} + k + 1]));
        }}
      }} else {{
        a{term}[s] = float2(0.0f);
      }}
    }}"""


def _epilogue_product(term: int) -> str:
    # The fragment's product with B[8 s + fm][c_col + c + {0, 1}]: B is down (R, IN), or up^T.
    return f"""
      float2 p{term} = float2(0.0f);
      LRF_UNROLL
      for (short s = 0; s < R{term} / 8; s++) {{
        const int k = 8 * s + fm;
        float2 b;
        if (TB) {{
          b = float2(
              (full || c < tgp_bn) ? float(up{term}[size_t(c_col + c) * R{term} + k]) : 0.0f,
              (full || c + 1 < tgp_bn)
                  ? float(up{term}[size_t(c_col + c + 1) * R{term} + k]) : 0.0f);
        }} else {{
          b = float2(
              (full || c < tgp_bn) ? float(down{term}[size_t(k) * IN + c_col + c]) : 0.0f,
              (full || c + 1 < tgp_bn)
                  ? float(down{term}[size_t(k) * IN + c_col + c + 1]) : 0.0f);
        }}
        lrf::Frag::mma(p{term}, a{term}[s], b, p{term});
      }}"""


def _source(count: int, main: int, *, transposed: bool) -> str:
    """The kernel body: the main term through the staged tile loop, the others in the epilogue.

    The tile is ``A @ B`` over the stored layout: ``up @ down`` for a weight stored (out, in),
    ``down^T @ up^T`` for one stored (in, out), which sums the same products in the same order.
    """
    small = [term for term in range(count) if term != main]
    factors = "".join(_epilogue_factors(term) for term in small)
    products = "".join(_epilogue_product(term) for term in small)
    folds = "\n        ".join(
        f"f = fma({'pm' if term == main else f'p{term}'}[e], scales[{term}], f);"
        for term in range(count)
    )
    # The staged operands' first rows and row strides; up and down may differ in dtype.
    if transposed:
        a_source, a_stride = f"down{main} + c_row", "IN"
        b_source, b_stride = f"up{main} + size_t(c_col) * R{main}", f"R{main}"
    else:
        a_source, a_stride = f"up{main} + size_t(c_row) * R{main}", f"R{main}"
        b_source, b_stride = f"down{main} + c_col", "IN"
    return f"""
  constexpr bool TA = {"true" if transposed else "false"};
  constexpr bool TB = TA;
  constexpr int PR = TA ? IN : OUT;
  constexpr int PC = TA ? OUT : IN;
  constexpr short BM = {_BM};
  constexpr short BN = {_BN};
  constexpr short BK = {_BK};
  constexpr short WM = {_WM};
  constexpr short WN = {_WN};
  constexpr short PAD = 4;
  constexpr short TGP = WM * WN * 32;
  constexpr short TM = BM / (8 * WM);
  constexpr short TN = BN / (8 * WN);
  constexpr short LDA = TA ? BM + PAD : BK + PAD;
  constexpr short LDB = TB ? BK + PAD : BN + PAD;
  // Shapes that tile exactly compile the full-tile path alone.
  constexpr bool ALIGNED = (PR % BM == 0) && (PC % BN == 0);
  using mma_t = lrf::BlockMMA<BM, BN, BK, WM, WN, TA, TB, LDA, LDB>;
  using loader_a_t = typename lrf::StagingLoader<
      FA, TA ? BK : BM, TA ? BM : BK, LDA, !TA, TGP>::type;
  using loader_b_t = typename lrf::StagingLoader<
      FB, TB ? BN : BK, TB ? BK : BN, LDB, TB, TGP>::type;

  const ushort sgid = simdgroup_index_in_threadgroup;
  const ushort lane = thread_index_in_simdgroup;
  const int c_row = int(threadgroup_position_in_grid.y) * BM;
  const int c_col = int(threadgroup_position_in_grid.x) * BN;
  const short tgp_bm = short(min(int(BM), PR - c_row));
  const short tgp_bn = short(min(int(BN), PC - c_col));
  const bool full = ALIGNED || (tgp_bm == BM && tgp_bn == BN);

  threadgroup float As[TA ? BK * (BM + PAD) : BM * (BK + PAD)];
  threadgroup float Bs[TB ? BN * (BK + PAD) : BK * (BN + PAD)];
  // The BF16 loaders write only each word's high half; the first barrier of the loop orders
  // these stores before them.
  for (short z = short(sgid * 32 + lane); z < short(sizeof(As) / sizeof(float)); z += TGP) {{
    As[z] = 0.0f;
  }}
  for (short z = short(sgid * 32 + lane); z < short(sizeof(Bs) / sizeof(float)); z += TGP) {{
    Bs[z] = 0.0f;
  }}

  thread loader_a_t loader_a({a_source}, {a_stride}, As, sgid, lane);
  thread loader_b_t loader_b({b_source}, {b_stride}, Bs, sgid, lane);
  thread mma_t mma_op(sgid, lane);
  if (full) {{
    for (int k = 0; k < R{main} / BK; k++) {{
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }}
  }} else {{
    const short2 dims_a = TA ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
    const short2 dims_b = TB ? short2(BK, tgp_bn) : short2(tgp_bn, BK);
    for (int k = 0; k < R{main} / BK; k++) {{
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(dims_a);
      loader_b.load_safe(dims_b);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }}
  }}

  // Fold every term onto the base value in place, then store: a store between the base reads
  // would order them, since the compiler cannot rule out aliasing.
  const short2 coord = lrf::Frag::coord(lane);
  const short fm = coord.y;
  const short fn = coord.x;
  const short tm = 8 * (sgid / WN);
  const short tn = 8 * (sgid % WN);
  const device T* base_tile = base + size_t(c_row) * PC + c_col;
  device T* out_tile = out + size_t(c_row) * PC + c_col;
  LRF_UNROLL
  for (short i = 0; i < TM; i++) {{
    const short r = tm + fm + i * (8 * WM);{factors}
    LRF_UNROLL
    for (short j = 0; j < TN; j++) {{
      const short c = tn + fn + j * (8 * WN);{products}
      thread auto& pm = mma_op.Ctile.at(i, j);
      LRF_UNROLL
      for (short e = 0; e < 2; e++) {{
        const bool inside = full || (r < tgp_bm && c + e < tgp_bn);
        float f = inside ? float(base_tile[int(r) * PC + c + e]) : 0.0f;
        {folds}
        pm[e] = f;
      }}
    }}
  }}
  LRF_UNROLL
  for (short i = 0; i < TM; i++) {{
    const short r = tm + fm + i * (8 * WM);
    LRF_UNROLL
    for (short j = 0; j < TN; j++) {{
      const short c = tn + fn + j * (8 * WN);
      thread auto& pm = mma_op.Ctile.at(i, j);
      LRF_UNROLL
      for (short e = 0; e < 2; e++) {{
        if (full || (r < tgp_bm && c + e < tgp_bn)) {{
          out_tile[int(r) * PC + c + e] = static_cast<T>(pm[e]);
        }}
      }}
    }}
  }}
"""


@cache
def _kernel(count: int, main: int, transposed: bool) -> MetalKernel:
    inputs = ["base", "scales"]
    for term in range(count):
        inputs += [f"up{term}", f"down{term}"]
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=f"kinomlx_low_rank_fusion_{count}_{main}{'_t' if transposed else ''}",
            input_names=inputs,
            output_names=["out"],
            header=_METAL.read_text(encoding="utf-8"),
            source=_source(count, main, transposed=transposed),
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )


def supports(
    base: mx.array, factors: Sequence[tuple[mx.array, mx.array]], *, transposed: bool
) -> bool:
    """Whether ``fuse`` takes this weight and these ``(up, down)`` factors.

    ``base`` is the logical ``(out, in)`` weight, or with ``transposed`` its stored
    ``(in, out)`` form; ``up`` is ``(out, rank)`` and ``down`` ``(rank, in)``.
    """
    if base.ndim != 2 or base.dtype not in _DTYPES or not 0 < len(factors) <= MAX_TERMS:
        return False
    out_dim, in_dim = (int(size) for size in (base.shape[::-1] if transposed else base.shape))
    if not 0 < out_dim * in_dim < _INDEX_LIMIT:
        return False
    ranks = []
    for up, down in factors:
        if up.ndim != 2 or down.ndim != 2 or up.dtype not in _DTYPES or down.dtype not in _DTYPES:
            return False
        rank = int(up.shape[1])
        if rank <= 0 or rank % RANK_STEP or int(down.shape[0]) != rank:
            return False
        if int(up.shape[0]) != out_dim or int(down.shape[1]) != in_dim:
            return False
        if max(out_dim, in_dim) * rank >= _INDEX_LIMIT:
            return False
        ranks.append(rank)
    main = _main_term(ranks)
    return all(rank <= SMALL_RANK for term, rank in enumerate(ranks) if term != main)


def fuse(
    base: mx.array, terms: Sequence[tuple[mx.array, mx.array, float]], *, transposed: bool
) -> mx.array:
    """``base`` with each ``(up, down, scale)`` term folded in, in order, in ``base``'s dtype
    and stored layout.

    Equals ``float(W)``, then ``mx.addmm(fused, up.astype(f32), down.astype(f32), alpha=scale)``
    per term, then a cast to ``base``'s dtype (and the transposed copy for a stored transpose),
    bit for bit where ``supports`` and ``available`` hold.
    """
    factors = [(up, down) for up, down, _scale in terms]
    if not supports(base, factors, transposed=transposed):
        raise ValueError("the low-rank fusion kernel does not take this weight or these factors")
    out_dim, in_dim = (int(size) for size in (base.shape[::-1] if transposed else base.shape))
    rows, cols = (int(size) for size in base.shape)
    ranks = [int(up.shape[1]) for up, _down in factors]
    main = _main_term(ranks)
    up_main, down_main = factors[main]
    inputs: list[mx.array] = [base, mx.array([scale for _, _, scale in terms], dtype=mx.float32)]
    template: list[tuple[str, int | mx.Dtype]] = [
        ("T", base.dtype),
        ("OUT", out_dim),
        ("IN", in_dim),
        ("FA", down_main.dtype if transposed else up_main.dtype),
        ("FB", up_main.dtype if transposed else down_main.dtype),
    ]
    for term, (up, down) in enumerate(factors):
        inputs += [up, down]
        template.append((f"R{term}", ranks[term]))
    return _kernel(len(terms), main, transposed)(
        inputs=inputs,
        output_shapes=[(rows, cols)],
        output_dtypes=[base.dtype],
        grid=(_THREADS * -(-cols // _BN), -(-rows // _BM), 1),
        threadgroup=(_THREADS, 1, 1),
        template=template,
    )[0]


def _pattern(shape: tuple[int, ...], phase: float) -> mx.array:
    """Deterministic, irregular values in [-1, 1] (product code draws no random numbers)."""
    count = prod(shape)
    return mx.sin(mx.arange(count, dtype=mx.float32) * 0.7310585 + phase).reshape(shape)


@cache
def _available(transposed: bool) -> bool:
    # A ragged weight, so interior and edge tiles both run, a staged term and an epilogue term,
    # and scales that are not powers of two, where a separately rounded multiply and add would
    # differ from the FMA.
    out_dim, in_dim = 70, 100
    logical = (_pattern((out_dim, in_dim), 0.0) * 0.05).astype(mx.bfloat16)
    terms = [
        (
            (_pattern((out_dim, 32), 1.0) * 0.02).astype(mx.bfloat16),
            (_pattern((32, in_dim), 2.0) * 0.02).astype(mx.bfloat16),
            0.3,
        ),
        (
            (_pattern((out_dim, 16), 3.0) * 0.02).astype(mx.bfloat16),
            (_pattern((16, in_dim), 4.0) * 0.02).astype(mx.bfloat16),
            1.7,
        ),
    ]
    expected = logical.astype(mx.float32)
    for up, down, scale in terms:
        expected = mx.addmm(expected, up.astype(mx.float32), down.astype(mx.float32), alpha=scale)
    expected = expected.astype(logical.dtype)
    stored = mx.contiguous(logical.T) if transposed else logical
    try:
        fused = fuse(stored, terms, transposed=transposed)
        mx.eval(fused)
    except RuntimeError:
        return False
    result = fused.T if transposed else fused
    return bool(mx.array_equal(result, expected).item())


def available(*, transposed: bool) -> bool:
    """Whether the kernel builds, runs and gives the chain's bits here, once per process and
    layout."""
    return _available(transposed)


def applies(
    base: mx.array, factors: Sequence[tuple[mx.array, mx.array]], *, transposed: bool
) -> bool:
    """Whether to fuse these factors through the kernel: switched on, supported, available."""
    return (
        _enabled
        and supports(base, factors, transposed=transposed)
        and available(transposed=transposed)
    )


__all__ = [
    "MAX_TERMS",
    "RANK_STEP",
    "SMALL_RANK",
    "applies",
    "available",
    "enabled",
    "fuse",
    "set_enabled",
    "supports",
]
