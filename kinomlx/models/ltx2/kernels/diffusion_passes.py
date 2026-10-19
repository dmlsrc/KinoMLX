"""FP16 passes of the LTX-2.5 diffusion video decoder's blocks, bit-identical to their parts.

``rms_norm`` is ``kinomlx.kernels.fused_ops.rms_norm`` (FP32 reduction and scale, one FP16
rounding) followed, with ``scale1p`` and ``shift``, by the block's FP16 AdaLN modulation
``y * scale1p + shift``, one rounding per operation, in one pass. One SIMD group takes a row
with half4 accesses: lane ``l`` folds channels ``128 g + 4 l`` to ``128 g + 4 l + 3`` of each
128-channel group ``g``, the channels the threadgroup kernel's thread ``32 g + l`` folds, in the
same order; each group's sum is the same ``simd_sum`` and the second stage sums the same lane
vector, so every bit matches. It takes widths that are multiples of 128 up to 1,024 and runs at
memory speed: on an M1 Max the final stage's norm of 2.6 million 256-wide rows took 7.6 ms
against 14.3.

``silu_mul`` is ``fused_ops.silu(gate) * up`` for a ``(rows, 2 * hidden)`` gate/up product, four
hidden units a thread.

``qk_norm_rope`` is the decoder's attention projection up to its scores: the Q and K products
of ``x`` with their biases (``mx.addmm``'s rounding), the per-head Q and K RMS norms, the query
scale and the 3D rotary embedding, without writing the raw projections. A threadgroup of four
SIMD groups takes 32 tokens and one head; group ``g`` runs the 32x32 MPP product for Q (``g`` < 2)
or K, columns ``32 (g % 2)`` of the head, as ``kinomlx.kernels.mpp_linear`` runs that tile. Its
epilogue keeps the norm's arithmetic exactly: a lane of MPP's 32x32 destination holds channel
pairs ``c, c + 1`` (a rotary pair) and a norm channel group lies in lanes ``l`` and ``l ^ 1``, so
the fold runs across the lane pair in the norm kernel's order, and the row's sum follows
``simd_sum``'s own tree (the xor butterfly), the two 32-column halves exchanging their sums
through threadgroup memory. Both facts are properties of the GPU, so ``available`` checks them
once per process; where either fails, callers keep the separate passes. On an M1 Max the
decoder's final-stage Q and K took 76.8 ms against 84.0 for two products and a separate norm and
rotary pass, and the decode's active peak fell by the two raw projections.
"""

from functools import cache
from typing import NamedTuple, cast

import mlx.core as mx

from kinomlx.kernels import mpp_linear
from kinomlx.kernels._typing import MetalKernel

HEAD_DIM = 64
# The FP32 cosines and sines of the T, H and W axes, each (length, size / 2).
RotaryTables = tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]
_TILE = 32
_NORM_ROWS = 8
# mx.fast.metal_kernel takes signed 32-bit grid dimensions.
_GRID_LIMIT = 2**31 - 1
_MPP_HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_RMS_NORM_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint row = thread_position_in_grid.y;
    const size_t row_offset = size_t(row) * FEATURE_DIM;
    constexpr uint GROUPS = FEATURE_DIM / 128;
    vec<T, 4> x4[GROUPS];
    float second = 0.0f;
    for (uint g = 0; g < GROUPS; ++g) {
        x4[g] = *reinterpret_cast<const device vec<T, 4>*>(inp + row_offset + 128 * g + 4 * lane);
        float folded = 0.0f;
        for (uint read = 0; read < 4; ++read) {
            const float x = float(x4[g][read]);
            folded = metal::fma(x, x, folded);
        }
        const float group_sum = simd_sum(folded);
        second = lane == g ? group_sum : second;
    }
    const float total = simd_sum(second);
    const float inverse_rms = metal::precise::rsqrt(total / float(FEATURE_DIM) + float(eps));
    for (uint g = 0; g < GROUPS; ++g) {
        vec<T, 4> y4;
        for (uint read = 0; read < 4; ++read) {
            const uint channel = 128 * g + 4 * lane + read;
            float result = float(x4[g][read]) * inverse_rms;
            result *= float(weight[channel]) + 0.0f;
            T y = T(result);
MODULATION
            y4[read] = y;
        }
        *reinterpret_cast<device vec<T, 4>*>(out + row_offset + 128 * g + 4 * lane) = y4;
    }
"""
_MODULATION = "            y = y * scale1p[channel];\n            y = y + shift[channel];"

_SILU_MUL_SOURCE = """
  const uint row = thread_position_in_grid.y;
  const uint col = 4 * thread_position_in_grid.x;
  if (col >= HIDDEN) {
    return;
  }
  const size_t at = size_t(row) * (2 * HIDDEN) + col;
  const vec<T, 4> gate = *reinterpret_cast<const device vec<T, 4>*>(gate_up + at);
  const vec<T, 4> up = *reinterpret_cast<const device vec<T, 4>*>(gate_up + at + HIDDEN);
  vec<T, 4> activated;
  for (uint i = 0; i < 4; ++i) {
    const float x = float(gate[i]);
    const float result = x / (1.0f + metal::precise::exp(-x));
    activated[i] = T(result) * up[i];
  }
  *reinterpret_cast<device vec<T, 4>*>(out + size_t(row) * HIDDEN + col) = activated;
"""

_QK_NORM_ROPE_SOURCE = """
  constexpr int D = 64;
  static_assert(DT + DH + DW == D, "the rotary split covers the head");
  static_assert(TM == 32 && TN == 32, "the epilogue follows the 32x32 destination layout");
  const int pid = int(threadgroup_position_in_grid.x);
  const int head = pid % HEADS;
  const int row = (pid / HEADS) * TM;
  const int M = dims[0];
  const uint lane = thread_index_in_simdgroup;
  const uint group = simdgroup_index_in_threadgroup;
  const bool is_key = group >= 2;
  const int half_col = int(group % 2) * TN;
  const int col = head * D + half_col;
  threadgroup float sums[4 * TM];

  tensor<device half, dextents<int32_t, 2>, tensor_inline> A(
      (device half*)a, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B(
      (device half*)(is_key ? wk : wq), dextents<int32_t, 2>(K, N));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> C(
      is_key ? k_out : q_out, dextents<int32_t, 2>(N, M));
  const device half* bias = is_key ? bk : bq;
  constexpr auto desc = matmul2d_descriptor(
      TM, TN, static_cast<int>(dynamic_extent), false, true);
  matmul2d<desc, execution_simdgroups<1>> op;
  // The tiles from pointers offset in 64-bit, so MPP's int32 coordinates never see a
  // volume past 2^31 elements.
  tensor<device half, dextents<int32_t, 2>, tensor_inline> A_rows(
      (device half*)a + size_t(row) * K, dextents<int32_t, 2>(K, TM));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> C_rows(
      (is_key ? k_out : q_out) + size_t(row) * N, dextents<int32_t, 2>(N, TM));
  auto tA = A_rows.template slice<dynamic_extent, TM>(0, 0);
  auto tB = B.template slice<dynamic_extent, TN>(0, col);
  using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(tA)>;
  using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(tB)>;
  auto xt = op.template get_destination_cooperative_tensor<AT, BT, half>();
  float part[4];
  int rows[4];
  {
    auto sum32 = op.template get_destination_cooperative_tensor<AT, BT, float>();
    op.run(tA, tB, sum32);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < 32; ++i) {
      const int cc = int(sum32.get_multidimensional_index(i)[0]);
      xt[i] = static_cast<half>(sum32[i] + float(bias[col + cc]));
    }
  }
  // Each lane's four rows: the sum of squares over its half of the head, folded across the lane
  // pair (channels 4g, 4g + 1 in lane l, 4g + 2, 4g + 3 in l ^ 1) and summed in simd_sum's tree
  // (groups 2m and 2m + 1 in lanes l and l ^ 8, then the column blocks within the lane).
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    float pair_sum[4];
#pragma clang loop unroll(full)
    for (int k = 0; k < 4; ++k) {
      const float e0 = float(xt[8 * j + 2 * k]);
      const float e1 = float(xt[8 * j + 2 * k + 1]);
      const float first = metal::fma(e1, e1, metal::fma(e0, e0, 0.0f));
      const float carried = simd_shuffle_xor(first, ushort(1));
      const float folded = metal::fma(e1, e1, metal::fma(e0, e0, carried));
      pair_sum[k] = folded + simd_shuffle_xor(folded, ushort(8));
    }
    part[j] = (pair_sum[0] + pair_sum[1]) + (pair_sum[2] + pair_sum[3]);
  }
  // Lanes holding channels 4g + 2, 4g + 3 (l & 1) hold their rows' half sums, and lanes l, l ^ 8
  // agree: the ones with (l & 8) == 0 publish them.
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    rows[j] = int(xt.get_multidimensional_index(uint16_t(8 * j))[1]);
    if ((lane & 9u) == 1u) {
      sums[group * TM + rows[j]] = part[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint first_half = group & ~1u;
  float inverse[4];
  uint t[4];
  uint h[4];
  uint w[4];
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    const float row_sum = sums[first_half * TM + rows[j]] + sums[(first_half + 1) * TM + rows[j]];
    inverse[j] = metal::precise::rsqrt(row_sum / float(D) + float(eps));
    const uint token = uint(row + rows[j]);
    t[j] = token / (GH * GW);
    h[j] = (token / GW) % GH;
    w[j] = token % GW;
  }

  // Norm weight, the query scale (D^-0.5, an FP16 multiply) and the rotation of each pair.
#pragma clang loop unroll(full)
  for (int k = 0; k < 4; ++k) {
    const int c = half_col + int(xt.get_multidimensional_index(uint16_t(2 * k))[0]);
    const float weight0 = float(is_key ? k_weight[c] : q_weight[c]);
    const float weight1 = float(is_key ? k_weight[c + 1] : q_weight[c + 1]);
#pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
      const int i = 8 * j + 2 * k;
      const device float* cos_table;
      const device float* sin_table;
      int pair;
      if (c < DT) {
        cos_table = cos_t + t[j] * (DT / 2);
        sin_table = sin_t + t[j] * (DT / 2);
        pair = c / 2;
      } else if (c < DT + DH) {
        cos_table = cos_h + h[j] * (DH / 2);
        sin_table = sin_h + h[j] * (DH / 2);
        pair = (c - DT) / 2;
      } else {
        cos_table = cos_w + w[j] * (DW / 2);
        sin_table = sin_w + w[j] * (DW / 2);
        pair = (c - DT - DH) / 2;
      }
      float result0 = float(xt[i]) * inverse[j];
      result0 *= weight0;
      half y0 = half(result0);
      float result1 = float(xt[i + 1]) * inverse[j];
      result1 *= weight1;
      half y1 = half(result1);
      if (!is_key) {
        y0 = y0 * half(0.125h);
        y1 = y1 * half(0.125h);
      }
      const float even = float(y0);
      const float odd = float(y1);
      const float cosine = cos_table[pair];
      const float sine = sin_table[pair];
      const float even_cos = even * cosine;
      const float odd_sin = odd * sine;
      const float even_sin = even * sine;
      const float odd_cos = odd * cosine;
      const float rotated_first = even_cos - odd_sin;
      const float rotated_second = even_sin + odd_cos;
      xt[i] = half(rotated_first);
      xt[i + 1] = half(rotated_second);
    }
  }
  xt.store(C_rows.template slice<TN, TM>(col, 0));
"""

# Whether a lane of a 32x32 destination holds, in slot i, column c0 + 8 ((i % 8) / 2) + i % 2 of
# row r0 + 8 (i / 8) with c0 = 2 (l & 1) + 4 ((l >> 3) & 1) and r0 from bits 1, 2 and 4.
_LAYOUT_PROBE_SOURCE = """
  const uint lane = thread_index_in_simdgroup;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> A(
      (device half*)a, dextents<int32_t, 2>(32, 32));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B(
      (device half*)w, dextents<int32_t, 2>(32, 32));
  constexpr auto desc = matmul2d_descriptor(32, 32, static_cast<int>(dynamic_extent), false, true);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto tA = A.template slice<dynamic_extent, 32>(0, 0);
  auto tB = B.template slice<dynamic_extent, 32>(0, 0);
  using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(tA)>;
  using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(tB)>;
  auto sum32 = op.template get_destination_cooperative_tensor<AT, BT, float>();
  auto rounded = op.template get_destination_cooperative_tensor<AT, BT, half>();
  const int c0 = 2 * int(lane & 1u) + 4 * int((lane >> 3) & 1u);
  const int r0 = int((lane >> 1) & 1u) + 2 * int((lane >> 2) & 1u) + 4 * int((lane >> 4) & 1u);
  bool matches = sum32.get_capacity() == 32 && rounded.get_capacity() == 32;
  for (uint16_t i = 0; i < 32 && matches; ++i) {
    const auto at = sum32.get_multidimensional_index(i);
    const auto at_half = rounded.get_multidimensional_index(i);
    const int column = c0 + 8 * ((i % 8) / 2) + i % 2;
    const int row = r0 + 8 * (i / 8);
    matches = sum32.is_valid_element(i) && int(at[0]) == column && int(at[1]) == row
        && int(at_half[0]) == column && int(at_half[1]) == row;
  }
  ok[lane] = matches ? 1 : 0;
"""

# Whether simd_sum equals the xor butterfly 1, 2, 4, 8, 16, and a half-warp's first four steps
# its sum with the other half at zero, on values spread over 2^-12 to 2^12.
_SUM_ORDER_PROBE_SOURCE = """
  const uint id = thread_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  const float magnitude = metal::exp2(float(int((id * 7u) % 25u) - 12));
  const float value = metal::sin(float(id) * 0.7310585f + 0.5f) * magnitude;
  float butterfly = value;
  butterfly += simd_shuffle_xor(butterfly, ushort(1));
  butterfly += simd_shuffle_xor(butterfly, ushort(2));
  butterfly += simd_shuffle_xor(butterfly, ushort(4));
  butterfly += simd_shuffle_xor(butterfly, ushort(8));
  const float halves = butterfly;
  butterfly += simd_shuffle_xor(butterfly, ushort(16));
  const float whole = simd_sum(value);
  const float low = simd_sum(lane < 16 ? value : 0.0f);
  const float high = simd_sum(lane < 16 ? 0.0f : value);
  const bool matches = whole == butterfly && halves == (lane < 16 ? low : high);
  ok[id] = matches ? 1 : 0;
"""


@cache
def _norm_kernel(modulate: bool) -> MetalKernel:
    inputs = ["inp", "weight", "eps"] + (["scale1p", "shift"] if modulate else [])
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=f"ltx_diffusion_rms_norm_{'modulated' if modulate else 'plain'}",
            input_names=inputs,
            output_names=["out"],
            source=_RMS_NORM_SOURCE.replace("MODULATION", _MODULATION if modulate else ""),
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )


@cache
def _silu_mul_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_diffusion_silu_mul",
            input_names=["gate_up"],
            output_names=["out"],
            source=_SILU_MUL_SOURCE,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )


@cache
def _qk_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_diffusion_qk_norm_rope",
            input_names=[
                "a",
                "wq",
                "bq",
                "wk",
                "bk",
                "dims",
                "q_weight",
                "k_weight",
                "eps",
                "cos_t",
                "sin_t",
                "cos_h",
                "sin_h",
                "cos_w",
                "sin_w",
            ],
            output_names=["q_out", "k_out"],
            header=_MPP_HEADER,
            source=_QK_NORM_ROPE_SOURCE,
            ensure_row_contiguous=True,
        ),
    )


def rms_norm_applies(
    value: mx.array,
    weight: mx.array,
    scale1p: mx.array | None = None,
    shift: mx.array | None = None,
) -> bool:
    """Whether ``rms_norm`` takes FP16 ``(rows, C)`` rows: ``C`` a multiple of 128 up to 1,024,
    a ``(C,)`` weight, and FP16 ``(C,)`` modulation vectors when given."""
    if value.ndim != 2 or value.dtype != mx.float16 or weight.ndim != 1:
        return False
    width = int(value.shape[1])
    if (scale1p is None) != (shift is None):
        return False
    for vector in (scale1p, shift):
        if vector is not None and (vector.dtype != mx.float16 or tuple(vector.shape) != (width,)):
            return False
    return (
        width % 128 == 0
        and 0 < width <= 1024
        and tuple(weight.shape) == (width,)
        and value.size > 0
    )


def rms_norm(
    value: mx.array,
    weight: mx.array,
    scale1p: mx.array | None = None,
    shift: mx.array | None = None,
    *,
    eps: float = 1e-6,
) -> mx.array:
    """``fused_ops.rms_norm(value, weight, eps)``, then ``* scale1p + shift`` in FP16 when given.

    Call this where ``rms_norm_applies`` holds.
    """
    if not rms_norm_applies(value, weight, scale1p, shift):
        raise ValueError(f"the row norm does not take {tuple(value.shape)} {value.dtype} rows")
    rows, width = (int(size) for size in value.shape)
    modulation = [] if scale1p is None or shift is None else [scale1p, shift]
    return _norm_kernel(bool(modulation))(
        inputs=[value, weight, eps, *modulation],
        output_shapes=[value.shape],
        output_dtypes=[mx.float16],
        grid=(32, rows, 1),
        threadgroup=(32, _NORM_ROWS, 1),
        template=[("T", mx.float16), ("FEATURE_DIM", width)],
    )[0]


def silu_mul_applies(gate_up: mx.array, hidden: int) -> bool:
    """Whether ``silu_mul`` takes this FP16 ``(rows, 2 * hidden)`` gate/up product: four hidden
    units a thread."""
    return (
        gate_up.ndim == 2
        and gate_up.dtype == mx.float16
        and int(gate_up.shape[1]) == 2 * hidden
        and hidden > 0
        and hidden % 4 == 0
        and gate_up.size > 0
    )


def silu_mul(gate_up: mx.array, hidden: int) -> mx.array:
    """``fused_ops.silu(gate) * up`` of an FP16 ``(rows, 2 * hidden)`` gate/up product.

    Call this where ``silu_mul_applies`` holds.
    """
    if not silu_mul_applies(gate_up, hidden):
        raise ValueError(
            f"the SiLU pass does not take a {tuple(gate_up.shape)} {gate_up.dtype} gate/up "
            f"product of {hidden} hidden units"
        )
    rows = int(gate_up.shape[0])
    return _silu_mul_kernel()(
        inputs=[gate_up],
        template=[("T", mx.float16), ("HIDDEN", hidden)],
        grid=(hidden // 4, rows, 1),
        threadgroup=(min(256, hidden // 4), 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.float16],
    )[0]


@cache
def _layout_matches() -> bool:
    kernel = cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_diffusion_mpp_layout_probe",
            input_names=["a", "w"],
            output_names=["ok"],
            header=_MPP_HEADER,
            source=_LAYOUT_PROBE_SOURCE,
            compile_options={"math_mode": "safe"},
        ),
    )
    zeros = mx.zeros((32, 32), dtype=mx.float16)
    try:
        (ok,) = kernel(
            inputs=[zeros, zeros],
            output_shapes=[(32,)],
            output_dtypes=[mx.int32],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
        )
        mx.eval(ok)
    except RuntimeError:
        return False
    return bool(mx.all(ok == 1).item())


@cache
def _sum_order_matches() -> bool:
    kernel = cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_diffusion_sum_order_probe",
            input_names=["unused"],
            output_names=["ok"],
            source=_SUM_ORDER_PROBE_SOURCE,
            compile_options={"math_mode": "safe"},
        ),
    )
    threads = 32 * 4096
    try:
        (ok,) = kernel(
            inputs=[mx.zeros((1,), dtype=mx.float32)],
            output_shapes=[(threads,)],
            output_dtypes=[mx.int32],
            grid=(threads, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(ok)
    except RuntimeError:
        return False
    return bool(mx.all(ok == 1).item())


def available() -> bool:
    """Whether ``qk_norm_rope`` keeps the separate passes' bits on this GPU: MPP's 32x32
    destination layout and ``simd_sum``'s order are the ones its epilogue follows, and the MPP
    products are enabled and exact. Evaluated once per process."""
    return (
        mpp_linear.enabled()
        and mpp_linear.rows_available(bias=True, residual=False)
        and _layout_matches()
        and _sum_order_matches()
    )


class QKWeights(NamedTuple):
    """The Q and K projections of one attention module and their per-head RMS norm weights:
    FP16 ``(heads * 64, C)`` weights with ``(heads * 64,)`` biases, and two ``(64,)`` norm
    weights of one dtype."""

    q_weight: mx.array
    q_bias: mx.array
    k_weight: mx.array
    k_bias: mx.array
    q_norm: mx.array
    k_norm: mx.array


def _qk_grid(rows: int, heads: int) -> int:
    """Threads the fused Q/K pass launches for ``rows``: 128 a 32-token tile and head."""
    return 128 * (rows // _TILE) * heads


def rows_addressable(rows: int, *, heads: int) -> bool:
    """Whether ``rows`` tokens of ``heads`` 64-wide heads launch within the passes' 32-bit
    counts. An MLX dimension already holds at most ``2**31 - 1`` rows, which the norm and SiLU
    grids index one each; the fused Q/K pass launches 128 threads a 32-token tile and head and
    binds first, at ``(2**31 - 1) // (128 * heads)`` tiles: 134,217,696 tokens for four heads.
    ``qk_norm_rope_applies`` requires it, and the decode planner proposes no tile past it."""
    return rows <= _GRID_LIMIT and _qk_grid(rows, heads) <= _GRID_LIMIT


def qk_norm_rope_applies(
    x: mx.array,
    weights: QKWeights,
    *,
    heads: int,
    dim_split: tuple[int, int, int],
) -> bool:
    """Whether ``qk_norm_rope`` takes FP16 ``(tokens, C)`` rows for ``heads`` 64-wide heads:
    whole 32-token tiles within the launch grid (``rows_addressable``), products MLX sums in
    MPP's order, FP16 projections, a rotary split that covers the head on channel pairs, and a
    GPU where ``available`` holds."""
    if x.ndim != 2 or x.dtype != mx.float16 or heads <= 0:
        return False
    rows, width = (int(size) for size in x.shape)
    n = heads * HEAD_DIM
    shapes = (
        (weights.q_weight, (n, width)),
        (weights.q_bias, (n,)),
        (weights.k_weight, (n, width)),
        (weights.k_bias, (n,)),
    )
    if any(array.dtype != mx.float16 or tuple(array.shape) != shape for array, shape in shapes):
        return False
    norms = (weights.q_norm, weights.k_norm)
    if any(tuple(norm.shape) != (HEAD_DIM,) for norm in norms):
        return False
    if weights.q_norm.dtype != weights.k_norm.dtype:
        return False
    return (
        rows % _TILE == 0
        and rows_addressable(rows, heads=heads)
        and sum(dim_split) == HEAD_DIM
        and all(size > 0 and size % 2 == 0 for size in dim_split)
        and mpp_linear.rows_supported(rows, width, n)
        and available()
    )


def qk_norm_rope(
    x: mx.array,
    weights: QKWeights,
    *,
    grid: tuple[int, int, int],
    heads: int,
    dim_split: tuple[int, int, int],
    tables: RotaryTables,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Normalized, scaled and rotated ``(tokens, heads * 64)`` queries and keys of FP16 rows.

    ``tables`` holds the FP32 cosines and sines of the T, H and W axes, each
    ``(length, size / 2)`` for the axis's rotary size, for ``grid``'s token order. Call this
    where ``qk_norm_rope_applies`` holds.
    """
    if not qk_norm_rope_applies(x, weights, heads=heads, dim_split=dim_split):
        raise ValueError(f"the fused Q/K pass does not take {tuple(x.shape)} {x.dtype} rows")
    rows, width = (int(size) for size in x.shape)
    if rows != grid[0] * grid[1] * grid[2]:
        raise ValueError(f"{rows} rows do not hold a {grid} token grid")
    dim_t, dim_h, dim_w = dim_split
    shapes = [(length, size // 2) for length, size in zip(grid, dim_split, strict=True)]
    for table, shape in zip(tables, (shape for shape in shapes for _ in range(2)), strict=True):
        if table.dtype != mx.float32 or tuple(table.shape) != shape:
            raise ValueError("the rotary tables are FP32 (length, size / 2) per axis")
    n = heads * HEAD_DIM
    q, k = _qk_kernel()(
        inputs=[
            x,
            weights.q_weight,
            weights.q_bias,
            weights.k_weight,
            weights.k_bias,
            mx.array([rows], dtype=mx.int32),
            weights.q_norm,
            weights.k_norm,
            eps,
            *tables,
        ],
        template=[
            ("N", n),
            ("K", width),
            ("TM", _TILE),
            ("TN", _TILE),
            ("HEADS", heads),
            ("GH", grid[1]),
            ("GW", grid[2]),
            ("DT", dim_t),
            ("DH", dim_h),
            ("DW", dim_w),
        ],
        grid=(_qk_grid(rows, heads), 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n), (rows, n)],
        output_dtypes=[mx.float16, mx.float16],
    )
    return q, k


__all__ = [
    "HEAD_DIM",
    "QKWeights",
    "RotaryTables",
    "available",
    "qk_norm_rope",
    "qk_norm_rope_applies",
    "rms_norm",
    "rms_norm_applies",
    "rows_addressable",
    "silu_mul",
    "silu_mul_applies",
]
