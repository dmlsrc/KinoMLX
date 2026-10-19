"""FP16 and BF16 linear products through Metal Performance Primitives' ``matmul2d``.

MPP's tensor operations are compiled by the GPU driver from Metal tensors, which MLX's
custom-kernel compiler provides on macOS 27. Each output tile is one ``matmul2d`` into an FP32
register tile. The bias is added to the FP32 sum and the value rounded once into the op's
destination tile of the output type, which MPP stores: ``mx.addmm``'s epilogue.

FP16. On an M1 Max ``matmul2d`` ran LTX's FP16 attention projections 3 to 9% faster than MLX's
GEMM from 1,024 rows up: 101 ms against 110 for a 30,576 x 4096 by 4096 x 4096 product with its
bias, about 10.1 TFLOP/s. Where MLX runs its regular GEMM kernel the result is bit-identical to
``mx.addmm`` (and, without a bias, to ``@``).

BF16. The M1 has no BF16 arithmetic, and MLX's BF16 GEMM ran LTX's video products at about
8 TFLOP/s. Here each operand is converted to FP16 scaled by the power of two that puts its
largest magnitude in [2^14, 2^15). A BF16 value then keeps every bit unless it lies about 2^28
or more below the largest value of its tensor, and the FP32 sum, unscaled exactly before the
bias and rounded once, is ``mx.addmm``'s in BF16 bit for bit unless such a value reaches it.
Unscaled FP16 would not do: LTX's attention outputs reach the output projections with most
values below FP16's normal range. Conversions included, the 30,576-row projection above took
103 ms against 127, and the 4096 -> 16384 feed-forward up projection 409 against 509. Converting
a weight on every call cost a 1,344-token LTX forward 0.54 s of its 5.54; a caller that keeps a
``prepare``d weight and passes it as the weight converts once (``restore`` gives the BF16
weight back).

Speed decides the layouts. A weight stored ``(in, out)`` with rows of 16,384 elements ran at
3.4 TFLOP/s, and a 16,384-element inner dimension lost to MLX's FP16 GEMM (492 against 429 ms)
and gained at most 3% on its BF16 GEMM, so ``supports`` refuses both. Products on an
``(out, in)`` weight walk eight 32-row stripes per column stripe so the weight's columns serve
several stripes while cached: 404 against 450 ms at 16,384 columns, the same at 4,096.

MLX sums an inner dimension that is not a multiple of 16, and the shapes it splits along K (at
most 2,048 16x16 output tiles), differently; ``supports`` refuses the first, and its size
thresholds exclude the second.

``available`` checks once per process and kernel form that a small ragged product builds, runs,
and equals ``mx.addmm`` exactly; otherwise callers keep MLX. Each shape compiles its own kernel
once per machine, and the system shader cache serves later processes.

Run-time rows. ``linear_rows`` is an FP16 form for callers whose row counts vary, such as a
decoder's token tiles: the row count is read at run time, so one build serves every row count
of an inner and output width, and an optional residual is added in FP16 after the product and
its bias round, as a separate addition would. It takes products of any width with more than
2,048 output tiles, which MLX never splits along K: the diffusion video decoder's 256-wide
products ran 7 to 10% faster than MLX's GEMM (9.1 to 10.2 TFLOP/s), bit-identical. Reading the
row count at run time cost under 1%, where reading the widths too cost 5 to 7%.
``rows_available`` checks its forms as ``available`` does.
"""

from functools import cache
from math import prod
from typing import NamedTuple, cast

import mlx.core as mx

from ._typing import MetalKernel

MIN_ROWS = 1024
MIN_WIDTH = 1024
# MLX sums along K in a different order (split-K) only for products of at most this many
# 16x16 output tiles; the run-time-row form takes larger products of any width.
_SPLIT_TILES = 2048
_TILE = 32
# mx.fast.metal_kernel takes signed 32-bit grid dimensions, and the kernels count rows and
# output tiles in int.
_GRID_LIMIT = 2**31 - 1
# Inner dimensions, and the rows of (in, out) weights, longer than this run faster on MLX.
_LONG = 8192
# Row stripes per column stripe for (out, in) weights.
_STRIPES = 8
_DTYPES = (mx.float16, mx.bfloat16)
_TYPE_NAMES = {mx.float16: "half", mx.bfloat16: "bfloat"}

_SCALES = """
// The scale exponent from the FP32 exponent field of a tensor's largest magnitude: peak *
// 2^shift lies in [2^14, 2^15), clamped to [-15, 24]. Nothing finite below 2^30 overflows FP16;
// infinities and NaNs pass through, so a product that would be non-finite on MLX stays so.
inline int mpp_shift(float peak) {
  return clamp(141 - int((as_type<uint>(peak) >> 23) & 0xffu), -15, 24);
}

// 2^exponent built from its bits, so every scale and unscale is exact.
inline float mpp_power(int exponent) {
  return as_type<float>(uint(127 + exponent) << 23);
}
"""

_HEADER = (
    """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
    + _SCALES
)

_SCALE_SOURCE = """
  const uint i = thread_position_in_grid.x;
  const float scaled = float(x[i]) * mpp_power(mpp_shift(float(peak[0])));
  out[i] = static_cast<half>(scaled);
"""

_RESTORE_SOURCE = """
  const uint i = thread_position_in_grid.x;
  out[i] = static_cast<bfloat>(float(x[i]) * mpp_power(-mpp_shift(float(peak[0]))));
"""

_enabled = True


class Operand(NamedTuple):
    """The FP16 values ``linear`` multiplies for one operand, and the largest magnitude that set
    their scale (``None`` for an FP16 operand, which is used as it is)."""

    values: mx.array
    peak: mx.array | None


def set_enabled(enabled: bool) -> None:
    """Use MPP wherever it applies (the default), or keep MLX's GEMM everywhere.

    The switch also covers the Winograd convolution's MPP products.
    """
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    """Whether MPP products are switched on."""
    return _enabled


def rows_addressable(rows: int, n: int) -> bool:
    """Whether a ``rows x n`` product's rows and 32x32 output tiles stay within the kernels'
    32-bit counts: the rows and the tile past them in ``int``, the tiles in the launch grid.
    ``supports`` and ``rows_supported`` require it."""
    return rows + _TILE - 1 <= _GRID_LIMIT and -(-rows // _TILE) * -(-n // _TILE) <= _GRID_LIMIT


def _shape_supported(rows: int, k: int, n: int, *, transposed: bool) -> bool:
    if rows < MIN_ROWS or k < MIN_WIDTH or n < MIN_WIDTH or k % 16 or k > _LONG:
        return False
    return rows_addressable(rows, n) and (transposed or n <= _LONG)


def _tile(
    a: str,
    b: str,
    c: str,
    *,
    out: str,
    bias: bool,
    scaled: bool,
    edge: bool,
    residual: bool = False,
) -> str:
    """Multiply into an FP32 tile, unscale, add the bias, round once into the output tile.

    With ``residual`` the rounded value is then added to the residual in the output type, as a
    separate addition after ``mx.addmm`` rounds. The FP32 tile and the output-type tile of one
    operation lay out their elements identically.
    """
    column = "col + int(sum32.get_multidimensional_index(i)[0])"
    value = "sum32[i] * unscale" if scaled else "sum32[i]"
    if bias:
        value += f" + float(bias[{column}])"
    guard = f"if ({column} < N) " if bias and edge else ""
    if residual:
        row_index = "row + int(sum32.get_multidimensional_index(i)[1])"
        guard = f"if ({column} < N && {row_index} < M) " if edge else ""
        value = f"static_cast<{out}>({value}) + res[size_t({row_index}) * N + {column}]"
    return f"""
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype({a})>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype({b})>;
    auto sum32 = op.template get_destination_cooperative_tensor<AT, BT, float>();
    op.run({a}, {b}, sum32);
    auto rounded = op.template get_destination_cooperative_tensor<AT, BT, {out}>();
    for (uint16_t i = 0; i < sum32.get_capacity(); ++i) {{
      if (sum32.is_valid_element(i)) {{
        {guard}rounded[i] = static_cast<{out}>({value});
      }}
    }}
    rounded.store({c});"""


def _source(
    *,
    transposed: bool,
    bias: bool,
    out: str,
    runtime_rows: bool = False,
    residual: bool = False,
) -> str:
    # Tensor extent 0 is the contiguous dimension. A transposed weight keeps the checkpoint's
    # (out, in) layout; otherwise it is stored (in, out). Interior tiles take fixed-extent
    # slices, which skip the edge checks; run() takes lvalues, so the slices are named. With
    # runtime_rows the row count comes from a one-element input instead of a template constant.
    scaled = out == "bfloat"
    rows = "\n  const int M = dims[0];" if runtime_rows else ""
    right = "(K, N)" if transposed else "(N, K)"
    right_tile = (
        "slice<dynamic_extent, TN>(0, col)" if transposed else "slice<TN, dynamic_extent>(col, 0)"
    )
    right_edge = "slice(0, col)" if transposed else "slice(col, 0)"
    if transposed:
        # Walk _STRIPES row stripes for each column stripe, then the next column stripe.
        order = f"""
  const int n_tiles = (N + TN - 1) / TN;
  const int m_tiles = (M + TM - 1) / TM;
  const int pid = int(tg.y) * n_tiles + int(tg.x);
  const int first = (pid / ({_STRIPES} * n_tiles)) * {_STRIPES};
  const int stripes = min(m_tiles - first, {_STRIPES});
  const int within = pid % ({_STRIPES} * n_tiles);
  const int row = (first + within % stripes) * TM;
  const int col = (within / stripes) * TN;"""
    else:
        order = """
  const int row = int(tg.y) * TM;
  const int col = int(tg.x) * TN;"""
    unscale = (
        "\n  const float unscale = mpp_power(-mpp_shift(float(peak_a[0])) - mpp_shift(float(peak_w[0])));"
        if scaled
        else ""
    )
    tiles = {
        name: _tile(*names, out=out, bias=bias, scaled=scaled, edge=edge, residual=residual)
        for name, names, edge in (
            ("interior", ("tA", "tB", "tC"), False),
            ("edge", ("eA", "eB", "eC"), True),
        )
    }
    return f"""
  const uint3 tg = threadgroup_position_in_grid;{rows}{order}{unscale}
  tensor<device half, dextents<int32_t, 2>, tensor_inline> A(
      (device half*)a, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B(
      (device half*)w, dextents<int32_t, 2>{right});
  tensor<device {out}, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(
      TM, TN, static_cast<int>(dynamic_extent), false, {"true" if transposed else "false"});
  matmul2d<desc, execution_simdgroups<1>> op;
  if (row + TM <= M && col + TN <= N) {{
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A_rows(
        (device half*)a + size_t(row) * K, dextents<int32_t, 2>(K, TM));
    tensor<device {out}, dextents<int32_t, 2>, tensor_inline> C_rows(
        c + size_t(row) * N, dextents<int32_t, 2>(N, TM));
    auto tA = A_rows.template slice<dynamic_extent, TM>(0, 0);
    auto tB = B.template {right_tile};
    auto tC = C_rows.template slice<TN, TM>(col, 0);{tiles["interior"]}
  }} else {{
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A_edge(
        (device half*)a + size_t(row) * K, dextents<int32_t, 2>(K, M - row));
    tensor<device {out}, dextents<int32_t, 2>, tensor_inline> C_edge(
        c + size_t(row) * N, dextents<int32_t, 2>(N, M - row));
    auto eA = A_edge.slice(0, 0);
    auto eB = B.{right_edge};
    auto eC = C_edge.slice(col, 0);{tiles["edge"]}
  }}
"""


@cache
def _kernel(
    transposed: bool, bias: bool, out: str, runtime_rows: bool = False, residual: bool = False
) -> MetalKernel:
    name = (
        f"kinomlx_mpp_linear_{'nt' if transposed else 'nn'}_{out}"
        + ("_bias" if bias else "")
        + ("_rows" if runtime_rows else "")
        + ("_residual" if residual else "")
    )
    inputs = (
        ["a", "w"]
        + (["bias"] if bias else [])
        + (["peak_a", "peak_w"] if out == "bfloat" else [])
        + (["dims"] if runtime_rows else [])
        + (["res"] if residual else [])
    )
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=["c"],
            header=_HEADER,
            source=_source(
                transposed=transposed,
                bias=bias,
                out=out,
                runtime_rows=runtime_rows,
                residual=residual,
            ),
            ensure_row_contiguous=True,
        ),
    )


@cache
def _scale_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="kinomlx_mpp_linear_scaled_fp16",
            input_names=["x", "peak"],
            output_names=["out"],
            header=_SCALES,
            source=_SCALE_SOURCE,
            ensure_row_contiguous=True,
        ),
    )


@cache
def _restore_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="kinomlx_mpp_linear_restored_bf16",
            input_names=["x", "peak"],
            output_names=["out"],
            header=_SCALES,
            source=_RESTORE_SOURCE,
            ensure_row_contiguous=True,
        ),
    )


def _scaled(values: mx.array) -> Operand:
    """``values`` in FP16 scaled by a power of two, with the largest magnitude that sets it."""
    peak = mx.max(mx.abs(values)).reshape(1)
    scaled = _scale_kernel()(
        inputs=[values, peak],
        output_shapes=[values.shape],
        output_dtypes=[mx.float16],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    return Operand(scaled, peak)


def fp16_operand(value: mx.array) -> Operand:
    """The FP16 input ``linear`` multiplies for ``value``: itself in FP16, else a scaled copy.

    Callers that project one input several times convert it once and pass it as ``operand``.
    """
    return Operand(value, None) if value.dtype == mx.float16 else _scaled(value)


def prepare(weight: mx.array) -> Operand:
    """The weight ``linear`` multiplies: a BF16 weight's scaled FP16 copy with its largest
    magnitude, or an FP16 weight as it is.

    A caller that keeps the result and passes it as the weight converts once, not per call;
    ``restore`` gives the BF16 weight back.
    """
    return fp16_operand(weight)


def restore(prepared: Operand) -> mx.array:
    """The BF16 weight a ``prepare``d one holds, exact unless a value lay about 2^28 below the
    largest of its tensor."""
    if prepared.peak is None:
        raise ValueError("only a prepared BF16 weight restores to BF16")
    return _restore_kernel()(
        inputs=[prepared.values, prepared.peak],
        output_shapes=[prepared.values.shape],
        output_dtypes=[mx.bfloat16],
        grid=(prepared.values.size, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def _launch(
    rows: Operand,
    right: Operand,
    bias: mx.array | None,
    *,
    transposed: bool,
    dtype: mx.Dtype,
) -> mx.array:
    """``rows @ right(.T) + bias`` in ``dtype``, from a 2D operand and a prepared weight."""
    m, k = (int(size) for size in rows.values.shape)
    n = int(right.values.shape[0] if transposed else right.values.shape[1])
    inputs = [rows.values, right.values] + ([] if bias is None else [bias])
    if rows.peak is not None and right.peak is not None:
        inputs += [rows.peak, right.peak]
    return _kernel(transposed, bias is not None, _TYPE_NAMES[dtype])(
        inputs=inputs,
        output_shapes=[(m, n)],
        output_dtypes=[dtype],
        grid=(32 * -(-n // _TILE), -(-m // _TILE), 1),
        threadgroup=(32, 1, 1),
        template=[("M", m), ("N", n), ("K", k), ("TM", _TILE), ("TN", _TILE)],
    )[0]


def _pattern(shape: tuple[int, ...], phase: float) -> mx.array:
    """Deterministic, irregular values in [-1, 1] (product code draws no random numbers)."""
    count = prod(shape)
    return mx.sin(mx.arange(count, dtype=mx.float32) * 0.7310585 + phase).reshape(shape)


@cache
def _available(transposed: bool, bias: bool, dtype_name: str) -> bool:
    # A ragged product, so interior and edge tiles both run, small enough that MLX does not
    # split it along K.
    dtype = mx.float16 if dtype_name == "half" else mx.bfloat16
    rows = _pattern((70, 64), 0.0).astype(dtype)
    shape = (96, 64) if transposed else (64, 96)
    weight = (_pattern(shape, 1.0) * 0.125).astype(dtype)
    offset = (_pattern((96,), 2.0) * 0.5).astype(dtype) if bias else None
    right = weight.T if transposed else weight
    try:
        product = _launch(
            fp16_operand(rows), prepare(weight), offset, transposed=transposed, dtype=dtype
        )
        mx.eval(product)
    except RuntimeError:
        return False
    expected = rows @ right if offset is None else mx.addmm(offset, rows, right)
    return bool(mx.array_equal(product, expected).item())


def available(*, transposed: bool, bias: bool, dtype: mx.Dtype = mx.float16) -> bool:
    """Whether this kernel form builds, runs, and gives ``mx.addmm``'s bits here.

    Evaluated once per process and form.
    """
    return dtype in _DTYPES and _available(transposed, bias, _TYPE_NAMES[dtype])


def supports(rows: int, k: int, n: int, *, transposed: bool, dtype: mx.Dtype) -> bool:
    """Whether MPP sums ``rows x k`` by ``k x n`` as MLX does, at a size and layout where it
    was faster."""
    return dtype in _DTYPES and _shape_supported(rows, k, n, transposed=transposed)


def _weight_dtype(weight: mx.array | Operand) -> tuple[mx.array, mx.Dtype]:
    """A weight's array and the dtype it stands for: BF16 for a prepared BF16 weight."""
    if isinstance(weight, Operand):
        return weight.values, (weight.values.dtype if weight.peak is None else mx.bfloat16)
    return weight, weight.dtype


def applies(
    value: mx.array, weight: mx.array | Operand, bias: mx.array | None, *, transposed: bool
) -> bool:
    """Whether ``linear`` computes this product: enabled, FP16 or BF16, supported, available.

    ``weight`` may be a ``prepare``d weight.
    """
    array, weight_dtype = _weight_dtype(weight)
    if not _enabled or value.ndim < 2 or array.ndim != 2:
        return False
    dtype = value.dtype
    if dtype not in _DTYPES or weight_dtype != dtype:
        return False
    k = int(value.shape[-1])
    n, inner = (int(size) for size in (array.shape if transposed else array.shape[::-1]))
    if inner != k:
        return False
    if bias is not None and (bias.dtype != dtype or bias.ndim != 1 or int(bias.shape[0]) != n):
        return False
    rows = prod(int(size) for size in value.shape[:-1])
    return supports(rows, k, n, transposed=transposed, dtype=dtype) and available(
        transposed=transposed, bias=bias is not None, dtype=dtype
    )


def linear(
    value: mx.array,
    weight: mx.array | Operand,
    bias: mx.array | None,
    *,
    transposed: bool,
    operand: Operand | None = None,
) -> mx.array:
    """``mx.addmm(bias, value, weight.T)`` for an ``(out, in)`` weight with ``transposed``, or
    ``mx.addmm(bias, value, weight)`` for an ``(in, out)`` weight, through MPP.

    ``weight`` may be a ``prepare``d weight, and ``operand`` is ``fp16_operand(value)`` when
    the caller has it. Call this where ``applies`` holds; the result then equals
    ``mx.addmm``'s within the limits the module docstring states.
    """
    dtype = value.dtype
    array, weight_dtype = _weight_dtype(weight)
    if weight_dtype != dtype:
        raise ValueError(
            f"MPP multiplies a {dtype} input by a weight of that dtype, not {weight_dtype}"
        )
    k = int(value.shape[-1])
    n = int(array.shape[0] if transposed else array.shape[1])
    rows = prod(int(size) for size in value.shape[:-1])
    if not supports(rows, k, n, transposed=transposed, dtype=dtype):
        raise ValueError(f"MPP does not take a {rows} x {k} by {k} x {n} {dtype} product")
    if operand is None:
        operand = fp16_operand(value)
    elif (
        operand.values.shape != value.shape
        or operand.values.dtype != mx.float16
        or (operand.peak is None) != (dtype == mx.float16)
    ):
        raise ValueError("operand must be fp16_operand(value)")
    flat = Operand(operand.values.reshape(rows, k), operand.peak)
    right = weight if isinstance(weight, Operand) else prepare(weight)
    product = _launch(flat, right, bias, transposed=transposed, dtype=dtype)
    return product.reshape(*value.shape[:-1], n)


def rows_supported(rows: int, k: int, n: int) -> bool:
    """Whether an FP16 ``rows x k`` product with an ``(n, k)`` weight, its row count read at run
    time, sums as MLX does at any width.

    MLX sums along K in a different order only for products of at most 2,048 16x16 output tiles,
    so more tiles than that keep its order whatever the widths. The inner dimension must be a
    multiple of 16 and at most 8,192, and the rows and 32x32 output tiles within the kernel's
    32-bit counts (``rows_addressable``).
    """
    tiles = -(-rows // 16) * -(-n // 16)
    return tiles > _SPLIT_TILES and k % 16 == 0 and 0 < k <= _LONG and rows_addressable(rows, n)


def _launch_rows(
    value: mx.array, weight: mx.array, bias: mx.array | None, residual: mx.array | None
) -> mx.array:
    m, k = (int(size) for size in value.shape)
    n = int(weight.shape[0])
    inputs = (
        [value, weight]
        + ([] if bias is None else [bias])
        + [mx.array([m], dtype=mx.int32)]
        + ([] if residual is None else [residual])
    )
    return _kernel(True, bias is not None, "half", True, residual is not None)(
        inputs=inputs,
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
        grid=(32 * -(-n // _TILE), -(-m // _TILE), 1),
        threadgroup=(32, 1, 1),
        template=[("N", n), ("K", k), ("TM", _TILE), ("TN", _TILE)],
    )[0]


@cache
def _rows_available(bias: bool, residual: bool) -> bool:
    rows = _pattern((70, 64), 0.0).astype(mx.float16)
    weight = (_pattern((96, 64), 1.0) * 0.125).astype(mx.float16)
    offset = (_pattern((96,), 2.0) * 0.5).astype(mx.float16) if bias else None
    added = (_pattern((70, 96), 3.0) * 0.5).astype(mx.float16) if residual else None
    try:
        product = _launch_rows(rows, weight, offset, added)
        mx.eval(product)
    except RuntimeError:
        return False
    expected = rows @ weight.T if offset is None else mx.addmm(offset, rows, weight.T)
    if added is not None:
        expected = added + expected
    return bool(mx.array_equal(product, expected).item())


def rows_available(*, bias: bool, residual: bool) -> bool:
    """Whether the run-time-row FP16 form builds, runs, and gives ``residual + mx.addmm``'s bits
    here. Evaluated once per process and form."""
    return _rows_available(bias, residual)


def rows_applies(
    value: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    residual: mx.array | None = None,
) -> bool:
    """Whether ``linear_rows`` computes this product: enabled, FP16, supported, available."""
    if not _enabled or value.ndim != 2 or weight.ndim != 2:
        return False
    if value.dtype != mx.float16 or weight.dtype != mx.float16:
        return False
    rows, k = (int(size) for size in value.shape)
    n, inner = (int(size) for size in weight.shape)
    if inner != k:
        return False
    if bias is not None and (bias.dtype != mx.float16 or tuple(bias.shape) != (n,)):
        return False
    if residual is not None and (
        residual.dtype != mx.float16 or tuple(residual.shape) != (rows, n)
    ):
        return False
    return rows_supported(rows, k, n) and rows_available(
        bias=bias is not None, residual=residual is not None
    )


def linear_rows(
    value: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    residual: mx.array | None = None,
) -> mx.array:
    """``residual + mx.addmm(bias, value, weight.T)`` for FP16 ``(rows, k)`` rows and an
    ``(n, k)`` weight, through MPP with the row count read at run time.

    One build serves every row count of an inner and output width. The product and its bias
    round to FP16 once, as ``mx.addmm`` rounds; the residual is then added in FP16. Call this
    where ``rows_applies`` holds.
    """
    if not rows_applies(value, weight, bias, residual):
        raise ValueError(
            f"MPP's run-time-row form does not take {tuple(value.shape)} rows by a "
            f"{tuple(weight.shape)} weight here"
        )
    return _launch_rows(value, weight, bias, residual)


__all__ = [
    "MIN_ROWS",
    "MIN_WIDTH",
    "Operand",
    "applies",
    "available",
    "enabled",
    "fp16_operand",
    "linear",
    "linear_rows",
    "prepare",
    "restore",
    "rows_addressable",
    "rows_applies",
    "rows_available",
    "rows_supported",
    "set_enabled",
    "supports",
]
