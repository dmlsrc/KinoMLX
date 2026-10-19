"""Banded F(6x6, 3x3) Winograd for the LTX convolutional VAE decoder.

Each 3x3x3 convolution transforms its input in bands of tile rows. Through MPP a band's three
temporal taps are multiplied and summed in one FP32 product per transform plane, for the kept
frames only; otherwise they are three FP16 products, summed in the output transform.
"""

from functools import cache
from typing import cast

import mlx.core as mx

from . import mpp_linear
from ._typing import MetalKernel

_KERNELS: dict[str, MetalKernel] = {}
_BAND_BUDGET_BYTES = 128 << 20
_COPY_INDEX_LIMIT = 1 << 31

_IN_8 = """
#define WG_IN8(d0, d1, d2, d3, d4, d5, d6, d7, v)            \\
    {                                                        \\
        float _t1 = (d2) + (d6) - (d4) * 4.25f;              \\
        float _t2 = (d1) + (d5) - (d3) * 4.25f;              \\
        float _t3 = (d6) + (d2) * 0.25f - (d4) * 1.25f;      \\
        float _t4 = (d1) * 0.5f - (d3) * 2.5f + (d5) * 2.0f; \\
        float _t5 = (d6) + ((d2) - (d4) * 1.25f) * 4.0f;     \\
        float _t6 = (d1) * 2.0f - (d3) * 2.5f + (d5) * 0.5f; \\
        v[0] = (d0) - (d6) + ((d4) - (d2)) * 5.25f;          \\
        v[1] = _t1 + _t2;                                    \\
        v[2] = _t1 - _t2;                                    \\
        v[3] = _t3 + _t4;                                    \\
        v[4] = _t3 - _t4;                                    \\
        v[5] = _t5 + _t6;                                    \\
        v[6] = _t5 - _t6;                                    \\
        v[7] = (d7) - (d1) + ((d3) - (d5)) * 5.25f;          \\
    }
"""

# y = A^T Y, factored: 8 transformed values in, 6 spatial out.
_OUT_8 = """
#define WG_OUT8(y0, y1, y2, y3, y4, y5, y6, y7, o)                  \\
    {                                                               \\
        float _e0 = (y1) + (y2); float _e1 = (y1) - (y2);           \\
        float _e2 = (y3) + (y4); float _e3 = (y3) - (y4);           \\
        float _e4 = (y5) + (y6); float _e5 = (y5) - (y6);           \\
        o[0] = (y0) + _e0 + _e2 + _e4;                              \\
        o[1] = _e1 + _e3 * 2.0f + _e5 * 0.5f;                       \\
        o[2] = _e0 + _e2 * 4.0f + _e4 * 0.25f;                      \\
        o[3] = _e1 + _e3 * 8.0f + _e5 * 0.125f;                     \\
        o[4] = _e0 + _e2 * 16.0f + _e4 * 0.0625f;                   \\
        o[5] = _e1 + _e3 * 32.0f + _e5 * 0.03125f + (y7);           \\
    }
"""

_HEADER = (
    _IN_8
    + _OUT_8
    + """
inline float wg_dot3(float a0, float b0, float a1, float b1, float a2, float b2) {
    return fma(a2, b2, fma(a1, b1, fma(a0, b0, 0.0f)));
}
"""
)
_FILTER_SRC = r"""
    uint group = thread_position_in_grid.x;
    uint packed = thread_position_in_grid.y;
    uint row_group = thread_position_in_grid.z;
    if (group * 4u >= (uint)COUT || packed >= 3u * (uint)CIN) return;

    uint tap = packed / (uint)CIN;
    uint channel = packed - tap * (uint)CIN;
    const device half* src = raw + ((ulong)packed * 9ul * (ulong)COUT_TOTAL
                                    + (ulong)OUT_START + group * 4u);
    float4 k[9];
    for (uint n = 0u; n < 9u; ++n) {
        k[n] = float4(*reinterpret_cast<const device half4*>(src + (ulong)n * COUT_TOTAL));
    }
    float gm[24];
    for (uint n = 0u; n < 24u; ++n) gm[n] = g[n];

    for (uint r = row_group * 4u; r < row_group * 4u + 4u; ++r) {
        float4 x0, x1, x2;
        for (uint f = 0u; f < 4u; ++f) {
            x0[f] = wg_dot3(gm[r * 3u], k[0][f], gm[r * 3u + 1u], k[3][f],
                             gm[r * 3u + 2u], k[6][f]);
            x1[f] = wg_dot3(gm[r * 3u], k[1][f], gm[r * 3u + 1u], k[4][f],
                             gm[r * 3u + 2u], k[7][f]);
            x2[f] = wg_dot3(gm[r * 3u], k[2][f], gm[r * 3u + 1u], k[5][f],
                             gm[r * 3u + 2u], k[8][f]);
        }
        for (uint s = 0u; s < 8u; ++s) {
            float4 u;
            for (uint f = 0u; f < 4u; ++f) {
                u[f] = wg_dot3(x0[f], gm[s * 3u], x1[f], gm[s * 3u + 1u],
                               x2[f], gm[s * 3u + 2u]);
            }
            ulong offset = (((ulong)tap * 64ul + r * 8u + s) * (ulong)CIN
                            + channel) * (ulong)COUT + group * 4u;
            *reinterpret_cast<device half4*>(out + offset) = half4(u);
        }
    }
"""
_INPUT_SRC = r"""
    uint c = thread_position_in_grid.x;
    uint tw = threadgroup_position_in_grid.y;
    uint z = threadgroup_position_in_grid.z;
    uint frame = z / (uint)TH;
    uint th = z - frame * (uint)TH;

    // TH0 shifts which tile rows this call covers, so a convolution can be run
    // in horizontal bands without changing any tile's arithmetic.
    int row0 = (int)((th + (uint)TH0) * 6u) - 1;
    int col0 = (int)(tw * 6u) - 1;
    ulong xbase = (ulong)frame * (ulong)H * (ulong)W * (ulong)C;

    float d[8][8];
    for (uint i = 0u; i < 8u; ++i) {
        int r = row0 + (int)i;
        bool rok = (r >= 0) && (r < (int)H);
        ulong rbase = xbase + (ulong)max(r, 0) * (ulong)W * (ulong)C;
        for (uint j = 0u; j < 8u; ++j) {
            int q = col0 + (int)j;
            d[i][j] = (rok && q >= 0 && q < (int)W)
                ? (float)inp[rbase + (ulong)q * (ulong)C + c] : 0.0f;
        }
    }

    float v[8];
    float z1[8][8];
    for (uint j = 0u; j < 8u; ++j) {
        WG_IN8(d[0][j], d[1][j], d[2][j], d[3][j],
               d[4][j], d[5][j], d[6][j], d[7][j], v);
        for (uint p = 0u; p < 8u; ++p) z1[p][j] = v[p];
    }

    ulong tile = ((ulong)frame * (ulong)TH + (ulong)th) * (ulong)TW + (ulong)tw;
    ulong nt = (ulong)T * (ulong)TH * (ulong)TW;
    for (uint p = 0u; p < 8u; ++p) {
        WG_IN8(z1[p][0], z1[p][1], z1[p][2], z1[p][3],
               z1[p][4], z1[p][5], z1[p][6], z1[p][7], v);
        for (uint q = 0u; q < 8u; ++q) {
            out[((ulong)(p * 8u + q) * nt + tile) * (ulong)C + c] = (T_IO)v[q];
        }
    }
"""
# The output transforms load a tile's 8x8 transformed values into y and share the rest: A^T y A,
# the bias, and one rounding into the output for each of the tile's 6x6 values.
_OUTPUT_HEAD = r"""
    uint c = thread_position_in_grid.x;
    uint tw = threadgroup_position_in_grid.y;
    uint zz = threadgroup_position_in_grid.z;
    uint frame = zz / (uint)TH;
    uint th = zz - frame * (uint)TH;

    ulong sp = (ulong)TH * (ulong)TW;
    ulong spatial = (ulong)th * (ulong)TW + (ulong)tw;
    float y[8][8];
    float g[6][8];
    float o[6];
"""
# Three FP16 tap products, summed here for every padded frame; the first two frames are dropped.
_TAP_PRODUCTS_LOAD = r"""
    ulong f0 = (ulong)(frame >= 2u ? frame - 2u : 0u);
    ulong f1 = (ulong)(frame >= 1u ? frame - 1u : 0u);
    ulong f2 = (ulong)frame;
    for (uint p = 0u; p < 8u; ++p) {
        for (uint q = 0u; q < 8u; ++q) {
            ulong plane = (ulong)(p * 8u + q);
            y[p][q] = (float)ys0[((plane * (ulong)N0 * sp + f0 * sp + spatial)
                                  * (ulong)COUT) + c]
                    + (float)ys1[((plane * (ulong)N1 * sp + f1 * sp + spatial)
                                  * (ulong)COUT) + c]
                    + (float)ys2[((plane * (ulong)N2 * sp + f2 * sp + spatial)
                                  * (ulong)COUT) + c];
        }
    }
"""
# The three taps' FP32 sum, for the kept frames only.
_TAP_SUM_LOAD = r"""
    for (uint p = 0u; p < 8u; ++p) {
        for (uint q = 0u; q < 8u; ++q) {
            ulong plane = (ulong)(p * 8u + q);
            y[p][q] = ys[((plane * (ulong)NF * sp + (ulong)frame * sp + spatial)
                          * (ulong)COUT) + c];
        }
    }
"""
_OUTPUT_TAIL = r"""
    // Contract the rows first so the store loop runs along columns.
    for (uint q = 0u; q < 8u; ++q) {
        WG_OUT8(y[0][q], y[1][q], y[2][q], y[3][q], y[4][q], y[5][q], y[6][q], y[7][q], o);
        for (uint m = 0u; m < 6u; ++m) g[m][q] = o[m];
    }

    float b = (float)bias[c];
    ulong obase = (ulong)frame * (ulong)H * (ulong)W * (ulong)COUT;
    for (uint m = 0u; m < 6u; ++m) {
        WG_OUT8(g[m][0], g[m][1], g[m][2], g[m][3], g[m][4], g[m][5], g[m][6], g[m][7], o);
        uint row = th * 6u + m;
        if (row >= (uint)H) continue;
        ulong rbase = obase + ((ulong)row * (ulong)W + (ulong)(tw * 6u)) * (ulong)COUT + c;
        for (uint n = 0u; n < 6u; ++n) {
            if (tw * 6u + n >= (uint)W) continue;
            float val = o[n] + b;
            out[rbase + (ulong)n * (ulong)COUT] = (T_IO)val;
        }
    }
"""
_OUTPUT_SRC = _OUTPUT_HEAD + _TAP_PRODUCTS_LOAD + _OUTPUT_TAIL
_TAP_SUM_OUTPUT_SRC = _OUTPUT_HEAD + _TAP_SUM_LOAD + _OUTPUT_TAIL

_MPP_HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""


def _tap_sum_tile(a: str, b: str, c: str) -> str:
    return f"""
        using AT = __tensor_ops_detail::__remove_addrspace_t<decltype({a}0)>;
        using BT = __tensor_ops_detail::__remove_addrspace_t<decltype({b}0)>;
        auto sum32 = op.template get_destination_cooperative_tensor<AT, BT, float>();
        for (uint16_t i = 0; i < sum32.get_capacity(); ++i) {{
            sum32[i] = 0.0f;
        }}
        op.run({a}0, {b}0, sum32);
        op.run({a}1, {b}1, sum32);
        op.run({a}2, {b}2, sum32);
        sum32.store({c});"""


# One transform plane's three temporal taps summed in an FP32 tile. Tap k reads the transformed
# input in place from k frames on, so kept frame f sums padded frames f, f + 1 and f + 2 against
# taps 0, 1 and 2. 32x32 output tiles, one SIMD group each, in matmul2d's multiply-accumulate
# mode; the sum equals MLX's FP32 product of the three row windows laid side by side along K
# against the stacked filters, bit for bit.
_MPP_TAP_SUM_SRC = (
    r"""
    const uint3 tg = threadgroup_position_in_grid;
    device half* a_ptr = (device half*)a + (ulong)tg.z * (ulong)ROWS * (ulong)K;
    const ulong shift = (ulong)SHIFT * (ulong)K;
    const ulong plane = (ulong)tg.z * (ulong)K * (ulong)N;
    device float* c_ptr = c + (ulong)tg.z * (ulong)M * (ulong)N;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A0(a_ptr, dextents<int32_t, 2>(K, M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A1(
        a_ptr + shift, dextents<int32_t, 2>(K, M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> A2(
        a_ptr + 2ul * shift, dextents<int32_t, 2>(K, M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B0(
        (device half*)b0 + plane, dextents<int32_t, 2>(N, K));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B1(
        (device half*)b1 + plane, dextents<int32_t, 2>(N, K));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> B2(
        (device half*)b2 + plane, dextents<int32_t, 2>(N, K));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c_ptr, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(
        32, 32, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<1>> op;
    const int row = int(tg.y) * 32;
    const int col = int(tg.x) * 32;
    if (row + 32 <= M && col + 32 <= N) {
        auto tA0 = A0.template slice<dynamic_extent, 32>(0, row);
        auto tA1 = A1.template slice<dynamic_extent, 32>(0, row);
        auto tA2 = A2.template slice<dynamic_extent, 32>(0, row);
        auto tB0 = B0.template slice<32, dynamic_extent>(col, 0);
        auto tB1 = B1.template slice<32, dynamic_extent>(col, 0);
        auto tB2 = B2.template slice<32, dynamic_extent>(col, 0);
        auto tC = C.template slice<32, 32>(col, row);"""
    + _tap_sum_tile("tA", "tB", "tC")
    + r"""
    } else {
        auto eA0 = A0.slice(0, row);
        auto eA1 = A1.slice(0, row);
        auto eA2 = A2.slice(0, row);
        auto eB0 = B0.slice(col, 0);
        auto eB1 = B1.slice(col, 0);
        auto eB2 = B2.slice(col, 0);
        auto eC = C.slice(col, row);"""
    + _tap_sum_tile("eA", "eB", "eC")
    + r"""
    }
"""
)


@cache
def _mpp_tap_sum_kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_vae_wg_mpp_tap_sum",
            input_names=["a", "b0", "b1", "b2"],
            output_names=["c"],
            header=_MPP_HEADER,
            source=_MPP_TAP_SUM_SRC,
            ensure_row_contiguous=True,
        ),
    )


def _mpp_tap_sum(
    transformed: mx.array, filters: tuple[mx.array, ...], tiles_per_frame: int, frames: int
) -> mx.array:
    """The three taps' FP32 sum for the kept frames: (64, (frames - 2) * tiles_per_frame, N)."""
    planes, total, channels = (int(value) for value in transformed.shape)
    output = int(filters[0].shape[-1])
    rows = (frames - 2) * tiles_per_frame
    return _mpp_tap_sum_kernel()(
        inputs=[transformed, *filters],
        output_shapes=[(planes, rows, output)],
        output_dtypes=[mx.float32],
        grid=(32 * -(-output // 32), -(-rows // 32), planes),
        threadgroup=(32, 1, 1),
        template=[
            ("M", rows),
            ("N", output),
            ("K", channels),
            ("ROWS", total),
            ("SHIFT", tiles_per_frame),
        ],
    )[0]


@cache
def _mpp_tap_sum_available() -> bool:
    # Ragged in rows and columns, deterministic values. The reference lays the three row windows
    # side by side along K, which MLX's FP32 product sums in the kernel's order.
    planes, frames, tiles, channels, output = 2, 5, 7, 64, 72
    inputs = planes * frames * tiles * channels
    values = mx.sin(
        mx.arange(inputs + 3 * planes * channels * output, dtype=mx.float32) * 0.7310585
    )
    transformed = values[:inputs].reshape(planes, frames * tiles, channels).astype(mx.float16)
    stacked = (values[inputs:].reshape(planes, 3 * channels, output) * 0.125).astype(mx.float16)
    filters = tuple(
        mx.contiguous(stacked[:, index * channels : (index + 1) * channels]) for index in range(3)
    )
    rows = (frames - 2) * tiles
    windows = mx.concatenate(
        [transformed[:, index * tiles : index * tiles + rows] for index in range(3)], axis=-1
    )
    try:
        product = _mpp_tap_sum(transformed, filters, tiles, frames)
        mx.eval(product)
    except RuntimeError:
        return False
    expected = mx.matmul(windows.astype(mx.float32), stacked.astype(mx.float32))
    return bool(mx.array_equal(product, expected).item())


def _uses_tap_sum(transformed: mx.array, tiles_per_frame: int, frames: int) -> bool:
    return (
        transformed.dtype == mx.float16
        and int(transformed.shape[-1]) % 16 == 0
        and (frames - 2) * tiles_per_frame > 1
        and mpp_linear.enabled()
        and _mpp_tap_sum_available()
    )


def _tap_products(
    transformed: mx.array, filters: tuple[mx.array, ...], tiles_per_frame: int, frames: int
) -> list[mx.array]:
    """The three temporal taps' FP16 products; tap k reads all but the last 2 - k frames' tiles."""
    rows = [(frames - (2 - index)) * tiles_per_frame for index in range(3)]
    return [mx.matmul(transformed[:, : rows[index]], filters[index]) for index in range(3)]


_G_T = (
    (1.0, 0.0, 0.0),
    (-2 / 9, -2 / 9, -2 / 9),
    (-2 / 9, 2 / 9, -2 / 9),
    (1 / 90, 1 / 45, 2 / 45),
    (1 / 90, -1 / 45, 2 / 45),
    (32 / 45, 16 / 45, 8 / 45),
    (32 / 45, -16 / 45, 8 / 45),
    (0.0, 0.0, 1.0),
)


@cache
def _weight_matrix() -> mx.array:
    matrix = mx.array(_G_T, dtype=mx.float32)
    mx.eval(matrix)
    return matrix


def _weight_transform(weight: mx.array) -> mx.array:
    """Transform (3*COUT, 3, 3, CIN) weights to (64, CIN, 3*COUT)."""
    output, _kh, _kw, channels = weight.shape
    matrix = _weight_matrix()
    values = weight.astype(mx.float32).transpose(0, 3, 1, 2).reshape(output * channels, 3, 3)
    values = mx.matmul(matrix, values)
    values = mx.matmul(values, matrix.transpose())
    return values.reshape(output, channels, 64).transpose(2, 1, 0)


def _fused_filter_transform_slab(raw: mx.array, start: int, output: int) -> mx.array:
    taps = 3
    if raw.ndim != 3 or raw.shape[0] % taps or raw.shape[1] != 9:
        raise ValueError("Winograd filter slab requires a packed three-tap filter")
    channels = int(raw.shape[0]) // 3
    output_total = int(raw.shape[-1])
    if start % 4 or output % 4 or start + output > output_total:
        raise ValueError("Winograd filter slab must cover aligned output channels")
    return _kernel("ltx_vae_wg_filter_transform", _FILTER_SRC, ("raw", "g"))(
        inputs=[raw, _weight_matrix().reshape(-1)],
        output_shapes=[(taps, 64, channels, output)],
        output_dtypes=[mx.float16],
        grid=(output // 4, taps * channels, 2),
        threadgroup=(min(output // 4, 32), 8, 1),
        template=[
            ("CIN", channels),
            ("COUT", output),
            ("COUT_TOTAL", output_total),
            ("OUT_START", start),
        ],
    )[0]


def _fused_filter_transform(weight: mx.array) -> mx.array:
    """Transform OTHWI filters directly to three FP16 Winograd tap slabs."""
    output, taps, _kh, _kw, channels = (int(value) for value in weight.shape)
    if taps != 3 or output % 4 or weight.dtype != mx.float16:
        raise ValueError("fused Winograd filter transform requires FP16 3-tap weights")
    raw = mx.contiguous(weight.transpose(1, 4, 2, 3, 0).reshape(taps * channels, 9, output))
    return _fused_filter_transform_slab(raw, 0, output)


def _kernel(name: str, source: str, inputs: tuple[str, ...]) -> MetalKernel:
    cached = _KERNELS.get(name)
    if cached is None:
        cached = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name=name,
                input_names=list(inputs),
                output_names=["out"],
                source=source,
                header=_HEADER,
                ensure_row_contiguous=True,
                compile_options={"math_mode": "safe"},
            ),
        )
        _KERNELS[name] = cached
    return cached


def _input_transform(x: mx.array, band_start: int, band_rows: int) -> tuple[mx.array, int]:
    frames, height, width, channels = (int(value) for value in x.shape)
    tile_width = (width + 5) // 6
    transformed = _kernel("ltx_vae_wg_input", _INPUT_SRC, ("inp",))(
        inputs=[x],
        output_shapes=[(64, frames * band_rows * tile_width, channels)],
        output_dtypes=[x.dtype],
        grid=(channels, tile_width, frames * band_rows),
        threadgroup=(min(channels, 256), 1, 1),
        template=[
            ("T_IO", x.dtype),
            ("C", channels),
            ("H", height),
            ("W", width),
            ("T", frames),
            ("TH", band_rows),
            ("TW", tile_width),
            ("TH0", band_start),
        ],
    )[0]
    return transformed, tile_width


def _output_transform(
    products: list[mx.array],
    bias: mx.array,
    frames: int,
    height: int,
    width: int,
    band_rows: int,
    tile_width: int,
    dtype: mx.Dtype,
) -> mx.array:
    channels = int(bias.size)
    tile_count = band_rows * tile_width
    lengths = [int(value.shape[1]) // tile_count for value in products]
    return _kernel("ltx_vae_wg_fused_tap_output", _OUTPUT_SRC, ("ys0", "ys1", "ys2", "bias"))(
        inputs=[*products, bias],
        output_shapes=[(frames, height, width, channels)],
        output_dtypes=[dtype],
        grid=(channels, tile_width, frames * band_rows),
        threadgroup=(min(channels, 256), 1, 1),
        template=[
            ("T_IO", dtype),
            ("COUT", channels),
            ("N0", lengths[0]),
            ("N1", lengths[1]),
            ("N2", lengths[2]),
            ("H", height),
            ("W", width),
            ("T", frames),
            ("TH", band_rows),
            ("TW", tile_width),
        ],
    )[0]


def _tap_sum_output_transform(
    product: mx.array,
    bias: mx.array,
    kept: int,
    height: int,
    width: int,
    band_rows: int,
    tile_width: int,
    dtype: mx.Dtype,
) -> mx.array:
    channels = int(bias.size)
    return _kernel("ltx_vae_wg_tap_sum_output", _TAP_SUM_OUTPUT_SRC, ("ys", "bias"))(
        inputs=[product, bias],
        output_shapes=[(kept, height, width, channels)],
        output_dtypes=[dtype],
        grid=(channels, tile_width, kept * band_rows),
        threadgroup=(min(channels, 256), 1, 1),
        template=[
            ("T_IO", dtype),
            ("COUT", channels),
            ("NF", kept),
            ("H", height),
            ("W", width),
            ("TH", band_rows),
            ("TW", tile_width),
        ],
    )[0]


def _band_output(
    transformed: mx.array,
    filters: tuple[mx.array, ...],
    bias: mx.array,
    frames: int,
    height: int,
    width: int,
    band_rows: int,
    tile_width: int,
    dtype: mx.Dtype,
) -> mx.array:
    """One band's output for the kept frames, all but the first two padded ones."""
    tiles = band_rows * tile_width
    if _uses_tap_sum(transformed, tiles, frames):
        product = _mpp_tap_sum(transformed, filters, tiles, frames)
        return _tap_sum_output_transform(
            product, bias, frames - 2, height, width, band_rows, tile_width, dtype
        )
    products = _tap_products(transformed, filters, tiles, frames)
    full = _output_transform(products, bias, frames, height, width, band_rows, tile_width, dtype)
    return full[2:]


class WinogradConv3d:
    """Run banded Winograd with resident or transient transformed filters."""

    def __init__(
        self,
        *,
        transient_output_block: int | None = None,
    ) -> None:
        if transient_output_block is not None and (
            transient_output_block <= 0 or transient_output_block % 4
        ):
            raise ValueError("transient Winograd output block must be a positive multiple of four")
        self._transient_output_block = transient_output_block
        self._source_weight: mx.array | None = None
        self._transformed_taps: tuple[mx.array, mx.array, mx.array] | None = None

    @staticmethod
    def supports(x: mx.array, weight: mx.array) -> bool:
        if x.ndim != 5 or weight.ndim != 5 or x.shape[0] != 1:
            return False
        batch, frames, height, width, channels = (int(value) for value in x.shape)
        del batch
        output, taps, kh, kw, weight_channels = (int(value) for value in weight.shape)
        residual = (
            (channels, output) in ((128, 128), (256, 256), (256, 512), (512, 512))
            and frames >= (48 if channels == 512 else 64)
            and height >= (96 if channels == 128 else 48)
            and width >= (128 if channels == 128 else 96)
        )
        early_projection = (
            (channels, output) == (512, 4096) and frames >= 31 and height >= 28 and width >= 48
        ) or ((channels, output) == (1024, 4096) and frames >= 16 and height >= 14 and width >= 24)
        early_residual = (
            (channels, output) == (512, 512) and frames >= 31 and height >= 28 and width >= 48
        ) or ((channels, output) == (1024, 1024) and frames >= 16 and height >= 14 and width >= 24)
        return (
            x.dtype == mx.float16
            and weight.dtype == x.dtype
            and (taps, kh, kw) == (3, 3, 3)
            and channels == weight_channels
            and (residual or early_projection or early_residual)
            and (frames + 2) * height * width * output < _COPY_INDEX_LIMIT
        )

    def _prepare(self, weight: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if weight is self._source_weight and self._transformed_taps is not None:
            return self._transformed_taps
        if weight.dtype == mx.float16 and mx.default_device() == mx.gpu:
            transformed = _fused_filter_transform(weight)
            selected = tuple(mx.contiguous(transformed[index]) for index in range(3))
        else:
            layout = weight.transpose(0, 4, 1, 2, 3)
            output = int(weight.shape[0])
            taps = mx.contiguous(
                mx.concatenate([layout[:, :, index] for index in range(3)], axis=0).transpose(
                    0, 2, 3, 1
                )
            )
            transformed = mx.contiguous(_weight_transform(taps).astype(weight.dtype))
            selected = (
                mx.contiguous(transformed[:, :, :output]),
                mx.contiguous(transformed[:, :, output : 2 * output]),
                mx.contiguous(transformed[:, :, 2 * output :]),
            )
        mx.eval(*selected)
        self._source_weight = weight
        self._transformed_taps = (selected[0], selected[1], selected[2])
        return self._transformed_taps

    def _call_transient_slabs(
        self,
        padded: mx.array,
        weight: mx.array,
        bias: mx.array,
        frames: int,
        height: int,
        width: int,
        channels: int,
        output: int,
        tile_rows: int,
        tile_width: int,
    ) -> mx.array:
        block = self._transient_output_block
        if block is None or output % block:
            raise ValueError("transient Winograd output block must divide output channels")
        raw = mx.contiguous(weight.transpose(1, 4, 2, 3, 0).reshape(3 * channels, 9, output))
        mx.eval(raw)
        working = 64 * frames * tile_rows * tile_width * (channels * 2 + 3 * block * 2 + block * 4)
        bands = max(1, (working + _BAND_BUDGET_BYTES - 1) // _BAND_BUDGET_BYTES)
        band_rows = (tile_rows + bands - 1) // bands
        inputs = []
        for start in range(0, tile_rows, band_rows):
            count = min(band_rows, tile_rows - start)
            transformed, _ = _input_transform(padded, start, count)
            mx.eval(transformed)
            inputs.append((start, count, transformed))

        outputs: list[list[mx.array]] = [[] for _ in inputs]
        for channel_start in range(0, output, block):
            transformed_filter = _fused_filter_transform_slab(raw, channel_start, block)
            filters = tuple(mx.contiguous(transformed_filter[index]) for index in range(3))
            mx.eval(*filters)
            for band_index, (start, count, transformed) in enumerate(inputs):
                band_height = min(count * 6, height - start * 6)
                result = _band_output(
                    transformed,
                    filters,
                    bias[channel_start : channel_start + block],
                    frames,
                    band_height,
                    width,
                    count,
                    tile_width,
                    padded.dtype,
                )
                mx.eval(result)
                outputs[band_index].append(result)
            del transformed_filter, filters

        joined_bands = []
        for channel_outputs in outputs:
            band = mx.concatenate(channel_outputs, axis=-1)
            mx.eval(band)
            joined_bands.append(band)
        joined = mx.concatenate(joined_bands, axis=1) if len(joined_bands) > 1 else joined_bands[0]
        return joined[None]

    def __call__(
        self,
        x: mx.array,
        weight: mx.array,
        bias: mx.array,
    ) -> mx.array:
        """Convolve BTHWC with one temporal edge repeat on each side."""
        if (
            x.ndim != 5
            or x.shape[0] != 1
            or weight.ndim != 5
            or tuple(weight.shape[1:4]) != (3, 3, 3)
            or x.shape[-1] != weight.shape[-1]
            or x.dtype != weight.dtype
        ):
            raise ValueError("Winograd Conv3d requires batch-one BTHWC and OTHWI 3x3x3 weights")
        padded = mx.concatenate([x[0, :1], x[0], x[0, -1:]], axis=0)
        frames, height, width, channels = (int(value) for value in padded.shape)
        output = int(weight.shape[0])
        tile_rows = (height + 5) // 6
        tile_width = (width + 5) // 6
        if self._transient_output_block is not None:
            return self._call_transient_slabs(
                padded, weight, bias, frames, height, width, channels, output, tile_rows, tile_width
            )
        working = (
            64 * frames * tile_rows * tile_width * (channels * 2 + 3 * output * 2 + output * 4)
        )
        bands = max(1, (working + _BAND_BUDGET_BYTES - 1) // _BAND_BUDGET_BYTES)
        band_rows = (tile_rows + bands - 1) // bands
        filters = self._prepare(weight)
        outputs = []
        for start in range(0, tile_rows, band_rows):
            count = min(band_rows, tile_rows - start)
            transformed, tile_width = _input_transform(padded, start, count)
            band_height = min(count * 6, height - start * 6)
            band_output = _band_output(
                transformed, filters, bias, frames, band_height, width, count, tile_width, x.dtype
            )
            # Finish this band before constructing the next one's products.
            mx.eval(band_output)
            outputs.append(band_output)
            del transformed
        joined = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return joined[None]


__all__ = ["WinogradConv3d"]
