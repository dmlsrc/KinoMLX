"""Precision-sensitive operations backed by Metal kernels and faithful formulas."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn

from ._typing import MetalKernel

_LOW_PRECISION_DTYPES = frozenset({mx.bfloat16, mx.float16})
_THREADGROUP_SIZE = 256
_GROUP_NORM_MAX_THREADGROUP_SIZE = 1024
_GROUP_NORM_READS_PER_THREAD = 4
_RMS_NORM_MAX_THREADGROUP_SIZE = 256
_RMS_NORM_READS_PER_THREAD = 4
_KERNELS: dict[str, MetalKernel] = {}

# The fast tanh matches PyTorch/MPS's realized low-precision GELU. Its small
# difference from the explicit stock-FP32 tanh is the source of the test
# tolerance; accumulation itself remains FP32. At |inner| >= 10, tanh has
# already rounded to its saturated low-precision result, so the clamp is
# semantics-preserving while preventing the fast intrinsic's inf/inf NaN on
# the observed large Gemma activations.
_GELU_APPROX_SOURCE = r"""
    uint elem = thread_position_in_grid.x;
    float x = float(inp[elem]);
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    float tanh_inner = metal::fast::tanh(metal::clamp(inner, -10.0f, 10.0f));
    float result = 0.5f * x * (1.0f + tanh_inner);
    out[elem] = T(result);
"""

# SiLU uses the precise exponential because it was required for elementwise
# agreement with the explicit FP32/PyTorch anchors. Unlike fast tanh, exp has
# well-defined saturation through infinity here, so it needs no safety clamp.
_SILU_SOURCE = r"""
    uint elem = thread_position_in_grid.x;
    float x = float(inp[elem]);
    float result = x / (1.0f + metal::precise::exp(-x));
    out[elem] = T(result);
"""

_GROUP_NORM_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_groups = (threads_per_threadgroup.x + 31u) / 32u;

    threadgroup float local_sums[32];
    threadgroup float statistics[2];

    uint batch = row / NUM_GROUPS;
    uint group = row - batch * NUM_GROUPS;
    uint input_batch_offset = batch * SPATIAL_SIZE * CHANNELS;

    float sum = 0.0f;
    for (uint base = lid * READS_PER_THREAD;
         base < GROUP_ELEMENTS;
         base += threads_per_threadgroup.x * READS_PER_THREAD) {
        for (uint read = 0u; read < READS_PER_THREAD; ++read) {
            uint linear = base + read;
            if (linear < GROUP_ELEMENTS) {
                uint spatial = linear / GROUP_SIZE;
                uint channel = linear - spatial * GROUP_SIZE + group * GROUP_SIZE;
                sum += float(inp[input_batch_offset + spatial * CHANNELS + channel]);
            }
        }
    }

    sum = simd_sum(sum);
    if (simd_lane == 0u) {
        local_sums[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float partial = simd_lane < simd_groups ? local_sums[simd_lane] : 0.0f;
        partial = simd_sum(partial);
        if (simd_lane == 0u) {
            statistics[0] = partial / float(GROUP_ELEMENTS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = statistics[0];

    float variance_sum = 0.0f;
    for (uint base = lid * READS_PER_THREAD;
         base < GROUP_ELEMENTS;
         base += threads_per_threadgroup.x * READS_PER_THREAD) {
        for (uint read = 0u; read < READS_PER_THREAD; ++read) {
            uint linear = base + read;
            if (linear < GROUP_ELEMENTS) {
                uint spatial = linear / GROUP_SIZE;
                uint channel = linear - spatial * GROUP_SIZE + group * GROUP_SIZE;
                float centered =
                    float(inp[input_batch_offset + spatial * CHANNELS + channel]) - mean;
                variance_sum += centered * centered;
            }
        }
    }

    variance_sum = simd_sum(variance_sum);
    if (simd_lane == 0u) {
        local_sums[simd_group] = variance_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float partial = simd_lane < simd_groups ? local_sums[simd_lane] : 0.0f;
        partial = simd_sum(partial);
        if (simd_lane == 0u) {
            statistics[1] = metal::precise::rsqrt(
                partial / float(GROUP_ELEMENTS) + float(eps)
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float normalizer = statistics[1];

    for (uint base = lid * READS_PER_THREAD;
         base < GROUP_ELEMENTS;
         base += threads_per_threadgroup.x * READS_PER_THREAD) {
        for (uint read = 0u; read < READS_PER_THREAD; ++read) {
            uint linear = base + read;
            if (linear < GROUP_ELEMENTS) {
                uint spatial = linear / GROUP_SIZE;
                uint channel = linear - spatial * GROUP_SIZE + group * GROUP_SIZE;
                uint offset = input_batch_offset + spatial * CHANNELS + channel;
                float normalized = (float(inp[offset]) - mean) * normalizer;
                out[offset] = T(
                    normalized * float(weight[channel]) + float(bias[channel])
                );
            }
        }
    }
"""

_RMS_NORM_SOURCE = r"""
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint simd_groups = (threads_per_threadgroup.x + 31u) / 32u;

    threadgroup float local_sums[32];
    threadgroup float inverse_rms;
    uint row_offset = row * FEATURE_DIM;

    float sum_squares = 0.0f;
    for (uint base = lid * READS_PER_THREAD;
         base < FEATURE_DIM;
         base += threads_per_threadgroup.x * READS_PER_THREAD) {
        for (uint read = 0u; read < READS_PER_THREAD; ++read) {
            uint channel = base + read;
            if (channel < FEATURE_DIM) {
                float x = float(inp[row_offset + channel]);
                sum_squares = metal::fma(x, x, sum_squares);
            }
        }
    }

    sum_squares = simd_sum(sum_squares);
    if (simd_lane == 0u) {
        local_sums[simd_group] = sum_squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float partial = simd_lane < simd_groups ? local_sums[simd_lane] : 0.0f;
        partial = simd_sum(partial);
        if (simd_lane == 0u) {
            inverse_rms = metal::precise::rsqrt(
                partial / float(FEATURE_DIM) + float(eps)
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = lid * READS_PER_THREAD;
         base < FEATURE_DIM;
         base += threads_per_threadgroup.x * READS_PER_THREAD) {
        for (uint read = 0u; read < READS_PER_THREAD; ++read) {
            uint channel = base + read;
            if (channel < FEATURE_DIM) {
                float result = float(inp[row_offset + channel]) * inverse_rms;
                result *= float(weight[channel]) + float(weight_offset);
                out[row_offset + channel] = T(result);
            }
        }
    }
"""


def _kernel(
    name: str,
    source: str,
    *,
    input_names: tuple[str, ...] = ("inp",),
) -> MetalKernel:
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name=f"kinomlx_{name}_fp32_opmath",
                input_names=list(input_names),
                output_names=["out"],
                source=source,
                ensure_row_contiguous=True,
                compile_options={"math_mode": "safe"},
            ),
        )
        _KERNELS[name] = kernel
    return kernel


def _metal_elementwise(value: mx.array, *, name: str, source: str) -> mx.array:
    threadgroup_size = min(_THREADGROUP_SIZE, value.size)
    return _kernel(name, source)(
        inputs=[value],
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
        grid=(value.size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        template=[("T", value.dtype)],
    )[0]


def _needs_metal_opmath(value: mx.array) -> bool:
    return value.dtype in _LOW_PRECISION_DTYPES and value.size > 0 and mx.default_device() == mx.gpu


def _fp32_opmath_fallback(
    value: mx.array,
    operation: Callable[[mx.array], mx.array],
) -> mx.array:
    if value.dtype not in _LOW_PRECISION_DTYPES:
        return operation(value)
    dtype = value.dtype
    return operation(value.astype(mx.float32)).astype(dtype)


def _reduction_threadgroup_size(
    elements: int,
    *,
    reads_per_thread: int,
    max_threadgroup_size: int,
) -> int:
    threads = (elements + reads_per_thread - 1) // reads_per_thread
    threads = ((threads + 31) // 32) * 32
    return min(max(threads, 32), max_threadgroup_size)


def _metal_group_norm(value: mx.array, norm: nn.GroupNorm) -> mx.array:
    weight = norm.weight
    bias = norm.bias
    if weight is None or bias is None:
        raise ValueError("Metal GroupNorm requires affine weight and bias parameters")
    batch = value.shape[0]
    channels = value.shape[-1]
    groups = norm.num_groups
    group_size = channels // groups
    spatial_size = value.size // (batch * channels)
    group_elements = spatial_size * group_size
    threadgroup_size = _reduction_threadgroup_size(
        group_elements,
        reads_per_thread=_GROUP_NORM_READS_PER_THREAD,
        max_threadgroup_size=_GROUP_NORM_MAX_THREADGROUP_SIZE,
    )
    kernel = _kernel(
        "group_norm",
        _GROUP_NORM_SOURCE,
        input_names=("inp", "weight", "bias", "eps"),
    )
    return kernel(
        inputs=[value, weight, bias, norm.eps],
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
        grid=(batch * groups * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        template=[
            ("T", value.dtype),
            ("CHANNELS", channels),
            ("NUM_GROUPS", groups),
            ("GROUP_SIZE", group_size),
            ("SPATIAL_SIZE", spatial_size),
            ("GROUP_ELEMENTS", group_elements),
            ("READS_PER_THREAD", _GROUP_NORM_READS_PER_THREAD),
        ],
    )[0]


def _metal_rms_norm(
    value: mx.array,
    weight: mx.array,
    eps: float,
    weight_offset: float,
) -> mx.array:
    feature_dim = value.shape[-1]
    rows = value.size // feature_dim
    threadgroup_size = _reduction_threadgroup_size(
        feature_dim,
        reads_per_thread=_RMS_NORM_READS_PER_THREAD,
        max_threadgroup_size=_RMS_NORM_MAX_THREADGROUP_SIZE,
    )
    kernel = _kernel(
        "rms_norm_weighted",
        _RMS_NORM_SOURCE,
        input_names=("inp", "weight", "eps", "weight_offset"),
    )
    return kernel(
        inputs=[value, weight, eps, weight_offset],
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
        grid=(rows * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        template=[
            ("T", value.dtype),
            ("FEATURE_DIM", feature_dim),
            ("READS_PER_THREAD", _RMS_NORM_READS_PER_THREAD),
        ],
    )[0]


def gelu_approx(value: mx.array) -> mx.array:
    """Apply tanh GELU with FP32 opmath and preserve the input dtype.

    Stock MLX evaluates ``nn.gelu_approx`` at the BF16/FP16 input dtype, which
    introduces intermediate rounding that is materially larger than the
    PyTorch CPU and MPS behavior used by the LTX reference. This one-pass
    kernel reads the low-precision value once, keeps the polynomial and tanh
    in FP32 registers, and rounds once when storing the low-precision result.
    The tanh input is clamped at its saturated range, matching PyTorch's MPS
    low-precision kernel and preventing the fast intrinsic from overflowing
    to NaN for large, otherwise finite Gemma activations.

    FP32 inputs stay on stock MLX because they already have the required
    precision and may fuse with surrounding graph operations. CPU execution
    uses the same FP32-opmath boundary through ordinary MLX operations because
    ``mx.fast.metal_kernel`` is GPU-only.
    """
    if _needs_metal_opmath(value):
        return _metal_elementwise(
            value,
            name="gelu_tanh",
            source=_GELU_APPROX_SOURCE,
        )
    return _fp32_opmath_fallback(value, nn.gelu_approx)


def silu(value: mx.array) -> mx.array:
    """Apply SiLU with FP32 opmath and preserve the input dtype.

    Stock MLX's low-precision ``nn.silu`` rounds the multiply and sigmoid
    expression more aggressively than the PyTorch CPU and MPS realization
    used by the LTX reference. This one-pass kernel widens once at load, keeps
    the exponential and arithmetic in FP32 registers, and rounds once at the
    BF16/FP16 store without materializing an FP32 activation tensor.

    FP32 inputs stay on stock MLX because no precision widening is needed.
    CPU execution uses an explicit FP32-opmath MLX fallback because custom
    Metal kernels cannot execute on the CPU backend.
    """
    if _needs_metal_opmath(value):
        return _metal_elementwise(
            value,
            name="silu",
            source=_SILU_SOURCE,
        )
    return _fp32_opmath_fallback(value, nn.silu)


def group_norm(value: mx.array, norm: nn.GroupNorm) -> mx.array:
    """Apply PyTorch-compatible GroupNorm with FP32 reduction and affine math.

    Stock MLX GroupNorm keeps a low-precision input in low precision, which
    produces materially more error than PyTorch CPU and MPS for the LTX
    spatial upscaler. An explicit FP32 wrapper fixes the arithmetic but its
    PyTorch-compatible reshape transposes every group into a large contiguous
    FP32 temporary. This kernel instead traverses the native channels-last
    tensor directly, computes each group's statistics and affine transform in
    FP32, and writes one low-precision output allocation.

    The custom reduction is used only for affine, PyTorch-compatible
    GroupNorm, the form used by LTX. Non-affine or alternate-layout MLX layers
    retain their own grouping semantics through an explicit FP32 fallback.
    FP32 inputs stay on stock MLX because no widening or temporary avoidance
    is needed, and CPU execution uses the same explicit FP32 fallback because
    Metal kernels are GPU-only.
    """
    if value.dtype not in _LOW_PRECISION_DTYPES:
        return norm(value)
    if (
        value.size > 0
        and value.ndim >= 2
        and mx.default_device() == mx.gpu
        and norm.pytorch_compatible
        and "weight" in norm
        and norm.dims == value.shape[-1]
        and norm.weight is not None
        and norm.bias is not None
        and tuple(norm.weight.shape) == (value.shape[-1],)
        and tuple(norm.bias.shape) == (value.shape[-1],)
    ):
        return _metal_group_norm(value, norm)
    dtype = value.dtype
    return norm(value.astype(mx.float32)).astype(dtype)


def rms_norm(
    value: mx.array,
    weight: mx.array | None = None,
    eps: float = 1e-6,
    *,
    weight_offset: float = 0.0,
) -> mx.array:
    """Apply RMSNorm with the precision policy required by each LTX form.

    Stock MLX and PyTorch MPS realize learned BF16/FP16 RMSNorm with
    low-precision affine math. On captured Gemma, connector, and transformer
    rows that path differs from PyTorch CPU's FP32-opmath result by roughly
    0.2--0.6 percent relative L2. This kernel reduces squares, evaluates the
    reciprocal root, and applies the learned scale in FP32 registers before
    one low-precision store. ``weight_offset`` keeps Gemma's checkpoint
    spelling, ``1 + weight``, inside FP32 instead of rounding the addition at
    the input dtype before normalization.

    Unweighted RMSNorm deliberately stays on stock MLX on the GPU. Actual LTX
    captures either match the PyTorch CPU result after the required output
    cast or land on the same BF16 value with the custom reduction, while that
    custom path costs about 45 percent more at the largest stage-2 shape. CPU
    low-precision execution uses an explicit FP32 fallback; FP32 inputs retain
    stock MLX because no widening is needed.
    """
    if weight is None:
        if value.dtype in _LOW_PRECISION_DTYPES and mx.default_device() == mx.cpu:
            dtype = value.dtype
            return mx.fast.rms_norm(value.astype(mx.float32), None, eps).astype(dtype)
        return mx.fast.rms_norm(value, None, eps).astype(value.dtype)

    valid_weight = (
        value.ndim >= 1 and weight.ndim == 1 and tuple(weight.shape) == (value.shape[-1],)
    )
    if _needs_metal_opmath(value) and valid_weight:
        return _metal_rms_norm(value, weight, eps, weight_offset)

    if value.dtype in _LOW_PRECISION_DTYPES:
        dtype = value.dtype
        effective_weight = weight.astype(mx.float32) + weight_offset
        return mx.fast.rms_norm(
            value.astype(mx.float32),
            effective_weight,
            eps,
        ).astype(dtype)
    return mx.fast.rms_norm(value, weight + weight_offset, eps).astype(value.dtype)


def pixel_norm(
    value: mx.array,
    *,
    axis: int,
    eps: float,
) -> mx.array:
    """Apply the LTX reference's stepwise, parameter-free PixelNorm formula.

    This deliberately does not use stock ``mx.fast.rms_norm``: that fused
    spelling has different low-precision rounding from the reference's
    ``square -> mean -> add epsilon -> sqrt -> divide`` sequence. It also does
    not use a custom reduction kernel because preserving those intermediate
    boundaries matters more than eliminating a cold VAE temporary, and the
    audio and video VAEs normalize different axes and layouts.
    """
    rms = mx.sqrt(mx.mean(value * value, axis=axis, keepdims=True) + eps)
    return value / rms


__all__ = ["gelu_approx", "group_norm", "pixel_norm", "rms_norm", "silu"]
