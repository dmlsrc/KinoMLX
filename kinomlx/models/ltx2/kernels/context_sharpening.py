"""A per-frame spatial unsharp mask over a channels-last video tensor, in one pass.

``value + amount * (value - blur(value))`` for a ``(batch, frames, height, width, channels)``
tensor, where ``blur`` is a Gaussian over height and width only, frame by frame and channel by
channel, with replicated edges. The LTX-2.5 diffusion VAE uses it to sharpen its final-stage
context (see ``video_vae.sharpening``).

The kernel gives each thread four adjacent channels of one position and sums their frame's
``(2 r + 1) ** 2`` neighborhood in FP32, rows first, with the edges replicated by index
clamping; it reads the input and writes the output once and rounds once at the store. Four
channels per thread took 52 ms against 125 ms for one on an M1 Max, for a (1, 121, 144, 256,
256) FP16 tensor at sigma 0.5 (92 against 202 ms at sigma 1), with identical results. The MLX
fallback, used on the CPU or when the channels are not a multiple of four, computes the same
separable Gaussian with edge padding in FP32, columns first, so the two differ by FP32
summation order only.
"""

import math
from functools import cache
from typing import cast

import mlx.core as mx

from kinomlx.kernels._typing import MetalKernel

_THREADS = 256

_SOURCE = r"""
    const uint quad = thread_position_in_grid.x;
    const uint h = thread_position_in_grid.y;
    const uint n = thread_position_in_grid.z;
    const int height = dims[0];
    const int width = dims[1];
    const int channels = dims[2];
    if (int(quad) * 4 >= width * channels) {
        return;
    }
    const int wc = int(quad) * 4;
    const int w = wc / channels;
    const int c = wc - w * channels;
    const ulong row_stride = (ulong)width * (ulong)channels;
    const ulong frame = (ulong)n * (ulong)height * row_stride;
    float4 blurred = float4(0.0f);
    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        const int y = clamp(int(h) + dy, 0, height - 1);
        const ulong row = frame + (ulong)y * row_stride + (ulong)c;
        float4 acc = float4(0.0f);
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            const int x = clamp(w + dx, 0, width - 1);
            const ulong at = row + (ulong)x * (ulong)channels;
            const float4 v = float4(
                float(inp[at]), float(inp[at + 1]), float(inp[at + 2]), float(inp[at + 3]));
            acc = fma(v, float4(taps[dx + RADIUS]), acc);
        }
        blurred = fma(acc, float4(taps[dy + RADIUS]), blurred);
    }
    const ulong index = frame + (ulong)h * row_stride + (ulong)wc;
    const float4 center = float4(
        float(inp[index]), float(inp[index + 1]), float(inp[index + 2]), float(inp[index + 3]));
    const float4 result = center + amount[0] * (center - blurred);
    out[index] = T(result.x);
    out[index + 1] = T(result.y);
    out[index + 2] = T(result.z);
    out[index + 3] = T(result.w);
"""


def gaussian_taps(sigma: float) -> tuple[float, ...]:
    """Normalized Gaussian taps of standard deviation ``sigma``, reaching ``ceil(3 sigma)``."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite number greater than 0")
    radius = max(1, math.ceil(3 * sigma))
    weights = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-radius, radius + 1)]
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


@cache
def _kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_spatial_unsharp_mask",
            input_names=["inp", "taps", "amount", "dims"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )


def _applies(value: mx.array) -> bool:
    return (
        mx.default_device() == mx.gpu
        and value.dtype in (mx.float16, mx.bfloat16, mx.float32)
        and int(value.shape[4]) % 4 == 0
        and int(value.shape[3]) * int(value.shape[4]) < 1 << 31
        and int(value.shape[0]) * int(value.shape[1]) < 1 << 31
    )


def _reference(value: mx.array, taps: tuple[float, ...], amount: float) -> mx.array:
    radius = (len(taps) - 1) // 2
    source = value.astype(mx.float32)
    blurred = source
    for axis in (2, 3):
        length = int(blurred.shape[axis])
        pad = [(0, 0)] * 5
        pad[axis] = (radius, radius)
        padded = mx.pad(blurred, pad, mode="edge")
        acc = mx.zeros_like(blurred)
        for index, tap in enumerate(taps):
            window = [slice(None)] * 5
            window[axis] = slice(index, index + length)
            acc = acc + tap * padded[tuple(window)]
        blurred = acc
    return (source + amount * (source - blurred)).astype(value.dtype)


def spatial_unsharp_mask(value: mx.array, *, sigma: float, amount: float) -> mx.array:
    """``value + amount * (value - blur(value))`` with a per-frame spatial Gaussian blur.

    ``value`` is ``(batch, frames, height, width, channels)``; ``sigma`` is in its grid cells.
    The result keeps ``value``'s dtype.
    """
    if value.ndim != 5:
        raise ValueError("spatial unsharp masking needs a (batch, frames, height, width, channels)")
    if not math.isfinite(amount):
        raise ValueError("amount must be finite")
    taps = gaussian_taps(sigma)
    if value.size == 0:
        return value
    if not _applies(value):
        return _reference(value, taps, amount)
    batch, frames, height, width, channels = (int(size) for size in value.shape)
    quads = width * channels // 4
    return _kernel()(
        inputs=[
            value,
            mx.array(taps, dtype=mx.float32),
            mx.array([amount], dtype=mx.float32),
            mx.array([height, width, channels], dtype=mx.int32),
        ],
        output_shapes=[value.shape],
        output_dtypes=[value.dtype],
        grid=(-(-quads // _THREADS) * _THREADS, height, batch * frames),
        threadgroup=(_THREADS, 1, 1),
        template=[("T", value.dtype), ("RADIUS", (len(taps) - 1) // 2)],
    )[0]


__all__ = ["gaussian_taps", "spatial_unsharp_mask"]
