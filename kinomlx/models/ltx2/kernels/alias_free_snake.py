"""The vocoder's alias-free SnakeBeta activation in one pass.

BigVGAN's anti-aliased activation upsamples a channels-first FP32 signal ``(batch, channels,
length)`` by two (replicate padding of 5, a 12-tap transposed convolution, a factor of two and a
crop of 15 on each side), applies SnakeBeta ``u + sin(a u) ** 2 / (b + 1e-9)`` with per-channel
``a = exp(alpha)`` and ``b = exp(beta)``, and low-pass filters back down (replicate padding of 5
and 6 and a 12-tap convolution at stride two). MLX runs both filters as single-channel
convolutions with one batch row per channel, which on an M1 Max took 53 and 19 ms of an 81 ms
layer at 16 channels of 1.44 million samples.

Here one threadgroup computes the SnakeBeta values its outputs need into threadgroup memory, each
from the six upsampler taps that reach it, and each thread then takes the 12-tap low-pass of its
outputs; replicate padding becomes index clamping. SnakeBeta uses MLX's own functions
(``precise::sin``, ``pow``) in MLX's order. The filter sums run in tap order, where MLX's
convolutions sum in its matrix units, so outputs differ by FP32 rounding: at most 4.8e-7, on
values up to 4.7, over every activation shape of a 30-second LTX-2.3 vocoder run, whose BF16
waveform kept 99.4% of its samples (77 dB SNR against MLX's). That vocoder took 2.2 s against
12.7 s.
"""

from functools import cache
from typing import cast

import mlx.core as mx

from kinomlx.kernels._typing import MetalKernel

# The activation's geometry, which the kernel's indexing assumes.
RATIO = 2
TAPS = 12
UP_PAD = 5
UP_CROP = 15
DOWN_PADS = (5, 6)
_THREADS = 256
_OUTPUTS = 512
# SnakeBeta values one threadgroup's outputs read: 2 * outputs plus the low-pass's reach.
_WINDOW = 2 * _OUTPUTS + TAPS - 2

_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint segment = threadgroup_position_in_grid.x;
    const uint c = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const int channels = dims[0];
    const int length = dims[1];
    const ulong row = ((ulong)b * (ulong)channels + c) * (ulong)length;
    const int n0 = int(segment) * OUTPUTS;
    const float a = alpha_exp[c];
    const float d = beta_eps[c];
    threadgroup float window[WINDOW];

    // SnakeBeta at upsampled positions 2 n0 - 5 onward, clamped: the low-pass's replicate pad.
    for (int w = int(tid); w < WINDOW; w += THREADS) {
        const int position = clamp(2 * n0 - DOWN_PAD + w, 0, 2 * length - 1);
        // Twice the transposed convolution's output at position + crop: the taps of its parity,
        // each reading the replicate-padded input.
        const int m = position + UP_CROP;
        float acc = 0.0f;
        for (int k = m & 1; k < TAPS; k += 2) {
            const int source = clamp((m - k) / 2 - UP_PAD, 0, length - 1);
            acc = fma(x[row + (ulong)source], up[k], acc);
        }
        const float u = 2.0f * acc;
        // MLX's SnakeBeta, operation for operation.
        const float scaled = a * u;
        const float sine = metal::precise::sin(scaled);
        const float squared = metal::pow(sine, 2.0f);
        const float ratio = squared / d;
        window[w] = u + ratio;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = int(tid); r < OUTPUTS; r += THREADS) {
        const int n = n0 + r;
        if (n >= length) break;
        float acc = 0.0f;
        for (int k = 0; k < TAPS; ++k) {
            acc = fma(window[2 * r + k], down[k], acc);
        }
        out[row + (ulong)n] = acc;
    }
"""

_enabled = True


def set_enabled(enabled: bool) -> None:
    """Run the activation through the kernel wherever it applies (the default), or keep MLX's."""
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    """Whether the activation kernel is switched on."""
    return _enabled


@cache
def _kernel() -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="ltx_alias_free_snake",
            input_names=["x", "up", "down", "alpha_exp", "beta_eps", "dims"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )


def applies(x: mx.array, up: mx.array, down: mx.array) -> bool:
    """Whether channels-first FP32 ``x`` with these 12-tap FP32 filters takes the kernel."""
    return (
        _enabled
        and mx.default_device() == mx.gpu
        and x.ndim == 3
        and x.dtype == mx.float32
        and min(x.shape) > 0
        and int(x.shape[2]) < 1 << 30
        and up.size == TAPS
        and down.size == TAPS
        and up.dtype == mx.float32
        and down.dtype == mx.float32
    )


def activate(
    x: mx.array, up: mx.array, down: mx.array, alpha: mx.array, beta: mx.array
) -> mx.array:
    """Upsample by two, SnakeBeta with log-parameters ``alpha`` and ``beta``, downsample by two."""
    if not applies(x, up, down):
        raise ValueError("the alias-free SnakeBeta kernel does not apply to this input")
    batch, channels, length = (int(size) for size in x.shape)
    # As SnakeBeta computes them, so the kernel's SnakeBeta sees MLX's values.
    alpha_exp = mx.exp(alpha)
    beta_eps = mx.exp(beta) + 1e-9
    return _kernel()(
        inputs=[
            x,
            up.reshape(-1),
            down.reshape(-1),
            alpha_exp,
            beta_eps,
            mx.array([channels, length], dtype=mx.int32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
        grid=(-(-length // _OUTPUTS) * _THREADS, channels, batch),
        threadgroup=(_THREADS, 1, 1),
        template=[
            ("OUTPUTS", _OUTPUTS),
            ("WINDOW", _WINDOW),
            ("THREADS", _THREADS),
            ("TAPS", TAPS),
            ("UP_PAD", UP_PAD),
            ("UP_CROP", UP_CROP),
            ("DOWN_PAD", DOWN_PADS[0]),
        ],
    )[0]


__all__ = [
    "DOWN_PADS",
    "RATIO",
    "TAPS",
    "UP_CROP",
    "UP_PAD",
    "activate",
    "applies",
    "enabled",
    "set_enabled",
]
