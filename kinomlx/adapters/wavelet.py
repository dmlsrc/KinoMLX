"""One-level WaveFT synthesis using MLX transposed convolution."""

import math

import mlx.core as mx

from .formats import AdapterOperation
from .permutation import torch_cpu_permutation_prefix
from .wavelet_filters import LOWPASS


def wavelet_delta(operation: AdapterOperation, base: mx.array) -> mx.array:
    options = operation.linear_options
    if options is None or base.ndim != 2 or operation.part != "weight":
        raise ValueError("WaveFT requires a configured Linear weight target")
    count = options.rank
    spectrum = operation.factors["spectrum"]
    if count > base.size or spectrum.shape != (count,):
        raise ValueError("WaveFT frequency count does not fit its Linear target")
    positions = mx.array(torch_cpu_permutation_prefix(base.size, count, options.seed))
    out, inp = base.shape
    if not options.variant:
        return (
            mx.zeros((base.size,), dtype=mx.float32)
            .at[positions]
            .add(spectrum.astype(mx.float32))
            .reshape(base.shape)
            * options.scale
        )
    coefficients = LOWPASS[options.variant]
    length = len(coefficients)
    reduction = length - 2
    height = (out + reduction + 1) // 2 * 2
    width = (inp + reduction + 1) // 2 * 2
    rows = positions // inp + (height - out) // 2
    columns = positions % inp + (width - inp) // 2
    packed = (
        mx.zeros((height, width), dtype=mx.float32)
        .at[rows, columns]
        .add(spectrum.astype(mx.float32))
    )
    h, w = height // 2, width // 2
    bands = mx.stack([packed[:h, :w], packed[:h, w:], packed[h:, :w], packed[h:, w:]], axis=-1)
    low = mx.array(coefficients, dtype=mx.float32)
    if options.variant.startswith("coif"):
        low = mx.array([value * math.sqrt(2) for value in coefficients], dtype=mx.float32)
    high = low[::-1] * mx.where(mx.arange(length) % 2, -1.0, 1.0)
    # Four separable synthesis filters: approximation, horizontal detail,
    # vertical detail, diagonal detail. Cropping removes filter support, not
    # model tokens or learned coefficients.
    kernels = mx.stack(
        [low[:, None] * low, high[:, None] * low, low[:, None] * high, high[:, None] * high],
        axis=-1,
    )
    image = mx.conv_transpose2d(bands[None], kernels[None], stride=2, padding=reduction)[0, :, :, 0]
    start_h, start_w = (image.shape[0] - out) // 2, (image.shape[1] - inp) // 2
    return image[start_h : start_h + out, start_w : start_w + inp] * options.scale
