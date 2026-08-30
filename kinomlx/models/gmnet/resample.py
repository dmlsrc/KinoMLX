"""Reference-matched MLX resamplers for GMNet pre/post-processing.

GMNet's shipped checkpoints were trained and evaluated against three
specific resamplers, and matching their kernels matters more than kernel
elegance:

- the local-branch input is downscaled with OpenCV ``INTER_CUBIC``
  (Keys cubic, a = -0.75, half-pixel mapping, replicated border, no
  antialias prefilter);
- the predicted gain map is upscaled back with OpenCV ``INTER_LINEAR``
  (tent, half-pixel mapping, replicated border);
- the 256x256 global-branch thumbnail follows the upstream project's own
  ``imresize`` (MATLAB-style Keys cubic, a = -0.5, antialiased by kernel
  stretching on downscale, symmetric border, normalized weights).

Everything here is separable: each axis is either a small fixed tap
gather or one dense weight-matrix matmul. Pure MLX, float32.
"""

from __future__ import annotations

import math

import mlx.core as mx

_CV2_CUBIC_A = -0.75
_MATLAB_CUBIC_A = -0.5
_MATLAB_SUPPORT = 4.0


def _keys_cubic(distance: float, a: float) -> float:
    """The Keys cubic convolution kernel with free parameter ``a``."""
    d = abs(distance)
    if d <= 1.0:
        return (a + 2.0) * d**3 - (a + 3.0) * d**2 + 1.0
    if d < 2.0:
        return a * (d**3 - 5.0 * d**2 + 8.0 * d - 4.0)
    return 0.0


def _half_pixel_taps(
    in_length: int,
    out_length: int,
    offsets: tuple[int, ...],
    kernel_a: float | None,
) -> list[tuple[mx.array, mx.array]]:
    """Clamped tap indices and weights for the OpenCV half-pixel mapping.

    ``kernel_a`` selects the Keys cubic; ``None`` selects the linear tent.
    Border handling replicates edge samples without reweighting, exactly as
    ``cv2.resize`` does.
    """
    scale = in_length / out_length
    taps: list[tuple[mx.array, mx.array]] = []
    indices: list[list[int]] = [[] for _ in offsets]
    weights: list[list[float]] = [[] for _ in offsets]
    for i in range(out_length):
        position = (i + 0.5) * scale - 0.5
        base = math.floor(position)
        fraction = position - base
        for slot, offset in enumerate(offsets):
            distance = fraction - offset
            if kernel_a is None:
                weight = max(0.0, 1.0 - abs(distance))
            else:
                weight = _keys_cubic(distance, kernel_a)
            indices[slot].append(min(max(base + offset, 0), in_length - 1))
            weights[slot].append(weight)
    for slot in range(len(offsets)):
        taps.append(
            (
                mx.array(indices[slot], dtype=mx.int32),
                mx.array(weights[slot], dtype=mx.float32),
            )
        )
    return taps


def _apply_taps(
    image: mx.array,
    taps: list[tuple[mx.array, mx.array]],
    axis: int,
) -> mx.array:
    shape = [1] * image.ndim
    shape[axis] = -1
    total: mx.array | None = None
    for indices, weights in taps:
        part = mx.take(image, indices, axis=axis) * weights.reshape(shape)
        total = part if total is None else total + part
    if total is None:
        raise ValueError("resampling requires at least one tap")
    return total


def _resize_half_pixel(
    image: mx.array,
    width: int,
    height: int,
    offsets: tuple[int, ...],
    kernel_a: float | None,
) -> mx.array:
    in_height, in_width = int(image.shape[0]), int(image.shape[1])
    result = image if image.dtype == mx.float32 else image.astype(mx.float32)
    if height != in_height:
        result = _apply_taps(result, _half_pixel_taps(in_height, height, offsets, kernel_a), 0)
    if width != in_width:
        result = _apply_taps(result, _half_pixel_taps(in_width, width, offsets, kernel_a), 1)
    return result


def resize_bicubic(image: mx.array, width: int, height: int) -> mx.array:
    """Resize HWC float pixels like ``cv2.resize(..., INTER_CUBIC)``."""
    return _resize_half_pixel(image, width, height, (-1, 0, 1, 2), _CV2_CUBIC_A)


def resize_bilinear(image: mx.array, width: int, height: int) -> mx.array:
    """Resize HWC float pixels like ``cv2.resize(..., INTER_LINEAR)``."""
    return _resize_half_pixel(image, width, height, (0, 1), None)


def _matlab_axis_matrix(in_length: int, out_length: int) -> mx.array:
    """One dense MATLAB-imresize row-weight matrix of shape (out, in).

    Antialiasing stretches the cubic by the scale factor on downscale;
    out-of-range taps reflect symmetrically (edge sample included) and every
    row is normalized to unit weight.
    """
    scale = out_length / in_length
    antialias = scale < 1.0
    kernel_width = _MATLAB_SUPPORT / scale if antialias else _MATLAB_SUPPORT
    tap_count = math.ceil(kernel_width) + 2

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for i in range(out_length):
        # MATLAB 1-based center mapping: u = (i + 1) / scale + 0.5 * (1 - 1 / scale).
        center = (i + 1) / scale + 0.5 * (1.0 - 1.0 / scale)
        left = math.floor(center - kernel_width / 2.0)
        weights = []
        for p in range(tap_count):
            distance = center - (left + p)
            if antialias:
                weights.append(scale * _keys_cubic(distance * scale, _MATLAB_CUBIC_A))
            else:
                weights.append(_keys_cubic(distance, _MATLAB_CUBIC_A))
        total = sum(weights)
        for p, weight in enumerate(weights):
            if weight == 0.0:
                continue
            index = left + p - 1  # to 0-based
            while index < 0 or index >= in_length:
                index = -index - 1 if index < 0 else 2 * in_length - 1 - index
            rows.append(i)
            columns.append(index)
            values.append(weight / total)

    matrix = mx.zeros((out_length, in_length), dtype=mx.float32)
    return matrix.at[
        mx.array(rows, dtype=mx.int32),
        mx.array(columns, dtype=mx.int32),
    ].add(mx.array(values, dtype=mx.float32))


def resize_bicubic_antialiased(image: mx.array, width: int, height: int) -> mx.array:
    """Resize HWC float pixels with MATLAB-style antialiased bicubic.

    This matches the upstream GMNet project's own ``imresize`` and is used
    for the 256x256 global-branch thumbnail, where the input may be
    decimated far below the plain 4-tap kernel's antialiasing limit.
    """
    in_height, in_width, channels = (
        int(image.shape[0]),
        int(image.shape[1]),
        int(image.shape[2]),
    )
    result = image if image.dtype == mx.float32 else image.astype(mx.float32)
    if height != in_height:
        matrix = _matlab_axis_matrix(in_height, height)
        result = (matrix @ result.reshape(in_height, in_width * channels)).reshape(
            height, in_width, channels
        )
    if width != in_width:
        matrix = _matlab_axis_matrix(in_width, width)
        swapped = result.transpose(1, 0, 2).reshape(in_width, height * channels)
        result = (matrix @ swapped).reshape(width, height, channels).transpose(1, 0, 2)
    return result


__all__ = [
    "resize_bicubic",
    "resize_bicubic_antialiased",
    "resize_bilinear",
]
