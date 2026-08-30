"""Resampler parity against direct NumPy oracles of the reference kernels."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.models.gmnet.resample import (
    resize_bicubic,
    resize_bicubic_antialiased,
    resize_bilinear,
)


def _keys_cubic(distance: float, a: float) -> float:
    d = abs(distance)
    if d <= 1.0:
        return (a + 2.0) * d**3 - (a + 3.0) * d**2 + 1.0
    if d < 2.0:
        return a * (d**3 - 5.0 * d**2 + 8.0 * d - 4.0)
    return 0.0


def _oracle_half_pixel_axis(data, out_length, offsets, kernel_a):
    """Direct per-pixel evaluation of the cv2-style half-pixel resampler."""
    in_length = data.shape[0]
    scale = in_length / out_length
    out = np.zeros((out_length, *data.shape[1:]), dtype=np.float64)
    for i in range(out_length):
        position = (i + 0.5) * scale - 0.5
        base = math.floor(position)
        fraction = position - base
        for offset in offsets:
            distance = fraction - offset
            if kernel_a is None:
                weight = max(0.0, 1.0 - abs(distance))
            else:
                weight = _keys_cubic(distance, kernel_a)
            index = min(max(base + offset, 0), in_length - 1)
            out[i] += weight * data[index]
    return out


def _oracle_resize(data, width, height, offsets, kernel_a):
    rows = _oracle_half_pixel_axis(data.astype(np.float64), height, offsets, kernel_a)
    swapped = np.swapaxes(rows, 0, 1)
    columns = _oracle_half_pixel_axis(swapped, width, offsets, kernel_a)
    return np.swapaxes(columns, 0, 1)


@pytest.mark.parametrize(("width", "height"), [(9, 7), (24, 31), (16, 12)])
def test_bicubic_matches_the_cv2_kernel_oracle(width, height):
    rng = np.random.default_rng(3)
    data = rng.uniform(size=(12, 16, 3)).astype(np.float32)
    produced = np.array(resize_bicubic(mx.array(data), width, height))
    expected = _oracle_resize(data, width, height, (-1, 0, 1, 2), -0.75)
    np.testing.assert_allclose(produced, expected, atol=1e-5)


@pytest.mark.parametrize(("width", "height"), [(9, 7), (32, 24), (16, 12)])
def test_bilinear_matches_the_cv2_kernel_oracle(width, height):
    rng = np.random.default_rng(5)
    data = rng.uniform(size=(12, 16, 1)).astype(np.float32)
    produced = np.array(resize_bilinear(mx.array(data), width, height))
    expected = _oracle_resize(data, width, height, (0, 1), None)
    np.testing.assert_allclose(produced, expected, atol=1e-6)


def test_same_size_is_identity():
    rng = np.random.default_rng(9)
    data = rng.uniform(size=(6, 8, 3)).astype(np.float32)
    for function in (resize_bicubic, resize_bilinear, resize_bicubic_antialiased):
        np.testing.assert_array_equal(np.array(function(mx.array(data), 8, 6)), data)


def _oracle_matlab_axis(data, out_length):
    """Direct evaluation of MATLAB-style antialiased bicubic along axis 0."""
    in_length = data.shape[0]
    scale = out_length / in_length
    antialias = scale < 1.0
    kernel_width = 4.0 / scale if antialias else 4.0
    tap_count = math.ceil(kernel_width) + 2
    out = np.zeros((out_length, *data.shape[1:]), dtype=np.float64)
    for i in range(out_length):
        center = (i + 1) / scale + 0.5 * (1.0 - 1.0 / scale)
        left = math.floor(center - kernel_width / 2.0)
        weights = []
        for p in range(tap_count):
            distance = center - (left + p)
            if antialias:
                weights.append(scale * _keys_cubic(distance * scale, -0.5))
            else:
                weights.append(_keys_cubic(distance, -0.5))
        total = sum(weights)
        for p, weight in enumerate(weights):
            index = left + p - 1
            while index < 0 or index >= in_length:
                index = -index - 1 if index < 0 else 2 * in_length - 1 - index
            out[i] += (weight / total) * data[index]
    return out


@pytest.mark.parametrize(("width", "height"), [(5, 4), (32, 24), (11, 3)])
def test_antialiased_bicubic_matches_the_matlab_oracle(width, height):
    rng = np.random.default_rng(17)
    data = rng.uniform(size=(12, 16, 3)).astype(np.float32)
    produced = np.array(resize_bicubic_antialiased(mx.array(data), width, height))
    rows = _oracle_matlab_axis(data.astype(np.float64), height)
    expected = np.swapaxes(_oracle_matlab_axis(np.swapaxes(rows, 0, 1), width), 0, 1)
    np.testing.assert_allclose(produced, expected, atol=1e-5)


def test_antialiased_bicubic_preserves_constants_under_heavy_decimation():
    data = mx.full((97, 203, 3), 0.6127, dtype=mx.float32)
    produced = np.array(resize_bicubic_antialiased(data, 16, 16))
    np.testing.assert_allclose(produced, 0.6127, atol=1e-5)


def test_heavy_decimation_tracks_the_area_mean():
    rng = np.random.default_rng(23)
    data = rng.uniform(size=(128, 128, 3)).astype(np.float32)
    produced = np.array(resize_bicubic_antialiased(mx.array(data), 8, 8))
    assert abs(float(produced.mean()) - float(data.mean())) < 0.01
