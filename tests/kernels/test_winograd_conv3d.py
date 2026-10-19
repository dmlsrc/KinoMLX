"""Parity and weight lifetime checks for the LTX VAE Winograd convolution."""

import mlx.core as mx
import pytest

from kinomlx.kernels import mpp_linear
from kinomlx.kernels import winograd_conv3d as vae_winograd
from kinomlx.kernels.winograd_conv3d import (
    WinogradConv3d,
    _fused_filter_transform,
    _weight_transform,
)


def _native(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    padded = mx.concatenate([x[:, :1], x, x[:, -1:]], axis=1)
    return mx.conv3d(padded, weight, padding=(0, 1, 1)) + bias


def test_fused_filter_transform_matches_fp32_matmuls() -> None:
    weight = mx.random.normal((32, 3, 3, 3, 32), key=mx.random.key(413), dtype=mx.float16)
    layout = weight.transpose(0, 4, 1, 2, 3)
    taps = mx.contiguous(
        mx.concatenate([layout[:, :, index] for index in range(3)], axis=0).transpose(0, 2, 3, 1)
    )
    reference = _weight_transform(taps).astype(mx.float16)
    fused = _fused_filter_transform(weight)
    mx.eval(reference, fused)
    for index in range(3):
        assert mx.array_equal(fused[index], reference[:, :, index * 32 : (index + 1) * 32]).item()


@pytest.mark.parametrize("through_mpp", [True, False])
def test_winograd_rectangular_channel_projection(
    monkeypatch: pytest.MonkeyPatch, through_mpp: bool
) -> None:
    monkeypatch.setattr(mpp_linear, "_enabled", through_mpp)
    x = mx.random.normal((1, 3, 12, 14, 16), key=mx.random.key(418), dtype=mx.float16)
    weight = mx.random.normal((32, 3, 3, 3, 16), key=mx.random.key(419), dtype=mx.float16)
    weight = weight * 0.02
    bias = mx.arange(32, dtype=mx.float16) * 0.001
    actual = WinogradConv3d()(x, weight, bias)
    monkeypatch.setattr(vae_winograd, "_BAND_BUDGET_BYTES", 100_000)
    slabbed = WinogradConv3d(transient_output_block=16)(x, weight, bias)
    expected = _native(x, weight, bias)
    mx.eval(actual, slabbed, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, slabbed).item()
    error = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    assert mx.mean(error).item() < 0.005
    assert mx.max(error).item() < 0.04


def test_winograd_matches_native_conv3d_and_reloads_filters() -> None:
    x = mx.random.normal((1, 3, 12, 12, 32), key=mx.random.key(411), dtype=mx.float16)
    weight = mx.random.normal((32, 3, 3, 3, 32), key=mx.random.key(412), dtype=mx.float16)
    weight = weight * 0.02
    bias = mx.arange(32, dtype=mx.float16) * 0.001
    kernel = WinogradConv3d()

    first = kernel(x, weight, bias)
    first_reference = _native(x, weight, bias)
    replacement = weight * 0.5
    second = kernel(x, replacement, bias)
    second_reference = _native(x, replacement, bias)
    mx.eval(first, first_reference, second, second_reference)

    assert first.shape == first_reference.shape == (1, 3, 12, 12, 32)
    assert mx.all(mx.isfinite(first)).item()
    assert mx.all(mx.isfinite(second)).item()
    for actual, expected in ((first, first_reference), (second, second_reference)):
        error = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
        assert mx.mean(error).item() < 0.005
        assert mx.max(error).item() < 0.04


@pytest.mark.parametrize("through_mpp", [True, False])
def test_evaluated_bands_match_single_band(
    monkeypatch: pytest.MonkeyPatch, through_mpp: bool
) -> None:
    monkeypatch.setattr(mpp_linear, "_enabled", through_mpp)
    x = mx.random.normal((1, 3, 12, 14, 16), key=mx.random.key(420), dtype=mx.float16)
    weight = mx.random.normal((32, 3, 3, 3, 16), key=mx.random.key(421), dtype=mx.float16)
    bias = mx.zeros((32,), dtype=mx.float16)
    single = WinogradConv3d()(x, weight, bias)

    monkeypatch.setattr(vae_winograd, "_BAND_BUDGET_BYTES", 100_000)
    banded = WinogradConv3d()(x, weight, bias)
    mx.eval(single, banded)

    assert mx.array_equal(single, banded).item()


def test_mpp_tap_sum_is_the_fp32_product_of_the_stacked_taps() -> None:
    if not vae_winograd._mpp_tap_sum_available():
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")
    planes, frames, tiles, channels, output = 3, 6, 11, 48, 40
    keys = mx.random.split(mx.random.key(424), 2)
    transformed = mx.random.normal(
        (planes, frames * tiles, channels), key=keys[0], dtype=mx.float16
    )
    stacked = mx.random.normal((planes, 3 * channels, output), key=keys[1], dtype=mx.float16)
    filters = tuple(
        mx.contiguous(stacked[:, index * channels : (index + 1) * channels]) for index in range(3)
    )
    rows = (frames - 2) * tiles
    windows = mx.concatenate(
        [transformed[:, index * tiles : index * tiles + rows] for index in range(3)], axis=-1
    )
    product = vae_winograd._mpp_tap_sum(transformed, filters, tiles, frames)
    expected = mx.matmul(windows.astype(mx.float32), stacked.astype(mx.float32))
    mx.eval(product, expected)
    assert product.dtype == mx.float32
    assert mx.array_equal(product, expected).item()


def test_mpp_tap_sum_is_closer_to_fp32_than_three_fp16_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vae_winograd._mpp_tap_sum_available():
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")
    x = mx.random.normal((1, 3, 12, 14, 16), key=mx.random.key(422), dtype=mx.float16)
    weight = mx.random.normal((32, 3, 3, 3, 16), key=mx.random.key(423), dtype=mx.float16)
    weight = weight * 0.02
    bias = mx.arange(32, dtype=mx.float16) * 0.001
    padded = mx.concatenate([x[:, :1], x, x[:, -1:]], axis=1).astype(mx.float32)
    reference = mx.conv3d(padded, weight.astype(mx.float32), padding=(0, 1, 1))
    reference = reference + bias.astype(mx.float32)
    monkeypatch.setattr(vae_winograd, "_BAND_BUDGET_BYTES", 100_000)
    errors = {}
    for enabled in (True, False):
        monkeypatch.setattr(mpp_linear, "_enabled", enabled)
        for block in (None, 16):
            result = WinogradConv3d(transient_output_block=block)(x, weight, bias)
            error = mx.sqrt(mx.mean(mx.square(result.astype(mx.float32) - reference)))
            errors[enabled, block] = error.item()
    for block in (None, 16):
        assert errors[True, block] < errors[False, block]
