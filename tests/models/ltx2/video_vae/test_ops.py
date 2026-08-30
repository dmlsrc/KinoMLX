"""Patch packing and latent-statistics tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.video_vae.blocks import (
    patchify_spatial_bfhwc,
    to_native_conv3d_layout,
    unpatchify_spatial_bfhwc,
)
from kinomlx.models.ltx2.video_vae.ops import (
    PerChannelStatistics,
    _compiled_patchify,
    _compiled_unpatchify,
    patchify,
    unpatchify,
)


@pytest.mark.parametrize(
    ("shape", "patch_size_hw", "patch_size_t"),
    [
        ((1, 2, 4, 6), 2, 1),
        ((1, 2, 4, 4, 6), 2, 2),
        ((1, 2, 3, 4, 6), 2, 1),
    ],
)
def test_patchify_unpatchify_round_trip(
    shape: tuple[int, ...],
    patch_size_hw: int,
    patch_size_t: int,
) -> None:
    count = 1
    for dimension in shape:
        count *= dimension
    source = mx.arange(count).reshape(shape)
    restored = unpatchify(
        patchify(source, patch_size_hw, patch_size_t),
        patch_size_hw,
        patch_size_t,
    )
    mx.eval(restored)
    assert tuple(restored.shape) == shape
    assert mx.array_equal(restored, source).item()


def test_patchify_matches_upstream_channel_order() -> None:
    source = mx.array([[[[0, 1], [2, 3]]]], dtype=mx.float32)
    packed = patchify(source, patch_size_hw=2)
    mx.eval(packed)
    assert packed.reshape(-1).tolist() == [0.0, 2.0, 1.0, 3.0]


def test_patch_compilation_is_lazy_and_cached() -> None:
    _compiled_patchify.cache_clear()
    _compiled_unpatchify.cache_clear()
    assert _compiled_patchify.cache_info().currsize == 0
    assert _compiled_unpatchify.cache_info().currsize == 0
    source = mx.zeros((1, 1, 2, 2))
    packed = patchify(source, 2)
    unpatchify(packed, 2)
    assert _compiled_patchify.cache_info().currsize == 1
    assert _compiled_unpatchify.cache_info().currsize == 1


def test_bfhwc_spatial_patch_round_trip() -> None:
    source = mx.arange(1 * 2 * 4 * 6 * 3).reshape(1, 2, 4, 6, 3)
    restored = unpatchify_spatial_bfhwc(
        patchify_spatial_bfhwc(source, patch_size=2),
        patch_size=2,
    )
    mx.eval(restored)
    assert mx.array_equal(restored, source).item()


def test_per_channel_statistics_default_to_identity() -> None:
    statistics = PerChannelStatistics(latent_channels=2)
    source = mx.arange(8, dtype=mx.float32).reshape(1, 2, 1, 2, 2)
    assert mx.array_equal(statistics.normalize(source), source).item()
    assert mx.array_equal(statistics.denormalize(source), source).item()
    assert mx.array_equal(statistics.un_normalize(source), source).item()


def test_per_channel_statistics_round_trip_checkpoint_affine() -> None:
    statistics = PerChannelStatistics(latent_channels=2)
    statistics.mean_of_means = mx.array([2.0, -3.0])
    statistics.std_of_means = mx.array([0.5, 4.0])
    source = mx.arange(8, dtype=mx.float32).reshape(1, 2, 1, 2, 2)
    restored = statistics.denormalize(statistics.normalize(source))
    assert mx.allclose(restored, source).item()


def test_conv3d_layout_accepts_native_and_converts_pytorch() -> None:
    native = mx.zeros((3, 2, 2, 2, 4))
    assert to_native_conv3d_layout(native, tuple(native.shape)) is native
    pytorch = mx.zeros((3, 4, 2, 2, 2))
    converted = to_native_conv3d_layout(pytorch, tuple(native.shape))
    assert tuple(converted.shape) == tuple(native.shape)
