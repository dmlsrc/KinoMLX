from __future__ import annotations

import mlx.core as mx

from kinomlx.media.signals import ColorPrimaries
from kinomlx.videotoolbox.hlg import prepare_hlg_scene_linear, scene_linear_to_hlg_codes


def test_native_hlg_matches_frozen_apple_transfer_vectors() -> None:
    linear = mx.array(
        [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 12.0],
        dtype=mx.float32,
    )
    frame = mx.stack((linear, linear, linear), axis=-1)[None, :, :]
    converted = scene_linear_to_hlg_codes(frame, primaries=ColorPrimaries.REC709)
    expected = mx.array(
        [
            0.0,
            0.1308229715,
            0.2558123469,
            0.3414685130,
            0.5002175570,
            0.6322637200,
            0.7498773336,
            0.8607344627,
            0.9681388736,
            1.0,
        ],
        dtype=mx.float32,
    )
    assert mx.allclose(converted[0, :, 0], expected, rtol=0.0, atol=2e-5)
    assert mx.allclose(converted[0, :, 1], expected, rtol=0.0, atol=2e-5)
    assert mx.allclose(converted[0, :, 2], expected, rtol=0.0, atol=2e-5)


def test_acescg_hlg_delivery_preserves_wide_gamut_until_bt2020_conversion() -> None:
    acescg = mx.array(
        [[[20.0, 0.5, 0.25], [0.2, 4.0, 1.0]]],
        dtype=mx.float32,
    )
    prepared = prepare_hlg_scene_linear(acescg, primaries=ColorPrimaries.ACESCG)
    direct = scene_linear_to_hlg_codes(acescg, primaries=ColorPrimaries.ACESCG)

    assert mx.array_equal(prepared, acescg)
    assert tuple(direct.shape) == tuple(acescg.shape)
    assert bool(mx.all(mx.isfinite(direct)).item())
