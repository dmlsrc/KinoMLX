"""Behavioral tests for ``kinomlx.io.image``."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from kinomlx.io.image import _resize_bilinear, load_image, save_image

# ---------------------------------------------------------------------------
# Basic shape / dtype / range contract
# ---------------------------------------------------------------------------


def test_load_image_returns_hwc_float32_in_unit_range(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.new("RGB", (64, 48), color=(128, 200, 50)).save(src)
    arr = load_image(src)
    assert arr.shape == (48, 64, 3)
    assert arr.dtype == mx.float32
    # 128/255, 200/255, 50/255 - all in [0, 1].
    np_arr = np.asarray(arr)
    assert np_arr.min() >= 0.0
    assert np_arr.max() <= 1.0


# ---------------------------------------------------------------------------
# Mode normalization - RGBA / grayscale -> RGB
# ---------------------------------------------------------------------------


def test_load_image_strips_alpha_from_rgba(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.new("RGBA", (16, 16), color=(255, 0, 0, 128)).save(src)
    arr = load_image(src)
    assert arr.shape == (16, 16, 3)


def test_load_image_expands_grayscale_to_rgb(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.new("L", (16, 16), color=128).save(src)
    arr = load_image(src)
    assert arr.shape == (16, 16, 3)
    np_arr = np.asarray(arr)
    # Grayscale -> RGB replicates the single channel.
    assert np.allclose(np_arr[..., 0], np_arr[..., 1])
    assert np.allclose(np_arr[..., 1], np_arr[..., 2])


# ---------------------------------------------------------------------------
# Resize + cover-crop
# ---------------------------------------------------------------------------


def test_load_with_size_matches_target_dims(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.new("RGB", (320, 240), color=(0, 128, 255)).save(src)
    arr = load_image(src, size=(160, 120))
    assert arr.shape == (120, 160, 3)


def test_cover_crop_handles_wider_source(tmp_path: Path) -> None:
    """Source 200x100, target 100x100 - height fits exactly, width is cropped."""
    src = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(src)
    arr = load_image(src, size=(100, 100))
    assert arr.shape == (100, 100, 3)


def test_cover_crop_handles_taller_source(tmp_path: Path) -> None:
    """Source 100x200, target 100x100 - width fits exactly, height is cropped."""
    src = tmp_path / "tall.png"
    Image.new("RGB", (100, 200), color=(255, 255, 255)).save(src)
    arr = load_image(src, size=(100, 100))
    assert arr.shape == (100, 100, 3)


def test_bilinear_resize_matches_align_corners_false_grid() -> None:
    values = mx.array([[0.0, 1.0], [2.0, 3.0]])
    source = mx.broadcast_to(values[:, :, None], (2, 2, 3))
    resized = _resize_bilinear(source, 4, 4)
    expected = np.array(
        [
            [0.0, 0.25, 0.75, 1.0],
            [0.5, 0.75, 1.25, 1.5],
            [1.5, 1.75, 2.25, 2.5],
            [2.0, 2.25, 2.75, 3.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(np.asarray(resized[:, :, 0]), expected)


# ---------------------------------------------------------------------------
# Round-trip - load -> save -> load
# ---------------------------------------------------------------------------


def test_round_trip_preserves_values_within_uint8_resolution(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    # Solid color in the middle of the dynamic range so uint8 quantization
    # gives the same value both directions.
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(src)
    original = load_image(src)

    out = tmp_path / "out.png"
    save_image(out, original)
    reloaded = load_image(out)

    # Round trip through uint8 is lossy at ~1/255 precision.
    assert np.allclose(np.asarray(original), np.asarray(reloaded), atol=1 / 255 + 1e-6)


def test_save_clips_out_of_range_values(tmp_path: Path) -> None:
    """Values outside [0, 1] don't crash; they clip cleanly to uint8."""
    arr = mx.array([[[-0.5, 1.5, 0.5]]], dtype=mx.float32)  # 1x1 image with one negative + one >1.
    out = tmp_path / "clipped.png"
    save_image(out, arr)
    # Reload and verify each channel landed in [0, 255] uint8 space (i.e. [0, 1] float).
    reloaded = np.asarray(load_image(out))
    assert reloaded[0, 0, 0] == 0.0  # -0.5 -> clipped to 0
    assert reloaded[0, 0, 1] == 1.0  # 1.5 -> clipped to 1
    assert abs(reloaded[0, 0, 2] - 0.5) < 1 / 255 + 1e-6  # 0.5 round-trips
