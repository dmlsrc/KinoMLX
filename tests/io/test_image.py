"""Behavioral tests for ``kinomlx.io.image``."""

from pathlib import Path

import mlx.core as mx

from kinomlx.io.image import _resize_bilinear, load_image, save_image
from tests._image_fixtures import solid_png

# ---------------------------------------------------------------------------
# Basic shape / dtype / range contract
# ---------------------------------------------------------------------------


def test_load_image_returns_hwc_float32_in_unit_range(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    solid_png(src, 64, 48, (128, 200, 50))
    arr = load_image(src)
    assert arr.shape == (48, 64, 3)
    assert arr.dtype == mx.float32
    # 128/255, 200/255, 50/255 - all in [0, 1].
    assert mx.min(arr).item() >= 0.0
    assert mx.max(arr).item() <= 1.0


# ---------------------------------------------------------------------------
# Mode normalization - RGBA / grayscale -> RGB
# ---------------------------------------------------------------------------


def test_load_image_strips_alpha_from_rgba(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    solid_png(src, 16, 16, (255, 0, 0, 128))
    arr = load_image(src)
    assert arr.shape == (16, 16, 3)


def test_load_image_expands_grayscale_to_rgb(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    solid_png(src, 16, 16, (128,))
    arr = load_image(src)
    assert arr.shape == (16, 16, 3)
    # Grayscale -> RGB replicates the single channel.
    assert mx.allclose(arr[..., 0], arr[..., 1], rtol=1e-5, atol=1e-8).item()
    assert mx.allclose(arr[..., 1], arr[..., 2], rtol=1e-5, atol=1e-8).item()


# ---------------------------------------------------------------------------
# Resize + cover-crop
# ---------------------------------------------------------------------------


def test_load_with_size_matches_target_dims(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    solid_png(src, 320, 240, (0, 128, 255))
    arr = load_image(src, size=(160, 120))
    assert arr.shape == (120, 160, 3)


def test_cover_crop_handles_wider_source(tmp_path: Path) -> None:
    """Source 200x100, target 100x100 - height fits exactly, width is cropped."""
    src = tmp_path / "wide.png"
    solid_png(src, 200, 100, (255, 255, 255))
    arr = load_image(src, size=(100, 100))
    assert arr.shape == (100, 100, 3)


def test_cover_crop_handles_taller_source(tmp_path: Path) -> None:
    """Source 100x200, target 100x100 - width fits exactly, height is cropped."""
    src = tmp_path / "tall.png"
    solid_png(src, 100, 200, (255, 255, 255))
    arr = load_image(src, size=(100, 100))
    assert arr.shape == (100, 100, 3)


def test_bilinear_resize_matches_align_corners_false_grid() -> None:
    values = mx.array([[0.0, 1.0], [2.0, 3.0]])
    source = mx.broadcast_to(values[:, :, None], (2, 2, 3))
    resized = _resize_bilinear(source, 4, 4)
    expected = mx.array(
        [
            [0.0, 0.25, 0.75, 1.0],
            [0.5, 0.75, 1.25, 1.5],
            [1.5, 1.75, 2.25, 2.5],
            [2.0, 2.25, 2.75, 3.0],
        ],
        dtype=mx.float32,
    )
    assert mx.allclose(resized[:, :, 0], expected, rtol=1e-5, atol=1e-8).item()


# ---------------------------------------------------------------------------
# Round-trip - load -> save -> load
# ---------------------------------------------------------------------------


def test_round_trip_preserves_values_within_uint8_resolution(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    # Solid color in the middle of the dynamic range so uint8 quantization
    # gives the same value both directions.
    solid_png(src, 8, 8, (100, 150, 200))
    original = load_image(src)

    out = tmp_path / "out.png"
    save_image(out, original)
    reloaded = load_image(out)

    # Round trip through uint8 is lossy at ~1/255 precision.
    assert mx.allclose(original, reloaded, atol=1 / 255 + 1e-6, rtol=1e-5).item()


def test_save_clips_out_of_range_values(tmp_path: Path) -> None:
    """Values outside [0, 1] don't crash; they clip cleanly to uint8."""
    arr = mx.array([[[-0.5, 1.5, 0.5]]], dtype=mx.float32)  # 1x1 image with one negative + one >1.
    out = tmp_path / "clipped.png"
    save_image(out, arr)
    # Reload and verify each channel landed in [0, 255] uint8 space (i.e. [0, 1] float).
    reloaded = load_image(out)
    assert reloaded[0, 0, 0] == 0.0  # -0.5 -> clipped to 0
    assert reloaded[0, 0, 1] == 1.0  # 1.5 -> clipped to 1
    assert abs(reloaded[0, 0, 2] - 0.5) < 1 / 255 + 1e-6  # 0.5 round-trips
