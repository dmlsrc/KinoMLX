"""Round-trip and helper tests for ``kinomlx.videotoolbox.pixel_buffers``.

The CoreVideo bridge marshals frames between MLX arrays and IOSurface-backed
CVPixelBuffers with zero-copy memoryviews - the same class of code as the audio
path and just as easy to get subtly wrong (row padding, channel order, a
vertical flip). These tests pin it:

- ``upload_frame_to_buffer`` -> ``read_pixel_buffer_rgb`` round-trips an fp16
  RGBA frame through a RGBAHalf CVPixelBuffer, exact at uint8 precision for flat
  colors and spatially-varying frames (orientation preserved), plus the uint8
  RGB -> fp16 promotion path.
- ``frame_pts`` lands bit-exact integer ticks for integer fps and within a
  microsecond for NTSC fractional rates.
- ``resolve_pixel_format`` / ``make_pixel_buffer_from_attrs`` and the small
  frame-buffer helpers behave as documented.

Everything here drives CoreVideo / CoreImage / CoreMedia, so the module is
tagged ``requires_avfoundation`` for explicit native-test selection.
"""

from __future__ import annotations

import CoreMedia
import mlx.core as mx
import pytest
import Quartz

from kinomlx.videotoolbox.pixel_buffers import (
    PIX_NV12,
    PIX_RGBAHALF,
    VIDEO_TIME_SCALE,
    _frame_buffer,
    _frame_is_fp16,
    frame_pts,
    make_pixel_buffer_from_attrs,
    make_pool_from_attrs,
    pool_create_buffer,
    read_pixel_buffer_rgb,
    read_rgbahalf_rgb,
    resolve_pixel_format,
    upload_frame_to_buffer,
)

pytestmark = pytest.mark.requires_avfoundation


def _rgbahalf_attrs(width: int, height: int) -> dict:
    return {
        Quartz.kCVPixelBufferPixelFormatTypeKey: PIX_RGBAHALF,
        Quartz.kCVPixelBufferWidthKey: width,
        Quartz.kCVPixelBufferHeightKey: height,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }


def _flat_rgba_fp16(height: int, width: int, rgb: tuple[float, float, float]) -> mx.array:
    """A (H, W, 4) fp16 frame filled with one opaque color."""
    frame = mx.zeros((height, width, 4), dtype=mx.float16)
    frame[..., 0] = rgb[0]
    frame[..., 1] = rgb[1]
    frame[..., 2] = rgb[2]
    frame[..., 3] = 1.0
    return frame


# --------------------------------------------------------------------------- #
# frame_pts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("frame_index", "fps", "expected_value"),
    [(0, 24.0, 0), (10, 24.0, 10000), (30, 30.0, 24000), (48, 48.0, 24000)],
)
def test_frame_pts_integer_fps_is_bit_exact(frame_index, fps, expected_value):
    t = frame_pts(frame_index, fps)
    assert t.timescale == VIDEO_TIME_SCALE
    assert t.value == expected_value
    assert CoreMedia.CMTimeGetSeconds(t) == pytest.approx(frame_index / fps)


def test_frame_pts_ntsc_is_within_a_microsecond():
    # 24000 does not divide 23.976 exactly; the tick rounds, staying sub-us.
    t = frame_pts(10, 23.976)
    assert t.timescale == VIDEO_TIME_SCALE
    assert CoreMedia.CMTimeGetSeconds(t) == pytest.approx(10 / 23.976, abs=1e-6)


# --------------------------------------------------------------------------- #
# resolve_pixel_format / make_pixel_buffer_from_attrs
# --------------------------------------------------------------------------- #


def test_resolve_pixel_format_passes_through_int():
    assert resolve_pixel_format({"PixelFormatType": PIX_NV12}) == PIX_NV12


def test_resolve_pixel_format_unwraps_single_element_sequence():
    # VTSuperResolutionScalerConfiguration hands its formats back as a
    # one-element NSArray rather than a bare int.
    assert resolve_pixel_format({"PixelFormatType": [PIX_NV12]}) == PIX_NV12


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({}, "do not contain"),
        ({"PixelFormatType": []}, "empty or invalid"),
        ({"PixelFormatType": True}, "integer FourCC"),
        ({"PixelFormatType": "420v"}, "invalid PixelFormatType"),
    ],
)
def test_resolve_pixel_format_rejects_malformed_attributes(attrs, message):
    with pytest.raises(ValueError, match=message):
        resolve_pixel_format(attrs)


def test_resolve_pixel_format_reports_original_wrapped_value() -> None:
    with pytest.raises(ValueError, match=r"\['420v'\]"):
        resolve_pixel_format({"PixelFormatType": ["420v"]})


def test_make_pixel_buffer_from_attrs_has_requested_geometry():
    pb = make_pixel_buffer_from_attrs(16, 8, _rgbahalf_attrs(16, 8))
    assert Quartz.CVPixelBufferGetWidth(pb) == 16
    assert Quartz.CVPixelBufferGetHeight(pb) == 8
    assert Quartz.CVPixelBufferGetPixelFormatType(pb) == PIX_RGBAHALF


def test_retained_buffers_exhaust_the_declared_pool_window() -> None:
    pool = make_pool_from_attrs(_rgbahalf_attrs(16, 8))
    assert pool is not None

    retained = [
        pool_create_buffer(pool, allocation_threshold=2),
        pool_create_buffer(pool, allocation_threshold=2),
    ]

    assert all(buffer is not None for buffer in retained)
    assert pool_create_buffer(pool, allocation_threshold=2) is None


# --------------------------------------------------------------------------- #
# frame-buffer helpers
# --------------------------------------------------------------------------- #


def test_frame_is_fp16_detects_dtype():
    assert _frame_is_fp16(mx.zeros((2, 2, 4), dtype=mx.float16))
    assert not _frame_is_fp16(mx.zeros((2, 2, 3), dtype=mx.uint8))


def test_frame_buffer_byte_length_matches_dtype():
    assert len(_frame_buffer(mx.zeros((4, 5, 4), dtype=mx.float16))) == 4 * 5 * 4 * 2
    assert len(_frame_buffer(mx.zeros((4, 5, 3), dtype=mx.uint8))) == 4 * 5 * 3


# --------------------------------------------------------------------------- #
# upload_frame_to_buffer -> read_pixel_buffer_rgb round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((0.0, 0.0, 0.0), (0, 0, 0)),
        ((1.0, 1.0, 1.0), (255, 255, 255)),
        ((0.5, 0.5, 0.5), (128, 128, 128)),
        ((1.0, 0.0, 0.0), (255, 0, 0)),
        ((0.0, 1.0, 0.0), (0, 255, 0)),
        ((0.0, 0.0, 1.0), (0, 0, 255)),
    ],
    ids=["black", "white", "mid-gray", "red", "green", "blue"],
)
def test_rgbahalf_roundtrip_flat_color(rgb, expected):
    height, width = 8, 16
    pb = make_pixel_buffer_from_attrs(width, height, _rgbahalf_attrs(width, height))
    upload_frame_to_buffer(_flat_rgba_fp16(height, width, rgb), pb)

    out = read_pixel_buffer_rgb(pb)
    assert tuple(int(x) for x in out.shape) == (height, width, 3)
    assert out.dtype == mx.uint8
    expected_frame = mx.broadcast_to(
        mx.array(expected, dtype=mx.uint8),
        (height, width, 3),
    )
    assert mx.array_equal(out, expected_frame)


def test_rgbahalf_roundtrip_preserves_orientation():
    """Top-half red / bottom-half blue must survive without a vertical flip."""
    height, width = 8, 16
    mid = height // 2
    frame = mx.zeros((height, width, 4), dtype=mx.float16)
    frame[:mid, :, 0] = 1.0
    frame[mid:, :, 2] = 1.0
    frame[..., 3] = 1.0

    pb = make_pixel_buffer_from_attrs(width, height, _rgbahalf_attrs(width, height))
    upload_frame_to_buffer(frame, pb)
    out = read_pixel_buffer_rgb(pb).astype(mx.float32)

    expected = mx.zeros((height, width, 3), dtype=mx.float32)
    expected[:mid, :, 0] = 255
    expected[mid:, :, 2] = 255
    assert mx.array_equal(out, expected)


def test_uint8_rgb_upload_to_rgbahalf_promotes_and_roundtrips():
    """uint8 RGB into a RGBAHalf buffer takes the inline fp16-promotion path."""
    height, width = 8, 16
    frame = mx.zeros((height, width, 3), dtype=mx.uint8)
    frame[..., 0] = 200
    frame[..., 1] = 100
    frame[..., 2] = 50

    pb = make_pixel_buffer_from_attrs(width, height, _rgbahalf_attrs(width, height))
    upload_frame_to_buffer(frame, pb)
    out = read_pixel_buffer_rgb(pb)

    expected = mx.broadcast_to(
        mx.array([200, 100, 50], dtype=mx.uint8),
        (height, width, 3),
    )
    assert mx.array_equal(out, expected)


def test_rgbahalf_direct_read_preserves_half_float_samples():
    """The writer feed must not quantize RGBAHalf through an RGBA8 render."""
    height, width = 8, 16
    rgb = mx.linspace(0.001, 0.999, height * width * 3, dtype=mx.float16).reshape(
        height,
        width,
        3,
    )
    alpha = mx.ones((height, width, 1), dtype=mx.float16)
    frame = mx.concatenate([rgb, alpha], axis=-1)
    pb = make_pixel_buffer_from_attrs(width, height, _rgbahalf_attrs(width, height))
    upload_frame_to_buffer(frame, pb)

    out = read_rgbahalf_rgb(pb)

    assert out.dtype == mx.float32
    assert mx.array_equal(out, rgb.astype(mx.float32))


def test_rgbahalf_host_adoption_stays_inside_locked_borrow_scope(monkeypatch):
    height, width = 8, 16
    frame = _flat_rgba_fp16(height, width, (0.2, 0.4, 0.6))
    pb = make_pixel_buffer_from_attrs(width, height, _rgbahalf_attrs(width, height))
    upload_frame_to_buffer(frame, pb)
    state = {"locked": False, "evaluated": False, "copies": []}
    original_lock = Quartz.CVPixelBufferLockBaseAddress
    original_unlock = Quartz.CVPixelBufferUnlockBaseAddress
    original_asarray = mx.asarray
    original_eval = mx.eval

    def lock(buffer, flags):
        result = original_lock(buffer, flags)
        state["locked"] = True
        return result

    def unlock(buffer, flags):
        assert state["locked"]
        assert state["evaluated"]
        state["locked"] = False
        return original_unlock(buffer, flags)

    def asarray(value, *, copy=None):
        assert state["locked"]
        state["copies"].append(copy)
        return original_asarray(value, copy=copy)

    def evaluate(*values):
        assert state["locked"]
        result = original_eval(*values)
        state["evaluated"] = True
        return result

    monkeypatch.setattr(Quartz, "CVPixelBufferLockBaseAddress", lock)
    monkeypatch.setattr(Quartz, "CVPixelBufferUnlockBaseAddress", unlock)
    monkeypatch.setattr(mx, "asarray", asarray)
    monkeypatch.setattr(mx, "eval", evaluate)

    out = read_rgbahalf_rgb(pb)

    assert out.shape == (height, width, 3)
    assert state == {"locked": False, "evaluated": True, "copies": [False]}


def test_rgbahalf_direct_read_rejects_other_pixel_formats():
    pb = make_pixel_buffer_from_attrs(16, 8, _nv12_attrs(16, 8))
    with pytest.raises(ValueError, match="expected RGBAHalf"):
        read_rgbahalf_rgb(pb)


# --------------------------------------------------------------------------- #
# NV12 (4:2:0 via CoreImage) upload -> read round-trip
# --------------------------------------------------------------------------- #


def _nv12_attrs(width: int, height: int) -> dict:
    return {
        Quartz.kCVPixelBufferPixelFormatTypeKey: PIX_NV12,
        Quartz.kCVPixelBufferWidthKey: width,
        Quartz.kCVPixelBufferHeightKey: height,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }


@pytest.mark.parametrize(
    "gray",
    [0, 128, 255],
    ids=["black", "mid-gray", "white"],
)
def test_nv12_roundtrip_achromatic_is_exact(gray):
    """uint8 RGB -> NV12 (4:2:0 via CoreImage) -> RGB. Achromatic colors carry
    neutral chroma, so they survive the subsampling exactly; saturated colors
    would drift under 4:2:0 and are intentionally not asserted here."""
    height, width = 8, 16
    frame = mx.full((height, width, 3), gray, dtype=mx.uint8)
    pb = make_pixel_buffer_from_attrs(width, height, _nv12_attrs(width, height))
    upload_frame_to_buffer(frame, pb)

    out = read_pixel_buffer_rgb(pb)
    assert mx.array_equal(out, frame)


def test_nv12_upload_is_deterministic():
    height, width = 8, 16
    frame = mx.zeros((height, width, 3), dtype=mx.uint8)
    frame[..., 1] = 200
    a = make_pixel_buffer_from_attrs(width, height, _nv12_attrs(width, height))
    b = make_pixel_buffer_from_attrs(width, height, _nv12_attrs(width, height))
    upload_frame_to_buffer(frame, a)
    upload_frame_to_buffer(frame, b)
    assert mx.array_equal(read_pixel_buffer_rgb(a), read_pixel_buffer_rgb(b))
