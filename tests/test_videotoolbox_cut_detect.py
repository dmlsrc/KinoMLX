"""Behavioral tests for the scene-cut detector (``kinomlx.videotoolbox.cut_detect``).

Pure MLX on the CPU (no pyobjc), so this runs everywhere. Pins the cut-decision
sequence, the histogram/thumbnail shapes and determinism, and the frame-format
coercion on deterministic frames.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.videotoolbox.cut_detect import (
    CutDetector,
    _frame_histogram,
    _frame_thumbnail,
    _to_uint8_rgb,
)


def _black(h: int = 64, w: int = 64) -> mx.array:
    return mx.zeros((h, w, 3), dtype=mx.uint8)


def _white(h: int = 64, w: int = 64) -> mx.array:
    return mx.full((h, w, 3), 255, dtype=mx.uint8)


# ---------------------------------------------------------------------------
# CutDetector decision sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "threshold"),
    [("simple", 0.2), ("hist", 0.5)],
)
def test_cut_detection_sequence(mode, threshold):
    det = CutDetector(mode, threshold)
    # black, black (no change), white (hard cut), white (no change).
    seq = [det.is_cut(f) for f in (_black(), _black(), _white(), _white())]
    assert seq == [False, False, True, False]


def test_first_frame_never_cuts():
    assert CutDetector("simple", 0.2).is_cut(_white()) is False


def test_off_mode_never_cuts():
    det = CutDetector("off", 0.2)
    assert [det.is_cut(f) for f in (_black(), _white(), _black())] == [False, False, False]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown cut-detect mode"):
        CutDetector("bogus", 0.2)


@pytest.mark.parametrize(
    ("mode", "threshold"),
    [("off", 0.0), ("simple", 0.25), ("hist", 0.5)],
)
def test_mode_defaults_resolve_at_construction(mode, threshold):
    assert CutDetector(mode).threshold == pytest.approx(threshold)


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError, match="finite and non-negative"):
        CutDetector("simple", threshold)


# ---------------------------------------------------------------------------
# Thumbnail / histogram helpers
# ---------------------------------------------------------------------------


def test_frame_thumbnail_downsamples_to_target():
    thumb = _frame_thumbnail(_white(64, 64), target_size=32)
    assert tuple(int(x) for x in thumb.shape) == (32, 32, 3)
    assert thumb.dtype == mx.uint8


def test_frame_histogram_shape_and_total():
    hist = _frame_histogram(_white(64, 64), bins=32)
    # Three concatenated 32-bin channel histograms.
    assert tuple(int(x) for x in hist.shape) == (96,)
    # Every pixel is counted once per channel: total == 3 * H * W.
    assert int(mx.sum(hist).item()) == 3 * 64 * 64


def test_frame_histogram_is_deterministic_and_content_sensitive():
    a = _frame_histogram(_white())
    b = _frame_histogram(_white())
    c = _frame_histogram(_black())
    assert mx.array_equal(a, b)  # same input -> identical histogram
    assert not mx.array_equal(a, c)  # white vs black differ


# ---------------------------------------------------------------------------
# Frame-format coercion
# ---------------------------------------------------------------------------


def test_to_uint8_rgb_drops_alpha():
    rgba = mx.zeros((4, 4, 4), dtype=mx.uint8)
    out = _to_uint8_rgb(rgba)
    assert tuple(int(x) for x in out.shape) == (4, 4, 3)
    assert out.dtype == mx.uint8


def test_to_uint8_rgb_scales_float_rgba():
    # fp16 RGBA in [0, 1] -> uint8 RGB in [0, 255].
    frame = mx.ones((4, 4, 4), dtype=mx.float16)
    out = _to_uint8_rgb(frame)
    assert tuple(int(x) for x in out.shape) == (4, 4, 3)
    assert out.dtype == mx.uint8
    assert int(mx.max(out).item()) == 255
