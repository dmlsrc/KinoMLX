"""BT.2100 PQ HEIC conversion and manifest contracts."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.media.signals import ColorPrimaries
from kinomlx.videotoolbox.heic import (
    HEIC_COMPRESSION_QUALITY,
    PQ_PEAK_NITS,
    PQ_REFERENCE_WHITE_NITS,
    scene_linear_to_pq_codes,
    write_heic_manifest,
)


def test_scene_linear_reference_white_maps_to_published_st2084_code() -> None:
    frame = mx.ones((1, 1, 3), dtype=mx.float32)

    codes = scene_linear_to_pq_codes(frame, primaries=ColorPrimaries.REC709)

    assert codes.dtype == mx.float32
    assert tuple(codes.shape) == (1, 1, 3)
    assert float(codes[0, 0, 0].item()) == pytest.approx(0.580688881, abs=2e-6)
    assert bool(mx.allclose(codes[..., 0], codes[..., 1]))
    assert bool(mx.allclose(codes[..., 1], codes[..., 2]))


def test_scene_linear_pq_conversion_clips_negative_and_peak_light() -> None:
    peak = PQ_PEAK_NITS / PQ_REFERENCE_WHITE_NITS
    frame = mx.array(
        [[[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [peak * 2.0] * 3]],
        dtype=mx.float32,
    )

    codes = scene_linear_to_pq_codes(frame, primaries=ColorPrimaries.REC709)

    assert float(codes[0, 0, 0].item()) < 1e-5
    assert float(codes[0, 1, 0].item()) < 1e-5
    assert float(codes[0, 2, 0].item()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "frame",
    [
        mx.zeros((1, 1), dtype=mx.float32),
        mx.zeros((1, 1, 4), dtype=mx.float32),
        mx.zeros((1, 1, 3), dtype=mx.float16),
        mx.full((1, 1, 3), float("nan"), dtype=mx.float32),
    ],
)
def test_scene_linear_pq_conversion_rejects_invalid_frames(frame: mx.array) -> None:
    with pytest.raises((TypeError, ValueError)):
        scene_linear_to_pq_codes(frame, primaries=ColorPrimaries.REC709)


def test_heic_manifest_declares_viewable_display_encoding(tmp_path: Path) -> None:
    path = write_heic_manifest(
        tmp_path,
        source_primaries=ColorPrimaries.ACESCG,
        frame_count=3,
        width=64,
        height=32,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "format": "heic-sequence",
        "codec": "hevc",
        "bit_depth": 10,
        "primaries": "bt2020",
        "transfer": "pq",
        "reference_white_nits": 203.0,
        "pq_peak_nits": 10000.0,
        "compression_quality": HEIC_COMPRESSION_QUALITY,
        "source_primaries": "acescg",
        "frame_count": 3,
        "width": 64,
        "height": 32,
        "pattern": "frame_%05d.heic",
    }
