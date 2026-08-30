from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from kinomlx.media.signals import (
    ColorPrimaries,
    ColorTransfer,
    ExrDeliverySpec,
    ExrSampleType,
)
from kinomlx.videotoolbox.exr import (
    read_exr_frame,
    save_exr_frame,
    write_exr_manifest,
)


def _delivery(transfer: ColorTransfer = ColorTransfer.LINEAR) -> ExrDeliverySpec:
    return ExrDeliverySpec(
        primaries=ColorPrimaries.ACESCG,
        transfer=transfer,
        sample_type=ExrSampleType.FLOAT16,
        color_space_tag="ACEScg" if transfer is ColorTransfer.LINEAR else "ACEScct",
    )


def test_native_half_exr_round_trip_retains_hdr_values(tmp_path: Path) -> None:
    frame = mx.array(
        [
            [[0.0, 0.5, 1.0], [4.0, 8.0, 16.0]],
            [[-0.25, 2.0, 12.0], [0.125, 3.0, 6.0]],
        ],
        dtype=mx.float32,
    )
    path = tmp_path / "frame_00000.exr"
    save_exr_frame(frame, path, delivery=_delivery())
    decoded = read_exr_frame(path)
    assert path.read_bytes()[:4] == bytes.fromhex("762f3101")
    assert decoded.dtype == mx.float32
    expected = frame.astype(mx.float16).astype(mx.float32)
    assert mx.allclose(decoded, expected, rtol=0.0, atol=2e-4)
    assert mx.array_equal(decoded[0, 1], expected[0, 1])


def test_exr_manifest_declares_nonstandard_log_authoring_semantics(tmp_path: Path) -> None:
    path = write_exr_manifest(
        tmp_path,
        delivery=_delivery(ColorTransfer.ACESCCT),
        frame_count=9,
        width=64,
        height=32,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["primaries"] == "acescg"
    assert payload["transfer"] == "acescct"
    assert payload["sample_type"] == "float16"
    assert payload["frame_count"] == 9
