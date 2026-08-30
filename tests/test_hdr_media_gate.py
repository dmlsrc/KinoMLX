"""End-to-end native EXR plus HEVC Main10 HLG media gate."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import Foundation
import mlx.core as mx
import pytest
import Quartz

from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.hdr import scene_linear_to_acescct, scene_linear_to_logc3
from kinomlx.media.signals import (
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    ExrDeliverySpec,
    ExrSampleType,
    OutputColorPlan,
)
from kinomlx.models.ltx2.runner import GenerationOutput
from kinomlx.models.ltx2.signals import ltx_hdr_working_signal
from kinomlx.output import HDRGenerationSink
from kinomlx.videotoolbox.exr import read_exr_frame


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is unavailable")
@pytest.mark.parametrize(
    ("transfer", "encode", "primaries"),
    [
        (ColorTransfer.ACESCCT, scene_linear_to_acescct, ColorPrimaries.ACESCG),
        (ColorTransfer.LOGC3, scene_linear_to_logc3, ColorPrimaries.REC709),
    ],
)
def test_hdr_working_stream_reaches_lossless_exr_and_tagged_hlg(
    tmp_path: Path,
    transfer: ColorTransfer,
    encode,
    primaries: ColorPrimaries,
) -> None:
    ramp = mx.linspace(0.0, 12.0, 64 * 64, dtype=mx.float32).reshape(64, 64, 1)
    linear = mx.concatenate((ramp, ramp * 0.5, ramp * 0.25), axis=-1)
    codes = encode(linear)
    signal = ltx_hdr_working_signal(
        transfer=transfer,
        width=64,
        height=64,
        fps=24.0,
    )
    generation = GenerationOutput(
        frames=VideoFrameStream(lambda: iter((codes,)), spec=signal, frame_count=1)
    )
    output = tmp_path / f"{transfer.value}.mp4"
    plan = OutputColorPlan(
        source=signal,
        deliveries=(
            ExrDeliverySpec(
                primaries=primaries,
                transfer=ColorTransfer.LINEAR,
                sample_type=ExrSampleType.FLOAT16,
                color_space_tag="linear-master",
            ),
            BT2020_HLG_DELIVERY,
        ),
    )

    artifacts = HDRGenerationSink(
        path=output,
        fps=24.0,
        heic_directory=tmp_path / f"{transfer.value}_heic",
    ).write(generation, plan)
    exr = read_exr_frame(artifacts.exr_frames / "frame_00000.exr")
    assert float(exr.max().item()) == pytest.approx(12.0, abs=0.01)
    manifest = json.loads((artifacts.exr_frames / "manifest.json").read_text())
    assert manifest["transfer"] == "linear"
    assert manifest["frame_count"] == 1

    heic_path = artifacts.heic_frames / "frame_00000.heic"
    source = Quartz.CGImageSourceCreateWithURL(
        Foundation.NSURL.fileURLWithPath_(str(heic_path)),
        None,
    )
    assert source is not None
    properties = Quartz.CGImageSourceCopyPropertiesAtIndex(source, 0, None)
    assert int(properties[Quartz.kCGImagePropertyDepth]) == 10
    assert "BT.2100 PQ" in str(properties[Quartz.kCGImagePropertyProfileName])
    heic_manifest = json.loads((artifacts.heic_frames / "manifest.json").read_text())
    assert heic_manifest["primaries"] == "bt2020"
    assert heic_manifest["transfer"] == "pq"
    assert heic_manifest["reference_white_nits"] == 203.0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_name,profile,pix_fmt,color_range,color_space,"
                "color_transfer,color_primaries,nb_frames"
            ),
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    expected = {
        "codec_name": "hevc",
        "profile": "Main 10",
        "pix_fmt": "yuv420p10le",
        "color_range": "tv",
        "color_space": "bt2020nc",
        "color_transfer": "arib-std-b67",
        "color_primaries": "bt2020",
        "nb_frames": "1",
    }
    assert {key: stream[key] for key in expected} == expected
