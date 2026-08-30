"""Native SDR reference-reader integration gates."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.hdr import scene_linear_to_acescct
from kinomlx.media.signals import (
    BT709_SDR_420_DELIVERY,
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    ExrDeliverySpec,
    ExrSampleType,
    OutputColorPlan,
)
from kinomlx.models.ltx2.runner import GenerationOutput
from kinomlx.models.ltx2.signals import ltx23_sdr_signal, ltx_hdr_working_signal
from kinomlx.output import HDRGenerationSink, VideoToolboxGenerationSink
from kinomlx.videotoolbox.reader import read_sdr_video_frames


def test_native_reader_decodes_color_managed_sdr_frames(tmp_path: Path) -> None:
    signal = ltx23_sdr_signal(width=64, height=32, fps=24.0)
    zeros = mx.zeros((32, 64), dtype=mx.float16)
    red = mx.stack((mx.full_like(zeros, 0.8), zeros, zeros), axis=-1)
    green = mx.stack((zeros, mx.full_like(zeros, 0.7), zeros), axis=-1)
    generation = GenerationOutput(
        frames=VideoFrameStream(
            lambda: iter((red, green)),
            spec=signal,
            frame_count=2,
        )
    )
    path = tmp_path / "reference.mp4"
    VideoToolboxGenerationSink(path=path, fps=24.0).write(
        generation,
        OutputColorPlan(source=signal, deliveries=(BT709_SDR_420_DELIVERY,)),
    )

    decoded = read_sdr_video_frames(path, max_frames=2)

    assert tuple(decoded.shape) == (2, 32, 64, 3)
    assert decoded.dtype == mx.float32
    assert float(decoded[0, :, :, 0].mean().item()) > 0.7
    assert float(decoded[0, :, :, 1].mean().item()) < 0.05
    assert float(decoded[1, :, :, 1].mean().item()) > 0.6
    assert float(decoded[1, :, :, 0].mean().item()) < 0.05


def test_native_reader_rejects_explicit_hlg_reference(tmp_path: Path) -> None:
    signal = ltx_hdr_working_signal(
        transfer=ColorTransfer.ACESCCT,
        width=64,
        height=32,
        fps=24.0,
    )
    codes = scene_linear_to_acescct(mx.ones((32, 64, 3), dtype=mx.float32))
    generation = GenerationOutput(
        frames=VideoFrameStream(
            lambda: iter((codes,)),
            spec=signal,
            frame_count=1,
        )
    )
    path = tmp_path / "hdr.mp4"
    HDRGenerationSink(path=path, fps=24.0).write(
        generation,
        OutputColorPlan(
            source=signal,
            deliveries=(
                ExrDeliverySpec(
                    primaries=ColorPrimaries.ACESCG,
                    transfer=ColorTransfer.LINEAR,
                    sample_type=ExrSampleType.FLOAT16,
                    color_space_tag="ACEScg",
                ),
                BT2020_HLG_DELIVERY,
            ),
        ),
    )

    with pytest.raises(ValueError, match="ordinary SDR Rec.709"):
        read_sdr_video_frames(path, max_frames=1)
