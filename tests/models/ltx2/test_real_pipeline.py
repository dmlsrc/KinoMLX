"""Explicit opt-in real-weight gate for the complete LTX-2.3 pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import AVFoundation as av
import CoreMedia
import Foundation
import mlx.core as mx
import pytest

from kinomlx.cli.config import OutputConfig
from kinomlx.cli.output import write_generation
from kinomlx.models.ltx2.pipelines.distilled import generate_distilled
from kinomlx.models.ltx2.runner import GenerationOutput, LTX2Runner
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.settings import Settings

_MAX_UNACCOUNTED_VAE_ENTRY_BYTES = 1 << 30


class _InspectingFrameStream:
    """Validate real generated frames while preserving one-pass terminal flow."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self.spec = source.spec
        self.frame_count = source.frame_count
        self.consumed = 0
        self.nonzero = False

    def __iter__(self):
        for frame in self._source:
            assert tuple(frame.shape) == (64, 64, 3)
            assert frame.dtype == mx.float16
            assert mx.all(mx.isfinite(frame)).item()
            assert mx.all((frame >= 0.0) & (frame <= 1.0)).item()
            self.consumed += 1
            self.nonzero = self.nonzero or bool(mx.any(frame != 0).item())
            yield frame

    def close(self) -> None:
        self._source.close()

    @property
    def receipts(self):
        return self._source.receipts


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.requires_avfoundation
def test_real_ltx23_joint_pipeline_writes_video_and_audio(tmp_path: Path) -> None:
    if os.environ.get("KINO_RUN_REAL_PIPELINE") != "1":
        pytest.skip("set KINO_RUN_REAL_PIPELINE=1 for the full real-weight smoke")
    infrastructure = Settings.from_env()
    model_settings = LTX2Settings.from_env()
    checkpoint = model_settings.weights_path
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("KINO_WEIGHTS_PATH is not configured")

    with LTX2Runner(model_settings, infrastructure=infrastructure).run(
        generate_distilled,
        DistilledRequest(
            prompt="A simple landscape.",
            width=64,
            height=64,
            frames=9,
            seed=7,
            generate_audio=True,
        ),
    ) as output:
        assert output.signal.dtype == "float16"
        assert output.frame_count == 9
        assert output.audio_waveform is not None
        assert tuple(output.audio_waveform.shape) == (1, 2, 17_760)
        assert mx.all(mx.isfinite(output.audio_waveform)).item()
        assert mx.any(output.audio_waveform != 0).item()
        assert output.audio_sample_rate == 48_000
        assert output.metadata["model_version"] == "2.3.0"
        assert output.metadata["video_shape"] == (1, 3, 9, 64, 64)

        frames = _InspectingFrameStream(output.frames)
        muxed = write_generation(
            GenerationOutput(
                frames=frames,
                audio_waveform=output.audio_waveform,
                audio_sample_rate=output.audio_sample_rate,
                metadata=output.metadata,
                diagnostics_provider=output.runtime_diagnostics,
            ),
            OutputConfig(path=tmp_path / "real-joint.mp4"),
            fps=24.0,
        )
        assert frames.consumed == 9
        assert frames.nonzero
        assert output.frames.closed
        diagnostics = output.runtime_diagnostics()
        vae_decode = diagnostics["vae_decode"]
        entry = vae_decode["entry_memory"]
        tiling = vae_decode["tiling"]
        assert 0 <= entry["unaccounted_active_bytes"] <= _MAX_UNACCOUNTED_VAE_ENTRY_BYTES, (
            "LTX-2.3 VAE entry retained a heavyweight predecessor component"
        )
        assert tiling["requested_mode"] == "auto"
        assert tiling["total_tiles"] == 1
    asset = av.AVURLAsset.URLAssetWithURL_options_(
        Foundation.NSURL.fileURLWithPath_(str(muxed)),
        None,
    )
    video_tracks = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    audio_tracks = asset.tracksWithMediaType_(av.AVMediaTypeAudio)
    assert len(video_tracks) == 1
    assert len(audio_tracks) == 1
    size = video_tracks[0].naturalSize()
    assert (int(size.width), int(size.height)) == (64, 64)

    video_duration = CoreMedia.CMTimeGetSeconds(video_tracks[0].timeRange().duration)
    audio_duration = CoreMedia.CMTimeGetSeconds(audio_tracks[0].timeRange().duration)
    assert video_duration == pytest.approx(9 / 24, abs=1 / 48_000)
    assert audio_duration == pytest.approx(17_760 / 48_000, abs=1 / 48_000)
    assert 0.0 <= video_duration - audio_duration <= 1 / 24
