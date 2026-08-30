"""End-to-end tests for ``kinomlx.videotoolbox.encode.encode_video_videotoolbox``.

These drive the real VideoToolbox HEVC encoder through AVAssetWriter: synthetic
frames go in, a playable ``.mp4`` comes out, and AVFoundation reads it back to
confirm the track layout, geometry, and duration. No model weights are needed
(VSR / temporal interpolation stay off), so the path under test is the core
video mux plus the audio mux + onset-mitigation wiring.

The audio path runs the ported ``kinomlx.audio`` onset mitigation before the
AudioTrack is built, so the with-audio cases also exercise that integration.

Frame-exact counting via AVAssetReader is unreliable for HEVC (B-frame decode
order yields out-of-order / NaN PTSes), so frame count is asserted via the
container duration instead, which is exact.

Tagged ``requires_avfoundation`` for explicit native-test selection.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import AVFoundation as av
import CoreMedia
import Foundation
import mlx.core as mx
import pytest

from kinomlx.media.signals import (
    BT709_SDR_420_DELIVERY,
    BT709_SDR_422_DELIVERY,
)
from kinomlx.models.ltx2.signals import ltx23_sdr_signal
from kinomlx.reporting import RecordingReporter
from kinomlx.videotoolbox import encode as encode_module
from kinomlx.videotoolbox.encode import (
    _normalize_audio_for_track,
    encode_video_videotoolbox,
)
from kinomlx.videotoolbox.pixel_buffers import (
    PIX_RGBAHALF,
    make_pixel_buffer_from_attrs,
    upload_frame_to_buffer,
)
from kinomlx.videotoolbox.writer import AVWriter

pytestmark = pytest.mark.requires_avfoundation

N, H, W, FPS = 12, 32, 48, 24.0
AUDIO_SR = 48000


def _frames(n: int = N, h: int = H, w: int = W) -> list[mx.array]:
    """N distinct LTX-2.3 encoder-boundary float16 RGB frames."""
    out = []
    for i in range(n):
        frame = mx.zeros((h, w, 3), dtype=mx.float16)
        frame[..., 0] = ((20 * i) % 256) / 255.0
        out.append(frame)
    return out


def _probe_mp4(path):
    """(video_tracks, audio_tracks, duration_s, natural_size) for an mp4."""
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.URLAssetWithURL_options_(url, None)
    video = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    audio = asset.tracksWithMediaType_(av.AVMediaTypeAudio)
    duration_s = CoreMedia.CMTimeGetSeconds(asset.duration())
    size = video[0].naturalSize() if video else None
    return video, audio, duration_s, size


def _track_duration_seconds(track) -> float:
    return CoreMedia.CMTimeGetSeconds(track.timeRange().duration)


def test_audio_track_boundary_unwraps_singleton_pipeline_batch():
    waveform = mx.arange(32, dtype=mx.float16).reshape(1, 2, 16)
    normalized = _normalize_audio_for_track(waveform)

    assert normalized.shape == (2, 16)
    assert normalized.dtype == mx.float32
    assert mx.array_equal(normalized, waveform[0].astype(mx.float32)).item()


def test_audio_track_boundary_rejects_multiple_batch_items():
    with pytest.raises(ValueError, match="exactly one item"):
        _normalize_audio_for_track(mx.zeros((2, 2, 16)))


def test_public_encoder_rejects_multiple_audio_batch_items(tmp_path):
    output = tmp_path / "multi_batch.mp4"
    with pytest.raises(ValueError, match="exactly one item"):
        encode_video_videotoolbox(
            _frames(n=2),
            output,
            fps=FPS,
            source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
            delivery=BT709_SDR_420_DELIVERY,
            audio_waveform=mx.zeros((2, 2, 4000)),
            audio_sample_rate=AUDIO_SR,
            verbose=False,
        )
    assert not output.exists()


def test_encode_video_only_roundtrip(tmp_path):
    reporter = RecordingReporter()
    out = encode_video_videotoolbox(
        _frames(),
        tmp_path / "v.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_420_DELIVERY,
        reporter=reporter,
        verbose=False,
    )

    assert out.exists()
    assert out.suffix == ".mp4"
    assert out.stat().st_size > 0

    video, audio, duration_s, size = _probe_mp4(out)
    assert len(video) == 1
    assert len(audio) == 0
    assert (int(size.width), int(size.height)) == (W, H)
    # Duration is exact: N frames at FPS -> N / FPS seconds.
    assert round(duration_s * FPS) == N
    assert reporter.events[0] == (
        "start",
        "HEVC encode",
        {"total": N, "unit": "frame"},
    )
    assert len([event for event in reporter.events if event[0] == "advance"]) == N
    assert reporter.events[-1] == ("end", "HEVC encode", {})


def test_lazy_vae_phase_precedes_terminal_encode_phase(tmp_path):
    reporter = RecordingReporter()

    def frames():
        reporter.phase_start("VAE decode tiles", total=1, unit="tile")
        reporter.phase_advance("VAE decode tiles")
        reporter.phase_end("VAE decode tiles")
        yield from _frames(n=2)

    encode_video_videotoolbox(
        frames(),
        tmp_path / "ordered.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_420_DELIVERY,
        n_source_frames=2,
        reporter=reporter,
        verbose=False,
    )

    assert reporter.events[:4] == [
        ("start", "VAE decode tiles", {"total": 1, "unit": "tile"}),
        ("advance", "VAE decode tiles", {"advance": 1.0}),
        ("end", "VAE decode tiles", {}),
        ("start", "HEVC encode", {"total": 2, "unit": "frame"}),
    ]


@pytest.mark.parametrize(
    ("cut_detect_mode", "expected_resets"),
    [("simple", [2]), ("off", [])],
)
def test_balanced_vsr_resets_history_at_generated_cut(
    monkeypatch,
    tmp_path,
    cut_detect_mode,
    expected_resets,
):
    reset_before_source_indices = []
    upscaled_indices = []

    class FakeVsrSession:
        def __init__(self, *_args, **_kwargs):
            self.dst_attrs = {"format": "rgba-half"}
            self.uses_frame_history = True

        def upscale_to_buffer(self, _frame, frame_index):
            upscaled_indices.append(frame_index)
            return f"upscaled-{frame_index}"

        def reset_frame_history(self):
            reset_before_source_indices.append(len(upscaled_indices))

        def flush_pools(self):
            pass

        def close(self):
            pass

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            self.adaptor = SimpleNamespace(pixelBufferPool=lambda: None)

        def append(self, _buffer):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        encode_module,
        "objc",
        SimpleNamespace(autorelease_pool=nullcontext),
    )
    monkeypatch.setattr(encode_module, "VsrSession", FakeVsrSession)
    monkeypatch.setattr(encode_module, "AVWriter", FakeWriter)
    monkeypatch.setattr(
        encode_module._pb,
        "resolve_pixel_format",
        lambda _attrs: encode_module._pb.PIX_RGBAHALF,
    )

    frames = [
        mx.zeros((H, W, 3), dtype=mx.float16),
        mx.zeros((H, W, 3), dtype=mx.float16),
        mx.ones((H, W, 3), dtype=mx.float16),
        mx.ones((H, W, 3), dtype=mx.float16),
    ]
    encode_video_videotoolbox(
        frames,
        tmp_path / "cut.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_422_DELIVERY,
        vsr_spatial_mode="balanced",
        cut_detect_mode=cut_detect_mode,
        verbose=False,
    )

    assert upscaled_indices == [0, 1, 2, 3]
    assert reset_before_source_indices == expected_resets


def test_vtfrc_uses_cut_safe_feed_for_generated_cut(monkeypatch, tmp_path):
    feed_calls = []

    class FakeVtfrcSession:
        def __init__(self, *_args, **_kwargs):
            self.src_attrs = {"format": "rgba-half"}
            self.dst_attrs = {"format": "rgba-half"}

        def make_source_buffer(self):
            return "source-buffer"

        def use_dst_pool(self, _pool):
            pass

        def feed(self, _buffer, frame_index):
            feed_calls.append(("normal", frame_index))
            return iter(())

        def feed_cut(self, _buffer, frame_index):
            feed_calls.append(("cut", frame_index))
            return iter(())

        def drain(self):
            return iter(())

        def close(self):
            pass

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            self.adaptor = SimpleNamespace(pixelBufferPool=lambda: None)

        def append(self, _buffer):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        encode_module,
        "objc",
        SimpleNamespace(autorelease_pool=nullcontext),
    )
    monkeypatch.setattr(encode_module, "VtfrcSession", FakeVtfrcSession)
    monkeypatch.setattr(encode_module, "AVWriter", FakeWriter)
    monkeypatch.setattr(encode_module._pb, "upload_frame_to_buffer", lambda *_args: None)
    monkeypatch.setattr(
        encode_module._pb,
        "resolve_pixel_format",
        lambda _attrs: encode_module._pb.PIX_RGBAHALF,
    )

    frames = [
        mx.zeros((H, W, 3), dtype=mx.float16),
        mx.zeros((H, W, 3), dtype=mx.float16),
        mx.ones((H, W, 3), dtype=mx.float16),
        mx.ones((H, W, 3), dtype=mx.float16),
    ]
    encode_video_videotoolbox(
        frames,
        tmp_path / "cut-temporal.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_422_DELIVERY,
        target_fps=60.0,
        verbose=False,
    )

    assert feed_calls == [
        ("normal", 0),
        ("normal", 1),
        ("cut", 2),
        ("normal", 3),
    ]


def test_encode_rewrites_non_mp4_extension(tmp_path):
    # The encoder always produces .mp4; a .mov request is rewritten.
    out = encode_video_videotoolbox(
        _frames(),
        tmp_path / "clip.mov",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_420_DELIVERY,
        verbose=False,
    )
    assert out.suffix == ".mp4"
    assert out.exists()


def test_encode_empty_frames_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        encode_video_videotoolbox(
            [],
            tmp_path / "v.mp4",
            fps=FPS,
            source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
            delivery=BT709_SDR_420_DELIVERY,
            verbose=False,
        )


def test_encode_with_audio_muxes_track(tmp_path):
    # 0.5 s of quiet stereo noise matching the 12-frame / 24 fps video. Default
    # auto onset mode does not fire on non-click content, so audio muxes as-is.
    samples = int(N / FPS * AUDIO_SR)
    wav = mx.random.normal(shape=(1, 2, samples), key=mx.random.key(7)) * 0.05

    out = encode_video_videotoolbox(
        _frames(),
        tmp_path / "av.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_420_DELIVERY,
        audio_waveform=wav,
        audio_sample_rate=AUDIO_SR,
        verbose=False,
    )

    video, audio, duration_s, _ = _probe_mp4(out)
    assert len(video) == 1
    assert len(audio) == 1
    assert round(duration_s * FPS) == N
    assert _track_duration_seconds(video[0]) == pytest.approx(N / FPS, abs=1 / AUDIO_SR)
    assert _track_duration_seconds(audio[0]) == pytest.approx(
        samples / AUDIO_SR,
        abs=1 / AUDIO_SR,
    )


def test_encode_with_audio_force_onset_trim(tmp_path):
    # Exercises the onset mitigation wired into the encoder: force always
    # zero-fills the lead. The muxed result must still carry an audio track.
    samples = int(N / FPS * AUDIO_SR)
    wav = mx.random.normal(shape=(2, samples), key=mx.random.key(8)) * 0.2

    out = encode_video_videotoolbox(
        _frames(),
        tmp_path / "av_trim.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=W, height=H, fps=FPS),
        delivery=BT709_SDR_420_DELIVERY,
        audio_waveform=wav,
        audio_sample_rate=AUDIO_SR,
        audio_onset_trim_mode="force",
        audio_onset_trim_ms=80.0,
        verbose=False,
    )

    video, audio, _, _ = _probe_mp4(out)
    assert len(video) == 1
    assert len(audio) == 1


def test_rgbahalf_writer_uses_explicit_yuv_feed(tmp_path):
    """Exercise the real RGBAHalf -> MLX BT.709 -> Main42210 writer path."""
    width, height, count = 160, 128, 4
    out = tmp_path / "rgbahalf.mp4"
    writer = AVWriter(
        out,
        width=width,
        height=height,
        fps=FPS,
        source_pixel_format=PIX_RGBAHALF,
        delivery=BT709_SDR_422_DELIVERY,
    )
    attrs = {
        "PixelFormatType": PIX_RGBAHALF,
        "Width": width,
        "Height": height,
        "IOSurfaceProperties": {},
    }

    for index in range(count):
        frame = mx.zeros((height, width, 4), dtype=mx.float16)
        frame[..., 0] = 0.1 + 0.1 * index
        frame[..., 1] = 0.5
        frame[..., 2] = 0.9 - 0.1 * index
        frame[..., 3] = 1.0
        pb = make_pixel_buffer_from_attrs(width, height, attrs)
        upload_frame_to_buffer(frame, pb)
        writer.append(pb)
    writer.finish()

    video, audio, duration_s, size = _probe_mp4(out)
    assert len(video) == 1
    assert len(audio) == 0
    assert (int(size.width), int(size.height)) == (width, height)
    assert round(duration_s * FPS) == count


def test_temporal_encode_keeps_the_final_source_period(tmp_path):
    """24 -> 60 fps emits all four source periods, including the drain."""
    source_count = 4
    target_fps = 60.0
    out = encode_video_videotoolbox(
        _frames(n=source_count, h=128, w=160),
        tmp_path / "temporal.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=160, height=128, fps=FPS),
        delivery=BT709_SDR_422_DELIVERY,
        target_fps=target_fps,
        verbose=False,
    )

    video, audio, duration_s, size = _probe_mp4(out)
    expected_frames = round(source_count * target_fps / FPS)
    assert len(video) == 1
    assert len(audio) == 0
    assert (int(size.width), int(size.height)) == (160, 128)
    assert round(duration_s * target_fps) == expected_frames


def test_temporal_encode_restarts_cleanly_at_hard_cut(tmp_path):
    """The real VT session accepts a held pre-cut period plus a restart."""
    target_fps = 60.0
    frames = [
        mx.zeros((128, 160, 3), dtype=mx.float16),
        mx.zeros((128, 160, 3), dtype=mx.float16),
        mx.ones((128, 160, 3), dtype=mx.float16),
        mx.ones((128, 160, 3), dtype=mx.float16),
    ]
    out = encode_video_videotoolbox(
        frames,
        tmp_path / "temporal-cut.mp4",
        fps=FPS,
        source_signal=ltx23_sdr_signal(width=160, height=128, fps=FPS),
        delivery=BT709_SDR_422_DELIVERY,
        target_fps=target_fps,
        verbose=False,
    )

    video, audio, duration_s, size = _probe_mp4(out)
    assert len(video) == 1
    assert len(audio) == 0
    assert (int(size.width), int(size.height)) == (160, 128)
    assert round(duration_s * target_fps) == 10
