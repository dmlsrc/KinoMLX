"""Generation-output glue tests without invoking VideoToolbox."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.cli.config import OutputConfig
from kinomlx.cli.output import write_generation
from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.signals import (
    BT709_SDR_422_DELIVERY,
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    ExrDeliverySpec,
    UnsupportedSignalError,
)
from kinomlx.models.ltx2.runner import GenerationOutput
from kinomlx.models.ltx2.signals import (
    SCENE_LINEAR_HDR_SIGNAL,
    ltx23_sdr_signal,
    ltx_hdr_working_signal,
)
from kinomlx.output import ArtifactSet, OutputError
from kinomlx.reporting import RecordingReporter


def _generation(
    *,
    frame: object | None = None,
    audio_waveform: object | None = None,
    audio_sample_rate: int | None = None,
) -> GenerationOutput:
    signal = ltx23_sdr_signal(width=64, height=64, fps=24.0)
    resolved_frame = mx.zeros((64, 64, 3), dtype=mx.float16) if frame is None else frame
    frames = VideoFrameStream(
        lambda: iter((resolved_frame,)),
        spec=signal,
        frame_count=1,
    )
    return GenerationOutput(
        frames=frames,
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
    )


def test_write_generation_maps_resolved_output_config(tmp_path: Path) -> None:
    calls = []
    output = tmp_path / "out.mp4"

    def fake_encode(frames, path, **kwargs):
        calls.append((frames, path, kwargs))
        return Path(path)

    reporter = RecordingReporter()
    generation = _generation(audio_waveform="audio", audio_sample_rate=24000)
    result = write_generation(
        generation,
        OutputConfig(
            path=output,
            vsr_spatial_mode="off",
            target_fps=48.0,
            cut_detect_mode="hist",
            cut_detect_threshold=0.6,
            audio_onset_trim="80.5",
        ),
        fps=24.0,
        reporter=reporter,
        encoder=fake_encode,
    )
    assert result == output
    frames, path, kwargs = calls[0]
    assert frames is generation.frames
    assert path == output
    assert kwargs["fps"] == 24.0
    assert kwargs["target_fps"] == 48.0
    assert kwargs["vsr_spatial_mode"] is None
    assert kwargs["cut_detect_mode"] == "hist"
    assert kwargs["cut_detect_threshold"] == pytest.approx(0.6)
    assert kwargs["audio_onset_trim_mode"] == "force"
    assert kwargs["audio_onset_trim_ms"] == pytest.approx(80.5)
    assert kwargs["native_verbose"] is False
    assert kwargs["source_signal"] == generation.signal
    assert kwargs["delivery"] == BT709_SDR_422_DELIVERY
    assert kwargs["n_source_frames"] == 1
    assert kwargs["reporter"] is reporter
    assert reporter.events == []
    assert generation.frames.closed


def test_write_generation_derives_opt_in_vae_frame_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import images

    output = tmp_path / "result.mp4"

    def save_image(_image: mx.array, path: Path) -> Path:
        path.write_bytes(b"png")
        return path

    def fake_encode(frames, path, **_kwargs):
        assert len(list(frames)) == 1
        return Path(path)

    monkeypatch.setattr(images, "save_image", save_image)
    generation = _generation()
    assert (
        write_generation(
            generation,
            OutputConfig(path=output, save_vae_frames=True),
            fps=24.0,
            encoder=fake_encode,
        )
        == output
    )
    directory = tmp_path / "result_vae_frames"
    assert (directory / "frame_000000.png").is_file()
    assert (directory / "manifest.json").is_file()


@pytest.mark.parametrize("failure", [RuntimeError("unavailable"), ValueError("bad frame")])
def test_write_generation_wraps_operational_encoder_failure(
    tmp_path: Path,
    failure: Exception,
) -> None:
    def fail(*_args, **_kwargs):
        raise failure

    reporter = RecordingReporter()
    with pytest.raises(OutputError, match=str(failure)):
        write_generation(
            _generation(),
            OutputConfig(path=tmp_path / "out.mp4"),
            fps=24.0,
            reporter=reporter,
            encoder=fail,
        )
    assert reporter.events == []


def test_write_generation_does_not_swallow_process_control(tmp_path: Path) -> None:
    failure = SystemExit("stop now")

    def fail(*_args, **_kwargs):
        raise failure

    with pytest.raises(SystemExit) as caught:
        write_generation(
            _generation(),
            OutputConfig(path=tmp_path / "out.mp4"),
            fps=24.0,
            encoder=fail,
        )
    assert caught.value is failure


def test_writer_failure_closes_partly_consumed_stream(tmp_path: Path) -> None:
    events = []
    signal = ltx23_sdr_signal(width=64, height=64, fps=24.0)

    def produce():
        try:
            yield mx.zeros((64, 64, 3), dtype=mx.float16)
            yield mx.zeros((64, 64, 3), dtype=mx.float16)
        finally:
            events.append("source-close")

    stream = VideoFrameStream(produce, spec=signal, frame_count=2)

    def fail(frames, *_args, **_kwargs):
        next(iter(frames))
        raise ValueError("writer failed")

    with pytest.raises(OutputError, match="writer failed"):
        write_generation(
            GenerationOutput(frames=stream),
            OutputConfig(path=tmp_path / "out.mp4"),
            fps=24.0,
            encoder=fail,
        )

    assert stream.closed
    assert events == ["source-close"]


def test_write_generation_requires_a_path() -> None:
    with pytest.raises(OutputError, match="output path is required"):
        write_generation(
            _generation(),
            OutputConfig(),
            fps=24.0,
        )


def test_write_generation_rejects_hdr_before_pulling_or_loading_encoder(tmp_path: Path) -> None:
    pulled = []
    stream = VideoFrameStream(
        lambda: pulled.append(True) or iter(("frame",)),
        spec=SCENE_LINEAR_HDR_SIGNAL,
        frame_count=1,
    )

    with pytest.raises(UnsupportedSignalError, match="SDR terminal cannot consume"):
        write_generation(
            GenerationOutput(frames=stream),
            OutputConfig(path=tmp_path / "out.mp4"),
            fps=24.0,
            encoder=lambda *_args, **_kwargs: pytest.fail("encoder must not open"),
        )

    assert pulled == []
    assert stream.closed


@pytest.mark.parametrize(
    ("authoring", "primaries", "transfer"),
    [
        ("SRGB_LINEAR", ColorPrimaries.REC709, ColorTransfer.LINEAR),
        ("ACESCG", ColorPrimaries.ACESCG, ColorTransfer.LINEAR),
        ("ACESCCT", ColorPrimaries.ACESCG, ColorTransfer.ACESCCT),
    ],
)
def test_write_generation_maps_hdr_authoring_to_dual_sink_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authoring: str,
    primaries: ColorPrimaries,
    transfer: ColorTransfer,
) -> None:
    import kinomlx.cli.output as output_module

    signal = ltx_hdr_working_signal(
        transfer=ColorTransfer.ACESCCT,
        width=64,
        height=64,
        fps=24.0,
    )
    generation = GenerationOutput(
        frames=VideoFrameStream(
            lambda: iter((mx.zeros((64, 64, 3), dtype=mx.float32),)),
            spec=signal,
            frame_count=1,
        )
    )
    captured = {}

    class Sink:
        def __init__(self, *, path: Path, **_kwargs) -> None:
            self.path = path
            captured["sink_kwargs"] = _kwargs

        def write(self, received, plan):
            captured["generation"] = received
            captured["plan"] = plan
            received.close()
            return ArtifactSet(video=self.path, exr_frames=tmp_path / "out_exr")

    monkeypatch.setattr(output_module, "HDRGenerationSink", Sink)
    output = tmp_path / "out.mp4"
    assert (
        write_generation(
            generation,
            OutputConfig(path=output, save_hdr_heic_frames=True),
            fps=24.0,
            hdr_authoring=authoring,
        )
        == output
    )
    plan = captured["plan"]
    assert plan.source == signal
    assert len(plan.deliveries) == 2
    assert isinstance(plan.deliveries[0], ExrDeliverySpec)
    assert plan.deliveries[0].primaries is primaries
    assert plan.deliveries[0].transfer is transfer
    assert plan.deliveries[1] == BT2020_HLG_DELIVERY
    assert captured["sink_kwargs"]["heic_directory"] == tmp_path / "out_heic"


@pytest.mark.parametrize("value", ["invalid", "-1", "nan", "inf"])
def test_output_config_rejects_invalid_audio_onset_policy(value: str) -> None:
    with pytest.raises(ValueError, match="audio-onset-trim"):
        OutputConfig(audio_onset_trim=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cut_detect_mode": "typo"},
        {"cut_detect_threshold": -0.1},
        {"cut_detect_threshold": float("nan")},
    ],
)
def test_output_config_rejects_invalid_cut_detection_policy(kwargs) -> None:
    with pytest.raises(ValueError, match="cut_detect"):
        OutputConfig(**kwargs)
