"""Typed generation-sink behavior at the public terminal boundary."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.signals import (
    BT709_SDR_420_DELIVERY,
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    ExrDeliverySpec,
    ExrSampleType,
    OutputColorPlan,
    UnsupportedSignalError,
)
from kinomlx.models.ltx2.runner import GenerationOutput
from kinomlx.models.ltx2.signals import (
    SCENE_LINEAR_HDR_SIGNAL,
    ltx23_sdr_signal,
    ltx_hdr_working_signal,
)
from kinomlx.output import (
    ArtifactSet,
    GenerationSink,
    HDRGenerationSink,
    OutputError,
    VideoToolboxGenerationSink,
)


def _generation() -> GenerationOutput:
    signal = ltx23_sdr_signal(width=2, height=2, fps=24.0)
    return GenerationOutput(
        frames=VideoFrameStream(
            lambda: iter((mx.zeros((2, 2, 3), dtype=mx.float16),)),
            spec=signal,
            frame_count=1,
        )
    )


def test_videotoolbox_sink_consumes_an_explicit_output_color_plan(tmp_path: Path) -> None:
    generation = _generation()
    output = tmp_path / "output.mp4"
    calls = []

    def encode(frames, path, **kwargs):
        calls.append((frames, path, kwargs))
        return Path(path)

    sink: GenerationSink = VideoToolboxGenerationSink(
        path=output,
        fps=24.0,
        encoder=encode,
    )
    plan = OutputColorPlan(
        source=generation.signal,
        deliveries=(BT709_SDR_420_DELIVERY,),
    )

    artifacts = sink.write(generation, plan)

    assert artifacts == ArtifactSet(video=output)
    assert calls[0][0] is generation.frames
    assert calls[0][2]["source_signal"] is plan.source
    assert calls[0][2]["delivery"] is BT709_SDR_420_DELIVERY
    assert generation.frames.closed


def test_sink_rejects_a_relabelled_source_before_opening_encoder(tmp_path: Path) -> None:
    generation = _generation()
    sink = VideoToolboxGenerationSink(
        path=tmp_path / "output.mp4",
        fps=24.0,
        encoder=lambda *_args, **_kwargs: pytest.fail("encoder must not open"),
    )

    with pytest.raises(UnsupportedSignalError, match="does not match"):
        sink.write(
            generation,
            OutputColorPlan(
                source=SCENE_LINEAR_HDR_SIGNAL,
                deliveries=(BT709_SDR_420_DELIVERY,),
            ),
        )

    assert generation.frames.closed


def test_sink_dumps_vae_frames_before_the_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import images

    generation = _generation()
    output = tmp_path / "output.mp4"
    directory = tmp_path / "output_vae_frames"
    events: list[str] = []

    def save_image(_image: mx.array, path: Path) -> Path:
        events.append(f"save:{path.name}")
        path.write_bytes(b"png")
        return path

    def encode(frames, path, **_kwargs):
        events.append("encode:start")
        assert len(list(frames)) == 1
        events.append("encode:end")
        return Path(path)

    monkeypatch.setattr(images, "save_image", save_image)
    sink = VideoToolboxGenerationSink(
        path=output,
        fps=24.0,
        encoder=encode,
        vae_frame_directory=directory,
    )
    plan = OutputColorPlan(
        source=generation.signal,
        deliveries=(BT709_SDR_420_DELIVERY,),
    )

    artifacts = sink.write(generation, plan)

    assert artifacts == ArtifactSet(video=output, vae_frames=directory)
    assert events == ["encode:start", "save:frame_000000.png", "encode:end"]
    assert (directory / "manifest.json").is_file()
    assert generation.frames.closed


def test_hdr_sink_fans_one_working_frame_to_exr_and_hlg_transactionally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import exr, heic, hlg, writer

    signal = ltx_hdr_working_signal(
        transfer=ColorTransfer.ACESCCT,
        width=2,
        height=1,
        fps=24.0,
    )
    working = mx.array(
        [[[0.5547945, 0.5547945, 0.5547945], [0.7594157, 0.7594157, 0.7594157]]],
        dtype=mx.float32,
    )
    generation = GenerationOutput(
        frames=VideoFrameStream(lambda: iter((working,)), spec=signal, frame_count=1)
    )
    output = tmp_path / "output.mp4"
    exr_seen: list[mx.array] = []
    heic_seen: list[tuple[mx.array, ColorPrimaries]] = []
    hlg_seen: list[mx.array] = []

    class FakeWriter:
        def __init__(self, path: Path, **_kwargs) -> None:
            self.path = path
            self.adaptor = object()

        def append(self, payload: mx.array) -> None:
            hlg_seen.append(payload)

        def finish(self) -> None:
            self.path.write_bytes(b"mp4")

        def cancel(self) -> None:
            self.path.unlink(missing_ok=True)

    def save(frame: mx.array, path: Path, **_kwargs) -> Path:
        exr_seen.append(frame)
        path.write_bytes(b"exr")
        return path

    def convert(frame: mx.array, _adaptor: object, **_kwargs) -> mx.array:
        return frame

    def save_heic(
        frame: mx.array,
        path: Path,
        *,
        primaries: ColorPrimaries,
        **_kwargs,
    ) -> Path:
        heic_seen.append((frame, primaries))
        path.write_bytes(b"heic")
        return path

    monkeypatch.setattr(writer, "AVWriter", FakeWriter)
    monkeypatch.setattr(exr, "save_exr_frame", save)
    monkeypatch.setattr(heic, "save_pq_heic_frame", save_heic)
    monkeypatch.setattr(hlg, "make_hlg_pixel_buffer", convert)
    exr_delivery = ExrDeliverySpec(
        primaries=ColorPrimaries.ACESCG,
        transfer=ColorTransfer.ACESCCT,
        sample_type=ExrSampleType.FLOAT16,
        color_space_tag="ACEScct",
    )
    plan = OutputColorPlan(
        source=signal,
        deliveries=(exr_delivery, BT2020_HLG_DELIVERY),
    )

    artifacts = HDRGenerationSink(
        path=output,
        fps=24.0,
        heic_directory=tmp_path / "output_heic",
    ).write(generation, plan)

    assert artifacts.video == output
    assert artifacts.exr_frames == tmp_path / "output_exr"
    assert output.read_bytes() == b"mp4"
    assert (artifacts.exr_frames / "frame_00000.exr").read_bytes() == b"exr"
    assert artifacts.heic_frames == tmp_path / "output_heic"
    assert (artifacts.heic_frames / "frame_00000.heic").read_bytes() == b"heic"
    assert (artifacts.heic_frames / "manifest.json").is_file()
    assert mx.array_equal(exr_seen[0], working)
    assert heic_seen[0][1] is ColorPrimaries.ACESCG
    assert float(heic_seen[0][0][0, 0, 0].item()) == pytest.approx(1.0, rel=2e-5)
    assert float(heic_seen[0][0][0, 1, 0].item()) == pytest.approx(12.0, rel=2e-5)
    assert float(hlg_seen[0][0, 0, 0].item()) == pytest.approx(1.0, rel=2e-5)
    assert float(hlg_seen[0][0, 1, 0].item()) == pytest.approx(12.0, rel=2e-5)
    assert generation.frames.closed


def test_hdr_sink_fanout_stays_frame_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import exr, hlg, writer

    signal = ltx_hdr_working_signal(
        transfer=ColorTransfer.LOGC3,
        width=2,
        height=1,
        fps=24.0,
    )
    frames = tuple(mx.full((1, 2, 3), value, dtype=mx.float32) for value in (0.2, 0.4, 0.6))
    generation = GenerationOutput(
        frames=VideoFrameStream(lambda: iter(frames), spec=signal, frame_count=3)
    )
    events: list[str] = []

    class FakeWriter:
        def __init__(self, path: Path, **_kwargs) -> None:
            self.path = path
            self.adaptor = object()

        def append(self, _payload: mx.array) -> None:
            events.append("append")

        def finish(self) -> None:
            events.append("finish")
            self.path.write_bytes(b"mp4")

        def cancel(self) -> None:
            events.append("cancel")

    def save(frame: mx.array, path: Path, **_kwargs) -> Path:
        events.append(f"exr:{float(frame[0, 0, 0].item()):.1f}")
        path.write_bytes(b"exr")
        return path

    def convert(frame: mx.array, _adaptor: object, **_kwargs) -> mx.array:
        events.append("hlg")
        return frame

    monkeypatch.setattr(writer, "AVWriter", FakeWriter)
    monkeypatch.setattr(exr, "save_exr_frame", save)
    monkeypatch.setattr(hlg, "make_hlg_pixel_buffer", convert)
    plan = OutputColorPlan(
        source=signal,
        deliveries=(
            ExrDeliverySpec(
                primaries=ColorPrimaries.REC709,
                transfer=ColorTransfer.LOGC3,
                sample_type=ExrSampleType.FLOAT16,
                color_space_tag="LogC3",
            ),
            BT2020_HLG_DELIVERY,
        ),
    )

    HDRGenerationSink(path=tmp_path / "bounded.mp4", fps=24.0).write(generation, plan)

    assert events == [
        "exr:0.2",
        "hlg",
        "append",
        "exr:0.4",
        "hlg",
        "append",
        "exr:0.6",
        "hlg",
        "append",
        "finish",
    ]
    assert generation.frames.closed


def test_hdr_sink_cancels_and_removes_transaction_on_frame_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import exr, heic, hlg, writer

    signal = ltx_hdr_working_signal(
        transfer=ColorTransfer.ACESCCT,
        width=2,
        height=1,
        fps=24.0,
    )
    working = mx.full((1, 2, 3), 0.5, dtype=mx.float32)
    generation = GenerationOutput(
        frames=VideoFrameStream(
            lambda: iter((working, working)),
            spec=signal,
            frame_count=2,
        )
    )
    canceled = False

    class FakeWriter:
        def __init__(self, path: Path, **_kwargs) -> None:
            self.path = path
            self.adaptor = object()

        def append(self, _payload: mx.array) -> None:
            pass

        def finish(self) -> None:
            pytest.fail("failed fan-out must not finish the writer")

        def cancel(self) -> None:
            nonlocal canceled
            canceled = True

    calls = 0

    def convert(frame: mx.array, _adaptor: object, **_kwargs) -> mx.array:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected conversion failure")
        return frame

    def save(_frame: mx.array, path: Path, **_kwargs) -> Path:
        path.write_bytes(b"exr")
        return path

    def save_heic(_frame: mx.array, path: Path, **_kwargs) -> Path:
        path.write_bytes(b"heic")
        return path

    monkeypatch.setattr(writer, "AVWriter", FakeWriter)
    monkeypatch.setattr(exr, "save_exr_frame", save)
    monkeypatch.setattr(heic, "save_pq_heic_frame", save_heic)
    monkeypatch.setattr(hlg, "make_hlg_pixel_buffer", convert)
    output = tmp_path / "failed.mp4"
    plan = OutputColorPlan(
        source=signal,
        deliveries=(
            ExrDeliverySpec(
                primaries=ColorPrimaries.ACESCG,
                transfer=ColorTransfer.ACESCCT,
                sample_type=ExrSampleType.FLOAT16,
                color_space_tag="ACEScct",
            ),
            BT2020_HLG_DELIVERY,
        ),
    )

    with pytest.raises(OutputError, match="injected conversion failure"):
        HDRGenerationSink(
            path=output,
            fps=24.0,
            heic_directory=tmp_path / "failed_heic",
        ).write(generation, plan)

    assert canceled
    assert not output.exists()
    assert not (tmp_path / "failed_exr").exists()
    assert not (tmp_path / "failed_heic").exists()
    assert not tuple(tmp_path.glob(".failed-hdr-*"))
    assert generation.frames.closed


def test_hdr_sink_refuses_to_overwrite_audio_sidecar_before_opening_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kinomlx.videotoolbox import writer

    signal = ltx_hdr_working_signal(
        transfer=ColorTransfer.ACESCCT,
        width=2,
        height=1,
        fps=24.0,
    )
    generation = GenerationOutput(
        frames=VideoFrameStream(
            lambda: iter((mx.full((1, 2, 3), 0.5, dtype=mx.float32),)),
            spec=signal,
            frame_count=1,
        ),
        audio_waveform=mx.zeros((2, 480), dtype=mx.float32),
        audio_sample_rate=48_000,
    )
    output = tmp_path / "audio.mp4"
    output.with_suffix(".wav").write_bytes(b"existing")
    monkeypatch.setattr(
        writer,
        "AVWriter",
        lambda *_args, **_kwargs: pytest.fail("writer must not open"),
    )
    plan = OutputColorPlan(
        source=signal,
        deliveries=(
            ExrDeliverySpec(
                primaries=ColorPrimaries.ACESCG,
                transfer=ColorTransfer.ACESCCT,
                sample_type=ExrSampleType.FLOAT16,
                color_space_tag="ACEScct",
            ),
            BT2020_HLG_DELIVERY,
        ),
    )

    with pytest.raises(OutputError, match="audio sidecar already exists"):
        HDRGenerationSink(
            path=output,
            fps=24.0,
            save_audio_sidecar=True,
        ).write(generation, plan)

    assert output.with_suffix(".wav").read_bytes() == b"existing"
    assert generation.frames.closed
