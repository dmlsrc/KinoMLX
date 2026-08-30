"""Pure-helper tests for the VideoToolbox encode / vsr / writer modules.

These cover the small, side-effect-free helpers that need neither a GPU encode
session nor a downloadable VSR model: scale/profile selection, human-readable
size formatting, audio-shape normalization, frame peeking, and the HEVC
settings dict. The full hardware paths (encode_video_videotoolbox, AVWriter,
VsrSession, VtfrcSession) are integration-grade and left for a separate pass.

Only test_hevc_video_settings touches a real AVFoundation symbol table, so it
alone carries the requires_avfoundation marker; the rest are dependency-free and
run anywhere MLX imports.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import weakref
from fractions import Fraction
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.media.signals import (
    BT709_SDR_420_DELIVERY,
    BT709_SDR_422_DELIVERY,
    BT2020_HLG_DELIVERY,
    UnsupportedSignalError,
)
from kinomlx.models.ltx2.signals import SCENE_LINEAR_HDR_SIGNAL, ltx23_sdr_signal
from kinomlx.videotoolbox import encode, pixel_buffers, temporal, vsr, writer, yuv
from kinomlx.videotoolbox.errors import VideoToolboxUnavailableError


def test_yuv_compile_is_deferred_until_first_conversion() -> None:
    code = """
import mlx.core as mx

def fail(*_args, **_kwargs):
    raise AssertionError("mx.compile called during import")

mx.compile = fail
import kinomlx.videotoolbox.yuv
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize(
    ("source_signal", "delivery", "message"),
    [
        (SCENE_LINEAR_HDR_SIGNAL, BT709_SDR_420_DELIVERY, "cannot consume"),
        (
            ltx23_sdr_signal(width=64, height=64, fps=24.0),
            BT2020_HLG_DELIVERY,
            "cannot produce hlg",
        ),
    ],
)
def test_public_videotoolbox_terminal_rejects_hdr_before_frame_pull(
    source_signal,
    delivery,
    message,
) -> None:
    pulled = []

    def frames():
        pulled.append(True)
        yield mx.zeros((64, 64, 3), dtype=mx.float16)

    with pytest.raises(UnsupportedSignalError, match=message):
        encode.encode_video_videotoolbox(
            frames(),
            "unused.mp4",
            fps=24.0,
            source_signal=source_signal,
            delivery=delivery,
            n_source_frames=1,
            verbose=False,
        )

    assert pulled == []


def test_public_encoder_rejects_bad_cut_policy_before_frame_pull() -> None:
    pulled = []

    def frames():
        pulled.append(True)
        yield object()

    with pytest.raises(ValueError, match="unknown cut-detect mode"):
        encode.encode_video_videotoolbox(
            frames(),
            "unused.mp4",
            fps=24.0,
            source_signal=ltx23_sdr_signal(width=64, height=64, fps=24.0),
            delivery=BT709_SDR_420_DELIVERY,
            n_source_frames=1,
            cut_detect_mode="typo",
            verbose=False,
        )

    assert pulled == []


@pytest.mark.parametrize("n_source_frames", [None, 1])
def test_public_encoder_rejects_unbatched_mlx_frame_source(
    n_source_frames: int | None,
) -> None:
    with pytest.raises(ValueError, match=r"\(T,H,W,C\)"):
        encode.encode_video_videotoolbox(
            mx.zeros((64, 64, 3), dtype=mx.float16),
            "unused.mp4",
            fps=24.0,
            source_signal=ltx23_sdr_signal(width=64, height=64, fps=24.0),
            delivery=BT709_SDR_420_DELIVERY,
            n_source_frames=n_source_frames,
            verbose=False,
        )


# --------------------------------------------------------------------------- #
# vsr.scale_for_mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("mode", "scale"), [("fast", 2), ("balanced", 4), ("image", 4)])
def test_scale_for_mode(mode, scale):
    assert vsr.scale_for_mode(mode) == scale


def test_scale_for_mode_rejects_unknown():
    with pytest.raises(ValueError, match="unknown VSR spatial-mode"):
        vsr.scale_for_mode("bogus")


@pytest.mark.parametrize(
    ("spatial_mode", "target_fps", "expected"),
    [
        (None, None, "HEVC encode"),
        ("fast", None, "VSR fast 2x + HEVC encode"),
        ("balanced", None, "VSR balanced 4x + HEVC encode"),
        (None, 60.0, "VTFRC + HEVC encode"),
        ("image", 60.0, "VSR image 4x + VTFRC + HEVC encode"),
    ],
)
def test_output_phase_name_exposes_active_native_stages(
    spatial_mode,
    target_fps,
    expected,
) -> None:
    assert (
        encode._output_phase_name(
            vsr_spatial_mode=spatial_mode,
            source_fps=24.0,
            target_fps=target_fps,
        )
        == expected
    )


@pytest.mark.parametrize(("width", "height"), [(127, 256), (256, 127)])
def test_sub_128_balanced_falls_back_to_image(width, height, caplog):
    with caplog.at_level(logging.WARNING, logger=vsr.__name__):
        assert not vsr._balanced_uses_video_input(width, height, "balanced")
    assert "falling back" in caplog.text
    assert "below 128" in caplog.text

    assert vsr._balanced_uses_video_input(128, 128, "balanced")
    assert not vsr._balanced_uses_video_input(64, 64, "image")


def test_balanced_video_source_gets_bt709_matrix_attachment(monkeypatch):
    calls = []
    quartz = SimpleNamespace(
        kCVImageBufferYCbCrMatrixKey="matrix-key",
        kCVImageBufferYCbCrMatrix_ITU_R_709_2="bt709",
        kCVAttachmentMode_ShouldPropagate="propagate",
        CVBufferSetAttachment=lambda *args: calls.append(args),
    )
    monkeypatch.setattr(vsr, "Quartz", quartz)

    session = object.__new__(vsr.VsrSession)
    session._video_input = True
    session._tag_source_matrix("source-buffer")
    assert calls == [("source-buffer", "matrix-key", "bt709", "propagate")]

    session._video_input = False
    session._tag_source_matrix("image-buffer")
    assert len(calls) == 1


def test_vsr_reset_frame_history_drops_previous_pair() -> None:
    session = object.__new__(vsr.VsrSession)
    session._video_input = True
    session._prev_src_frame = "previous-source"
    session._prev_dst_frame = "previous-output"

    assert session.uses_frame_history
    session.reset_frame_history()

    assert session._prev_src_frame is None
    assert session._prev_dst_frame is None


def test_stateless_vsr_reports_no_frame_history() -> None:
    session = object.__new__(vsr.VsrSession)
    session._video_input = False

    assert not session.uses_frame_history


def test_balanced_vsr_video_input_allows_third_source_submission(monkeypatch):
    """The third submission must fit while VT still retains the prior pair."""

    class SourceBuffer:
        pass

    source_refs = []
    live_source_counts = []

    def pool_create_buffer(_pool, *, allocation_threshold):
        live_count = sum(ref() is not None for ref in source_refs)
        live_source_counts.append(live_count)
        if live_count >= allocation_threshold:
            return None
        buffer = SourceBuffer()
        source_refs.append(weakref.ref(buffer))
        return buffer

    class FakeFrame:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithBuffer_presentationTimeStamp_(self, buffer, _pts):
            self.buffer = buffer
            return self

    class FakeParameters:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
            self,
            source_frame,
            previous_frame,
            previous_output_frame,
            _optical_flow,
            _submission_mode,
            destination_frame,
        ):
            self.source_frame = source_frame
            self.previous_frame = previous_frame
            self.previous_output_frame = previous_output_frame
            self.destination_frame = destination_frame
            return self

    class RetainingProcessor:
        def __init__(self):
            self.last_parameters = None

        def processWithParameters_error_(self, parameters, _error):
            # VT keeps the submitted parameter graph alive until the next
            # sequential submission. Together with VsrSession's explicit
            # previous frame, this leaves two distinct sources live while the
            # third source is acquired.
            self.last_parameters = parameters
            return True, None

    fake_vt = SimpleNamespace(
        VTFrameProcessorFrame=FakeFrame,
        VTSuperResolutionScalerParameters=FakeParameters,
        VTSuperResolutionScalerParametersSubmissionModeSequential="sequential",
    )
    monkeypatch.setattr(vsr, "vt", fake_vt)
    monkeypatch.setattr(vsr._pb, "pool_create_buffer", pool_create_buffer)
    monkeypatch.setattr(vsr._pb, "upload_frame_to_buffer", lambda _frame, _buffer: None)
    monkeypatch.setattr(vsr._pb, "frame_pts", lambda index, _cadence: index)

    session = object.__new__(vsr.VsrSession)
    session._src_pool = "source-pool"
    session._video_input = True
    session._prev_src_frame = None
    session._prev_dst_frame = None
    session.mode = "balanced"
    session.cadence = Fraction(24, 1)
    session.processor = RetainingProcessor()
    session._make_dst_buffer = object
    session._tag_source_matrix = lambda _buffer: None

    outputs = [session.upscale_to_buffer("pixels", index) for index in range(3)]

    assert len(outputs) == 3
    assert live_source_counts == [0, 1, 2]


def test_hq_vsr_rejects_inputs_above_empirical_caps(monkeypatch):
    fake_hq = SimpleNamespace(
        isSupported=lambda: True,
        supportedScaleFactors=lambda: [4],
    )
    monkeypatch.setattr(
        vsr,
        "vt",
        SimpleNamespace(VTSuperResolutionScalerConfiguration=fake_hq),
    )

    with pytest.raises(VideoToolboxUnavailableError, match="max 1920x1080"):
        vsr._validate_combination(1921, 1080, 4, "balanced")


def test_native_stderr_disabled_path_performs_no_descriptor_operations(monkeypatch):
    monkeypatch.setattr(
        vsr,
        "_duplicate_stderr",
        lambda: pytest.fail("disabled path duplicated stderr"),
    )
    entered = False
    with vsr._suppress_native_stderr(enabled=False):
        entered = True
    assert entered


def test_native_stderr_open_failure_closes_saved_descriptor(monkeypatch):
    failure = OSError("cannot open /dev/null")
    closed = []
    monkeypatch.setattr(vsr, "_duplicate_stderr", lambda: 10)
    monkeypatch.setattr(
        vsr,
        "_open_devnull",
        lambda: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(vsr, "_close_fd", closed.append)

    with pytest.raises(OSError, match="cannot open") as caught, vsr._suppress_native_stderr():
        pytest.fail("body entered after setup failure")
    assert caught.value is failure
    assert closed == [10]


def test_native_stderr_preserves_body_failure_and_cleanup_order(monkeypatch):
    failure = RuntimeError("body failed")
    redirected = []
    closed = []
    monkeypatch.setattr(vsr, "_duplicate_stderr", lambda: 10)
    monkeypatch.setattr(vsr, "_open_devnull", lambda: 11)
    monkeypatch.setattr(vsr, "_redirect_stderr", redirected.append)
    monkeypatch.setattr(vsr, "_close_fd", closed.append)

    with pytest.raises(RuntimeError) as caught, vsr._suppress_native_stderr():
        raise failure
    assert caught.value is failure
    assert redirected == [11, 10]
    assert closed == [11, 10]


# --------------------------------------------------------------------------- #
# Exact video cadence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cadence",
    [Fraction(30000, 1001), Fraction(60000, 1001)],
)
def test_ntsc_frame_grid_does_not_accumulate_rounding_drift(cadence):
    scale = pixel_buffers.VIDEO_TIME_SCALE
    one_hour_index = round(3600 * cadence)
    for index in (0, 1, 2, 3, 1000, one_hour_index):
        expected = round(Fraction(index * scale) / cadence)
        assert pixel_buffers.frame_ticks(index, cadence) == expected

    exact_end = Fraction(one_hour_index, 1) / cadence
    encoded_end = Fraction(pixel_buffers.frame_ticks(one_hour_index, cadence), scale)
    assert abs(encoded_end - exact_end) <= Fraction(1, 2 * scale)


@pytest.mark.parametrize("cadence", [True, 0, -24, float("nan"), float("inf")])
def test_invalid_frame_cadence_is_rejected(cadence):
    with pytest.raises(ValueError, match="cadence"):
        pixel_buffers.frame_ticks(0, cadence)


# --------------------------------------------------------------------------- #
# VTFRC final-period drain
# --------------------------------------------------------------------------- #


def test_temporal_drain_emits_final_source_period(monkeypatch):
    frame_calls = []
    parameter_calls = []

    class FakeFrame:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithBuffer_presentationTimeStamp_(self, buffer, pts):
            frame_calls.append((buffer, pts))
            return (buffer, pts)

    class FakeParameters:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithSourceFrame_nextFrame_opticalFlow_interpolationPhase_submissionMode_destinationFrames_(
            self,
            *args,
        ):
            parameter_calls.append(args)
            return args

    fake_vt = SimpleNamespace(
        VTFrameProcessorFrame=FakeFrame,
        VTFrameRateConversionParameters=FakeParameters,
        VTFrameRateConversionParametersSubmissionModeSequential="sequential",
    )
    monkeypatch.setattr(temporal, "vt", fake_vt)
    monkeypatch.setattr(
        temporal._pb,
        "frame_pts",
        lambda index, cadence: (index, cadence),
    )

    session = object.__new__(temporal.VtfrcSession)
    session.source_cadence = Fraction(24, 1)
    session.target_cadence = Fraction(60, 1)
    session.source_fps = 24.0
    session.target_fps = 60.0
    session._prev_src_pb = "last-source"
    session._prev_src_index = 1
    session._next_target_index = 3
    session._make_dst_buffer = lambda: f"dst-{len(frame_calls)}"
    session.processor = SimpleNamespace(
        processWithParameters_error_=lambda _params, _error: (True, None),
    )

    outputs = list(session.drain())

    assert len(outputs) == 2
    assert frame_calls[0][0] == "last-source"
    assert frame_calls[1][0] == "last-source"
    assert parameter_calls[0][2] is None
    assert parameter_calls[0][3] == pytest.approx([0.2, 0.6])
    assert session._next_target_index == 5
    assert session._prev_src_pb is None


def test_temporal_cut_holds_previous_period_and_restarts_session(monkeypatch):
    frame_calls = []
    parameter_calls = []
    events = []
    destinations = []

    class FakeFrame:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithBuffer_presentationTimeStamp_(self, buffer, pts):
            frame_calls.append((buffer, pts))
            return (buffer, pts)

    class FakeParameters:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithSourceFrame_nextFrame_opticalFlow_interpolationPhase_submissionMode_destinationFrames_(
            self,
            *args,
        ):
            parameter_calls.append(args)
            return args

    class FreshProcessor:
        def startSessionWithConfiguration_error_(self, config, _error):
            events.append(("start", config))
            return True, None

    class FakeProcessorClass:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return FreshProcessor()

    class OldProcessor:
        def processWithParameters_error_(self, _params, _error):
            events.append(("process", "held-period"))
            return True, None

        def endSession(self):
            events.append(("end", "old-session"))

    fake_vt = SimpleNamespace(
        VTFrameProcessorFrame=FakeFrame,
        VTFrameRateConversionParameters=FakeParameters,
        VTFrameRateConversionParametersSubmissionModeSequential="sequential",
        VTFrameProcessor=FakeProcessorClass,
    )
    monkeypatch.setattr(temporal, "vt", fake_vt)
    monkeypatch.setattr(
        temporal._pb,
        "frame_pts",
        lambda index, cadence: (index, cadence),
    )

    session = object.__new__(temporal.VtfrcSession)
    session.source_cadence = Fraction(24, 1)
    session.target_cadence = Fraction(60, 1)
    session._prev_src_pb = "before-cut"
    session._prev_src_index = 0
    session._next_target_index = 0
    session.config = "temporal-config"
    session.processor = OldProcessor()

    def make_destination():
        destination = f"dst-{len(destinations)}"
        destinations.append(destination)
        return destination

    session._make_dst_buffer = make_destination

    outputs = list(session.feed_cut("after-cut", 1))

    assert outputs == ["dst-0", "dst-1", "dst-2"]
    assert parameter_calls[0][0][0] == "before-cut"
    assert parameter_calls[0][1][0] == "before-cut"
    assert parameter_calls[0][3] == pytest.approx([0.0, 0.4, 0.8])
    assert events == [
        ("process", "held-period"),
        ("end", "old-session"),
        ("start", "temporal-config"),
    ]
    assert session._prev_src_pb == "after-cut"
    assert session._prev_src_index == 1
    assert session._next_target_index == 3


def test_temporal_target_grid_uses_exact_half_open_intervals():
    session = object.__new__(temporal.VtfrcSession)
    session.source_cadence = Fraction(24000, 1001)
    session.target_cadence = Fraction(60000, 1001)
    session._next_target_index = 0

    emitted = []
    source_periods = 1000
    for source_index in range(source_periods):
        indices = session._target_indices_in_pair(source_index)
        emitted.extend(indices)
        if indices:
            session._next_target_index = indices[-1] + 1

    assert emitted == list(range(2500))


def test_temporal_rejects_unknown_mode_before_framework_setup():
    with pytest.raises(ValueError, match="unknown temporal mode"):
        temporal.VtfrcSession(160, 128, 24.0, 60.0, mode="typo")


def test_temporal_feed_rejects_nonconsecutive_source_indices():
    session = object.__new__(temporal.VtfrcSession)
    session._prev_src_pb = "previous"
    session._prev_src_index = 3
    with pytest.raises(ValueError, match="must be consecutive"):
        list(session.feed("next", 5))


def test_temporal_source_pool_exhaustion_never_falls_back_unbounded(monkeypatch):
    fallback_calls = []
    session = object.__new__(temporal.VtfrcSession)
    session._src_pool = "bounded-source-pool"
    session.in_w = 160
    session.in_h = 128
    session.src_attrs = {"format": "source"}
    monkeypatch.setattr(temporal._pb, "pool_create_buffer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        temporal._pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: fallback_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="bounded VTFRC source pool is exhausted"):
        session.make_source_buffer()

    assert fallback_calls == []


# --------------------------------------------------------------------------- #
# Explicit RGBAHalf encoder conversion
# --------------------------------------------------------------------------- #


def test_writer_converts_rgbahalf_to_yuv_before_append(monkeypatch):
    events = []
    monkeypatch.setattr(writer._pb, "read_rgbahalf_rgb", lambda pb: f"rgb:{pb}")
    monkeypatch.setattr(
        writer._pb,
        "pool_create_buffer",
        lambda pool, **_kwargs: f"yuv:{pool}",
    )
    monkeypatch.setattr(writer._pb, "frame_pts", lambda index, cadence: (index, cadence))
    monkeypatch.setattr(
        writer._yuv,
        "rgb_to_yuv422_10",
        lambda rgb, dst: events.append(("convert", rgb, dst)),
    )

    class FakeAdaptor:
        def pixelBufferPool(self):
            return "pool"

        def appendPixelBuffer_withPresentationTime_(self, pb, pts):
            events.append(("append", pb, pts))
            return True

    instance = object.__new__(writer.AVWriter)
    instance._yuv_feed = True
    instance.video_input = SimpleNamespace(isReadyForMoreMediaData=lambda: True)
    instance.writer = SimpleNamespace(status=lambda: 1, error=lambda: None)
    instance.adaptor = FakeAdaptor()
    instance.frame_count = 0
    instance.cadence = Fraction(30000, 1001)
    instance.label = "test"

    instance.append("rgba-half")

    assert events == [
        ("convert", "rgb:rgba-half", "yuv:pool"),
        ("append", "yuv:pool", (0, Fraction(30000, 1001))),
    ]
    assert instance.frame_count == 1


def test_native_cleanup_attempts_every_close_after_writer_failure() -> None:
    events = []

    class Target:
        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail

        def finish(self):
            events.append(f"finish:{self.name}")
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

        def close(self):
            events.append(f"close:{self.name}")

    primary = RuntimeError("frame processing failed")
    with pytest.raises(RuntimeError, match="frame processing failed") as caught:
        encode._close_native_chain(
            Target("writer", fail=True),
            Target("original"),
            Target("temporal"),
            Target("spatial"),
            primary_failure=primary,
        )

    assert caught.value is primary
    assert events == [
        "finish:writer",
        "finish:original",
        "close:temporal",
        "close:spatial",
    ]
    assert any("primary writer finish also failed" in note for note in primary.__notes__)


def test_pool_acquisition_always_passes_allocation_threshold(monkeypatch) -> None:
    calls = []
    fake_quartz = SimpleNamespace(
        kCVPixelBufferPoolAllocationThresholdKey="threshold",
        CVPixelBufferPoolCreatePixelBufferWithAuxAttributes=lambda allocator, pool, aux, out: (
            calls.append((allocator, pool, aux, out)) or (0, "buffer")
        ),
    )
    monkeypatch.setattr(pixel_buffers, "Quartz", fake_quartz)

    result = pixel_buffers.pool_create_buffer("pool", allocation_threshold=8)

    assert result == "buffer"
    assert calls == [(None, "pool", {"threshold": 8}, None)]


def test_native_sessions_drop_history_configuration_and_pools(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        pixel_buffers,
        "flush_pool",
        lambda pool: events.append(f"flush:{pool}"),
    )

    spatial = object.__new__(vsr.VsrSession)
    spatial.processor = SimpleNamespace(endSession=lambda: events.append("end:spatial"))
    spatial._prev_src_frame = "src-history"
    spatial._prev_dst_frame = "dst-history"
    spatial._src_pool = "src-pool"
    spatial._dst_pool = "dst-pool"
    spatial.src_attrs = {"source": 1}
    spatial.dst_attrs = {"destination": 1}
    spatial.config = "config"
    spatial.close()

    temporal_session = object.__new__(temporal.VtfrcSession)
    temporal_session.processor = SimpleNamespace(endSession=lambda: events.append("end:temporal"))
    temporal_session._prev_src_pb = "history"
    temporal_session._prev_src_index = 12
    temporal_session._next_target_index = 30
    temporal_session._src_pool = "temporal-source-pool"
    temporal_session._dst_pool = "temporal-pool"
    temporal_session.src_attrs = {"source": 1}
    temporal_session.dst_attrs = {"destination": 1}
    temporal_session.config = "config"
    temporal_session.close()

    assert events == [
        "end:spatial",
        "flush:src-pool",
        "flush:dst-pool",
        "end:temporal",
        "flush:temporal-source-pool",
        "flush:temporal-pool",
    ]
    assert spatial._prev_src_frame is None
    assert spatial._prev_dst_frame is None
    assert spatial._src_pool is None
    assert spatial._dst_pool is None
    assert spatial.src_attrs == {}
    assert spatial.dst_attrs == {}
    assert spatial.config is None
    assert temporal_session._prev_src_pb is None
    assert temporal_session._prev_src_index == -1
    assert temporal_session._next_target_index == 0
    assert temporal_session._src_pool is None
    assert temporal_session._dst_pool is None
    assert temporal_session.src_attrs == {}
    assert temporal_session.dst_attrs == {}
    assert temporal_session.config is None


def test_writer_pins_video_clock_and_propagates_producer_padding(
    monkeypatch,
    tmp_path,
):
    events = []
    adaptor_attrs = []

    class FakeWriter:
        def initWithURL_fileType_error_(self, _url, _file_type, _error):
            return self, None

        def canAddInput_(self, _input):
            return True

        def addInput_(self, _input):
            events.append("add-input")

        def setMovieTimeScale_(self, scale):
            events.append(("movie-scale", scale))

        def startWriting(self):
            events.append("start-writing")
            return True

        def startSessionAtSourceTime_(self, _time):
            events.append("start-session")

    class FakeWriterClass:
        @staticmethod
        def alloc():
            return FakeWriter()

    class FakeInput:
        def setExpectsMediaDataInRealTime_(self, _value):
            pass

        def setMediaTimeScale_(self, scale):
            events.append(("media-scale", scale))

    class FakeInputClass:
        @staticmethod
        def assetWriterInputWithMediaType_outputSettings_(_media_type, _settings):
            return FakeInput()

    class FakeAdaptorClass:
        @staticmethod
        def assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            _input,
            attrs,
        ):
            adaptor_attrs.append(attrs)
            return SimpleNamespace()

    fake_av = SimpleNamespace(
        AVAssetWriter=FakeWriterClass,
        AVAssetWriterInput=FakeInputClass,
        AVAssetWriterInputPixelBufferAdaptor=FakeAdaptorClass,
        AVFileTypeMPEG4="mp4",
        AVMediaTypeVideo="video",
        AVVideoCodecKey="codec",
        AVVideoCodecTypeHEVC="hevc",
        AVVideoWidthKey="width",
        AVVideoHeightKey="height",
        AVVideoColorPropertiesKey="color",
        AVVideoColorPrimariesKey="primaries",
        AVVideoColorPrimaries_ITU_R_709_2="bt709",
        AVVideoTransferFunctionKey="transfer",
        AVVideoTransferFunction_ITU_R_709_2="bt709",
        AVVideoYCbCrMatrixKey="matrix",
        AVVideoYCbCrMatrix_ITU_R_709_2="bt709",
        AVVideoCompressionPropertiesKey="compression",
        AVVideoProfileLevelKey="profile",
        AVVideoQualityKey="quality",
    )
    fake_quartz = SimpleNamespace(
        kCVPixelBufferPixelFormatTypeKey="pixel-format",
        kCVPixelBufferWidthKey="width",
        kCVPixelBufferHeightKey="height",
        kCVPixelBufferIOSurfacePropertiesKey="iosurface",
        kCVPixelBufferExtendedPixelsLeftKey="left",
        kCVPixelBufferExtendedPixelsRightKey="right",
        kCVPixelBufferExtendedPixelsTopKey="top",
        kCVPixelBufferExtendedPixelsBottomKey="bottom",
    )
    monkeypatch.setattr(writer, "av", fake_av)
    monkeypatch.setattr(writer, "Quartz", fake_quartz)
    monkeypatch.setattr(
        writer,
        "Foundation",
        SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda path: path)),
    )
    monkeypatch.setattr(
        writer,
        "CoreMedia",
        SimpleNamespace(CMTimeMake=lambda value, scale: (value, scale)),
    )

    instance = writer.AVWriter(
        tmp_path / "clock.mp4",
        width=160,
        height=128,
        fps=Fraction(30000, 1001),
        source_pixel_format=pixel_buffers.PIX_NV12,
        delivery=BT709_SDR_420_DELIVERY,
        source_attrs={"right": 32, "bottom": 16, "ignored": 99},
    )

    assert ("media-scale", pixel_buffers.VIDEO_TIME_SCALE) in events
    assert ("movie-scale", pixel_buffers.VIDEO_TIME_SCALE) in events
    assert events.index(("movie-scale", pixel_buffers.VIDEO_TIME_SCALE)) < events.index(
        "start-writing"
    )
    assert instance.cadence == Fraction(30000, 1001)
    assert adaptor_attrs == [
        {
            "pixel-format": pixel_buffers.PIX_NV12,
            "width": 160,
            "height": 128,
            "iosurface": {},
            "right": 32,
            "bottom": 16,
        }
    ]


def test_writer_rejects_bad_cadence_before_touching_output(tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep-me")

    with pytest.raises(ValueError, match="cadence"):
        writer.AVWriter(
            output,
            width=160,
            height=128,
            fps=float("nan"),
            source_pixel_format=pixel_buffers.PIX_NV12,
            delivery=BT709_SDR_420_DELIVERY,
        )

    assert output.read_bytes() == b"keep-me"


def test_yuv_neutral_range_endpoints():
    rgb = mx.stack(
        [
            mx.zeros((2, 2, 3), dtype=mx.float32),
            mx.ones((2, 2, 3), dtype=mx.float32),
        ]
    ).reshape(4, 2, 3)
    luma, chroma = yuv._compute_planes(rgb)
    mx.eval(luma, chroma)

    assert mx.all(luma[:2] == (64 << 6)).item()
    assert mx.all(luma[2:] == (940 << 6)).item()
    assert mx.all(chroma == (512 << 6)).item()


def test_yuv_bt709_roundtrip_preserves_flat_colors():
    # Keep each scanline one color so 4:2:2 chroma decimation is lossless; the
    # remaining error is only 10-bit quantization.
    colors = mx.array(
        [[0.08, 0.22, 0.91], [0.93, 0.18, 0.07], [0.15, 0.82, 0.31]],
        dtype=mx.float32,
    )
    rgb = mx.broadcast_to(colors[:, None, :], (3, 8, 3))
    luma, chroma = yuv._compute_planes(rgb)
    y = ((luma >> 6).astype(mx.float32) - 64.0) / 876.0
    cb = ((chroma[:, 0::2] >> 6).astype(mx.float32) - 512.0) / 896.0
    cr = ((chroma[:, 1::2] >> 6).astype(mx.float32) - 512.0) / 896.0
    cb = mx.repeat(cb, 2, axis=1)
    cr = mx.repeat(cr, 2, axis=1)
    red = y + 2.0 * (1.0 - yuv._KR) * cr
    blue = y + 2.0 * (1.0 - yuv._KB) * cb
    green = (y - yuv._KR * red - yuv._KB * blue) / yuv._KG
    reconstructed = mx.stack([red, green, blue], axis=-1)

    assert float(mx.max(mx.abs(reconstructed - rgb)).item()) < 0.002


def test_yuv_chroma_decimation_is_cosited_121():
    """4:2:2 chroma is centered on the even luma columns decoders assume."""
    mx.random.seed(9)
    height, width = 4, 16
    rgb = mx.random.uniform(shape=(height, width, 3))
    _, chroma = yuv._compute_planes(rgb)
    cb = (chroma.reshape(height, width // 2, 2)[..., 0] >> 6).astype(mx.float32)

    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = yuv._KR * red + yuv._KG * green + yuv._KB * blue
    full_cb = ((blue - luma) / (2.0 * (1.0 - yuv._KB))) * 896.0 + 512.0
    left = mx.concatenate([full_cb[:, :1], full_cb[:, 1:-1:2]], axis=1)
    expected = mx.clip(
        mx.round((left + 2.0 * full_cb[:, 0::2] + full_cb[:, 1::2]) * 0.25),
        0,
        1023,
    )

    assert float(mx.max(mx.abs(cb - expected)).item()) <= 1.0

    # The filter weights sum to one, so constant chroma is preserved exactly.
    flat = mx.full((2, 8, 3), 0.25)
    _, flat_chroma = yuv._compute_planes(flat)
    flat_cb = (flat_chroma.reshape(2, 4, 2)[..., 0] >> 6).astype(mx.float32)
    assert float(mx.max(flat_cb) - mx.min(flat_cb)) == 0.0


# --------------------------------------------------------------------------- #
# encode._human_size
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1048576, "1.0 MiB"),
        (1073741824, "1.0 GiB"),
        (1099511627776, "1.0 TiB"),
    ],
)
def test_human_size(n, text):
    assert encode._human_size(n) == text


# --------------------------------------------------------------------------- #
# encode._native_hevc_profile
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("delivery", "expected"),
    [
        (BT709_SDR_420_DELIVERY, encode.HEVC_PROFILE_MAIN10),
        (BT709_SDR_422_DELIVERY, encode.HEVC_PROFILE_MAIN422_10),
    ],
)
def test_native_hevc_profile_comes_from_delivery(delivery, expected):
    assert encode._native_hevc_profile(delivery) == expected


# --------------------------------------------------------------------------- #
# encode._normalize_audio_for_track
# --------------------------------------------------------------------------- #


def test_normalize_audio_drops_batch_dim():
    out = encode._normalize_audio_for_track(mx.zeros((1, 2, 100)))
    assert tuple(int(x) for x in out.shape) == (2, 100)
    assert out.dtype == mx.float32


def test_normalize_audio_passes_through_2d():
    out = encode._normalize_audio_for_track(mx.zeros((2, 100)))
    assert tuple(int(x) for x in out.shape) == (2, 100)


def test_normalize_audio_rejects_1d():
    with pytest.raises(ValueError, match="must be"):
        encode._normalize_audio_for_track(mx.zeros((100,)))


# --------------------------------------------------------------------------- #
# encode._peek_frames
# --------------------------------------------------------------------------- #


def test_peek_frames_4d_array():
    first, it, total = encode._peek_frames(mx.zeros((5, 4, 4, 3), dtype=mx.uint8))
    assert tuple(int(x) for x in first.shape) == (4, 4, 3)
    assert total == 5
    # The returned iterator re-yields the peeked first frame, so all 5 survive.
    assert sum(1 for _ in it) == 5


def test_peek_frames_list():
    first, it, total = encode._peek_frames([mx.zeros((4, 4, 3))] * 3)
    assert total == 3
    assert sum(1 for _ in it) == 3


def test_peek_frames_generator_has_unknown_total():
    def gen():
        yield mx.zeros((4, 4, 3))
        yield mx.ones((4, 4, 3))

    first, it, total = encode._peek_frames(gen())
    assert total is None
    assert sum(1 for _ in it) == 2  # first frame chained back in


def test_peek_frames_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        encode._peek_frames([])


def test_source_frame_payload_must_match_immutable_signal() -> None:
    signal = ltx23_sdr_signal(width=4, height=3, fps=24.0)
    encode._validate_source_frame(mx.zeros((3, 4, 3), dtype=mx.float16), signal)

    with pytest.raises(ValueError, match="source frame shape"):
        encode._validate_source_frame(mx.zeros((3, 4, 4), dtype=mx.float16), signal)
    with pytest.raises(ValueError, match="source frame dtype"):
        encode._validate_source_frame(mx.zeros((3, 4, 3), dtype=mx.uint8), signal)


# --------------------------------------------------------------------------- #
# writer.hevc_video_settings (needs the real AVFoundation key table)
# --------------------------------------------------------------------------- #


@pytest.mark.requires_avfoundation
def test_hevc_video_settings_embeds_geometry_and_profile():
    import AVFoundation as av

    settings = writer.hevc_video_settings(640, 480, 0.7, BT709_SDR_420_DELIVERY)

    assert settings[av.AVVideoCodecKey] == av.AVVideoCodecTypeHEVC
    assert settings[av.AVVideoWidthKey] == 640
    assert settings[av.AVVideoHeightKey] == 480
    compression = settings[av.AVVideoCompressionPropertiesKey]
    assert compression[av.AVVideoProfileLevelKey] == encode.HEVC_PROFILE_MAIN10
    assert compression[av.AVVideoQualityKey] == pytest.approx(0.7)
