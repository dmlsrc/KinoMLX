"""AVAssetWriter wrapper: HEVC video + optional ALAC/AAC audio, no ffmpeg.

The writer takes a stream of CVPixelBuffers and encodes them as HEVC Main10
4:2:0 or Main42210 4:2:2 10-bit BT.709 at the target fps. NV12 producers can
feed the adaptor pool directly. RGBAHalf producers are converted explicitly
to 10-bit BT.709 4:2:2 so color does not depend on producer-specific IOSurface
metadata. Audio (if attached) is pulled by AVAssetWriter on a dedicated
dispatch queue so it does not stall the video append loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

import AVFoundation as av
import CoreMedia
import Foundation
import libdispatch
import Quartz

from kinomlx.media.signals import (
    ChromaSubsampling,
    ColorTransfer,
    EncodedVideoDeliverySpec,
    VideoCodecProfile,
    validate_encoded_delivery,
)

from . import pixel_buffers as _pb
from . import yuv as _yuv
from .audio import AudioTrack, audio_writer_settings

_log = logging.getLogger(__name__)

# HEVC profile identifiers (Apple-stable strings; not exposed as PyObjC consts)
HEVC_PROFILE_MAIN10 = "HEVC_Main10_AutoLevel"  # 4:2:0 10-bit
HEVC_PROFILE_MAIN422_10 = "HEVC_Main42210_AutoLevel"  # 4:2:2 10-bit (Range Extensions)


class _AssetWriter(Protocol):
    def status(self) -> int: ...

    def error(self) -> object | None: ...

    def endSessionAtSourceTime_(self, end_time: object) -> None: ...

    def finishWritingWithCompletionHandler_(self, callback: Callable[[], None]) -> None: ...

    def cancelWriting(self) -> None: ...


class _WriterInput(Protocol):
    def isReadyForMoreMediaData(self) -> bool: ...

    def markAsFinished(self) -> None: ...

    def appendSampleBuffer_(self, sample_buffer: object) -> bool: ...

    def requestMediaDataWhenReadyOnQueue_usingBlock_(
        self,
        queue: object,
        callback: Callable[[], None],
    ) -> None: ...


def hevc_video_settings(
    width: int,
    height: int,
    quality: float,
    delivery: EncodedVideoDeliverySpec,
) -> dict[object, object]:
    """AVAssetWriterInput output settings for HEVC at the given size + profile."""
    delivery = validate_encoded_delivery(delivery)
    profile = (
        HEVC_PROFILE_MAIN10
        if delivery.profile is VideoCodecProfile.MAIN10
        else HEVC_PROFILE_MAIN422_10
    )
    hdr = delivery.transfer is ColorTransfer.HLG
    color_properties = (
        {
            av.AVVideoColorPrimariesKey: av.AVVideoColorPrimaries_ITU_R_2020,
            av.AVVideoTransferFunctionKey: av.AVVideoTransferFunction_ITU_R_2100_HLG,
            av.AVVideoYCbCrMatrixKey: av.AVVideoYCbCrMatrix_ITU_R_2020,
        }
        if hdr
        else {
            av.AVVideoColorPrimariesKey: av.AVVideoColorPrimaries_ITU_R_709_2,
            av.AVVideoTransferFunctionKey: av.AVVideoTransferFunction_ITU_R_709_2,
            av.AVVideoYCbCrMatrixKey: av.AVVideoYCbCrMatrix_ITU_R_709_2,
        }
    )
    return {
        av.AVVideoCodecKey: av.AVVideoCodecTypeHEVC,
        av.AVVideoWidthKey: width,
        av.AVVideoHeightKey: height,
        av.AVVideoColorPropertiesKey: color_properties,
        av.AVVideoCompressionPropertiesKey: {
            av.AVVideoProfileLevelKey: profile,
            av.AVVideoQualityKey: quality,
        },
    }


class AVWriter:
    """AVAssetWriter wrapping a HEVC video input + optional audio input.

    Construction kicks off `startWriting` + `startSessionAtSourceTime`. If
    `audio_track` is supplied, an audio AVAssetWriterInput is added and a
    GCD callback is scheduled to pull samples from the track as the encoder
    consumes them.

    Per-frame API:
        writer.append(pb)           # pixel buffer in the configured source format
    Finalize:
        writer.finish()             # waits for audio drain + finishWriting
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        source_pixel_format: int,
        delivery: EncodedVideoDeliverySpec,
        quality: float = 0.65,
        label: str = "video",
        audio_track: AudioTrack | None = None,
        audio_codec: str = "alac",
        source_attrs: Mapping[object, object] | None = None,
    ) -> None:
        delivery = validate_encoded_delivery(delivery)
        cadence = _pb.rational_cadence(fps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        url = Foundation.NSURL.fileURLWithPath_(str(output_path))
        writer, err = av.AVAssetWriter.alloc().initWithURL_fileType_error_(
            url,
            av.AVFileTypeMPEG4,
            None,
        )
        if writer is None:
            raise RuntimeError(f"AVAssetWriter init failed for {output_path}: {err}")
        native_writer = cast(_AssetWriter, writer)

        # Video input + pixel buffer adaptor ---------------------------------
        video_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
            av.AVMediaTypeVideo,
            hevc_video_settings(width, height, quality, delivery),
        )
        if video_input is None:
            raise RuntimeError(f"AVAssetWriter could not create video input for {output_path}")
        native_video_input = cast(_WriterInput, video_input)
        video_input.setExpectsMediaDataInRealTime_(False)
        video_input.setMediaTimeScale_(_pb.VIDEO_TIME_SCALE)

        # RGBAHalf inputs take an explicit BT.709 RGB-to-YUV path. AssetWriter's
        # internal conversion depends on producer-specific IOSurface metadata
        # and otherwise gives uploaded frames a path-dependent color shift.
        self._yuv_feed = source_pixel_format == _pb.PIX_RGBAHALF
        if self._yuv_feed and delivery.chroma is not ChromaSubsampling.YUV422:
            raise ValueError("RGBAHalf writer feed currently requires BT.709 4:2:2 delivery")
        adaptor_format = _yuv.PIX_422YCBCR10_VIDEO if self._yuv_feed else source_pixel_format
        src_attrs: dict[object, object] = {
            Quartz.kCVPixelBufferPixelFormatTypeKey: adaptor_format,
            Quartz.kCVPixelBufferWidthKey: width,
            Quartz.kCVPixelBufferHeightKey: height,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        }
        # When this adaptor pool is also a VT producer's destination pool,
        # retain all required padding. Missing extended-pixel attributes makes
        # VTFrameProcessor reject otherwise valid geometries with status -19730.
        if source_attrs is not None:
            for key in (
                Quartz.kCVPixelBufferExtendedPixelsLeftKey,
                Quartz.kCVPixelBufferExtendedPixelsRightKey,
                Quartz.kCVPixelBufferExtendedPixelsTopKey,
                Quartz.kCVPixelBufferExtendedPixelsBottomKey,
            ):
                if key in source_attrs:
                    src_attrs[key] = source_attrs[key]
        adaptor = av.AVAssetWriterInputPixelBufferAdaptor.assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            video_input,
            src_attrs,
        )
        if adaptor is None:
            raise RuntimeError(
                f"AVAssetWriter could not create pixel-buffer adaptor for {output_path}"
            )
        native_adaptor = cast(_pb.PixelBufferAdaptor, adaptor)
        if not writer.canAddInput_(video_input):
            raise RuntimeError(f"AVAssetWriter cannot add video input for {output_path}")
        writer.addInput_(video_input)

        # Optional audio input -----------------------------------------------
        native_audio_input: _WriterInput | None = None
        audio_state: tuple[AudioTrack, _WriterInput] | None = None
        if audio_track is not None:
            audio_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
                av.AVMediaTypeAudio,
                audio_writer_settings(audio_codec, audio_track.sample_rate, audio_track.channels),
            )
            if audio_input is None:
                raise RuntimeError(
                    f"AVAssetWriter could not create audio input ({audio_codec}) for {output_path}"
                )
            native_audio_input = cast(_WriterInput, audio_input)
            audio_input.setExpectsMediaDataInRealTime_(False)
            if not writer.canAddInput_(audio_input):
                raise RuntimeError(
                    f"AVAssetWriter cannot add audio input ({audio_codec}) for {output_path}"
                )
            writer.addInput_(audio_input)
            audio_state = (audio_track, native_audio_input)

        # Start the writer ---------------------------------------------------
        writer.setMovieTimeScale_(_pb.VIDEO_TIME_SCALE)
        if not writer.startWriting():
            raise RuntimeError(f"AVAssetWriter.startWriting failed: {writer.error()}")
        writer.startSessionAtSourceTime_(CoreMedia.CMTimeMake(0, _pb.VIDEO_TIME_SCALE))

        self.writer = native_writer
        self.video_input = native_video_input
        self.audio_input = native_audio_input
        self.adaptor = native_adaptor
        self.cadence = cadence
        self.fps = float(self.cadence)
        self.label = label
        self.path = output_path
        self.frame_count = 0
        self.audio_track = audio_track
        self._audio_codec = audio_codec
        self.delivery = delivery

        profile = (
            HEVC_PROFILE_MAIN10
            if delivery.profile is VideoCodecProfile.MAIN10
            else HEVC_PROFILE_MAIN422_10
        )

        audio_desc = f", audio={audio_codec}" if native_audio_input is not None else ""
        color_label = "BT.2020/HLG" if delivery.transfer is ColorTransfer.HLG else "BT.709"
        _log.info(
            f"[{label}] AVAssetWriter -> {output_path} "
            f"(HEVC {profile} {color_label} q={quality}{audio_desc})"
        )

        # Audio pump (GCD pull pattern) --------------------------------------
        self._audio_done = threading.Event()
        self._audio_progress = [0]
        self._audio_sample_count = audio_state[0].n_samples if audio_state is not None else 0
        if audio_state is not None:
            audio_track, audio_input = audio_state
            self._audio_queue = libdispatch.dispatch_queue_create(
                f"kinomlx.audio.{label}".encode(),
                None,
            )
            n_samples = audio_track.n_samples
            chunk_frames = max(4096, audio_track.sample_rate // 4)  # ~250 ms

            def pump() -> None:
                try:
                    while audio_input.isReadyForMoreMediaData():
                        pos = self._audio_progress[0]
                        if pos >= n_samples:
                            audio_input.markAsFinished()
                            self._audio_done.set()
                            return
                        end = min(pos + chunk_frames, n_samples)
                        sb = audio_track.make_sample_buffer(pos, end)
                        if sb is None or not audio_input.appendSampleBuffer_(sb):
                            self._audio_done.set()
                            raise RuntimeError(
                                f"[{label}] audio appendSampleBuffer failed at "
                                f"{pos}: {self.writer.error()}"
                            )
                        self._audio_progress[0] = end
                except Exception:
                    self._audio_done.set()
                    raise

            audio_input.requestMediaDataWhenReadyOnQueue_usingBlock_(
                self._audio_queue,
                pump,
            )
        else:
            self._audio_done.set()
            self._audio_queue = None

    # ------------------------------------------------------------------------
    # Internal: wait-with-status-check
    # ------------------------------------------------------------------------

    def _wait_for_ready(self, input_obj: _WriterInput, what: str) -> None:
        """Block until input_obj.isReadyForMoreMediaData(). Bail with a clean
        error if the writer enters Failed/Cancelled, or after 30 s of no
        progress (so a stuck writer surfaces as a visible failure, not a hang).
        """
        waited = 0.0
        while not input_obj.isReadyForMoreMediaData():
            status = self.writer.status()
            if status in (3, 4):  # Failed, Cancelled
                raise RuntimeError(
                    f"[{self.label}] writer entered status={status} while waiting on "
                    f"{what}: {self.writer.error()}"
                )
            time.sleep(0.001)
            waited += 0.001
            if waited > 30.0:
                raise RuntimeError(
                    f"[{self.label}] {what} input never became ready (waited 30s, status={status})"
                )

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def append(self, pb: object) -> None:
        """Append one video frame at the next PTS (frame_count/fps)."""
        self._wait_for_ready(self.video_input, "video")
        if self._yuv_feed:
            rgb = _pb.read_rgbahalf_rgb(pb)
            pool = self.adaptor.pixelBufferPool()
            yuv_pb = (
                _pb.pool_create_buffer(
                    pool,
                    allocation_threshold=_pb.WRITER_POOL_LIMIT,
                )
                if pool is not None
                else None
            )
            if yuv_pb is None:
                raise RuntimeError(f"[{self.label}] YUV pool buffer allocation failed")
            _yuv.rgb_to_yuv422_10(rgb, yuv_pb)
            pb = yuv_pb
        pts = _pb.frame_pts(self.frame_count, self.cadence)
        if not self.adaptor.appendPixelBuffer_withPresentationTime_(pb, pts):
            raise RuntimeError(
                f"[{self.label}] appendPixelBuffer failed at frame {self.frame_count}: "
                f"status={self.writer.status()} error={self.writer.error()}"
            )
        self.frame_count += 1

    def finish(self) -> None:
        """Mark inputs finished, drain audio, end session, finishWriting."""
        self.video_input.markAsFinished()
        if self.audio_input is not None and not self._audio_done.wait(timeout=120.0):
            raise RuntimeError(
                f"[{self.label}] audio pump didn't finish (progress="
                f"{self._audio_progress[0]}/{self._audio_sample_count})"
            )
        self.writer.endSessionAtSourceTime_(_pb.frame_pts(self.frame_count, self.cadence))
        done = threading.Event()
        self.writer.finishWritingWithCompletionHandler_(lambda: done.set())
        done.wait()
        if self.writer.status() != 2:  # AVAssetWriterStatusCompleted = 2
            raise RuntimeError(
                f"[{self.label}] AVAssetWriter finished with status "
                f"{self.writer.status()}: {self.writer.error()}"
            )

    def cancel(self) -> None:
        """Cancel an unfinished transaction and remove its partial container."""
        try:
            self.writer.cancelWriting()
        finally:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                _log.warning("[%s] could not remove partial output %s", self.label, self.path)
