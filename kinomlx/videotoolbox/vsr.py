"""VideoToolbox Super Resolution (spatial upscale) session wrapper.

`VsrSession` wraps VTSuperResolutionScalerConfiguration (HQ, scale=4) or
VTLowLatencySuperResolutionScalerConfiguration (LL, scale=2) plus its
VTFrameProcessor and the source/dst CVPixelBufferPools. The caller hands
in a frame (uint8 RGB or fp16 RGBA) and gets back a destination buffer
ready to feed straight into AVAssetWriter.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Protocol

import mlx.core as mx
import Quartz
import VideoToolbox as vt

from . import pixel_buffers as _pb
from .errors import VideoToolboxUnavailableError

_log = logging.getLogger(__name__)
_NATIVE_STDERR_LOCK = threading.RLock()


class _DownloadableConfiguration(Protocol):
    def configurationModelStatus(self) -> object: ...

    def downloadConfigurationModelWithCompletionHandler_(
        self,
        callback: Callable[[object | None], None],
    ) -> None: ...

    def configurationModelPercentageAvailable(self) -> float: ...


def _duplicate_stderr() -> int:
    return os.dup(2)


def _open_devnull() -> int:
    return os.open(os.devnull, os.O_WRONLY)


def _redirect_stderr(source: int) -> None:
    os.dup2(source, 2)


def _close_fd(fd: int) -> None:
    os.close(fd)


@contextmanager
def _suppress_native_stderr(*, enabled: bool = True) -> Iterator[None]:
    """Temporarily swallow native writes to process-global file descriptor 2.

    VideoToolbox logs Metal pipeline compilation through ``NSLog``, bypassing
    Python's ``sys.stderr`` and Rich's shared console. Suppress only the brief
    session-start call; API return values still carry real failures. The CLI's
    ``KINO_VERBOSE`` setting disables suppression for native diagnostics.
    """
    if not enabled:
        yield
        return

    # Serialize the complete process-global save/redirect/restore interval.
    # RLock also makes nested construction in one thread safe.
    with _NATIVE_STDERR_LOCK:
        sys.stderr.flush()
        saved_fd = _duplicate_stderr()
        try:
            devnull_fd = _open_devnull()
        except BaseException:
            with suppress(OSError):
                _close_fd(saved_fd)
            raise

        primary: BaseException | None = None
        cleanup_failures: list[tuple[str, BaseException]] = []

        def _cleanup(label: str, operation: Callable[[int], None], fd: int) -> None:
            try:
                operation(fd)
            except BaseException as exc:  # noqa: BLE001 - preserved below
                cleanup_failures.append((label, exc))

        try:
            _redirect_stderr(devnull_fd)
            try:
                yield
            finally:
                _cleanup("restore stderr", _redirect_stderr, saved_fd)
        except BaseException as exc:
            primary = exc
        finally:
            _cleanup("close /dev/null", _close_fd, devnull_fd)
            _cleanup("close saved stderr", _close_fd, saved_fd)

        if primary is None and cleanup_failures:
            _, primary = cleanup_failures.pop(0)
        if primary is not None:
            for label, failure in cleanup_failures:
                if failure is primary:
                    continue
                with suppress(BaseException):
                    primary.add_note(f"{label} also failed: {type(failure).__name__}: {failure}")
            raise primary


def scale_for_mode(mode: str) -> int:
    """Map a VSR spatial mode to its forced scale factor.

    VideoToolbox couples the spatial-mode choice to the scale: LowLatency
    is 2x-only, the HQ classes are 4x-only.  Centralized here so call sites
    don't reinvent the mapping.
    """
    if mode == "fast":
        return 2
    if mode in ("balanced", "image"):
        return 4
    raise ValueError(f"unknown VSR spatial-mode: {mode!r}")


# The HQ scaler has no public dimension-query API. These per-dimension input
# caps are the limits at which configuration initialization remains valid.
HQ_MAX_INPUT_W = 1920
HQ_MAX_INPUT_H = 1080
HQ_VIDEO_MIN_INPUT_DIM = 128


def _balanced_uses_video_input(width: int, height: int, mode: str) -> bool:
    """Choose HQ Video input only where its prior-frame path is reliable."""
    if mode != "balanced":
        return False
    if width >= HQ_VIDEO_MIN_INPUT_DIM and height >= HQ_VIDEO_MIN_INPUT_DIM:
        return True
    _log.warning(
        "balanced VideoToolbox SR at %dx%d is falling back to deterministic "
        "image mode because Video input is unreliable when either dimension "
        "is below %d pixels",
        width,
        height,
        HQ_VIDEO_MIN_INPUT_DIM,
    )
    return False


def _validate_combination(width: int, height: int, scale: int, mode: str) -> None:
    """Check the (input size, scale, mode) combo is something VT supports.

    VSR's HQ and LL classes each only support specific scale factors (and LL
    additionally restricts input size to <= 960x960). Failing fast here gives
    a clear error message instead of an opaque init/startSession failure.
    """
    if mode == "fast":
        cls = vt.VTLowLatencySuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise VideoToolboxUnavailableError("LowLatency VSR is not supported on this device")
        ok = list(cls.supportedScaleFactorsForFrameWidth_frameHeight_(width, height))
        if not ok:
            mn = cls.minimumDimensions()
            mx = cls.maximumDimensions()
            raise VideoToolboxUnavailableError(
                f"VideoToolbox fast SR does not support {width}x{height} input. "
                f"Allowed: {mn.width}x{mn.height} to {mx.width}x{mx.height}."
            )
        if float(scale) not in [float(s) for s in ok]:
            raise VideoToolboxUnavailableError(
                f"VideoToolbox fast SR at {width}x{height} supports scale={ok}, "
                f"requested scale={scale}."
            )
    else:
        cls = vt.VTSuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise VideoToolboxUnavailableError("High-quality VSR is not supported on this device")
        ok = [int(s) for s in cls.supportedScaleFactors()]
        if scale not in ok:
            raise VideoToolboxUnavailableError(
                f"VideoToolbox {mode} SR supports scale={ok}, requested scale={scale}. "
                "Use fast mode for 2x."
            )
        if width > HQ_MAX_INPUT_W or height > HQ_MAX_INPUT_H:
            fits_fast = width <= 960 and height <= 960
            hint = (
                "Use fast mode for a 2x upscale (input must be <= 960x960)."
                if fits_fast
                else "Downscale the input before using VideoToolbox SR."
            )
            raise VideoToolboxUnavailableError(
                f"VideoToolbox {mode} SR (4x) does not support {width}x{height} "
                f"input (max {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}). {hint}"
            )


def _wait_for_model_download(config: _DownloadableConfiguration) -> None:
    """Block until HQ VSR's downloadable model is ready, printing progress."""
    status = config.configurationModelStatus()
    if status == vt.VTSuperResolutionScalerConfigurationModelStatusReady:
        return
    _log.info(f"VSR model not ready (status={status}); requesting download...")
    done = threading.Event()
    err_box: list[object | None] = [None]

    def completion(error: object | None) -> None:
        err_box[0] = error
        done.set()

    config.downloadConfigurationModelWithCompletionHandler_(completion)
    last_reported = -1
    while not done.is_set():
        pct = int(config.configurationModelPercentageAvailable() * 100)
        if pct // 5 != last_reported // 5:
            _log.info(f"  model download: {pct}%")
            last_reported = pct
        done.wait(timeout=0.5)
    if err_box[0] is not None:
        raise RuntimeError(f"VSR model download failed: {err_box[0]}")
    _log.info("  model download: done")


class VsrSession:
    """Spatial VSR processor; balanced Video input also uses frame history.

    Spatial modes:
      "fast"      VTLowLatencySuperResolutionScalerConfiguration. scale=2,
                  input <= 960x960. NV12 source. Per-frame, no frame history.
      "balanced"  VTSuperResolutionScalerConfiguration InputType=Video.
                  scale=4. RGBAHalf source. Uses prev source + prev output to
                  inform the per-frame upscale.  Default for video; slightly
                  crisper motion edges at the cost of slightly more
                  frame-to-frame variation than image mode. Inputs below 128
                  pixels in either dimension fall back to Image input.
      "image"     VTSuperResolutionScalerConfiguration InputType=Image. scale=4.
                  RGBAHalf source. Per-frame deterministic upscale, no
                  prev-frame feedback.  Apple documents this as for stills,
                  but on real video it produces measurably lower frame-to-frame
                  second-difference than balanced - a legitimate alternative
                  if you prefer the smoother / less-edge-boosted trade-off.

    The terminal resets previous-frame state at detected hard cuts. This is
    relevant to both generated and supplied frame streams: a single generated
    clip can still contain an intentional or accidental scene transition.
    """

    def __init__(
        self,
        in_w: int,
        in_h: int,
        mode: str,
        fps: float = 24.0,
        *,
        native_verbose: bool = False,
    ):
        cadence = _pb.rational_cadence(fps)
        if mode not in ("fast", "balanced", "image"):
            raise ValueError(f"VsrSession only supports VideoToolbox modes, got {mode!r}")
        scale = scale_for_mode(mode)
        _validate_combination(in_w, in_h, scale, mode)
        self.in_w, self.in_h = in_w, in_h
        self.scale = scale
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.mode = mode
        self.cadence = cadence
        self.fps = float(self.cadence)
        self._video_input = _balanced_uses_video_input(in_w, in_h, mode)

        if mode == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc().initWithFrameWidth_frameHeight_scaleFactor_(
                in_w, in_h, float(scale)
            )
            if self.config is None:
                raise RuntimeError("LowLatency VSR config init returned nil")
        else:
            input_type = (
                vt.VTSuperResolutionScalerConfigurationInputTypeVideo
                if self._video_input
                else vt.VTSuperResolutionScalerConfigurationInputTypeImage
            )
            cls = vt.VTSuperResolutionScalerConfiguration
            self.config = cls.alloc().initWithFrameWidth_frameHeight_scaleFactor_inputType_usePrecomputedFlow_qualityPrioritization_revision_(
                in_w,
                in_h,
                scale,
                input_type,
                False,
                vt.VTSuperResolutionScalerConfigurationQualityPrioritizationNormal,
                cls.defaultRevision(),
            )
            if self.config is None:
                raise RuntimeError(
                    f"High-quality VSR config init returned nil for {in_w}x{in_h} "
                    f"input at {scale}x. The HQ scaler accepts up to "
                    f"{HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}; check the input dimensions."
                )
            _wait_for_model_download(self.config)

        self.processor = vt.VTFrameProcessor.alloc().init()
        # Session start compiles the VSR Metal pipeline and NSLogs compile
        # chatter directly to fd 2. Errors still return through ``err``.
        with _suppress_native_stderr(enabled=not native_verbose):
            ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(
                f"VTFrameProcessor.startSessionWithConfiguration_error_ failed: {err}"
            )

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        _log.info(
            f"VSR session ready (mode={mode}, {in_w}x{in_h} -> {self.out_w}x{self.out_h}, "
            f"src fmt {_pb.resolve_pixel_format(self.src_attrs):#x}, "
            f"dst fmt {_pb.resolve_pixel_format(self.dst_attrs):#x})"
        )

        self._prev_src_frame: object | None = None
        self._prev_dst_frame: object | None = None

        # Src pool: two buffers in flight at any time (current + prev_src).
        self._src_pool = _pb.make_pool_from_attrs(self.src_attrs)
        if self._src_pool is None:
            _log.warning("src pool creation failed; falling back to per-frame allocation")
        # Dst pool: typically set by the caller to the AVAssetWriter adaptor's
        # pool for zero-copy from VSR output to encoder.
        self._dst_pool: object | None = None

    def use_dst_pool(self, pool: object) -> None:
        """Wire the writer's adaptor pixelBufferPool() as VSR's dst source -
        zero-copy from VSR output straight into the encoder's queue.
        """
        self._dst_pool = pool

    @property
    def uses_frame_history(self) -> bool:
        """Whether this configuration feeds prior frames into VideoToolbox."""
        return self._video_input

    def reset_frame_history(self) -> None:
        """Drop the previous-frame chain before processing a post-cut frame."""
        self._prev_src_frame = None
        self._prev_dst_frame = None

    def flush_pools(self) -> None:
        """Release excess cached buffers in the src pool (and dst pool if we
        own it - we usually don't; the writer's adaptor owns the dst pool).

        Pool caching is what makes hot-path buffer allocation fast, but at
        steady state the cache should be ~3 buffers. Periodic flushing
        reclaims peak-watermark allocations that the workload no longer
        needs (e.g. an early VAE chunk that briefly inflated buffer demand).
        """
        _pb.flush_pool(self._src_pool)

    def close(self) -> None:
        processor = self.processor
        self.processor = None
        try:
            if processor is not None:
                processor.endSession()
        finally:
            self._prev_src_frame = None
            self._prev_dst_frame = None
            _pb.flush_pool(self._src_pool)
            _pb.flush_pool(self._dst_pool)
            self._src_pool = None
            self._dst_pool = None
            self.src_attrs = {}
            self.dst_attrs = {}
            self.config = None

    # ------------------------------------------------------------------------
    # Internal: buffer factories
    # ------------------------------------------------------------------------

    def _make_src_buffer(self) -> object:
        if self._src_pool is not None:
            allocation_threshold = (
                _pb.VSR_VIDEO_INPUT_SOURCE_POOL_LIMIT
                if self._video_input
                else _pb.VSR_STATELESS_SOURCE_POOL_LIMIT
            )
            pb = _pb.pool_create_buffer(
                self._src_pool,
                allocation_threshold=allocation_threshold,
            )
            if pb is not None:
                return pb
            raise RuntimeError("bounded VSR source pool is exhausted")
        return _pb.make_pixel_buffer_from_attrs(self.in_w, self.in_h, self.src_attrs)

    def _make_dst_buffer(self) -> object:
        if self._dst_pool is not None:
            pb = _pb.pool_create_buffer(
                self._dst_pool,
                allocation_threshold=_pb.VSR_DESTINATION_POOL_LIMIT,
            )
            if pb is not None:
                return pb
            raise RuntimeError("bounded VSR destination pool is exhausted")
        return _pb.make_pixel_buffer_from_attrs(self.out_w, self.out_h, self.dst_attrs)

    def _tag_source_matrix(self, src_pb: object) -> None:
        """Tag balanced Video input with its required BT.709 matrix."""
        if not self._video_input:
            return
        Quartz.CVBufferSetAttachment(
            src_pb,
            Quartz.kCVImageBufferYCbCrMatrixKey,
            Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2,
            Quartz.kCVAttachmentMode_ShouldPropagate,
        )

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def upscale_to_buffer(self, frame: mx.array, frame_index: int) -> object:
        """Upscale one frame. Returns the dst CVPixelBuffer (RGBAHalf for HQ,
        NV12 for LL) ready to append to AVWriter.
        """
        src_pb = self._make_src_buffer()
        _pb.upload_frame_to_buffer(frame, src_pb)
        self._tag_source_matrix(src_pb)
        dst_pb = self._make_dst_buffer()
        pts = _pb.frame_pts(frame_index, self.cadence)
        src_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            src_pb,
            pts,
        )
        dst_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            dst_pb,
            pts,
        )

        if self.mode == "fast":
            params = vt.VTLowLatencySuperResolutionScalerParameters.alloc().initWithSourceFrame_destinationFrame_(
                src_frame, dst_frame
            )
        else:
            use_video_history = self._video_input
            params = vt.VTSuperResolutionScalerParameters.alloc().initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
                src_frame,
                self._prev_src_frame if use_video_history else None,
                self._prev_dst_frame if use_video_history else None,
                None,
                vt.VTSuperResolutionScalerParametersSubmissionModeSequential,
                dst_frame,
            )

        ok, err = self.processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(f"VSR processWithParameters failed at frame {frame_index}: {err}")
        if self._video_input:
            self._prev_src_frame = src_frame
            self._prev_dst_frame = dst_frame
        else:
            self._prev_src_frame = None
            self._prev_dst_frame = None
        return dst_pb
