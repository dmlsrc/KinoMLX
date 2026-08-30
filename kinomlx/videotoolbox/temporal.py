"""VideoToolbox Frame Rate Conversion (temporal upscaler) session wrapper.

Wraps VTFrameRateConversionConfiguration + VTFrameRateConversionParameters
to convert between arbitrary source and target frame rates. Unlike VSR,
the configuration takes only the frame dimensions + quality; the rate
conversion ratio is driven entirely by per-pair interpolation phases.

Per source frame pair (frame_N at PTS_N, frame_N+1 at PTS_N+1), we
compute the set of target output PTSes that fall in [PTS_N, PTS_N+1)
and their phases (where phase = (target_pts - PTS_N) / (PTS_N+1 - PTS_N)
in [0, 1)). VT's API takes a phase array and a matching destinationFrames
array, so a single call produces all interpolated frames for that pair.

Cleanly handles arbitrary float fps both sides:
  15 -> 30   exact 2x; phases always [0.5].
  24 -> 60   2.5x; phases cycle [0, 0.4, 0.8], [0.2, 0.6], ...
  24 -> 24   identity; phase array per pair is [0.0] = source pass-through
             (caller should detect this and skip the stage entirely).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import VideoToolbox as vt

from . import pixel_buffers as _pb
from .errors import VideoToolboxUnavailableError

_log = logging.getLogger(__name__)


class VtfrcSession:
    """Per-pair temporal interpolator with arbitrary source/target fps.

    Construction takes frame dimensions, source fps, target fps, and a
    mode setting (Normal or Quality prioritization). The session buffers
    one source frame at a time and emits all output frames that fall in
    the gap when the next source frame arrives.

    Usage:
        session = VtfrcSession(width, height, source_fps=24, target_fps=60)
        session.use_dst_pool(av_writer.adaptor.pixelBufferPool())
        for src_idx, src_pb in enumerate(source_buffers):
            for dst_pb in session.feed(src_pb, src_idx):
                av_writer.append(dst_pb)
        for dst_pb in session.drain():
            av_writer.append(dst_pb)

    `feed()` yields zero or more interpolated frames per source frame. The
    first source frame just buffers (yields nothing); subsequent frames
    trigger interpolation between the buffered prev and the incoming curr.
    At a detected hard cut, `feed_cut()` holds the previous frame across its
    source period and restarts the sequential VT session before buffering the
    first post-cut frame. `drain()` emits the final held source period.
    """

    # mode enum (rate-conversion quality prioritization)
    MODE_NORMAL = "normal"
    MODE_HIGH = "high"

    def __init__(
        self,
        in_w: int,
        in_h: int,
        source_fps: float,
        target_fps: float,
        *,
        mode: str = MODE_NORMAL,
    ):
        self.source_cadence = _pb.rational_cadence(source_fps)
        self.target_cadence = _pb.rational_cadence(target_fps)
        if mode not in (self.MODE_NORMAL, self.MODE_HIGH):
            raise ValueError(f"unknown temporal mode: {mode!r}")
        if not vt.VTFrameRateConversionConfiguration.isSupported():
            raise VideoToolboxUnavailableError(
                "VideoToolbox frame-rate conversion is not supported on this device"
            )

        self.in_w, self.in_h = in_w, in_h
        self.source_fps = float(self.source_cadence)
        self.target_fps = float(self.target_cadence)
        self.mode = mode

        q = (
            vt.VTFrameRateConversionConfigurationQualityPrioritizationQuality
            if mode == self.MODE_HIGH
            else vt.VTFrameRateConversionConfigurationQualityPrioritizationNormal
        )
        cls = vt.VTFrameRateConversionConfiguration
        self.config = cls.alloc().initWithFrameWidth_frameHeight_usePrecomputedFlow_qualityPrioritization_revision_(
            in_w,
            in_h,
            False,
            q,
            cls.defaultRevision(),
        )
        if self.config is None:
            raise RuntimeError("VTFrameRateConversionConfiguration init returned nil")

        self.processor = vt.VTFrameProcessor.alloc().init()
        ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(f"VTFrameProcessor (rate conversion) startSession failed: {err}")

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        _log.info(
            f"Temporal session ready ({self.source_fps:.3f}fps -> "
            f"{self.target_fps:.3f}fps "
            f"@ {in_w}x{in_h}, mode={mode}, "
            f"src fmt {_pb.resolve_pixel_format(self.src_attrs):#x}, "
            f"dst fmt {_pb.resolve_pixel_format(self.dst_attrs):#x})"
        )

        self._src_pool = _pb.make_pool_from_attrs(self.src_attrs)
        if self._src_pool is None:
            _log.warning("temporal source pool creation failed; using lexical allocations")
        self._dst_pool: object | None = None
        self._dst_allocation_threshold = max(
            4,
            math.ceil(self.target_fps / self.source_fps) + _pb.WRITER_POOL_LIMIT,
        )

        # Per-pair state ----------------------------------------------------
        # Track the next integer target-frame index so each source interval can
        # discover its outputs without re-emitting a previous target.
        # source frame N is at time N / source_fps.
        # target frame M is at time M / target_fps.
        # M / target_fps in [N/source_fps, (N+1)/source_fps) means
        #   M in [N * (target/source), (N+1) * (target/source))
        self._prev_src_pb: object | None = None
        self._prev_src_index: int = -1  # source frame index of buffered prev
        self._next_target_index: int = 0  # next target frame index to emit

    def use_dst_pool(self, pool: object) -> None:
        """Wire AVWriter's adaptor pool for zero-copy output."""
        self._dst_pool = pool

    def close(self) -> None:
        processor = self.processor
        self.processor = None
        try:
            if processor is not None:
                processor.endSession()
        finally:
            self._prev_src_pb = None
            self._prev_src_index = -1
            self._next_target_index = 0
            _pb.flush_pool(self._src_pool)
            _pb.flush_pool(self._dst_pool)
            self._src_pool = None
            self._dst_pool = None
            self.src_attrs = {}
            self.dst_attrs = {}
            self.config = None

    # ------------------------------------------------------------------------
    # Internal buffer factory
    # ------------------------------------------------------------------------

    def make_source_buffer(self) -> object:
        """Acquire one bounded source buffer for a direct frame upload."""
        if self._src_pool is not None:
            pb = _pb.pool_create_buffer(
                self._src_pool,
                allocation_threshold=_pb.TEMPORAL_SOURCE_POOL_LIMIT,
            )
            if pb is not None:
                return pb
            raise RuntimeError("bounded VTFRC source pool is exhausted")
        return _pb.make_pixel_buffer_from_attrs(self.in_w, self.in_h, self.src_attrs)

    def _make_dst_buffer(self) -> object:
        if self._dst_pool is not None:
            pb = _pb.pool_create_buffer(
                self._dst_pool,
                allocation_threshold=self._dst_allocation_threshold,
            )
            if pb is not None:
                return pb
            raise RuntimeError("bounded VTFRC destination pool is exhausted")
        return _pb.make_pixel_buffer_from_attrs(self.in_w, self.in_h, self.dst_attrs)

    def _process_targets(
        self,
        source_pb: object,
        next_pb: object,
        *,
        source_index: int,
        next_source_index: int,
        target_indices: list[int],
        phases: list[float],
        failure_context: str,
    ) -> list[object]:
        """Run one bounded VT request for an already-resolved source period."""
        dest_buffers = [self._make_dst_buffer() for _ in target_indices]
        source_pts = _pb.frame_pts(source_index, self.source_cadence)
        next_pts = _pb.frame_pts(next_source_index, self.source_cadence)
        source_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            source_pb, source_pts
        )
        next_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            next_pb, next_pts
        )

        dest_frames = []
        for target_index, dest_pb in zip(target_indices, dest_buffers, strict=True):
            dest_pts = _pb.frame_pts(target_index, self.target_cadence)
            dest_frames.append(
                vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                    dest_pb, dest_pts
                )
            )

        params = vt.VTFrameRateConversionParameters.alloc().initWithSourceFrame_nextFrame_opticalFlow_interpolationPhase_submissionMode_destinationFrames_(
            source_frame,
            next_frame,
            None,
            phases,
            vt.VTFrameRateConversionParametersSubmissionModeSequential,
            dest_frames,
        )
        processor = self.processor
        if processor is None:
            raise RuntimeError("VTFRC session is closed")
        ok, err = processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(f"{failure_context}: {err}")

        del source_frame, next_frame, dest_frames, params
        return dest_buffers

    def _restart_processor(self) -> None:
        """Drop VideoToolbox's sequential history and open a fresh session."""
        processor = self.processor
        self.processor = None
        if processor is not None:
            processor.endSession()

        fresh = vt.VTFrameProcessor.alloc().init()
        ok, err = fresh.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(f"VTFRC restart after scene cut failed: {err}")
        self.processor = fresh

    # ------------------------------------------------------------------------
    # Phase / target-index math
    # ------------------------------------------------------------------------

    def _target_indices_in_pair(self, src_index: int) -> list[int]:
        """Target frame indices M such that M's PTS falls in
        [src_index / source_fps, (src_index + 1) / source_fps).
        """
        # Start at next_target_index so we never re-emit. The loop guards
        # below filter the exact source-frame interval.
        start = self._next_target_index
        start_time = src_index / self.source_cadence
        end_time = (src_index + 1) / self.source_cadence
        out = []
        m = start
        # Exact rational comparisons keep a target on the right boundary in
        # the next pair without relying on an arbitrary floating epsilon.
        while m / self.target_cadence < end_time:
            if m / self.target_cadence >= start_time:
                out.append(m)
            m += 1
            # Refuse a pathological ratio instead of silently truncating it.
            if m - start > 10_000:
                raise ValueError(
                    "temporal conversion produces more than 10,000 target "
                    "frames in one source period"
                )
        return out

    def _phases_for_targets(self, target_indices: list[int], src_index: int) -> list[float]:
        """For each target index M, return phase = (M/target - src/source) /
        (1/source) clamped to [0, 1). Phase 0 = source frame, phase 1 = next.
        """
        phases = []
        for m in target_indices:
            phase = float(m * self.source_cadence / self.target_cadence - src_index)
            # Retain a defensive clamp around the exact rational calculation.
            if phase < 0.0:
                phase = 0.0
            elif phase >= 1.0:
                phase = 1.0 - 1e-9
            phases.append(phase)
        return phases

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def feed(self, src_pb: object, src_index: int) -> Iterator[object]:
        """Feed one source frame. Yields the interpolated destination buffers
        whose PTSes fall in [prev_src_pts, this_src_pts).

        For the very first source frame, this is empty (no pair yet). For
        subsequent frames we compute the target indices in the prev->curr
        gap, build a single VTFrameRateConversionParameters with the phase
        and destination arrays, run one VT call, and yield each output pb.
        """
        if self._prev_src_pb is None:
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            return
        if src_index != self._prev_src_index + 1:
            raise ValueError(
                "VTFRC source indices must be consecutive; got "
                f"{self._prev_src_index} then {src_index}"
            )

        target_indices = self._target_indices_in_pair(self._prev_src_index)
        if not target_indices:
            # Identity / downsample case: no output frame falls in this gap.
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            return

        phases = self._phases_for_targets(target_indices, self._prev_src_index)
        dest_buffers = self._process_targets(
            self._prev_src_pb,
            src_pb,
            source_index=self._prev_src_index,
            next_source_index=src_index,
            target_indices=target_indices,
            phases=phases,
            failure_context=(
                "VTFRC processWithParameters failed at source pair "
                f"{self._prev_src_index}->{src_index}"
            ),
        )

        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = src_pb
        self._prev_src_index = src_index

        # Yield one buffer at a time and drop our local list reference once
        # we hand it over - the writer retains what it needs.
        while dest_buffers:
            yield dest_buffers.pop(0)

    def feed_cut(self, src_pb: object, src_index: int) -> Iterator[object]:
        """Feed the first post-cut source without interpolating across the cut.

        The preceding source period still owns its target timestamps, so it is
        emitted as an exact hold of the last pre-cut frame. The sequential VT
        processor is then restarted to discard any internal optical history.
        """
        if self._prev_src_pb is None:
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            return
        if src_index != self._prev_src_index + 1:
            raise ValueError(
                "VTFRC source indices must be consecutive; got "
                f"{self._prev_src_index} then {src_index}"
            )

        target_indices = self._target_indices_in_pair(self._prev_src_index)
        dest_buffers: list[object] = []
        if target_indices:
            phases = self._phases_for_targets(target_indices, self._prev_src_index)
            dest_buffers = self._process_targets(
                self._prev_src_pb,
                self._prev_src_pb,
                source_index=self._prev_src_index,
                next_source_index=src_index,
                target_indices=target_indices,
                phases=phases,
                failure_context=(
                    "VTFRC held-period processing failed before scene cut at "
                    f"source frame {src_index}"
                ),
            )
            self._next_target_index = target_indices[-1] + 1

        self._restart_processor()
        self._prev_src_pb = src_pb
        self._prev_src_index = src_index
        while dest_buffers:
            yield dest_buffers.pop(0)

    def drain(self) -> Iterator[object]:
        """Emit target frames in the buffered final source-frame period.

        ``feed`` cannot emit that period because no real next source frame
        arrives. Use the final frame as both endpoints so every remaining
        interpolation phase is an exact hold of that frame.
        """
        if self._prev_src_pb is None:
            return

        target_indices = self._target_indices_in_pair(self._prev_src_index)
        if not target_indices:
            self._prev_src_pb = None
            return

        phases = self._phases_for_targets(target_indices, self._prev_src_index)
        dest_buffers = self._process_targets(
            self._prev_src_pb,
            self._prev_src_pb,
            source_index=self._prev_src_index,
            next_source_index=self._prev_src_index + 1,
            target_indices=target_indices,
            phases=phases,
            failure_context=(
                f"VTFRC drain failed for final source period at frame {self._prev_src_index}"
            ),
        )
        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = None
        while dest_buffers:
            yield dest_buffers.pop(0)
