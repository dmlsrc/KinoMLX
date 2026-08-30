"""High-level VideoToolbox encode helper for KinoMLX pipelines.

`encode_video_videotoolbox()` is the AVAssetWriter-backed encode path
(no ffmpeg). It consumes a typed, closeable frame stream lazily and emits
an HEVC mp4 - no full-video staging and no on-disk WAV unless
`save_audio_sidecar=True`.

Two optional post-VAE stages can be inserted between the frame source
and the writer:

  vsr_spatial_mode={fast,balanced,image}   VideoToolbox Super Resolution.
                                            Scale is forced by the mode
                                            (fast=2x, balanced/image=4x).
  target_fps=FLOAT                          VideoToolbox Frame Rate
                                            Conversion to the requested
                                            output rate.

Both default off. Compatible NV12 stages share destination pools directly.
RGBAHalf output retains its producer pool and takes the writer's explicit
MLX RGB-to-10-bit-YUV conversion path.

Frames may be supplied as:
  - a closeable iterator of owned `(H, W, 3)` float16 MLX RGB frames;
  - a finite sequence with the same per-frame contract; or
  - a `(T, H, W, 3)` float16 array for explicit compatibility callers.

The immutable source signal and terminal delivery are validated before the
first frame is pulled. The first frame is then peeked to verify geometry and
dtype; the rest are consumed with synchronous backpressure.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import objc

from kinomlx.media.signals import (
    ChromaSubsampling,
    EncodedVideoDeliverySpec,
    OutputColorPlan,
    UnsupportedSignalError,
    VideoCodecProfile,
    VideoSignalSpec,
    validate_sdr_output_plan,
)
from kinomlx.reporting import NullReporter, Reporter

from . import pixel_buffers as _pb
from .audio import AudioTrack
from .cut_detect import CutDetector
from .temporal import VtfrcSession
from .vsr import VsrSession, scale_for_mode
from .writer import HEVC_PROFILE_MAIN10, HEVC_PROFILE_MAIN422_10, AVWriter

_log = logging.getLogger(__name__)

type FrameSource = mx.array | Sequence[mx.array] | Iterable[mx.array]


class _ProgressBar(Protocol):
    def update(self, advance: int = 1) -> None: ...


class ProgressStack(Protocol):
    """Caller-owned progress surface consumed by the native encoder."""

    def add(self, *, total: int, desc: str, unit: str) -> _ProgressBar: ...

    def write(self, message: str, *, position: str = "above") -> None: ...


def _human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _peek_frames(
    frames: FrameSource,
) -> tuple[mx.array, Iterator[mx.array], int | None]:
    """Return (first_frame, full_iterator, total_or_none).

    Accepts a list, tuple, MLX frame batch, or iterator. For a four-dimensional
    MLX batch we iterate along axis 0. The returned iterator includes the first
    frame. ``total`` is known for a batch or sequence and unknown for a pure
    iterator that would have to be consumed to count.
    """
    if isinstance(frames, mx.array):
        if frames.ndim != 4:
            raise ValueError(
                "encode_video_videotoolbox: an MLX frame batch must have shape (T,H,W,C)"
            )
        if frames.shape[0] == 0:
            raise ValueError("encode_video_videotoolbox: empty frames array")
        return (
            frames[0],
            (frames[i] for i in range(frames.shape[0])),
            int(frames.shape[0]),
        )
    if isinstance(frames, Sequence):
        if not frames:
            raise ValueError("encode_video_videotoolbox: empty frames list")
        return frames[0], iter(frames), len(frames)
    # Generic iterator: consume one frame, then chain it back.
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("encode_video_videotoolbox: empty frames iterator") from None
    return first, itertools.chain([first], it), None


def _validate_source_frame(frame: mx.array, source_signal: VideoSignalSpec) -> None:
    """Require one payload to match the already-validated stream signal."""
    shape = tuple(getattr(frame, "shape", ()))
    expected = (source_signal.height, source_signal.width, 3)
    if shape != expected:
        raise ValueError(f"source frame shape {shape} does not match signal {expected}")
    dtype = str(getattr(frame, "dtype", "")).split(".")[-1]
    if dtype != source_signal.dtype:
        raise ValueError(
            f"source frame dtype {dtype!r} does not match signal {source_signal.dtype!r}"
        )


def _native_hevc_profile(delivery: EncodedVideoDeliverySpec) -> str:
    return (
        HEVC_PROFILE_MAIN10
        if delivery.profile is VideoCodecProfile.MAIN10
        else HEVC_PROFILE_MAIN422_10
    )


def _output_phase_name(
    *,
    vsr_spatial_mode: str | None,
    source_fps: float,
    target_fps: float | None,
) -> str:
    """Name every active native stage in the terminal frame-consumer row."""
    stages: list[str] = []
    if vsr_spatial_mode is not None:
        stages.append(f"VSR {vsr_spatial_mode} {scale_for_mode(vsr_spatial_mode)}x")
    if target_fps is not None and abs(target_fps - source_fps) > 1e-6:
        stages.append("VTFRC")
    stages.append("HEVC encode")
    return " + ".join(stages)


def _normalize_audio_for_track(audio_waveform: mx.array) -> mx.array:
    """Final encoder boundary: MLX ``(B,C,T)`` or ``(C,T)`` -> ``(C,T)`` f32.

    AudioTrack expects (channels, samples). Pipeline outputs are (B,C,T).
    """
    if not isinstance(audio_waveform, mx.array):
        raise TypeError(f"audio_waveform must be an MLX array, got {type(audio_waveform).__name__}")
    arr = audio_waveform
    if arr.dtype != mx.float32:
        arr = arr.astype(mx.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"batched audio_waveform must contain exactly one item; got shape {arr.shape}"
            )
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"audio_waveform must be (B,C,T) or (C,T); got shape {arr.shape}")
    return arr


def prepare_audio_track(
    audio_waveform: mx.array | None,
    audio_sample_rate: int | None,
    *,
    onset_trim_mode: str,
    onset_trim_ms: float | None,
    verbose: bool = False,
) -> AudioTrack | None:
    """Build the shared onset-mitigated native audio track for any video sink."""
    if audio_waveform is None:
        return None
    if audio_sample_rate is None:
        raise ValueError("audio_sample_rate is required when audio_waveform is provided")
    arr = _normalize_audio_for_track(audio_waveform)
    from ..audio import DEFAULT_TRIM_MS, mitigate_onset

    trim_ms = onset_trim_ms if onset_trim_ms is not None else DEFAULT_TRIM_MS
    onset_result = mitigate_onset(
        arr,
        int(audio_sample_rate),
        mode=onset_trim_mode,
        trim_ms=trim_ms,
    )
    if verbose and onset_result.applied:
        _log.info("  audio onset: %s", onset_result.detail)
    return AudioTrack(onset_result.samples, sample_rate=int(audio_sample_rate))


def _allocate_writer_src_buffer(
    adaptor: _pb.PixelBufferAdaptor | None,
    width: int,
    height: int,
    fmt: int,
) -> object:
    """Pull a writer-source buffer (zero-copy when the pool is ready)."""
    pool = adaptor.pixelBufferPool() if adaptor is not None else None
    if pool is not None:
        pb = _pb.pool_create_buffer(
            pool,
            allocation_threshold=_pb.WRITER_POOL_LIMIT,
        )
        if pb is not None:
            return pb
        raise RuntimeError("bounded writer source pool is exhausted")
    attrs: dict[object, object] = {
        "PixelFormatType": fmt,
        "Width": width,
        "Height": height,
        "IOSurfaceProperties": {},
    }
    return _pb.make_pixel_buffer_from_attrs(width, height, attrs)


def _close_native_chain(
    writer: AVWriter,
    writer_orig: AVWriter | None,
    vtfrc: VtfrcSession | None,
    vsr: VsrSession | None,
    *,
    primary_failure: BaseException | None = None,
) -> None:
    """Attempt every terminal cleanup and preserve the processing failure."""
    cleanup_failures: list[tuple[str, BaseException]] = []
    cleanup_steps = [
        ("primary writer finish", writer.finish),
        (
            "original writer finish",
            None if writer_orig is None else writer_orig.finish,
        ),
        ("temporal session close", None if vtfrc is None else vtfrc.close),
        ("spatial session close", None if vsr is None else vsr.close),
    ]
    for label, cleanup in cleanup_steps:
        if cleanup is None:
            continue
        try:
            cleanup()
        except BaseException as exc:
            cleanup_failures.append((label, exc))

    if primary_failure is not None:
        for label, failure in cleanup_failures:
            primary_failure.add_note(f"{label} also failed: {failure}")
        raise primary_failure.with_traceback(primary_failure.__traceback__)
    if cleanup_failures:
        label, failure = cleanup_failures[0]
        for later_label, later_failure in cleanup_failures[1:]:
            failure.add_note(f"{later_label} also failed: {later_failure}")
        failure.add_note(f"failed during {label}")
        raise failure.with_traceback(failure.__traceback__)


def _encode_video_videotoolbox_impl(
    first: mx.array,
    frame_iter: Iterator[mx.array],
    output_path: str | Path,
    *,
    fps: float,
    source_signal: VideoSignalSpec,
    delivery: EncodedVideoDeliverySpec,
    reporter: Reporter,
    output_phase: str,
    audio_waveform: mx.array | None = None,
    audio_sample_rate: int | None = None,
    audio_bit_depth: str = "float32",
    save_audio_sidecar: bool = False,
    audio_onset_trim_mode: str = "auto",
    audio_onset_trim_ms: float | None = None,
    vsr_spatial_mode: str | None = None,
    target_fps: float | None = None,
    vsr_temporal_mode: str = "normal",
    vsr_save_original: bool = False,
    cut_detector: CutDetector,
    encode_quality: float = 0.65,
    n_source_frames: int,
    progress_stack: ProgressStack | None = None,
    audio_codec: str = "alac",
    verbose: bool = True,
    native_verbose: bool = False,
) -> Path:
    """Encode frames into an HEVC mp4 via AVAssetWriter (no ffmpeg).

    Returns the actual output path; rewrites the extension to .mp4 if the
    caller supplied something else (matches encode_video_ffmpeg() behavior for
    the ffmpeg `default` tier - both produce .mp4).

    ``audio_waveform`` is an MLX ``(B,C,T)`` or ``(C,T)`` array. Pass
    ``None`` for video-only.

    `vsr_spatial_mode`:
      None        no spatial upscale (writer source = NV12).
      "fast"      VTLowLatency VSR, scale 2x, input <= 960x960.
      "balanced"  VT HQ VSR Video mode, scale 4x; prev-frame chain.
      "image"     VT HQ VSR Image mode, scale 4x; per-frame deterministic.

    `target_fps`:
      None or equal to fps   no temporal interpolation.
      otherwise              route through VTFrameRateConversion.

    `cut_detector` protects native stages that retain adjacent-frame history.
    Balanced Video-input VSR drops its previous-frame pair before the first
    post-cut frame; VTFRC holds the preceding interval and restarts instead of
    interpolating across the discontinuity.

    `audio_bit_depth` is accepted for API parity with encode_video_ffmpeg();
    AVAssetWriter always consumes float32 PCM internally regardless.

    `audio_onset_trim_mode` / `audio_onset_trim_ms` route through to
    `kinomlx.audio.onset.mitigate_onset()` before the AudioTrack is
    built.  The same cleaned waveform feeds both the muxed audio track
    and the optional sidecar (`save_audio_sidecar=True`).

    `vsr_save_original`: when True AND any VT post-processing is engaged
    (VSR spatial or VTFRC temporal), also write the un-processed source-
    resolution source-fps mp4 alongside the requested output, as
    `<stem>_orig.mp4`.  Both files share the same AudioTrack so each is
    playable standalone.  No-op when neither VSR nor VTFRC is active
    (there's nothing for "original" to differ from).  The companion
    writer mirrors the resolved primary delivery (RGBAHalf + Main42210
    for VSR HQ or temporal conversion; NV12 + Main10 for fast-only VSR)
    so the A/B comparison isn't precision-floor mismatched. Implementation cost
    per source frame is one additional source-buffer upload + one HEVC
    HW append; the AVAssetWriter's pump runs on its own GCD queue, so
    the second encode is largely parallel to the primary.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_h, in_w, _in_c = first.shape

    if vsr_spatial_mode is not None:
        scale = scale_for_mode(vsr_spatial_mode)
        out_w, out_h = in_w * scale, in_h * scale
    else:
        scale = 1
        out_w, out_h = in_w, in_h

    if target_fps is not None and abs(target_fps - fps) > 1e-6:
        do_temporal = True
        output_fps = target_fps
    else:
        do_temporal = False
        output_fps = fps

    profile = _native_hevc_profile(delivery)

    # ---- Setup phase ------------------------------------------------------
    # VsrSession / VtfrcSession / AVWriter constructors each print to stdout
    # unconditionally ("VSR session ready ...", "Temporal session ready ...",
    # "[encode] AVAssetWriter -> ..."), plus we add the chain description
    # and the optional audio sidecar line.  When `progress_stack` is alive
    # (the caller is showing a "VAE chunks" bar above us), those raw prints
    # would stomp on the bar row.  Redirect them through a StringIO and
    # emit the captured block via `progress_stack.write()` so the lines
    # land cleanly above the bars (scroll-message-above-bars style); when no stack is
    # supplied, the prints flow normally to stdout.
    _setup_buf: io.StringIO | None = None
    _setup_ctx: AbstractContextManager[object | None]
    if progress_stack is not None:
        _setup_buf = io.StringIO()
        _setup_ctx = contextlib.redirect_stdout(_setup_buf)
    else:
        _setup_ctx = contextlib.nullcontext()

    with _setup_ctx:
        # VSR session
        vsr: VsrSession | None = None
        if vsr_spatial_mode is not None:
            vsr = VsrSession(
                in_w,
                in_h,
                mode=vsr_spatial_mode,
                fps=fps,
                native_verbose=native_verbose,
            )

        # VTFRC session
        vtfrc: VtfrcSession | None = None
        if do_temporal:
            vtfrc = VtfrcSession(
                out_w,
                out_h,
                source_fps=fps,
                target_fps=output_fps,
                mode=vsr_temporal_mode,
            )
        cut_detection_active = cut_detector.mode != "off" and (
            (vsr is not None and vsr.uses_frame_history) or vtfrc is not None
        )
        # Without VSR, decoded frames must be uploaded into VTFRC's declared
        # source format. The writer adaptor can expose a different format
        # (notably explicit 10-bit 4:2:2 YUV) and is not a valid source pool.
        # Audio track
        audio_track = prepare_audio_track(
            audio_waveform,
            audio_sample_rate,
            onset_trim_mode=audio_onset_trim_mode,
            onset_trim_ms=audio_onset_trim_ms,
            verbose=verbose,
        )

        # Pick writer source format.  When VSR or VTFRC is active, the
        # writer source = the last stage's dst.  When neither is active,
        # the writer source = NV12 and we upload through CoreImage (keeps
        # the encoder's RGB->YUV cost in one place).
        if vtfrc is not None:
            writer_src_fmt = _pb.resolve_pixel_format(vtfrc.dst_attrs)
        elif vsr is not None:
            writer_src_fmt = _pb.resolve_pixel_format(vsr.dst_attrs)
        else:
            writer_src_fmt = (
                _pb.PIX_RGBAHALF if delivery.chroma is ChromaSubsampling.YUV422 else _pb.PIX_NV12
            )

        producer_attrs = (
            vtfrc.dst_attrs if vtfrc is not None else vsr.dst_attrs if vsr is not None else None
        )
        writer_yuv_feed = writer_src_fmt == _pb.PIX_RGBAHALF

        # Writer + pool wiring.  Zero-copy hookups: VTFRC writes into the
        # writer's adaptor pool when active; VSR writes into its own dst
        # pool when VTFRC is between (a copy at the VT call boundary), or
        # directly into the writer's adaptor pool when there is no VTFRC.
        # An explicit RGBAHalf-to-YUV writer has a YUV adaptor pool, so its
        # producer must instead retain its own RGBAHalf destination pool.
        writer = AVWriter(
            output_path,
            width=out_w,
            height=out_h,
            fps=output_fps,
            source_pixel_format=writer_src_fmt,
            delivery=delivery,
            quality=encode_quality,
            label="encode",
            source_attrs=None if writer_yuv_feed else producer_attrs,
            audio_track=audio_track,
            audio_codec=audio_codec,
        )
        if vtfrc is not None and not writer_yuv_feed:
            vtfrc.use_dst_pool(writer.adaptor.pixelBufferPool())
        elif vsr is not None and not writer_yuv_feed:
            vsr.use_dst_pool(writer.adaptor.pixelBufferPool())

        # Optional "save the un-processed original alongside the VSR/VTFRC
        # result" companion writer.  Only meaningful when some VT post-
        # processing is engaged; otherwise the primary writer IS the
        # original and a duplicate adds zero value.  Shares the same
        # AudioTrack - CMSampleBuffer is fresh per make_sample_buffer()
        # call so two GCD pumps on the same track are safe.
        #
        # Source format + HEVC profile follow the same delivery object as the
        # primary writer. Main42210 uses RGBAHalf until the explicit numeric
        # conversion; Main10 uses NV12. The three-channel float16 source stays
        # unquantized, and opacity is added only at the native RGBAHalf bridge.
        writer_orig: AVWriter | None = None
        orig_path: Path | None = None
        do_save_original = vsr_save_original and (vsr is not None or vtfrc is not None)
        if do_save_original:
            orig_path = output_path.with_name(f"{output_path.stem}_orig{output_path.suffix}")
            if delivery.chroma is ChromaSubsampling.YUV422:
                orig_src_fmt = _pb.PIX_RGBAHALF
                orig_profile = HEVC_PROFILE_MAIN422_10
            else:
                orig_src_fmt = _pb.PIX_NV12
                orig_profile = HEVC_PROFILE_MAIN10
            orig_yuv_feed = orig_src_fmt == _pb.PIX_RGBAHALF
            writer_orig = AVWriter(
                orig_path,
                width=in_w,
                height=in_h,
                fps=fps,
                source_pixel_format=orig_src_fmt,
                delivery=delivery,
                quality=encode_quality,
                label="encode_orig",
                audio_track=audio_track,
                audio_codec=audio_codec,
            )
        else:
            orig_yuv_feed = False

        # Optional audio sidecar WAV.
        sidecar_path: Path | None = None
        if audio_track is not None and save_audio_sidecar:
            sidecar_path = output_path.with_suffix(".wav")
            audio_track.save_wav(sidecar_path)
            if verbose:
                _log.info(
                    f"  audio sidecar: {sidecar_path}  "
                    f"({audio_bit_depth}, {audio_track.sample_rate} Hz)"
                )

        # Chain description (above the encode bar so users see what's running).
        stages: list[str] = []
        if vsr is not None:
            stages.append(f"VSR={vsr_spatial_mode}({scale}x)")
        if vtfrc is not None:
            stages.append(f"VTFRC={fps:g}->{output_fps:g}fps")
        chain = " + ".join(stages) if stages else "passthrough"
        if verbose:
            _log.info(f"  encode (videotoolbox): {chain} -> HEVC {profile}")
            _log.info(f"  -> {output_path}")
            if writer_orig is not None:
                _log.info(f"  + original passthrough -> HEVC {orig_profile} -> {orig_path}")

    # If we captured the setup output, route it above the caller's bar
    # stack now - single bars.write() call so the bars stay coherent.
    if _setup_buf is not None and progress_stack is not None:
        _setup_msg = _setup_buf.getvalue().rstrip("\n")
        if _setup_msg:
            progress_stack.write(_setup_msg)

    # PhaseBar gives a stable, fixed-column progress display. The wrapper has
    # resolved an exact source-frame total even for a streaming iterator.
    # Suppress the legacy bar entirely when verbose=False.
    #
    # When `progress_stack` is provided, the encoder shares the caller's
    # stack - useful when a caller wants a "VAE chunks" bar above the
    # encoder's native-chain row, both rendered in one cohesive display.
    # The caller owns the stack in that mode; we just add our row.
    # progress_stack is an injected, structurally typed progress stack.
    # KinoMLX's progress UI is rich-based and the
    # caller wires it in; without an injected stack the encoder runs bar-less
    # (the status summary is still printed).
    pbar: _ProgressBar | None = None
    if verbose and progress_stack is not None:
        pbar = progress_stack.add(
            total=n_source_frames,
            desc=output_phase,
            unit="frame",
        )

    started = time.perf_counter()
    n_in = 0
    n_out = 0
    n_orig = 0
    processing_failure: BaseException | None = None
    try:
        for src_frame in frame_iter:
            _validate_source_frame(src_frame, source_signal)
            is_cut = cut_detection_active and cut_detector.is_cut(src_frame)
            if is_cut:
                _log.info(
                    "  scene cut before source frame %d: resetting native frame history",
                    n_in + 1,
                )
                if vsr is not None and vsr.uses_frame_history:
                    vsr.reset_frame_history()
            with objc.autorelease_pool():
                # Companion "original" writer: source frame -> orig_src_fmt
                # (NV12 or RGBAHalf, matching the primary's precision
                # envelope) -> append.  Independent of VSR/VTFRC chain;
                # uses its own source buffer pool.  AVAssetWriter's audio
                # + video pumps run on their own GCD queues so this second
                # append is largely parallel to the primary chain's encode
                # pass. upload_frame_to_buffer dispatches the float16 RGB
                # source into the delivery-selected native pixel format.
                if writer_orig is not None:
                    orig_pb = _allocate_writer_src_buffer(
                        None if orig_yuv_feed else writer_orig.adaptor,
                        in_w,
                        in_h,
                        orig_src_fmt,
                    )
                    _pb.upload_frame_to_buffer(src_frame, orig_pb)
                    writer_orig.append(orig_pb)
                    n_orig += 1
                    del orig_pb

                if vsr is not None:
                    src_pb = vsr.upscale_to_buffer(src_frame, n_in)
                elif vtfrc is not None:
                    src_pb = vtfrc.make_source_buffer()
                    _pb.upload_frame_to_buffer(src_frame, src_pb)
                else:
                    src_pb = _allocate_writer_src_buffer(
                        writer.adaptor,
                        in_w,
                        in_h,
                        writer_src_fmt,
                    )
                    _pb.upload_frame_to_buffer(src_frame, src_pb)

                if vtfrc is not None:
                    temporal_outputs = (
                        vtfrc.feed_cut(src_pb, n_in) if is_cut else vtfrc.feed(src_pb, n_in)
                    )
                    for out_pb in temporal_outputs:
                        writer.append(out_pb)
                        n_out += 1
                        del out_pb
                else:
                    writer.append(src_pb)
                    n_out += 1
                del src_pb
            n_in += 1
            reporter.phase_advance(output_phase)
            if pbar is not None:
                pbar.update(1)
            # Periodic janitorial work: CIContext caches + src pool drain.
            if n_in % 64 == 0:
                _pb.clear_ci_caches()
                if vsr is not None:
                    vsr.flush_pools()
        if n_in != n_source_frames:
            raise RuntimeError(f"source produced {n_in} frames; expected {n_source_frames}")
        # Emit the buffered final source period.
        if vtfrc is not None:
            for out_pb in vtfrc.drain():
                writer.append(out_pb)
                n_out += 1
                del out_pb
    except BaseException as exc:
        processing_failure = exc

    _close_native_chain(
        writer,
        writer_orig,
        vtfrc,
        vsr,
        primary_failure=processing_failure,
    )

    if verbose:
        elapsed = time.perf_counter() - started
        size = output_path.stat().st_size
        orig_part = ""
        if writer_orig is not None and orig_path is not None:
            orig_size = orig_path.stat().st_size
            orig_part = (
                f" + original {_human_size(orig_size)} "
                f"({n_orig} src frame{'s' if n_orig != 1 else ''})"
            )
        done_msg = (
            f"  done: {_human_size(size)} in {elapsed:.1f}s "
            f"({n_in} src frame{'s' if n_in != 1 else ''}, "
            f"{n_out} written){orig_part}"
        )
        # In caller-managed-stack mode the bars are still alive at this
        # point (the caller closes them after we return), so a raw print
        # would stomp on the bottom bar's row.  Route through bars.write()
        # with position="below" so the visual order in the persisted
        # scrollback is "bars (frozen at 100%) -> done summary".  The
        # stack implementation owns any redraw/reset behavior around that
        # message; the caller still owns its eventual cleanup.
        if progress_stack is not None:
            progress_stack.write(done_msg, position="below")
        else:
            _log.info(done_msg)

    return output_path


def encode_video_videotoolbox(
    frames: FrameSource,
    output_path: str | Path,
    *,
    fps: float,
    source_signal: VideoSignalSpec,
    delivery: EncodedVideoDeliverySpec,
    audio_waveform: mx.array | None = None,
    audio_sample_rate: int | None = None,
    audio_bit_depth: str = "float32",
    save_audio_sidecar: bool = False,
    audio_onset_trim_mode: str = "auto",
    audio_onset_trim_ms: float | None = None,
    vsr_spatial_mode: str | None = None,
    target_fps: float | None = None,
    vsr_temporal_mode: str = "normal",
    vsr_save_original: bool = False,
    cut_detect_mode: str = "simple",
    cut_detect_threshold: float | None = None,
    encode_quality: float = 0.65,
    n_source_frames: int | None = None,
    progress_stack: ProgressStack | None = None,
    reporter: Reporter | None = None,
    audio_codec: str = "alac",
    verbose: bool = True,
    native_verbose: bool = False,
) -> Path:
    """Validate a bounded SDR encode and report one progress unit per frame."""
    validate_sdr_output_plan(OutputColorPlan(source=source_signal, deliveries=(delivery,)))
    if source_signal.cadence != _pb.rational_cadence(fps):
        raise UnsupportedSignalError(
            f"source cadence {source_signal.cadence} does not match encoder fps {fps}"
        )

    high_quality_vsr = vsr_spatial_mode in {"balanced", "image"}
    temporal = target_fps is not None and abs(target_fps - fps) > 1e-6
    needs_422 = delivery.chroma is ChromaSubsampling.YUV422
    producer_requires_422 = high_quality_vsr or temporal
    if producer_requires_422 and not needs_422:
        producer = "frame-rate conversion" if temporal else vsr_spatial_mode
        raise UnsupportedSignalError(f"VideoToolbox {producer} path requires 4:2:2 delivery")
    if vsr_spatial_mode == "fast" and not temporal and needs_422:
        raise UnsupportedSignalError("VideoToolbox fast path requires 4:2:0 delivery")

    # Construct before peeking so malformed terminal policy never consumes a
    # lazy decoder stream or opens a native session.
    cut_detector = CutDetector(cut_detect_mode, cut_detect_threshold)

    if isinstance(frames, mx.array) and frames.ndim != 4:
        raise ValueError("encode_video_videotoolbox: an MLX frame batch must have shape (T,H,W,C)")
    if n_source_frames is None:
        if isinstance(frames, mx.array):
            n_source_frames = int(frames.shape[0])
        elif isinstance(frames, Sequence):
            n_source_frames = len(frames)
        else:
            raise ValueError("n_source_frames is required for an unsized streaming source")
    if n_source_frames <= 0:
        raise ValueError("encode_video_videotoolbox: empty frames source")

    # Pull the first lazy frame before creating the terminal row. Decoder-owned
    # phases therefore appear first, while signal/delivery rejection above
    # still happens without consuming the stream.
    first, frame_iter, _peeked_total = _peek_frames(frames)
    _validate_source_frame(first, source_signal)
    sink = reporter if reporter is not None else NullReporter()
    output_phase = _output_phase_name(
        vsr_spatial_mode=vsr_spatial_mode,
        source_fps=fps,
        target_fps=target_fps,
    )
    sink.phase_start(output_phase, total=n_source_frames, unit="frame")
    try:
        return _encode_video_videotoolbox_impl(
            first,
            frame_iter,
            output_path,
            fps=fps,
            source_signal=source_signal,
            delivery=delivery,
            reporter=sink,
            output_phase=output_phase,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            audio_bit_depth=audio_bit_depth,
            save_audio_sidecar=save_audio_sidecar,
            audio_onset_trim_mode=audio_onset_trim_mode,
            audio_onset_trim_ms=audio_onset_trim_ms,
            vsr_spatial_mode=vsr_spatial_mode,
            target_fps=target_fps,
            vsr_temporal_mode=vsr_temporal_mode,
            vsr_save_original=vsr_save_original,
            cut_detector=cut_detector,
            encode_quality=encode_quality,
            n_source_frames=n_source_frames,
            progress_stack=progress_stack,
            audio_codec=audio_codec,
            verbose=verbose,
            native_verbose=native_verbose,
        )
    finally:
        sink.phase_end(output_phase)
