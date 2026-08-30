"""CVPixelBuffer + CMTime helpers for the VideoToolbox bridge.

VSR's source/dst formats (NV12, RGBAHalf), the BGRA buffer used for the
side-by-side comparison, and the CoreImage-based upload path for converting
MLX frames into IOSurface-backed CVPixelBuffers all live here. Plus the
fixed-timescale `_frame_pts` so VSR and AVWriter agree on PTSes for any
arbitrary fps.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from fractions import Fraction
from typing import Protocol, SupportsIndex, cast

import CoreMedia
import Foundation
import mlx.core as mx
import Quartz

from kinomlx.io.buffer import mlx_array_from_buffer

# FourCC pixel-format constants ----------------------------------------------
#
# CV uses big-endian four-character codes packed into a uint32. PIX_BGRA is
# the common 8-bit RGBA destination used for the comparison composite;
# PIX_RGBAHALF is the half-float RGBA source VSR HighQuality expects;
# PIX_NV12 is what LL VSR (and HEVC encoders) consume.
PIX_BGRA = int.from_bytes(b"BGRA", "big")  # 0x42475241
PIX_RGBAHALF = 1380411457  # 'RGhA' kCVPixelFormatType_64RGBAHalf
PIX_NV12 = 875704438  # '420v' kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange

# Proven in-flight windows. Every hot-path pool acquisition supplies one of
# these (or a ratio-derived temporal limit) to Core Video.
# AVAssetWriter can retain seven submitted surfaces while its hardware queue
# drains; one active upload brings the proven adaptor window to eight.
WRITER_POOL_LIMIT = 8
# Stateless VSR modes need only the active upload plus the framework-owned
# prior submission. Balanced VSR with Video input also supplies its explicit
# previous source, so the next submission needs a third source surface.
VSR_STATELESS_SOURCE_POOL_LIMIT = 2
VSR_VIDEO_INPUT_SOURCE_POOL_LIMIT = 3
VSR_DESTINATION_POOL_LIMIT = WRITER_POOL_LIMIT + 2
TEMPORAL_SOURCE_POOL_LIMIT = 2


# CMTime base for video PTS --------------------------------------------------
#
# 24000 lands bit-exact for 24/25/30/48/50/60 and 24000/1001. Other NTSC
# rates alternate adjacent integer durations on an exact rational index grid;
# no per-frame rounding error accumulates. Picked over 600, whose coarse ticks
# cannot represent those presentation times closely enough.
VIDEO_TIME_SCALE = 24000


class PixelBufferAdaptor(Protocol):
    """Native adaptor methods consumed by KinoMLX."""

    def pixelBufferPool(self) -> object | None: ...

    def appendPixelBuffer_withPresentationTime_(
        self,
        pixel_buffer: object,
        presentation_time: object,
    ) -> bool: ...


class _CIContext(Protocol):
    """CoreImage context selectors used across the native media modules."""

    def clearCaches(self) -> None: ...

    def render_toCVPixelBuffer_(self, image: object, pixel_buffer: object) -> object: ...

    def render_toBitmap_rowBytes_bounds_format_colorSpace_(
        self,
        image: object,
        bitmap: object,
        row_bytes: int,
        bounds: object,
        pixel_format: object,
        color_space: object,
    ) -> object: ...

    def createCGImage_fromRect_(self, image: object, rect: object) -> object | None: ...

    def createCGImage_fromRect_format_colorSpace_(
        self,
        image: object,
        rect: object,
        pixel_format: object,
        color_space: object,
    ) -> object | None: ...

    def render_toCVPixelBuffer_bounds_colorSpace_(
        self,
        image: object,
        pixel_buffer: object,
        bounds: object,
        color_space: object,
    ) -> object: ...


class _Indexable(Protocol):
    def __getitem__(self, index: int, /) -> object: ...


_ci_context: _CIContext | None = None
_srgb: object | None = None


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


def ci_context() -> _CIContext:
    """Shared CIContext for all RGB <-> CVPixelBuffer conversions."""
    global _ci_context
    if _ci_context is None:
        context = Quartz.CIContext.contextWithOptions_(None)
        if context is None:
            raise RuntimeError("cannot create shared CoreImage context")
        _ci_context = cast(_CIContext, context)
    return _ci_context


def clear_ci_caches() -> None:
    """Tell CIContext to drop its internal Metal/CG caches.

    CIContext caches intermediate compute resources (rendered tiles, GPU
    pipeline states, etc.) across render calls for performance. In a long
    loop that does one render per frame these caches grow continuously
    even though we never reuse a CIImage. Periodic clearCaches() releases
    them back to the system.
    """
    if _ci_context is not None:
        _ci_context.clearCaches()


def srgb_colorspace() -> object:
    """Shared sRGB CGColorSpace handle (cheap to create but reused for clarity)."""
    global _srgb
    if _srgb is None:
        _srgb = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
    if _srgb is None:
        raise RuntimeError("cannot create shared sRGB color space")
    return _srgb


# ---------------------------------------------------------------------------
# CMTime helpers
# ---------------------------------------------------------------------------


def rational_cadence(
    value: Fraction | int | float | str,
    *,
    max_denominator: int = 1_000_000,
) -> Fraction:
    """Return a positive exact cadence from a public numeric value."""
    if isinstance(value, bool):
        raise ValueError("cadence must be a positive number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cadence must be finite")
    try:
        cadence = (
            value
            if isinstance(value, Fraction)
            else Fraction(str(value)).limit_denominator(max_denominator)
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid cadence {value!r}") from exc
    if cadence <= 0:
        raise ValueError(f"cadence must be positive, got {value!r}")
    return cadence


def frame_ticks(
    frame_index: int,
    cadence: Fraction | int | float | str,
) -> int:
    """Round one complete rational frame position into the video time base."""
    exact_cadence = rational_cadence(cadence)
    return round(Fraction(frame_index * VIDEO_TIME_SCALE) / exact_cadence)


def frame_pts(
    frame_index: int,
    fps: Fraction | int | float | str,
) -> object:
    """Build a CMTime for a video frame index at the given fps.

    The complete rational index is rounded once. Unlike multiplying a rounded
    one-frame duration, the error therefore stays within half one output tick
    for arbitrarily long NTSC-family sequences.
    """
    return cast(object, CoreMedia.CMTimeMake(frame_ticks(frame_index, fps), VIDEO_TIME_SCALE))


# ---------------------------------------------------------------------------
# Pixel format inspection
# ---------------------------------------------------------------------------


def resolve_pixel_format(attrs: Mapping[object, object]) -> int:
    """Extract the PixelFormatType from a VT config's attributes dict.

    Quirk: VTSuperResolutionScalerConfiguration returns its supported source
    formats as a single-element NSArray, not a bare int. Unwrap if needed.
    """
    original_value = attrs.get("PixelFormatType")
    value = original_value
    if value is None:
        raise ValueError("pixel-buffer attributes do not contain PixelFormatType")
    if not isinstance(value, int):
        if not hasattr(value, "__getitem__"):
            raise ValueError(f"invalid PixelFormatType value: {value!r}")
        try:
            value = cast(_Indexable, value)[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("PixelFormatType collection is empty or invalid") from exc
    if isinstance(value, bool):
        raise ValueError("PixelFormatType must be an integer FourCC")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError(f"invalid PixelFormatType value: {original_value!r}") from exc


# ---------------------------------------------------------------------------
# CVPixelBuffer creation
# ---------------------------------------------------------------------------


def make_pixel_buffer_from_attrs(
    width: int,
    height: int,
    attrs: Mapping[object, object],
) -> object:
    """Allocate a fresh CVPixelBuffer from a VT config's attributes dict.

    Used as a fallback when a CVPixelBufferPool isn't available (e.g., before
    AVAssetWriter has been started); pools are preferred for hot paths.
    """
    fmt = resolve_pixel_format(attrs)
    err, pb = Quartz.CVPixelBufferCreate(None, width, height, fmt, attrs, None)
    if err != 0 or pb is None:
        raise RuntimeError(
            f"CVPixelBufferCreate({width}x{height}, fmt={fmt:#x}) failed: status={err}"
        )
    return cast(object, pb)


def make_pool_from_attrs(attrs: Mapping[object, object]) -> object | None:
    """Try to create a CVPixelBufferPool for the given attrs; None on failure.

    Caller should fall back to make_pixel_buffer_from_attrs if this returns
    None - some attribute combos don't pool cleanly.
    """
    err, pool = Quartz.CVPixelBufferPoolCreate(None, None, attrs, None)
    if err != 0 or pool is None:
        return None
    return cast(object, pool)


def pool_create_buffer(pool: object, *, allocation_threshold: int) -> object | None:
    """Pull from a pool without permitting growth past the declared window."""
    if (
        isinstance(allocation_threshold, bool)
        or not isinstance(allocation_threshold, int)
        or allocation_threshold <= 0
    ):
        raise ValueError("allocation_threshold must be a positive integer")
    threshold_key = Quartz.kCVPixelBufferPoolAllocationThresholdKey
    aux = {threshold_key: allocation_threshold}
    err, pb = Quartz.CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
        None,
        pool,
        aux,
        None,
    )
    if err != 0 or pb is None:
        return None
    return cast(object, pb)


def flush_pool(pool: object | None) -> None:
    """Release any excess cached buffers in a CVPixelBufferPool.

    Pools cache returned buffers for reuse (default age threshold ~1s) and
    don't expose `kCVPixelBufferPoolAllocationThresholdKey` by default -
    they grow to whatever peak buffer count the workload demands and stay
    there. For long runs that's a memory leak from the user's perspective.
    Calling `CVPixelBufferPoolFlush` with `kCVPixelBufferPoolFlushExcessBuffers`
    aggressively releases the cached-but-currently-unused buffers back to
    the system.
    """
    if pool is None:
        return
    # kCVPixelBufferPoolFlushExcessBuffers = 1
    Quartz.CVPixelBufferPoolFlush(pool, 1)


def make_bgra_buffer(
    adaptor: PixelBufferAdaptor | None,
    width: int,
    height: int,
) -> object:
    """Get a BGRA CVPixelBuffer for the comparison composite output.

    Prefers the AVAssetWriter adaptor's own pool (zero-copy into the encoder);
    falls back to fresh allocation if the pool isn't ready yet.
    """
    pool = adaptor.pixelBufferPool() if adaptor is not None else None
    if pool is not None:
        pb = pool_create_buffer(pool, allocation_threshold=WRITER_POOL_LIMIT)
        if pb is not None:
            return pb
        raise RuntimeError("bounded BGRA writer pool is exhausted")
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: PIX_BGRA,
        Quartz.kCVPixelBufferWidthKey: width,
        Quartz.kCVPixelBufferHeightKey: height,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    err, pb = Quartz.CVPixelBufferCreate(None, width, height, PIX_BGRA, attrs, None)
    if err != 0 or pb is None:
        raise RuntimeError(f"CVPixelBufferCreate({width}x{height}, BGRA) failed: {err}")
    return cast(object, pb)


# ---------------------------------------------------------------------------
# frame -> CVPixelBuffer
# ---------------------------------------------------------------------------


def _frame_is_fp16(frame: mx.array) -> bool:
    """Return whether an MLX frame uses float16 storage."""
    return str(frame.dtype).split(".")[-1] == "float16"


def _frame_buffer(frame: mx.array) -> memoryview:
    """A contiguous uint8-format memoryview over a frame's bytes, no copy.

    MLX arrays go through the buffer protocol (mx.contiguous + memoryview).
    This lets the caller memcpy straight from the array's unified memory into
    an IOSurface plane instead of materializing an intermediate ``bytes``
    object first.
    """
    return memoryview(mx.contiguous(frame)).cast("B")


def _opaque_fp16_rgba(frame: mx.array) -> mx.array:
    """Return one contiguous fp16 RGBA frame, adding alpha to HWC RGB."""
    f = frame
    if f.ndim != 3 or f.shape[2] not in (3, 4):
        raise ValueError(f"float16 frame must be HWC RGB or RGBA, got {tuple(f.shape)}")
    if f.dtype != mx.float16:
        f = f.astype(mx.float16)
    if f.shape[2] == 3:
        alpha = mx.ones((f.shape[0], f.shape[1], 1), dtype=mx.float16)
        f = mx.concatenate([f, alpha], axis=-1)
    f = mx.contiguous(f)
    mx.eval(f)
    return f


def write_fp16_rgba(rgba_fp16: mx.array, pb: object) -> None:
    """Memcpy an MLX ``(H,W,4)`` fp16 RGBA frame into a RGBAHalf CVPixelBuffer.

    Used for the HQ VSR source upload (RGBAHalf format) and any other case where
    we already have the exact destination layout. The base address is an
    objc.varlist whose `.as_buffer(n)` is a writable memoryview into the IOSurface
    plane; the source bytes come straight from the frame's buffer.
    """
    h, w = int(rgba_fp16.shape[0]), int(rgba_fp16.shape[1])
    # Zero-copy view of the frame's buffer; the single mv[:] = src memcpy below
    # goes straight from MLX's unified memory into the IOSurface plane, with no
    # intermediate bytes object.
    src = _frame_buffer(rgba_fp16)
    row = w * 8
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        mv = base.as_buffer(h * bpr)
        if bpr == row:
            mv[:] = src
        else:
            # Row-pad case: copy each row's bytes, skipping the destination pad.
            for r in range(h):
                mv[r * bpr : r * bpr + row] = src[r * row : (r + 1) * row]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def upload_frame_to_buffer(frame: mx.array, pb: object) -> None:
    """Upload `frame` into `pb`, dispatching on the buffer's pixel format.

    Accepted MLX inputs:
      - (H,W,3) uint8 RGB           : --video / ffmpeg rgb24 path
      - (H,W,4) fp16 RGBA           : --latent / chunk_to_rgba_fp16 path

    Accepted destinations:
      - NV12 ('420v')               : LowLatency VSR source
      - RGBAHalf ('RGhA')           : HighQuality VSR source

    The NV12 destination always goes through CoreImage so the sRGB->BT.709
    YUV conversion is correct. CIImage's source format is RGBA8 for uint8
    input and RGBAh for fp16 input - using RGBAh defers quantization to
    CIContext's render pass so the single 8-bit cast happens in YUV space
    rather than once in RGB and once in YUV.

    The RGBAHalf destination is a direct memcpy when the source is already
    fp16 RGBA. For uint8 input we promote to fp16 inline. All channel math runs
    in MLX and the bytes come from the array buffer, so no numpy - and the
    CoreImage / memcpy calls are unchanged, so chroma is byte-for-byte identical.
    """
    pix_fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
    h, w = int(frame.shape[0]), int(frame.shape[1])

    if pix_fmt == PIX_RGBAHALF:
        if _frame_is_fp16(frame):
            write_fp16_rgba(_opaque_fp16_rgba(frame), pb)
            return
        # uint8 RGB -> fp16 RGBA promotion (legacy / --video path).
        rgb = frame.astype(mx.float16) * mx.array(1.0 / 255.0, dtype=mx.float16)
        alpha = mx.ones((h, w, 1), dtype=mx.float16)
        write_fp16_rgba(mx.concatenate([rgb, alpha], axis=-1), pb)
        return

    # NV12 (and any other format CoreImage can render into). Pick the CIImage
    # source format from the input dtype: RGBAh for fp16, RGBA8 for uint8.
    if _frame_is_fp16(frame):
        rgba = _opaque_fp16_rgba(frame)
        src = _frame_buffer(rgba)
        data = Foundation.NSData.dataWithBytes_length_(src, len(src))
        ci_image = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
            data,
            w * 8,
            (w, h),
            Quartz.kCIFormatRGBAh,
            srgb_colorspace(),
        )
        ci_context().render_toCVPixelBuffer_(ci_image, pb)
        return

    # uint8 RGB -> opaque RGBA8 for CoreImage.
    alpha = mx.full((h, w, 1), 255, dtype=mx.uint8)
    src = _frame_buffer(mx.concatenate([frame, alpha], axis=-1))
    data = Foundation.NSData.dataWithBytes_length_(src, len(src))
    ci_image = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data,
        w * 4,
        (w, h),
        Quartz.kCIFormatRGBA8,
        srgb_colorspace(),
    )
    ci_context().render_toCVPixelBuffer_(ci_image, pb)


# ---------------------------------------------------------------------------
# CVPixelBuffer -> mlx
# ---------------------------------------------------------------------------


def read_rgbahalf_rgb(pb: object) -> mx.array:
    """Read RGBAHalf directly into a float32 ``(H, W, 3)`` MLX array.

    This avoids CoreImage's 8-bit render and retains the producer's half-float
    samples for the explicit RGB-to-YUV writer conversion.
    """
    w = Quartz.CVPixelBufferGetWidth(pb)
    h = Quartz.CVPixelBufferGetHeight(pb)
    pixel_format = int(Quartz.CVPixelBufferGetPixelFormatType(pb))
    if pixel_format != PIX_RGBAHALF:
        raise ValueError(f"expected RGBAHalf source format, got {pixel_format:#x}")
    Quartz.CVPixelBufferLockBaseAddress(pb, 1)
    try:
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        if bpr < w * 8 or bpr % 2:
            raise ValueError(f"invalid RGBAHalf row stride {bpr} for width {w}")
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        # MLX 0.32.1 can adopt page-aligned host storage on unified memory. The
        # borrowed view exists only while this retained pixel buffer is locked;
        # an ineligible plane takes an explicit owned-copy fallback. The
        # derived float32 result is evaluated before unlock, so no borrowed
        # view escapes this lexical scope.
        plane = memoryview(base.as_buffer(h * bpr))
        try:
            raw = mx.asarray(plane, copy=False)
        except RuntimeError, TypeError, ValueError:
            raw = mlx_array_from_buffer(plane)
        half = raw.view(mx.float16).reshape(h, bpr // 2)[:, : w * 4].reshape(h, w, 4)
        rgb = mx.contiguous(half[..., :3]).astype(mx.float32)
        mx.eval(rgb)
        del half, raw
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    return rgb


def read_pixel_buffer_rgb(pb: object) -> mx.array:
    """Read any CVPixelBuffer into a (H, W, 3) uint8 RGB mlx array via CoreImage.

    Goes through CIImage(CVPixelBuffer) + CIContext.render_toBitmap, so any
    source format (NV12, RGBAHalf, BGRA, ...) is handled uniformly. Slower
    than a direct memcpy for the trivial cases but correct everywhere.
    """
    w = Quartz.CVPixelBufferGetWidth(pb)
    h = Quartz.CVPixelBufferGetHeight(pb)
    ci_image = Quartz.CIImage.alloc().initWithCVPixelBuffer_(pb)
    buf = bytearray(w * h * 4)
    ci_context().render_toBitmap_rowBytes_bounds_format_colorSpace_(
        ci_image,
        buf,
        w * 4,
        ((0, 0), (w, h)),
        Quartz.kCIFormatRGBA8,
        srgb_colorspace(),
    )
    rgba = mlx_array_from_buffer(memoryview(buf)).reshape(h, w, 4)
    return mx.contiguous(rgba[..., :3])
