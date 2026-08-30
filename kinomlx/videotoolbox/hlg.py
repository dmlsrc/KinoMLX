"""Native scene-linear RGB to BT.2020/HLG P010 conversion."""

from __future__ import annotations

import Foundation
import mlx.core as mx
import objc
import Quartz

from kinomlx.io.buffer import mlx_array_from_buffer
from kinomlx.media.signals import ColorPrimaries

from . import pixel_buffers as _pb

PIX_P010_VIDEO = 2016686640  # kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange

_hlg_space: object | None = None
_linear_spaces: dict[ColorPrimaries, object] = {}


def _hlg_colorspace() -> object:
    global _hlg_space
    if _hlg_space is None:
        _hlg_space = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceITUR_2100_HLG)
    if _hlg_space is None:
        raise RuntimeError("cannot create BT.2100 HLG color space")
    return _hlg_space


def _linear_colorspace(primaries: ColorPrimaries) -> object:
    cached = _linear_spaces.get(primaries)
    if cached is not None:
        return cached
    if primaries is ColorPrimaries.ACESCG:
        name = Quartz.kCGColorSpaceACESCGLinear
    elif primaries is ColorPrimaries.REC709:
        name = Quartz.kCGColorSpaceExtendedLinearSRGB
    else:
        raise ValueError(f"HLG terminal does not support {primaries.value} source primaries")
    space = Quartz.CGColorSpaceCreateWithName(name)
    if space is None:
        raise RuntimeError(f"cannot create {primaries.value} scene-linear color space")
    _linear_spaces[primaries] = space
    return space


def _linear_ci_image(frame: mx.array, primaries: ColorPrimaries) -> object:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"HLG source frame must be HWC RGB, got {tuple(frame.shape)}")
    if frame.dtype != mx.float32:
        raise TypeError(f"HLG source frame must be float32, got {frame.dtype}")
    if not bool(mx.all(mx.isfinite(frame)).item()):
        raise ValueError("HLG source frame must contain only finite values")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    rgba = mx.concatenate([frame, mx.ones((height, width, 1), dtype=mx.float32)], axis=-1)
    rgba = mx.contiguous(rgba)
    mx.eval(rgba)
    raw = memoryview(rgba).cast("B")
    data = Foundation.NSData.dataWithBytes_length_(raw, len(raw))
    image = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data,
        width * 16,
        (width, height),
        Quartz.kCIFormatRGBAf,
        _linear_colorspace(primaries),
    )
    if image is None:
        raise RuntimeError("cannot create scene-linear CoreImage frame")
    return image


def prepare_hlg_scene_linear(
    frame: mx.array,
    *,
    primaries: ColorPrimaries,
) -> mx.array:
    """Resolve the HLG encoder's non-negative source-primary boundary."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"HLG source frame must be HWC RGB, got {tuple(frame.shape)}")
    if frame.dtype != mx.float32:
        raise TypeError(f"HLG source frame must be float32, got {frame.dtype}")
    if not bool(mx.all(mx.isfinite(frame)).item()):
        raise ValueError("HLG source frame must contain only finite values")
    if primaries not in {ColorPrimaries.ACESCG, ColorPrimaries.REC709}:
        raise ValueError(f"HLG terminal does not support {primaries.value} source primaries")
    # Clip only negative light in the declared source gamut. CoreImage then
    # converts ACEScg or Rec.709 directly into BT.2020/HLG, avoiding a lossy
    # clamp in the smaller Rec.709 gamut for ACEScg producers.
    prepared = mx.maximum(frame, 0.0)
    mx.eval(prepared)
    return prepared


def scene_linear_to_hlg_codes(
    frame: mx.array,
    *,
    primaries: ColorPrimaries,
) -> mx.array:
    """Return Apple's color-managed float32 BT.2020/HLG RGB result."""
    frame = prepare_hlg_scene_linear(frame, primaries=primaries)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    with objc.autorelease_pool():
        output = _pb.ci_context().createCGImage_fromRect_format_colorSpace_(
            _linear_ci_image(frame, primaries),
            Quartz.CGRectMake(0, 0, width, height),
            Quartz.kCIFormatRGBAf,
            _hlg_colorspace(),
        )
        if output is None:
            raise RuntimeError("CoreImage could not convert scene-linear RGB to HLG")
        row_bytes = int(Quartz.CGImageGetBytesPerRow(output))
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(output))
        raw = mlx_array_from_buffer(memoryview(data), dtype=mx.uint8)[: height * row_bytes]
        values = raw.view(mx.float32).reshape(height, row_bytes // 4)
        rgba = values[:, : width * 4].reshape(height, width, 4)
        rgb = mx.contiguous(rgba[:, :, :3])
        mx.eval(rgb)
        return rgb


def render_scene_linear_to_p010(
    frame: mx.array,
    pixel_buffer: object,
    *,
    primaries: ColorPrimaries,
) -> None:
    """Render one float32 scene-linear frame into a tagged P010 buffer."""
    frame = prepare_hlg_scene_linear(frame, primaries=primaries)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if int(Quartz.CVPixelBufferGetPixelFormatType(pixel_buffer)) != PIX_P010_VIDEO:
        raise ValueError("HLG render destination must be P010 video range")
    if (
        int(Quartz.CVPixelBufferGetWidth(pixel_buffer)),
        int(Quartz.CVPixelBufferGetHeight(pixel_buffer)),
    ) != (width, height):
        raise ValueError("HLG source and P010 destination geometry do not match")
    with objc.autorelease_pool():
        _pb.ci_context().render_toCVPixelBuffer_bounds_colorSpace_(
            _linear_ci_image(frame, primaries),
            pixel_buffer,
            Quartz.CGRectMake(0, 0, width, height),
            _hlg_colorspace(),
        )
    mode = Quartz.kCVAttachmentMode_ShouldPropagate
    for key, value in (
        (
            Quartz.kCVImageBufferColorPrimariesKey,
            Quartz.kCVImageBufferColorPrimaries_ITU_R_2020,
        ),
        (
            Quartz.kCVImageBufferTransferFunctionKey,
            Quartz.kCVImageBufferTransferFunction_ITU_R_2100_HLG,
        ),
        (
            Quartz.kCVImageBufferYCbCrMatrixKey,
            Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020,
        ),
        (
            Quartz.kCVImageBufferChromaLocationTopFieldKey,
            Quartz.kCVImageBufferChromaLocation_Left,
        ),
    ):
        Quartz.CVBufferSetAttachment(pixel_buffer, key, value, mode)


def make_hlg_pixel_buffer(
    frame: mx.array,
    adaptor: _pb.PixelBufferAdaptor,
    *,
    primaries: ColorPrimaries,
) -> object:
    """Acquire one bounded writer-pool buffer and fill it with HLG P010."""
    pool = adaptor.pixelBufferPool()
    if pool is None:
        raise RuntimeError("HLG writer did not expose a P010 pixel-buffer pool")
    buffer = _pb.pool_create_buffer(pool, allocation_threshold=_pb.WRITER_POOL_LIMIT)
    if buffer is None:
        raise RuntimeError("bounded HLG writer pool is exhausted")
    render_scene_linear_to_p010(frame, buffer, primaries=primaries)
    return buffer


__all__ = [
    "PIX_P010_VIDEO",
    "make_hlg_pixel_buffer",
    "prepare_hlg_scene_linear",
    "render_scene_linear_to_p010",
    "scene_linear_to_hlg_codes",
]
