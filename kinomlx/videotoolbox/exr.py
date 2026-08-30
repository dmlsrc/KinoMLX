"""Native half-float OpenEXR frame I/O through ImageIO."""

from __future__ import annotations

import json
from pathlib import Path

import Foundation
import mlx.core as mx
import objc
import Quartz

from kinomlx.io.buffer import mlx_array_from_buffer
from kinomlx.media.signals import ColorPrimaries, ColorTransfer, ExrDeliverySpec

_OPENEXR_UTI = "com.ilm.openexr-image"


def _color_space(primaries: ColorPrimaries) -> object:
    if primaries not in {ColorPrimaries.ACESCG, ColorPrimaries.REC709}:
        raise ValueError(f"EXR authoring does not support {primaries.value} primaries")
    # ImageIO preserves extended-linear component values while finalizing the
    # EXR. The adjacent manifest carries the authoring primaries independently
    # so ACEScg and log-coded values are never transformed on write.
    space = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceExtendedLinearSRGB)
    if space is None:
        raise RuntimeError(f"cannot create {primaries.value} EXR color space")
    return space


def _half_rgba_image(rgb: mx.array, primaries: ColorPrimaries) -> object:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"EXR frame must be HWC RGB, got {tuple(rgb.shape)}")
    if rgb.dtype not in (mx.float16, mx.float32, mx.bfloat16):
        raise TypeError(f"EXR frame must be floating point, got {rgb.dtype}")
    if not bool(mx.all(mx.isfinite(rgb)).item()):
        raise ValueError("EXR frame must contain only finite values")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    rgba = mx.concatenate(
        [rgb.astype(mx.float16), mx.ones((height, width, 1), dtype=mx.float16)],
        axis=-1,
    )
    rgba = mx.contiguous(rgba)
    mx.eval(rgba)
    raw = memoryview(rgba).cast("B")
    data = Foundation.NSData.dataWithBytes_length_(raw, len(raw))
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    bitmap_info = (
        Quartz.kCGBitmapFloatComponents
        | Quartz.kCGBitmapByteOrder16Little
        | Quartz.kCGImageAlphaPremultipliedLast
    )
    image = Quartz.CGImageCreate(
        width,
        height,
        16,
        64,
        width * 8,
        _color_space(primaries),
        bitmap_info,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )
    if image is None:
        raise RuntimeError("cannot create half-float EXR source image")
    return image


def save_exr_frame(
    frame: mx.array,
    path: Path | str,
    *,
    delivery: ExrDeliverySpec,
) -> Path:
    """Write one finite RGB frame as a native half-float OpenEXR image."""
    if delivery.sample_type.value != "float16":
        raise ValueError("native EXR terminal currently requires float16 samples")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with objc.autorelease_pool():
        destination = Quartz.CGImageDestinationCreateWithURL(
            Foundation.NSURL.fileURLWithPath_(str(target)),
            _OPENEXR_UTI,
            1,
            None,
        )
        if destination is None:
            raise RuntimeError(f"cannot create EXR destination {target}")
        Quartz.CGImageDestinationAddImage(
            destination,
            _half_rgba_image(frame, delivery.primaries),
            None,
        )
        if not Quartz.CGImageDestinationFinalize(destination):
            raise RuntimeError(f"cannot finalize EXR frame {target}")
    return target


def read_exr_frame(path: Path | str) -> mx.array:
    """Read ImageIO's native floating EXR payload without display conversion."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"EXR frame does not exist: {source_path}")
    with objc.autorelease_pool():
        source = Quartz.CGImageSourceCreateWithURL(
            Foundation.NSURL.fileURLWithPath_(str(source_path)),
            None,
        )
        if source is None:
            raise ValueError(f"cannot open EXR frame {source_path}")
        image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        if image is None:
            raise ValueError(f"cannot decode EXR frame {source_path}")
        width = int(Quartz.CGImageGetWidth(image))
        height = int(Quartz.CGImageGetHeight(image))
        bits = int(Quartz.CGImageGetBitsPerComponent(image))
        bits_per_pixel = int(Quartz.CGImageGetBitsPerPixel(image))
        row_bytes = int(Quartz.CGImageGetBytesPerRow(image))
        info = int(Quartz.CGImageGetBitmapInfo(image))
        if not info & int(Quartz.kCGBitmapFloatComponents):
            raise ValueError(f"EXR frame {source_path} did not decode as floating point")
        if bits not in (16, 32) or bits_per_pixel % bits:
            raise ValueError(
                f"unsupported EXR pixel layout: {bits}-bit components, {bits_per_pixel} bits/pixel"
            )
        channels = bits_per_pixel // bits
        if channels < 3:
            raise ValueError(f"EXR frame must contain at least 3 channels, got {channels}")
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
        raw = mlx_array_from_buffer(memoryview(data), dtype=mx.uint8)
        dtype = mx.float16 if bits == 16 else mx.float32
        row_values = row_bytes * 8 // bits
        values = raw.view(dtype).reshape(height, row_values)
        pixels = values[:, : width * channels].reshape(height, width, channels)
        rgb = mx.contiguous(pixels[:, :, :3].astype(mx.float32))
        mx.eval(rgb)
        return rgb


def write_exr_manifest(
    directory: Path | str,
    *,
    delivery: ExrDeliverySpec,
    frame_count: int,
    width: int,
    height: int,
) -> Path:
    """Declare EXR authoring semantics ImageIO cannot carry as a standard profile."""
    target = Path(directory) / "manifest.json"
    payload = {
        "schema_version": 1,
        "format": "openexr-sequence",
        "sample_type": delivery.sample_type.value,
        "primaries": delivery.primaries.value,
        "transfer": delivery.transfer.value,
        "color_space_tag": delivery.color_space_tag,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "pattern": "frame_%05d.exr",
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def validate_exr_delivery(delivery: ExrDeliverySpec) -> None:
    """Validate one authoring space supported by the native EXR writer."""
    if delivery.primaries not in {ColorPrimaries.REC709, ColorPrimaries.ACESCG}:
        raise ValueError("EXR delivery primaries must be Rec.709 or ACEScg")
    if delivery.transfer not in {
        ColorTransfer.LINEAR,
        ColorTransfer.ACESCCT,
        ColorTransfer.LOGC3,
    }:
        raise ValueError("EXR delivery must be linear, ACEScct, or LogC3")
    if delivery.transfer in {ColorTransfer.ACESCCT, ColorTransfer.LOGC3} and (
        delivery.primaries
        is not (
            ColorPrimaries.ACESCG
            if delivery.transfer is ColorTransfer.ACESCCT
            else ColorPrimaries.REC709
        )
    ):
        raise ValueError(f"{delivery.transfer.value} EXR delivery has incompatible primaries")


__all__ = [
    "read_exr_frame",
    "save_exr_frame",
    "validate_exr_delivery",
    "write_exr_manifest",
]
