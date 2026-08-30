"""Native scene-linear RGB to user-viewable BT.2100 PQ HEIC frames."""

from __future__ import annotations

import json
import math
from pathlib import Path

import Foundation
import mlx.core as mx
import objc
import Quartz

from kinomlx.io.atomic import write_text_atomic
from kinomlx.media.hdr import convert_scene_linear_primaries
from kinomlx.media.signals import ColorPrimaries

PQ_REFERENCE_WHITE_NITS = 203.0
PQ_PEAK_NITS = 10_000.0
HEIC_COMPRESSION_QUALITY = 0.95

_HEIC_UTI = "public.heic"
_PQ_M1 = 1305.0 / 8192.0
_PQ_M2 = 2523.0 / 32.0
_PQ_C1 = 107.0 / 128.0
_PQ_C2 = 2413.0 / 128.0
_PQ_C3 = 2392.0 / 128.0


def scene_linear_to_pq_codes(
    frame: mx.array,
    *,
    primaries: ColorPrimaries,
    reference_white_nits: float = PQ_REFERENCE_WHITE_NITS,
) -> mx.array:
    """Map scene-linear RGB into bounded float32 BT.2020/ST-2084 codes."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"PQ source frame must be HWC RGB, got {tuple(frame.shape)}")
    if frame.dtype != mx.float32:
        raise TypeError(f"PQ source frame must be float32, got {frame.dtype}")
    if not bool(mx.all(mx.isfinite(frame)).item()):
        raise ValueError("PQ source frame must contain only finite values")
    if primaries not in {ColorPrimaries.ACESCG, ColorPrimaries.REC709}:
        raise ValueError(f"PQ terminal does not support {primaries.value} source primaries")
    if not math.isfinite(reference_white_nits) or reference_white_nits <= 0.0:
        raise ValueError("PQ reference white must be finite and positive")

    source = mx.maximum(frame, 0.0)
    bt2020 = convert_scene_linear_primaries(
        source,
        source=primaries,
        target=ColorPrimaries.BT2020,
    )
    normalized = mx.clip(
        mx.maximum(bt2020, 0.0) * (reference_white_nits / PQ_PEAK_NITS),
        0.0,
        1.0,
    )
    powered = mx.power(normalized, _PQ_M1)
    codes = mx.power(
        (_PQ_C1 + _PQ_C2 * powered) / (1.0 + _PQ_C3 * powered),
        _PQ_M2,
    )
    return codes.astype(mx.float32)


def _pq_rgb_image(codes: mx.array) -> object:
    height, width = int(codes.shape[0]), int(codes.shape[1])
    rgb16 = mx.contiguous(mx.round(codes * 65535.0).astype(mx.uint16))
    mx.eval(rgb16)
    raw = memoryview(rgb16).cast("B")
    data = Foundation.NSData.dataWithBytes_length_(raw, len(raw))
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    color_space = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceITUR_2100_PQ)
    if color_space is None:
        raise RuntimeError("cannot create BT.2100 PQ color space")
    bitmap_info = Quartz.kCGBitmapByteOrder16Little | Quartz.kCGImageAlphaNone
    image = Quartz.CGImageCreate(
        width,
        height,
        16,
        48,
        width * 6,
        color_space,
        bitmap_info,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )
    if image is None:
        raise RuntimeError("cannot create 16-bit PQ source image")
    return image


def save_pq_heic_frame(
    frame: mx.array,
    path: Path | str,
    *,
    primaries: ColorPrimaries,
    compression_quality: float = HEIC_COMPRESSION_QUALITY,
) -> Path:
    """Write one scene-linear frame as a tagged 10-bit BT.2100 PQ HEIC."""
    if not math.isfinite(compression_quality) or not 0.0 <= compression_quality <= 1.0:
        raise ValueError("HEIC compression quality must be between 0 and 1")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with objc.autorelease_pool():
        codes = scene_linear_to_pq_codes(frame, primaries=primaries)
        destination = Quartz.CGImageDestinationCreateWithURL(
            Foundation.NSURL.fileURLWithPath_(str(target)),
            _HEIC_UTI,
            1,
            None,
        )
        if destination is None:
            raise RuntimeError(f"cannot create HEIC destination {target}")
        Quartz.CGImageDestinationAddImage(
            destination,
            _pq_rgb_image(codes),
            {Quartz.kCGImageDestinationLossyCompressionQuality: compression_quality},
        )
        if not Quartz.CGImageDestinationFinalize(destination):
            raise RuntimeError(f"cannot finalize HEIC frame {target}")
    return target


def write_heic_manifest(
    directory: Path | str,
    *,
    source_primaries: ColorPrimaries,
    frame_count: int,
    width: int,
    height: int,
    compression_quality: float = HEIC_COMPRESSION_QUALITY,
) -> Path:
    """Declare the display encoding applied to an adjacent HEIC sequence."""
    target = Path(directory) / "manifest.json"
    payload = {
        "schema_version": 1,
        "format": "heic-sequence",
        "codec": "hevc",
        "bit_depth": 10,
        "primaries": "bt2020",
        "transfer": "pq",
        "reference_white_nits": PQ_REFERENCE_WHITE_NITS,
        "pq_peak_nits": PQ_PEAK_NITS,
        "compression_quality": compression_quality,
        "source_primaries": source_primaries.value,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "pattern": "frame_%05d.heic",
    }
    write_text_atomic(
        target,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return target


__all__ = [
    "HEIC_COMPRESSION_QUALITY",
    "PQ_PEAK_NITS",
    "PQ_REFERENCE_WHITE_NITS",
    "save_pq_heic_frame",
    "scene_linear_to_pq_codes",
    "write_heic_manifest",
]
