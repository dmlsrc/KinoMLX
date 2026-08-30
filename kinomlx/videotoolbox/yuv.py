"""MLX RGB to 10-bit BT.709 video-range 4:2:2 conversion.

AVAssetWriter's internal RGB-to-YUV conversion depends on colorspace metadata
that is not equivalent across decoded, VideoToolbox-produced, and MLX-uploaded
RGBAHalf buffers. Converting explicitly gives every RGBAHalf path the same
BT.709 encoder feed.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import cast

import mlx.core as mx
import Quartz

# 10-bit video-range 4:2:2 biplanar ('x422'), consumed by HEVC Main42210.
# Keep the FourCC local because PyObjC does not expose a named constant for it.
PIX_422YCBCR10_VIDEO = int.from_bytes(b"x422", "big")

_KR = 0.2126
_KB = 0.0722
_KG = 1.0 - _KR - _KB


def _compute_planes(rgb: mx.array) -> tuple[mx.array, mx.array]:
    """Compute left-justified 10-bit luma and interleaved chroma planes."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = _KR * red + _KG * green + _KB * blue
    cb = (blue - luma) / (2.0 * (1.0 - _KB))
    cr = (red - luma) / (2.0 * (1.0 - _KR))
    height, width = rgb.shape[0], rgb.shape[1]
    luma10 = luma * 876.0 + 64.0
    cb10 = cb * 896.0 + 512.0
    cr10 = cr * 896.0 + 512.0

    # HEVC cannot signal chroma sample location for 4:2:2, so decoders assume
    # samples are co-sited with the even luma columns.  An adjacent-pair box
    # average is centered between columns and therefore decodes half a luma
    # sample out of phase.  Center a [1, 2, 1] / 4 filter on each even column.
    def cosited(chroma: mx.array) -> mx.array:
        left = mx.concatenate([chroma[:, :1], chroma[:, 1:-1:2]], axis=1)
        return (left + 2.0 * chroma[:, 0::2] + chroma[:, 1::2]) * 0.25

    cb_subsampled = cosited(cb10)
    cr_subsampled = cosited(cr10)
    luma_plane = (mx.clip(mx.round(luma10), 0, 1023).astype(mx.uint16)) << 6
    cb_plane = mx.clip(mx.round(cb_subsampled), 0, 1023).astype(mx.uint16)
    cr_plane = mx.clip(mx.round(cr_subsampled), 0, 1023).astype(mx.uint16)
    chroma_plane = mx.stack([cb_plane, cr_plane], axis=-1).reshape(height, width) << 6
    return luma_plane, chroma_plane


@cache
def _compiled_planes() -> Callable[[mx.array], tuple[mx.array, mx.array]]:
    """Compile the conversion kernel on first use, never during import."""
    return cast(Callable[[mx.array], tuple[mx.array, mx.array]], mx.compile(_compute_planes))


def rgb_to_yuv422_10(rgb: mx.array, dst_buffer: object) -> None:
    """Convert gamma-encoded ``(H, W, 3)`` RGB in [0, 1] to BT.709 4:2:2."""
    if not isinstance(rgb, mx.array):
        raise TypeError(f"rgb must be an MLX array, got {type(rgb).__name__}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {tuple(rgb.shape)}")
    if rgb.dtype not in (mx.float16, mx.float32, mx.bfloat16):
        raise TypeError(f"expected floating RGB in [0, 1], got {rgb.dtype}")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if width % 2:
        raise ValueError(f"10-bit 4:2:2 conversion requires even width, got {width}")
    dst_width = int(Quartz.CVPixelBufferGetWidth(dst_buffer))
    dst_height = int(Quartz.CVPixelBufferGetHeight(dst_buffer))
    dst_format = int(Quartz.CVPixelBufferGetPixelFormatType(dst_buffer))
    if (dst_width, dst_height) != (width, height):
        raise ValueError(
            f"RGB source is {width}x{height}, YUV destination is {dst_width}x{dst_height}"
        )
    if dst_format != PIX_422YCBCR10_VIDEO:
        raise ValueError(f"expected x422 destination format, got {dst_format:#x}")
    luma, chroma = _compiled_planes()(rgb)
    _write_planes(dst_buffer, (luma, chroma))


def _write_planes(buf: object, planes: tuple[mx.array, ...]) -> None:
    """Copy uint16 MLX planes into padded CVPixelBuffer plane storage."""
    Quartz.CVPixelBufferLockBaseAddress(buf, 0)
    try:
        for plane, arr in enumerate(planes):
            arr = mx.contiguous(arr)
            mx.eval(arr)
            rows, cols = int(arr.shape[0]), int(arr.shape[1])
            base = Quartz.CVPixelBufferGetBaseAddressOfPlane(buf, plane)
            bpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(buf, plane)
            mv = base.as_buffer(rows * bpr)
            src = memoryview(arr).cast("B")
            row_bytes = cols * 2
            if bpr == row_bytes:
                mv[: rows * row_bytes] = src
            else:
                for row in range(rows):
                    start = row * bpr
                    src_start = row * row_bytes
                    mv[start : start + row_bytes] = src[src_start : src_start + row_bytes]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buf, 0)
