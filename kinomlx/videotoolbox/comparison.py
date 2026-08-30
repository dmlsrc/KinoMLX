"""Side-by-side composite for `comparison.mp4`: NEAREST-upscaled pre vs VSR post.

The pre half is the input frame scaled up with `repeat` (true pixel
nearest-neighbor), the post half is the VSR output CVPixelBuffer. Both
halves are composited via CoreImage and rendered into a single BGRA buffer
in one GPU pass.

Accepts pre as either (H,W,3) uint8 RGB or (H,W,4) fp16/fp32 RGBA so it
works with both --video (uint8) and --latent HQ (fp16) chunk formats.
"""

from __future__ import annotations

import Foundation
import mlx.core as mx
import Quartz

from . import pixel_buffers as _pb


def render_comparison(
    pre_frame: mx.array,
    post_pb: object,
    scale: int,
    dest_pb: object,
) -> None:
    """Render the side-by-side comparison into `dest_pb` (BGRA).

    Layout: NEAREST-upscaled pre on the left, VSR post (post_pb) on the
    right. Output dimensions are (2 * pre_W * scale, pre_H * scale).
    """
    in_h, in_w = int(pre_frame.shape[0]), int(pre_frame.shape[1])
    out_w, out_h = in_w * scale, in_h * scale

    # NEAREST upscale via repeat - exact integer-replicated pixels.
    pre_up = mx.repeat(mx.repeat(pre_frame, scale, axis=0), scale, axis=1)

    # Coerce to uint8 RGBA8 for CIImage(kCIFormatRGBA8). The pre half is a
    # deliberately-blocky visualization, so 8-bit is fine. Channel math runs in
    # MLX and the bytes come from the array buffer, so no numpy.
    if str(pre_up.dtype).split(".")[-1] in ("float16", "float32"):
        rgba = mx.clip(pre_up * 255.0, 0, 255).astype(mx.uint8)
    else:
        rgba = pre_up
    if rgba.shape[-1] == 3:
        alpha = mx.full((out_h, out_w, 1), 255, dtype=mx.uint8)
        rgba = mx.concatenate([rgba, alpha], axis=-1)

    # memoryview view of the frame; NSData copies it, so no intermediate bytes.
    src = memoryview(mx.contiguous(rgba)).cast("B")
    data = Foundation.NSData.dataWithBytes_length_(src, len(src))
    pre_ci = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data,
        out_w * 4,
        (out_w, out_h),
        Quartz.kCIFormatRGBA8,
        _pb.srgb_colorspace(),
    )

    # Post comes in via its CVPixelBuffer - CoreImage reads it at whatever
    # bit depth the format provides (fp16 for RGBAHalf, 8-bit for NV12).
    post_ci = Quartz.CIImage.alloc().initWithCVPixelBuffer_(post_pb)
    post_ci_translated = post_ci.imageByApplyingTransform_(
        Quartz.CGAffineTransformMakeTranslation(out_w, 0),
    )
    composite = post_ci_translated.imageByCompositingOverImage_(pre_ci)

    _pb.ci_context().render_toCVPixelBuffer_bounds_colorSpace_(
        composite,
        dest_pb,
        ((0, 0), (2 * out_w, out_h)),
        _pb.srgb_colorspace(),
    )
