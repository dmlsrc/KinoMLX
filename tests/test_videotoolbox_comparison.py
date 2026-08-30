"""Structural test for the side-by-side comparison composite
(``kinomlx.videotoolbox.comparison``).

``render_comparison`` NEAREST-upscales the pre frame onto the left half and
composites the VSR post buffer onto the right half, in one CoreImage pass.
Rather than pin an opaque render hash, this asserts the layout directly: a flat
red pre frame must land on the left and a flat blue post buffer on the right, at
the documented output geometry (2 * pre_W * scale, pre_H * scale).

Tagged ``requires_avfoundation`` for explicit native-test selection.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
import Quartz

from kinomlx.videotoolbox import comparison
from kinomlx.videotoolbox import pixel_buffers as pb

pytestmark = pytest.mark.requires_avfoundation


def _make_buffer(fmt: int, w: int, h: int):
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: fmt,
        Quartz.kCVPixelBufferWidthKey: w,
        Quartz.kCVPixelBufferHeightKey: h,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    return pb.make_pixel_buffer_from_attrs(w, h, attrs)


def test_render_comparison_layout_pre_left_post_right():
    scale = 2
    in_h, in_w = 4, 4
    out_w, out_h = in_w * scale, in_h * scale  # post buffer is 8x8

    # Flat red pre frame (left half), flat blue post buffer (right half).
    pre = mx.zeros((in_h, in_w, 3), dtype=mx.uint8)
    pre[..., 0] = 255
    post = _make_buffer(pb.PIX_RGBAHALF, out_w, out_h)
    blue = mx.zeros((out_h, out_w, 4), dtype=mx.float16)
    blue[..., 2] = 1.0
    blue[..., 3] = 1.0
    pb.upload_frame_to_buffer(blue, post)

    dest = _make_buffer(pb.PIX_BGRA, 2 * out_w, out_h)
    comparison.render_comparison(pre, post, scale, dest)

    out = pb.read_pixel_buffer_rgb(dest).astype(mx.float32)
    assert tuple(int(x) for x in out.shape) == (out_h, 2 * out_w, 3)

    # Left half is the NEAREST-upscaled red pre; right half is the blue post.
    left = out[:, :out_w]
    right = out[:, out_w:]
    red = mx.broadcast_to(mx.array([255.0, 0.0, 0.0]), left.shape)
    blue_rgb = mx.broadcast_to(mx.array([0.0, 0.0, 255.0]), right.shape)
    assert mx.array_equal(left, red)
    assert mx.array_equal(right, blue_rgb)
