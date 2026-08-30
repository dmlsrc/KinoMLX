"""Tests for native still-image I/O (``kinomlx.videotoolbox.images``).

Covers the ImageIO / CoreImage / AppKit replacements for Pillow: decode,
encode, Lanczos resize, and text annotation - all MLX in, MLX out, no Pillow.
Tagged ``requires_avfoundation`` for explicit native-test selection.
"""

from __future__ import annotations

import struct

import Foundation
import mlx.core as mx
import pytest
import Quartz

from kinomlx.videotoolbox import images as I

pytestmark = pytest.mark.requires_avfoundation


def _gradient(h: int = 48, w: int = 64) -> mx.array:
    """(h, w, 3) uint8 gradient: red ramps over x, green over y, blue flat."""
    xr = mx.broadcast_to((mx.arange(w, dtype=mx.float32) * (255 / w))[None, :], (h, w))
    yg = mx.broadcast_to((mx.arange(h, dtype=mx.float32) * (255 / h))[:, None], (h, w))
    return mx.stack([xr, yg, mx.full((h, w), 100.0)], axis=2).astype(mx.uint8)


def test_save_load_png_byte_exact(tmp_path):
    src = _gradient()
    p = tmp_path / "rt.png"
    assert I.save_image(src, p) == p
    assert p.exists()
    rt = I.load_image_rgb(p)
    assert rt.shape == src.shape
    assert int(mx.max(mx.abs(rt.astype(mx.int32) - src.astype(mx.int32))).item()) == 0


def test_load_image_rgb_orientation_top_first(tmp_path):
    # Top quarter white, rest black; row 0 must come back as the white top.
    h, w = 40, 16
    src = mx.concatenate(
        [
            mx.full((h // 4, w, 3), 255, dtype=mx.uint8),
            mx.zeros((h - h // 4, w, 3), dtype=mx.uint8),
        ],
        axis=0,
    )
    p = tmp_path / "orient.png"
    I.save_image(src, p)
    rt = I.load_image_rgb(p)
    assert int(rt[0, 0, 0].item()) > 240
    assert int(rt[h - 1, 0, 0].item()) < 15


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        I.load_image_rgb("/nonexistent/path/nope.png")


def test_resize_lanczos_exact_shape():
    src = _gradient(48, 64)
    down = I.resize_lanczos(src, 32, 24)
    up = I.resize_lanczos(src, 100, 60)
    assert down.shape == (24, 32, 3)
    assert up.shape == (60, 100, 3)
    assert int(mx.max(down).item()) > 0
    assert int(mx.max(up).item()) > 0


def test_resize_lanczos_same_size_noop():
    src = _gradient(40, 50)
    out = I.resize_lanczos(src, 50, 40)
    assert out.shape == (40, 50, 3)
    assert int(mx.max(mx.abs(out.astype(mx.int32) - src.astype(mx.int32))).item()) == 0


def test_resize_lanczos_anamorphic_dims():
    # Independent x/y scaling (aspect change).
    out = I.resize_lanczos(_gradient(40, 40), 80, 20)
    assert out.shape == (20, 80, 3)


def test_grayscale_png_decodes_to_equal_rgb(tmp_path):
    val, w, h = 137, 8, 6
    p = tmp_path / "gray.png"
    buf = bytearray([val] * (w * h))
    cs = Quartz.CGColorSpaceCreateDeviceGray()
    ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, w, cs, Quartz.kCGImageAlphaNone)
    cg = Quartz.CGBitmapContextCreateImage(ctx)
    url = Foundation.NSURL.fileURLWithPath_(str(p))
    dest = Quartz.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
    Quartz.CGImageDestinationAddImage(dest, cg, None)
    Quartz.CGImageDestinationFinalize(dest)

    g = I.load_image_rgb(p)
    assert g.shape == (h, w, 3)
    # A gray source decodes to R == G == B.
    eq = int(mx.max(mx.abs(g[:, :, 0].astype(mx.int32) - g[:, :, 2].astype(mx.int32))).item())
    assert eq == 0
    assert abs(int(g[0, 0, 0].item()) - val) <= 2


def test_rgba_png_drops_alpha_without_compositing_hidden_rgb(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "alpha.png"
    source = image_module.new("RGBA", (4, 1))
    source.putdata(
        [
            (255, 0, 0, 0),
            (0, 255, 0, 128),
            (10, 20, 30, 64),
            (200, 100, 50, 255),
        ]
    )
    source.save(path)

    actual = I.load_image_rgb(path)
    expected = mx.array(
        list(source.convert("RGB").get_flattened_data()),
        dtype=mx.uint8,
    ).reshape(1, 4, 3)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_jpeg_exif_orientation_matches_displayed_pixels(tmp_path, orientation: int):
    image_module = pytest.importorskip("PIL.Image")
    image_ops = pytest.importorskip("PIL.ImageOps")
    base_path = tmp_path / "orientation-base.jpg"
    path = tmp_path / f"orientation-{orientation}.jpg"
    source = image_module.new("RGB", (32, 48), (0, 0, 0))
    for y in range(48):
        for x in range(32):
            source.putpixel((x, y), (x * 7 % 256, y * 5 % 256, (x + y) * 3 % 256))
    source.save(base_path, quality=100, subsampling=0)
    tiff = (
        struct.pack("<2sHIH", b"II", 42, 8, 1)
        + struct.pack("<HHI", 274, 3, 1)
        + struct.pack("<H", orientation)
        + b"\x00\x00"
        + struct.pack("<I", 0)
    )
    exif = b"Exif\x00\x00" + tiff
    marker = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    encoded = base_path.read_bytes()
    path.write_bytes(encoded[:2] + marker + encoded[2:])

    base = I.load_image_rgb(base_path)
    actual = I.load_image_rgb(path)
    expected = I._apply_exif_orientation(base, orientation)
    with image_module.open(path) as decoded:
        displayed = image_ops.exif_transpose(decoded)

    assert tuple(actual.shape[:2]) == (displayed.height, displayed.width)
    assert mx.array_equal(actual, expected).item()


def test_save_image_rejects_numpy_input(tmp_path):
    numpy = pytest.importorskip("numpy")
    arr = numpy.zeros((10, 12, 3), dtype=numpy.uint8)
    p = tmp_path / "np.png"

    with pytest.raises(TypeError, match="image must be an MLX array"):
        I.save_image(arr, p)
    assert not p.exists()


@pytest.mark.parametrize(
    "image",
    [
        mx.ones((8, 8, 3), dtype=mx.float32),
        mx.ones((8, 8, 3), dtype=mx.float16),
        mx.ones((8, 8, 3), dtype=mx.uint16),
    ],
)
def test_save_image_rejects_ambiguous_non_uint8_input(image, tmp_path):
    with pytest.raises(TypeError, match="expected a uint8 image"):
        I.save_image(image, tmp_path / "bad.png")


def test_same_size_resize_still_validates_dtype():
    image = mx.ones((8, 8, 3), dtype=mx.float32)
    with pytest.raises(TypeError, match="expected a uint8 image"):
        I.resize_lanczos(image, 8, 8)


def test_draw_labels_rejects_ambiguous_non_uint8_input():
    image = mx.ones((8, 8, 3), dtype=mx.float32)
    with pytest.raises(TypeError, match="expected a uint8 image"):
        I.draw_labels(image, [(0, 0, "x")])


def test_draw_labels_renders_top_anchored_upright():
    ann = I.draw_labels(mx.zeros((44, 80, 3), dtype=mx.uint8), [(6, 3, "F")], font_size=30)
    assert ann.shape == (44, 80, 3)
    lum = mx.max(ann.astype(mx.float32), axis=2)
    rows = [r for r in range(44) if float(mx.sum(lum[r]).item()) > 30]
    assert rows, "no text rendered"
    assert min(rows) < 14  # anchored near the top (y=3), not the bottom
    lo, hi = min(rows), max(rows)
    mid = (lo + hi) // 2
    top_ink = float(mx.sum(lum[lo : mid + 1]).item())
    bot_ink = float(mx.sum(lum[mid + 1 : hi + 1]).item())
    assert top_ink > bot_ink  # 'F' is top-heavy when upright
