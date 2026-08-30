"""Native still-image I/O: ImageIO + CoreImage + AppKit, no Pillow.

The still-image sister of `audio.py` (native audio I/O). It replaces Pillow with
the Apple frameworks this project already requires:

  - decode / encode   ImageIO (CGImageSource / CGImageDestination), via Quartz
  - Lanczos resize    CoreImage CILanczosScaleTransform (GPU, shared CIContext)
  - text annotation   AppKit NSStringDrawing (lazy-imported; diagnostic
                      contact-sheet tooling only)

Pixels cross into MLX straight from the CoreGraphics bitmap buffer
(`memoryview` -> `mx.array`), no numpy. `load_image_rgb` and `resize_lanczos`
back the runtime image-conditioning path; `save_image` and `draw_labels` back
the diagnostic scripts.

Color management: images are decoded into the shared sRGB context used by the
rest of the videotoolbox subsystem. Alpha is dropped rather than premultiplied,
so the result mirrors Pillow's `convert("RGB")`.
"""

from __future__ import annotations

from pathlib import Path

import Foundation
import mlx.core as mx
import objc
import Quartz

from kinomlx.io.buffer import mlx_array_from_buffer

from .pixel_buffers import ci_context, srgb_colorspace

# RGBX: 4 bytes/pixel, alpha byte ignored. Matches Pillow's convert("RGB")
# (keep raw RGB, drop alpha) rather than premultiplying by alpha.
_RGBX = Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault

_UTI_BY_SUFFIX = {
    ".png": "public.png",
    ".jpg": "public.jpeg",
    ".jpeg": "public.jpeg",
    ".tif": "public.tiff",
    ".tiff": "public.tiff",
    ".heic": "public.heic",
}


def _url(path: str | Path) -> object:
    return Foundation.NSURL.fileURLWithPath_(str(path))


def _cgimage_to_mx_rgb(cg: object) -> mx.array:
    """CGImage -> (H, W, 3) uint8 MLX array via an sRGB RGBX bitmap context."""
    w = int(Quartz.CGImageGetWidth(cg))
    h = int(Quartz.CGImageGetHeight(cg))
    row = w * 4
    buf = bytearray(row * h)
    ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, row, srgb_colorspace(), _RGBX)
    if ctx is None:
        raise ValueError("CGBitmapContextCreate failed (sRGB/RGBX)")
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, w, h), cg)
    return mlx_array_from_buffer(memoryview(buf), dtype=mx.uint8).reshape(h, w, 4)[:, :, :3]


def _cgimage_with_alpha_ignored(cg: object) -> object:
    """Return an alpha-less view of unpremultiplied 8-bit RGBA storage."""
    alpha = int(Quartz.CGImageGetAlphaInfo(cg))
    alpha_last = int(Quartz.kCGImageAlphaLast)
    alpha_first = int(Quartz.kCGImageAlphaFirst)
    if (
        alpha not in (alpha_last, alpha_first)
        or int(Quartz.CGImageGetBitsPerComponent(cg)) != 8
        or int(Quartz.CGImageGetBitsPerPixel(cg)) != 32
    ):
        return cg
    bitmap_info = int(Quartz.CGImageGetBitmapInfo(cg))
    bitmap_info &= ~int(Quartz.kCGBitmapAlphaInfoMask)
    bitmap_info |= int(
        Quartz.kCGImageAlphaNoneSkipLast
        if alpha == alpha_last
        else Quartz.kCGImageAlphaNoneSkipFirst
    )
    ignored = Quartz.CGImageCreate(
        Quartz.CGImageGetWidth(cg),
        Quartz.CGImageGetHeight(cg),
        Quartz.CGImageGetBitsPerComponent(cg),
        Quartz.CGImageGetBitsPerPixel(cg),
        Quartz.CGImageGetBytesPerRow(cg),
        Quartz.CGImageGetColorSpace(cg),
        bitmap_info,
        Quartz.CGImageGetDataProvider(cg),
        Quartz.CGImageGetDecode(cg),
        Quartz.CGImageGetShouldInterpolate(cg),
        Quartz.CGImageGetRenderingIntent(cg),
    )
    if ignored is None:
        raise ValueError("CGImageCreate failed while dropping image alpha")
    return ignored


def _apply_exif_orientation(image: mx.array, orientation: int) -> mx.array:
    """Apply the eight TIFF/EXIF display orientations to an HWC image."""
    if orientation == 1:
        return image
    if orientation == 2:
        return mx.flip(image, axis=1)
    if orientation == 3:
        return mx.flip(image, axis=(0, 1))
    if orientation == 4:
        return mx.flip(image, axis=0)
    transposed = image.transpose(1, 0, 2)
    if orientation == 5:
        return transposed
    if orientation == 6:
        return mx.flip(transposed, axis=1)
    if orientation == 7:
        return mx.flip(transposed, axis=(0, 1))
    if orientation == 8:
        return mx.flip(transposed, axis=0)
    return image


def _require_uint8_image(img: mx.array) -> mx.array:
    """Return a validated ``(H, W, 3|4)`` uint8 image.

    Non-uint8 input is deliberately rejected: its numeric scale is ambiguous,
    and casting a conventional ``[0, 1]`` float image would silently turn it
    almost entirely black.
    """
    if not isinstance(img, mx.array):
        raise TypeError(f"image must be an MLX array, got {type(img).__name__}")
    a = img
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) image, got shape {tuple(a.shape)}")
    if a.dtype != mx.uint8:
        raise TypeError(
            f"expected a uint8 image, got {a.dtype}; convert first, e.g. "
            "(mx.clip(x, 0, 1) * 255 + 0.5).astype(mx.uint8)"
        )
    return a


def _mx_to_cgimage(img: mx.array) -> object:
    """(H, W, 3|4) uint8 array-like -> CGImage (opaque; any alpha is ignored)."""
    a = _require_uint8_image(img)
    h, w = int(a.shape[0]), int(a.shape[1])
    if a.shape[2] == 3:
        a = mx.concatenate([a, mx.full((h, w, 1), 255, dtype=mx.uint8)], axis=2)
    mx.eval(a)
    # Independent mutable backing for the CGBitmapContext (CG renders into it
    # and retains the pointer); cast("B") fills it in one copy. Do not pass a
    # bare memoryview - CG would hold MLX-managed memory that MLX may recycle.
    buf = bytearray(memoryview(mx.contiguous(a)).cast("B"))
    ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, w * 4, srgb_colorspace(), _RGBX)
    if ctx is None:
        raise ValueError("CGBitmapContextCreate failed building CGImage")
    return Quartz.CGBitmapContextCreateImage(ctx)


def load_image_rgb(path: str | Path) -> mx.array:
    """Decode an image file to an (H, W, 3) uint8 MLX array (sRGB, alpha dropped).

    Native ImageIO replacement for `PIL.Image.open(path).convert("RGB")`. Reads
    any format ImageIO supports (PNG, JPEG, HEIF, TIFF, ...); grayscale and RGBA
    sources decode to 3-channel sRGB. Raises FileNotFoundError for a missing
    path, ValueError for an undecodable file.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with objc.autorelease_pool():
        src = Quartz.CGImageSourceCreateWithURL(_url(path), None)
        if src is None:
            raise ValueError(f"Failed to open image {path}: unrecognized format")
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if cg is None:
            raise ValueError(f"Failed to decode image {path}")
        properties = Quartz.CGImageSourceCopyPropertiesAtIndex(src, 0, None)
        orientation = 1
        if properties is not None:
            orientation = int(properties.get(Quartz.kCGImagePropertyOrientation, 1))
        image = _cgimage_to_mx_rgb(_cgimage_with_alpha_ignored(cg))
        return mx.contiguous(_apply_exif_orientation(image, orientation))


def resize_lanczos(img: mx.array, width: int, height: int) -> mx.array:
    """Lanczos-resample an (H, W, C) uint8 array to (height, width, 3) uint8.

    Native CoreImage `CILanczosScaleTransform` replacement for
    `PIL.Image.resize((width, height), LANCZOS)`. The resample is anamorphic
    (independent x and y), matching Pillow: `inputScale` sets the vertical
    factor and `inputAspectRatio` the horizontal stretch relative to it. The
    output is taken over an exact integer rect so the shape is precisely
    (height, width, 3).
    """
    a = _require_uint8_image(img)
    src_h, src_w = int(a.shape[0]), int(a.shape[1])
    if (src_w, src_h) == (int(width), int(height)):
        return a[:, :, :3]
    with objc.autorelease_pool():
        ci = Quartz.CIImage.imageWithCGImage_(_mx_to_cgimage(a))
        scale = height / src_h
        aspect = (width / src_w) / scale  # = (width/src_w) * (src_h/height)
        f = Quartz.CIFilter.filterWithName_("CILanczosScaleTransform")
        f.setValue_forKey_(ci, "inputImage")
        f.setValue_forKey_(float(scale), "inputScale")
        f.setValue_forKey_(float(aspect), "inputAspectRatio")
        out = f.valueForKey_("outputImage")
        # Exact integer rect -> the rendered CGImage is exactly width x height,
        # sidestepping any sub-pixel rounding in the filter's reported extent.
        cg_out = ci_context().createCGImage_fromRect_(
            out, Quartz.CGRectMake(0, 0, int(width), int(height))
        )
        if cg_out is None:
            raise ValueError("CILanczosScaleTransform render failed")
        return _cgimage_to_mx_rgb(cg_out)


def save_image(img: mx.array, path: str | Path) -> Path:
    """Encode an (H, W, 3|4) uint8 array to an image file via ImageIO.

    Native CGImageDestination replacement for
    `PIL.Image.fromarray(arr).save(path)`. The container is chosen from the path
    suffix (.png -> PNG, .jpg/.jpeg -> JPEG, .tif/.tiff -> TIFF, .heic -> HEIF);
    anything else defaults to PNG. Returns the path written.
    """
    path = Path(path)
    uti = _UTI_BY_SUFFIX.get(path.suffix.lower(), "public.png")
    with objc.autorelease_pool():
        dest = Quartz.CGImageDestinationCreateWithURL(_url(path), uti, 1, None)
        if dest is None:
            raise ValueError(f"Cannot create image destination for {path}")
        Quartz.CGImageDestinationAddImage(dest, _mx_to_cgimage(img), None)
        if not Quartz.CGImageDestinationFinalize(dest):
            raise ValueError(f"Failed to write image {path}")
    return path


def draw_labels(
    img: mx.array,
    labels: list[tuple[int, int, str]],
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    font_size: float = 12.0,
    font_name: str = "Helvetica",
) -> mx.array:
    """Draw text labels onto a copy of an (H, W, C) uint8 array; AppKit text.

    Native NSStringDrawing replacement for `PIL.ImageDraw.Draw(img).text(...)`.
    `labels` is a list of (x, y, text) in top-left pixel coordinates (y grows
    downward, like Pillow). Returns a new (H, W, 3) uint8 array. AppKit is
    imported lazily so the runtime image path (load/resize/save) never loads it.
    """
    import AppKit  # lazy: diagnostic contact-sheet tooling only

    a = _require_uint8_image(img)
    h, w = int(a.shape[0]), int(a.shape[1])
    rgba = mx.concatenate([a[:, :, :3], mx.full((h, w, 1), 255, dtype=mx.uint8)], axis=2)
    mx.eval(rgba)
    # Independent mutable backing for the CGBitmapContext (CG renders into it
    # and retains the pointer); cast("B") fills it in one copy. Do not pass a
    # bare memoryview - CG would hold MLX-managed memory that MLX may recycle.
    buf = bytearray(memoryview(mx.contiguous(rgba)).cast("B"))
    ctx = Quartz.CGBitmapContextCreate(buf, w, h, 8, w * 4, srgb_colorspace(), _RGBX)
    if ctx is None:
        raise ValueError("CGBitmapContextCreate failed for label drawing")

    with objc.autorelease_pool():
        # A CGBitmapContext stores row 0 at the top but draws with y=0 at the
        # bottom. Flip the CTM (translate to the top edge, invert y) so a label
        # placed at (x, y) lands y pixels from the TOP, matching Pillow's anchor.
        # The matching flipped=True below keeps glyphs upright under that flip.
        Quartz.CGContextTranslateCTM(ctx, 0, h)
        Quartz.CGContextScaleCTM(ctx, 1, -1)
        nsctx = AppKit.NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, True)
        AppKit.NSGraphicsContext.saveGraphicsState()
        AppKit.NSGraphicsContext.setCurrentContext_(nsctx)
        font = AppKit.NSFont.fontWithName_size_(
            font_name, font_size
        ) or AppKit.NSFont.systemFontOfSize_(font_size)
        nscolor = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
            color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0
        )
        attrs = {
            AppKit.NSFontAttributeName: font,
            AppKit.NSForegroundColorAttributeName: nscolor,
        }
        for x, y, text in labels:
            point = Foundation.NSMakePoint(float(x), float(y))
            Foundation.NSString.stringWithString_(str(text)).drawAtPoint_withAttributes_(
                point, attrs
            )
        AppKit.NSGraphicsContext.restoreGraphicsState()
        nsctx.flushGraphics()
    return mlx_array_from_buffer(memoryview(buf), dtype=mx.uint8).reshape(h, w, 4)[:, :, :3]
