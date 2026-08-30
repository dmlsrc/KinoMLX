"""Image I/O for typed ``(H, W, C)`` float32 tensors.

A thin wrapper over native ImageIO decode plus MLX bilinear resize (no Pillow,
no numpy). ``load_image`` returns shape ``(H, W, C)`` float32 in ``[0, 1]``;
``load_raw_exr`` preserves declared floating working or scene-linear samples.
Model-specific reshapes (NCHW
transpose, ``[-1, 1]`` re-normalization, batch / frame dims for video
pipelines) happen at the per-model conditioning layer
(``kinomlx/models/<name>/conditioning/``), not here.

The ``size`` argument uses ``(width, height)``; internal indexing is
``(H, W, C)`` to match the returned array.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from kinomlx.videotoolbox.images import load_image_rgb
from kinomlx.videotoolbox.images import save_image as _save_image_native


def load_image(
    path: Path | str,
    *,
    size: tuple[int, int] | None = None,
) -> mx.array:
    """Load an image into an ``mx.array`` of shape ``(H, W, C)`` float32 in ``[0, 1]``.

    Any input mode (RGB, RGBA, grayscale, paletted, ...) decodes to
    3-channel sRGB with alpha dropped, matching the old
    ``PIL.Image.convert("RGB")`` behavior.

    If ``size`` is given as ``(width, height)``, the image is fit via the
    reference align-corners-false bilinear cover resize plus center crop.
    Without ``size`` the original dimensions are preserved.
    """
    rgb = load_image_rgb(path)  # (H, W, 3) uint8, sRGB, alpha dropped
    image = rgb.astype(mx.float32) / 255.0
    if size is not None:
        image = _cover_crop(image, size)
    return image


def load_raw_exr(
    path: Path | str,
    *,
    size: tuple[int, int] | None = None,
) -> mx.array:
    """Load raw floating EXR RGB samples without display-space conversion."""
    from kinomlx.videotoolbox.exr import read_exr_frame

    image = read_exr_frame(path)
    if size is not None:
        image = _cover_crop(image, size)
    return image


def save_image(path: Path | str, image: mx.array) -> None:
    """Save an ``mx.array`` image to disk via native ImageIO.

    Expects shape ``(H, W, C)`` (or ``(H, W)`` grayscale) with values in
    ``[0, 1]``; values outside the range are clipped. The output container
    is inferred from the path extension (.png / .jpg / .tiff / .heic).
    """
    arr = mx.clip(image, 0.0, 1.0) * 255.0
    if arr.ndim == 2:  # grayscale -> 3-channel for the CGImage encoder
        arr = mx.broadcast_to(arr[:, :, None], (arr.shape[0], arr.shape[1], 3))
    _save_image_native(arr.astype(mx.uint8), path)


def _resize_bilinear(img: mx.array, width: int, height: int) -> mx.array:
    """Resize HWC float pixels with align-corners-false bilinear sampling."""
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    if (src_w, src_h) == (width, height):
        return img

    y = (mx.arange(height, dtype=mx.float32) + 0.5) * (src_h / height) - 0.5
    y = mx.clip(y, 0.0, float(src_h - 1))
    y0 = mx.floor(y).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, src_h - 1)
    y_weight = (y - y0.astype(mx.float32))[:, None, None]
    resized_y = mx.take(img, y0, axis=0) * (1.0 - y_weight) + mx.take(img, y1, axis=0) * y_weight

    x = (mx.arange(width, dtype=mx.float32) + 0.5) * (src_w / width) - 0.5
    x = mx.clip(x, 0.0, float(src_w - 1))
    x0 = mx.floor(x).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, src_w - 1)
    x_weight = (x - x0.astype(mx.float32))[None, :, None]
    return (
        mx.take(resized_y, x0, axis=1) * (1.0 - x_weight)
        + mx.take(resized_y, x1, axis=1) * x_weight
    )


def _cover_crop(img: mx.array, size: tuple[int, int]) -> mx.array:
    """Aspect-preserving cover-resize + center-crop to ``(width, height)``.

    Cover semantics: the resized image fully *covers* the target rectangle;
    the dimension with the wrong aspect gets cropped.
    """
    target_w, target_h = size
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    scale = max(target_h / src_h, target_w / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    resized = _resize_bilinear(img, new_w, new_h)
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return resized[top : top + target_h, left : left + target_w, :]
