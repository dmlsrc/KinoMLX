"""Native image resampling and shared source-reference fitting."""

from typing import Literal

import mlx.core as mx


def resize_rgb(
    pixels: mx.array, height: int, width: int, *, mode: Literal["bilinear", "bicubic", "area"]
) -> mx.array:
    """Separable half-pixel resampling; bicubic includes antialiasing."""
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or min(height, width) <= 0:
        raise ValueError("resize_rgb requires BHWC RGB and positive output dimensions")
    value = pixels.astype(mx.float32)
    for axis, output in ((1, height), (2, width)):
        source = value.shape[axis]
        if source == output:
            continue
        centers = (mx.arange(output, dtype=mx.float32) + 0.5) * (source / output) - 0.5
        indices = mx.arange(source, dtype=mx.float32)
        if mode == "area":
            start = mx.floor(mx.arange(output) * (source / output))
            end = mx.ceil((mx.arange(output) + 1) * (source / output))
            weights = ((indices[None] >= start[:, None]) & (indices[None] < end[:, None])).astype(
                mx.float32
            )
        elif mode == "bilinear":
            centers = mx.clip(centers, 0, source - 1)
            weights = mx.maximum(1 - mx.abs(indices[None] - centers[:, None]), 0)
        elif mode == "bicubic":
            scale = max(source / output, 1.0)
            distance = mx.abs((indices[None] - centers[:, None]) / scale)
            # Antialiased bicubic uses the Keys kernel with a=-0.5.
            inner = ((1.5 * distance - 2.5) * distance) * distance + 1
            outer = ((-0.5 * distance + 2.5) * distance - 4) * distance + 2
            weights = mx.where(distance < 1, inner, mx.where(distance < 2, outer, 0))
        else:
            raise ValueError(f"unsupported resize mode: {mode}")
        weights = weights / mx.sum(weights, axis=1, keepdims=True)
        value = mx.moveaxis(mx.moveaxis(value, axis, -1) @ weights.T, -1, axis)
    return value


def reference_fit_geometry(
    ih: int, iw: int, height: int, width: int, *, mode: str = "fit"
) -> tuple[int, int, int, int, int, int]:
    """One crop box and output grid shared by source pixels and source masks."""
    scale = min(height / ih, width / iw)
    fill = mode == "crop" or (ih * scale >= height * 0.92 and iw * scale >= width * 0.92)
    if mode not in {"fit", "crop"}:
        raise ValueError("reference fit mode must be fit or crop")
    if fill:
        scale = max(height / ih, width / iw)
        nh, nw = height, width
    else:
        nh = min(max(16, int(ih * scale) // 16 * 16), max(16, height // 16 * 16))
        nw = min(max(16, int(iw * scale) // 16 * 16), max(16, width // 16 * 16))
    ch, cw = min(ih, max(1, round(nh / scale))), min(iw, max(1, round(nw / scale)))
    top, left = (ih - ch) // 2, (iw - cw) // 2
    return top, left, ch, cw, nh, nw


def fit_reference(pixels: mx.array, height: int, width: int, *, mode: str = "fit") -> mx.array:
    """Fit reference pixels to the trained target grid before VAE encoding."""
    top, left, ch, cw, nh, nw = reference_fit_geometry(
        pixels.shape[1], pixels.shape[2], height, width, mode=mode
    )
    return mx.clip(
        resize_rgb(pixels[:, top : top + ch, left : left + cw], nh, nw, mode="bicubic"), 0, 1
    )
