"""Native image resampling and patch products for Krea conditioning."""

import math

import mlx.core as mx

from kinomlx.io.resampling import resize_rgb


def grounding_patches(
    pixels: mx.array, *, longest_side: int = 768
) -> tuple[mx.array, tuple[int, int]]:
    """Qwen3-VL still preprocessing and spatial-merge patch ordering."""
    if pixels.shape[0] != 1:
        raise ValueError("grounding_patches requires one image")
    height, width = pixels.shape[1:3]
    if longest_side and max(height, width) > longest_side:
        scale = longest_side / max(height, width)
        height, width = max(1, round(height * scale)), max(1, round(width * scale))
        pixels = resize_rgb(pixels, height, width, mode="area")
    nh, nw = round(height / 32) * 32, round(width / 32) * 32
    if nh * nw < 3136:
        scale = math.sqrt(3136 / (height * width))
        nh, nw = math.ceil(height * scale / 32) * 32, math.ceil(width * scale / 32) * 32
    elif nh * nw > 12845056:
        scale = math.sqrt(height * width / 12845056)
        nh, nw = (
            max(32, math.floor(height / scale / 32) * 32),
            max(32, math.floor(width / scale / 32) * 32),
        )
    if min(nh, nw) <= 0:
        raise ValueError("grounding image aspect ratio is too extreme")
    pixels = resize_rgb(pixels, nh, nw, mode="bilinear") * 2 - 1
    gh, gw = nh // 16, nw // 16
    frames = mx.repeat(pixels.transpose(0, 3, 1, 2), 2, axis=0)
    patches = frames.reshape(1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16)
    return patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 1536), (gh, gw)


def reference_boost_selection(pixels: mx.array, grid: tuple[int, int]) -> mx.array:
    """Reference-node mask semantics: area resize, then strict majority threshold.

    Callers crop source masks with reference_fit_geometry before reduction.
    RGB masks use channel mean; alpha is handled by the image I/O boundary.
    """
    reduced = resize_rgb(pixels, *grid, mode="area")
    return (mx.mean(reduced, axis=-1) > 0.5).reshape(-1)
