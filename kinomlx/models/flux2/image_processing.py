"""Output-relative reference geometry with native pixels and ordered spatial IDs."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.geometry import derive_output_size, pinned_axis
from kinomlx.io.image import load_image
from kinomlx.io.resampling import reference_fit_geometry, resize_rgb

from .geometry_rules import output_aspect_mode
from .request_constraints import OUTPUT_DIMENSION


def output_size(
    source: tuple[int, int] | None,
    *,
    reference_count: int,
    width: int | None = None,
    height: int | None = None,
    match_source_aspect: str | None = None,
) -> tuple[int, int]:
    """Resolve Klein defaults before positioning references against the target grid."""
    if reference_count < 0 or bool(reference_count) != (source is not None):
        raise ValueError("reference count and first source size must agree")
    for name, value in (("width", width), ("height", height)):
        if value is not None:
            try:
                OUTPUT_DIMENSION.validate(value)
            except ValueError as exc:
                raise ValueError(f"Klein {name}: {exc}") from exc
    mode = output_aspect_mode(reference_count, width, height, match_source_aspect)
    if mode is not None:
        assert source is not None
        result = derive_output_size(
            source, axis=pinned_axis(mode, source), pixels=width or height or 1024
        )
    else:
        result = width or 1024, height or 1024
    if min(result) < 64:
        raise ValueError("resolved Klein output dimensions must be at least 64 pixels")
    return result


@dataclass(frozen=True)
class ReferenceGeometry:
    """Sizes are width/height; crop_box is in the pre-crop resized image's pixels."""

    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    reference_size: tuple[int, int]
    offset_hw: tuple[float, float]
    policy: str

    @property
    def resize_kernel(self) -> str:
        return "bicubic"

    def to_metadata(self) -> dict[str, object]:
        return {**asdict(self), "resize_kernel": self.resize_kernel}

    @property
    def token_count(self) -> int:
        width, height = self.reference_size
        return width // 16 * (height // 16)


def reference_geometry(
    source: tuple[int, int],
    target: tuple[int, int],
    *,
    reference_count: int,
    reference_fit: str | None = None,
) -> ReferenceGeometry:
    width, height = source
    tw, th = target
    if min(width, height) <= 0:
        raise ValueError("Klein source dimensions must be positive")
    if min(tw, th) < 64 or tw % 16 or th % 16 or reference_count < 1:
        raise ValueError("Klein target requires 16-pixel dimensions and at least one reference")
    if reference_fit not in {None, "fit", "crop"}:
        raise ValueError("reference_fit must be fit or crop")
    mode = reference_fit or ("fit" if reference_count == 1 else None)
    if mode is not None:
        top, left, ch, cw, nh, nw = reference_fit_geometry(height, width, th, tw, mode=mode)
        offsets = ((th - nh) / 32, (tw - nw) / 32) if mode == "fit" else (0.0, 0.0)
        return ReferenceGeometry(
            source, source, (left, top, left + cw, top + ch), (nw, nh), offsets, mode
        )
    # Native multi-reference grids preserve independent source scale with the
    # practical Diffusers area cap, then center-crop to packed latent alignment.
    if min(width, height) < 64 or max(width / height, height / width) > 8:
        raise ValueError("Klein native references require sides >=64 and aspect ratio <=8")
    scale = min(1.0, math.sqrt(1024**2 / (width * height)))
    rw, rh = int(width * scale), int(height * scale)
    nw, nh = rw // 16 * 16, rh // 16 * 16
    left, top = (rw - nw) // 2, (rh - nh) // 2
    return ReferenceGeometry(
        source, (rw, rh), (left, top, left + nw, top + nh), (nw, nh), (0.0, 0.0), "native"
    )


def prepare_reference_pixels(pixels: mx.array, geometry: ReferenceGeometry) -> mx.array:
    """BHWC float RGB [0,1] to BCHW [-1,1], with recorded native bicubic sampling."""
    if (
        pixels.ndim != 4
        or pixels.shape[0] != 1
        or pixels.shape[-1] != 3
        or (pixels.shape[2], pixels.shape[1]) != geometry.source_size
    ):
        raise ValueError("reference pixels must be one BHWC RGB image matching source geometry")
    value = pixels
    if geometry.resized_size != geometry.source_size:
        rw, rh = geometry.resized_size
        value = resize_rgb(value, rh, rw, mode="bicubic")
    left, top, right, bottom = geometry.crop_box
    value = value[:, top:bottom, left:right]
    nw, nh = geometry.reference_size
    value = resize_rgb(value, nh, nw, mode="bicubic")
    return (mx.clip(value, 0, 1) * 2 - 1).transpose(0, 3, 1, 2)


def load_reference_pixels(path: Path, geometry: ReferenceGeometry) -> mx.array:
    return prepare_reference_pixels(load_image(path)[None], geometry)


def spatial_ids(
    height: int,
    width: int,
    *,
    reference_index: int | None = None,
    offset_hw: tuple[float, float] = (0.0, 0.0),
) -> mx.array:
    """Batched THWL coordinates; grid H/W offsets can contain half-token positions."""
    if min(height, width) < 1 or (reference_index is not None and reference_index < 0):
        raise ValueError("spatial grids must be positive and reference indices non-negative")
    h = mx.repeat(mx.arange(height, dtype=mx.float32), width) + offset_hw[0]
    w = mx.tile(mx.arange(width, dtype=mx.float32), height) + offset_hw[1]
    t = mx.full(
        h.shape, 0 if reference_index is None else 10 * (reference_index + 1), dtype=mx.float32
    )
    return mx.stack([t, h, w, mx.zeros_like(h)], axis=-1)[None]


def pack_tokens(patches: mx.array) -> mx.array:
    if patches.ndim != 4 or patches.shape[1] != 128 or min(patches.shape) <= 0:
        raise ValueError("Klein token packing requires nonempty BCHW 128-channel patches")
    batch, channels, height, width = patches.shape
    return patches.reshape(batch, channels, height * width).transpose(0, 2, 1)


def unpack_tokens(tokens: mx.array, height: int, width: int) -> mx.array:
    if tokens.ndim != 3 or min(height, width) < 1 or tokens.shape[1:] != (height * width, 128):
        raise ValueError("Klein token shape must match the declared grid and 128 channels")
    return tokens.transpose(0, 2, 1).reshape(tokens.shape[0], 128, height, width)
