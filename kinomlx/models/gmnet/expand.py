"""SDR-to-HDR expansion: preprocessing, inference, and reconstruction.

The numeric contract follows the upstream evaluation exactly:

1. the display-referred SDR image (RGB, ``[0, 1]``) is bicubic-downscaled
   by the variant's network scale for the local branch and antialias-resized
   to 256x256 for the global branch;
2. the network predicts a Qmax-scaled gain map, clamped to ``[0, 1]`` and
   bilinearly resized back to the source geometry;
3. scene-linear HDR is ``sdr ** 2.2 * 2 ** (gain_map * gain_stops)``,
   where ``gain_stops = log2(peak_over_sdr_white)``.

The result keeps 1.0 at SDR diffuse white, so it feeds the native EXR
terminal directly and the PQ HEIC terminal after its own reference-white
scaling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from kinomlx.io.safetensors import save_weights
from kinomlx.models.gmnet.catalog import GMNetVariantSpec
from kinomlx.models.gmnet.net import THUMBNAIL_SIZE
from kinomlx.models.gmnet.resample import (
    resize_bicubic,
    resize_bicubic_antialiased,
    resize_bilinear,
)

GAMMA_EXPONENT = 2.2

GAIN_MAP_SIDECAR_SUFFIX = ".gain_map.safetensors"


class _GMNetCallable(Protocol):
    def __call__(
        self,
        image: mx.array,
        thumbnail: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]: ...


@dataclass(frozen=True)
class ExpansionResult:
    """One expanded still plus the observational facts that produced it."""

    linear_rgb: mx.array
    """``(H, W, 3)`` float32 scene-linear Rec.709; 1.0 = SDR diffuse white."""

    gain_map: mx.array
    """``(H, W)`` float32 normalized gain in ``[0, 1]`` at source geometry."""

    qmax_normalized: float
    """The scalar Qmax head output (1.0 would use the full variant range)."""

    spec: GMNetVariantSpec

    @property
    def peak_linear(self) -> float:
        """Largest reconstructed linear value relative to SDR diffuse white."""
        return float(cast(int | float, mx.max(self.linear_rgb).item()))


def _validate_sdr_image(image: mx.array) -> mx.array:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expansion input must be HWC RGB, got {tuple(image.shape)}")
    if image.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise TypeError(f"expansion input must be floating point, got {image.dtype}")
    return mx.clip(image.astype(mx.float32), 0.0, 1.0)


def reconstruct_linear_hdr(
    image: mx.array,
    gain_map: mx.array,
    gain_stops: float,
) -> mx.array:
    """Apply one normalized gain map to a display-referred SDR image.

    ``image`` is HWC RGB in ``[0, 1]``; ``gain_map`` is ``(H, W)`` in
    ``[0, 1]``. Returns float32 scene-linear RGB with 1.0 at SDR diffuse
    white. The SDR decode is the upstream pure power 2.2, not piecewise
    sRGB - keep it that way for checkpoint fidelity.
    """
    sdr_linear = mx.power(_validate_sdr_image(image), GAMMA_EXPONENT)
    gain = mx.clip(gain_map.astype(mx.float32), 0.0, 1.0)[:, :, None]
    return sdr_linear * mx.power(2.0, gain * gain_stops)


def expand_image(
    model: _GMNetCallable,
    image: mx.array,
    spec: GMNetVariantSpec,
) -> ExpansionResult:
    """Expand one display-referred SDR still into scene-linear HDR."""
    source = _validate_sdr_image(image)
    height, width = int(source.shape[0]), int(source.shape[1])

    if spec.network_scale > 1:
        local_width = max(1, round(width / spec.network_scale))
        local_height = max(1, round(height / spec.network_scale))
        local = mx.clip(resize_bicubic(source, local_width, local_height), 0.0, 1.0)
    else:
        local = source
    thumbnail = mx.clip(
        resize_bicubic_antialiased(source, THUMBNAIL_SIZE, THUMBNAIL_SIZE),
        0.0,
        1.0,
    )

    _, scaled_gain, qmax = model(local[None], thumbnail[None])
    scaled_gain = mx.maximum(scaled_gain[0], 0.0)
    if tuple(scaled_gain.shape[:2]) != (height, width):
        scaled_gain = resize_bilinear(scaled_gain, width, height)
    gain_map = mx.clip(scaled_gain[:, :, 0], 0.0, 1.0)

    linear = reconstruct_linear_hdr(source, gain_map, spec.gain_stops)
    mx.eval(linear, gain_map)
    return ExpansionResult(
        linear_rgb=linear,
        gain_map=gain_map,
        qmax_normalized=float(cast(int | float, qmax[0, 0, 0, 0].item())),
        spec=spec,
    )


def write_gain_map_sidecar(
    path: Path | str,
    result: ExpansionResult,
    *,
    source_image: Path | None = None,
) -> Path:
    """Persist the gain map as a self-describing safetensors sidecar.

    The metadata carries everything needed to re-render the HDR without
    the model: the variant, its stop range, and the reconstruction law.
    """
    spec = result.spec
    metadata = {
        "producer": "gmnet",
        "artifact": "gain-map",
        "variant": spec.variant.value,
        "network_scale": str(spec.network_scale),
        "peak_over_sdr_white": f"{spec.peak_over_sdr_white:g}",
        "gain_stops_max": f"{spec.gain_stops:.8g}",
        "sdr_reference_white_nits": f"{spec.sdr_reference_white_nits:g}",
        "qmax_normalized": f"{result.qmax_normalized:.8g}",
        "reconstruction": (
            "linear = clip(sdr_rgb, 0, 1) ** 2.2 * 2 ** (gain_map * gain_stops_max); "
            "1.0 = SDR diffuse white"
        ),
    }
    if source_image is not None:
        metadata["source_image"] = source_image.name
        metadata["source_image_sha256"] = hashlib.sha256(source_image.read_bytes()).hexdigest()
    target = Path(path)
    save_weights(target, {"gain_map": result.gain_map}, metadata)
    return target


__all__ = [
    "GAIN_MAP_SIDECAR_SUFFIX",
    "GAMMA_EXPONENT",
    "ExpansionResult",
    "expand_image",
    "reconstruct_linear_hdr",
    "write_gain_map_sidecar",
]
