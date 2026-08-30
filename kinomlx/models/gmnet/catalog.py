"""GMNet variant facts and converted-weights resolution.

The two published checkpoints differ in training domain and in the
numeric contract baked into their outputs, so the variant is a typed
fact, not a filename convention:

- ``realworld`` - trained on photographed pairs; SDR reference white is
  203 nits, HDR peak 5x over it (about 1015 nits), and the network runs
  its local branch on a half-resolution input.
- ``synthetic`` - trained on HDR video frames; SDR reference white is
  100 nits, HDR peak 8x over it (800 nits), full-resolution local
  branch.

Both predict a gain map normalized so that 1.0 means the full
``log2(peak)`` stops of expansion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


class GMNetVariant(StrEnum):
    REALWORLD = "realworld"
    SYNTHETIC = "synthetic"


DEFAULT_VARIANT = GMNetVariant.REALWORLD


@dataclass(frozen=True)
class GMNetVariantSpec:
    """Facts one published checkpoint carries beyond its tensors."""

    variant: GMNetVariant
    network_scale: int
    peak_over_sdr_white: float
    sdr_reference_white_nits: float
    weights_filename: str
    source_filename: str
    source_url: str
    source_sha256: str

    @property
    def gain_stops(self) -> float:
        """Full expansion range in stops: ``log2(peak_over_sdr_white)``."""
        return math.log2(self.peak_over_sdr_white)


_SPECS = {
    GMNetVariant.REALWORLD: GMNetVariantSpec(
        variant=GMNetVariant.REALWORLD,
        network_scale=2,
        peak_over_sdr_white=5.0,
        sdr_reference_white_nits=203.0,
        weights_filename="gmnet_realworld.safetensors",
        source_filename="G_realworld.pth",
        source_url="https://github.com/qtlark/GMNet/raw/main/checkpoints/G_realworld.pth",
        source_sha256="83bf27bcdbf6eacfdef37f0e24ed6d79152b7386620c012ae509a59a895c875f",
    ),
    GMNetVariant.SYNTHETIC: GMNetVariantSpec(
        variant=GMNetVariant.SYNTHETIC,
        network_scale=1,
        peak_over_sdr_white=8.0,
        sdr_reference_white_nits=100.0,
        weights_filename="gmnet_synthetic.safetensors",
        source_filename="G_synthetic.pth",
        source_url="https://github.com/qtlark/GMNet/raw/main/checkpoints/G_synthetic.pth",
        source_sha256="887c940d492424cd44f029c6b09dd3bbe1bbec07126f15d41192828ff95e6880",
    ),
}


def variant_spec(variant: GMNetVariant | str) -> GMNetVariantSpec:
    """Return the immutable spec for one published variant."""
    return _SPECS[GMNetVariant(variant)]


def variant_for_source_sha256(digest: str) -> GMNetVariant | None:
    """Identify a published upstream checkpoint by its file hash."""
    for spec in _SPECS.values():
        if spec.source_sha256 == digest.lower():
            return spec.variant
    return None


def variant_weights_path(
    variant: GMNetVariant | str,
    cache_dir: Path | str,
) -> Path:
    """Return the convenient writable path for one converted checkpoint.

    Editable source checkouts keep derived weights beside the model-specific
    conversion documentation. Installed packages use the infrastructure cache
    instead of assuming site-packages is writable.
    """
    checkout_dir = _editable_checkout_weights_dir()
    if checkout_dir is not None:
        return checkout_dir / variant_spec(variant).weights_filename
    return (
        Path(cache_dir).expanduser() / "weights" / "gmnet" / variant_spec(variant).weights_filename
    )


def _editable_checkout_weights_dir() -> Path | None:
    """Return the repo-local weights directory only in an actual Git checkout."""
    repository = Path(__file__).resolve().parents[3]
    if (repository / ".git").exists() and (repository / "pyproject.toml").is_file():
        return WEIGHTS_DIR
    return None


def resolve_variant_weights(
    variant: GMNetVariant | str,
    override: Path | None = None,
    *,
    cache_dir: Path | str | None = None,
) -> Path:
    """Resolve the converted safetensors for ``variant``.

    An explicit ``override`` path wins. Editable development checkouts keep
    converted files beside this model's conversion documentation; installed
    packages use the configured KinoMLX cache. Weights are not distributed
    with the repository; the error points at the documented conversion step.
    """
    spec = variant_spec(variant)
    if override is not None:
        target = override.expanduser()
    else:
        if cache_dir is None:
            from kinomlx.settings import Settings

            cache_dir = Settings.from_env_fields("cache_dir").cache_dir
        target = variant_weights_path(spec.variant, cache_dir)
    if not target.is_file():
        raise FileNotFoundError(
            f"no converted GMNet weights at {target}; download {spec.source_filename} "
            f"from {spec.source_url} and run "
            f"'kinomlx weights convert gmnet {spec.source_filename}' "
            f"(see {WEIGHTS_DIR / 'README.md'})"
        )
    return target


__all__ = [
    "DEFAULT_VARIANT",
    "GMNetVariant",
    "GMNetVariantSpec",
    "WEIGHTS_DIR",
    "resolve_variant_weights",
    "variant_for_source_sha256",
    "variant_spec",
    "variant_weights_path",
]
