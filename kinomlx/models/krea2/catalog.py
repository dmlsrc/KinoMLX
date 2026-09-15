"""Published Krea 2 variants and their local acquisition facts."""

from dataclasses import dataclass
from enum import StrEnum

from kinomlx.huggingface import download_hint


class Krea2Variant(StrEnum):
    """Recipe-bearing Krea 2 checkpoint families."""

    RAW = "raw"
    TURBO = "turbo"


class Krea2VAEDecoder(StrEnum):
    """Supported Krea-compatible still-image decoder profiles."""

    QWEN = "qwen"
    UPSCALE2X = "upscale2x"

    @property
    def output_scale(self) -> int:
        """Return the decoder's spatial scale relative to the stock VAE."""
        return 2 if self is Krea2VAEDecoder.UPSCALE2X else 1

    @property
    def output_channels(self) -> int:
        """Return the packed channel width produced by the final convolution."""
        return 3 * self.output_scale**2


@dataclass(frozen=True)
class Krea2VariantSpec:
    """Small source and recipe identity for one official variant."""

    variant: Krea2Variant
    repo_id: str
    monolith_filename: str
    is_distilled: bool
    recipe_families: tuple[str, ...]
    default_steps: int
    default_guidance: float
    default_mu: float | None


_VARIANTS = {
    Krea2Variant.RAW: Krea2VariantSpec(
        variant=Krea2Variant.RAW,
        repo_id="krea/Krea-2-Raw",
        monolith_filename="raw.safetensors",
        is_distilled=False,
        recipe_families=("raw",),
        default_steps=52,
        default_guidance=3.5,
        default_mu=None,
    ),
    Krea2Variant.TURBO: Krea2VariantSpec(
        variant=Krea2Variant.TURBO,
        repo_id="krea/Krea-2-Turbo",
        monolith_filename="turbo.safetensors",
        is_distilled=True,
        recipe_families=("turbo",),
        default_steps=8,
        default_guidance=0.0,
        default_mu=1.15,
    ),
}

DEFAULT_VARIANT = Krea2Variant.TURBO
KREA2_VARIANT_CHOICES = tuple(item.value for item in Krea2Variant)
DEFAULT_VAE_DECODER = Krea2VAEDecoder.QWEN
KREA2_VAE_DECODER_CHOICES = tuple(item.value for item in Krea2VAEDecoder)
UPSCALE2X_VAE_REPO_ID = "spacepxl/Wan2.1-VAE-upscale2x"
UPSCALE2X_VAE_FILENAME = "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"


def variant_spec(variant: Krea2Variant | str) -> Krea2VariantSpec:
    """Return the immutable catalog entry for one variant."""
    try:
        selected = variant if isinstance(variant, Krea2Variant) else Krea2Variant(variant)
    except ValueError as exc:
        valid = ", ".join(KREA2_VARIANT_CHOICES)
        raise ValueError(f"Krea 2 variant must be one of: {valid}") from exc
    return _VARIANTS[selected]


def variant_from_distilled(is_distilled: bool) -> Krea2Variant:
    """Map the snapshot recipe declaration to a public variant."""
    return Krea2Variant.TURBO if is_distilled else Krea2Variant.RAW


def repository_id(variant: Krea2Variant | str) -> str:
    """Return the official Hugging Face repository id."""
    return variant_spec(variant).repo_id


def download_command(variant: Krea2Variant | str) -> str:
    """Return the exact user-run acquisition command, excluding only the root monolith."""
    spec = variant_spec(variant)
    return download_hint(spec.repo_id, "--exclude", spec.monolith_filename)


def upscale2x_vae_download_command() -> str:
    """Return the exact one-file acquisition command for the optional decoder."""
    return download_hint(UPSCALE2X_VAE_REPO_ID, UPSCALE2X_VAE_FILENAME)


__all__ = [
    "DEFAULT_VAE_DECODER",
    "DEFAULT_VARIANT",
    "KREA2_VAE_DECODER_CHOICES",
    "KREA2_VARIANT_CHOICES",
    "UPSCALE2X_VAE_FILENAME",
    "UPSCALE2X_VAE_REPO_ID",
    "Krea2VAEDecoder",
    "Krea2Variant",
    "Krea2VariantSpec",
    "download_command",
    "repository_id",
    "upscale2x_vae_download_command",
    "variant_from_distilled",
    "variant_spec",
]
