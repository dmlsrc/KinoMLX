"""Data-only Krea 2 source, component, and weight dtype settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.config_rules import RuleInputs
from kinomlx.modeling.dtypes import FLOAT_DTYPE_CHOICES, FloatDType
from kinomlx.settings import EnvironmentSettings

from .catalog import KREA2_VAE_DECODER_CHOICES, KREA2_VARIANT_CHOICES
from .settings_rules import (
    REFERENCE_FP16_ERROR,
    REFERENCE_PRECISION,
    VAE_SOURCE,
    VARIANT_SOURCE,
)

OPTIMIZED_DENOISE_MODE = "compact_unmasked_pretransposed"


def denoise_mode(reference_denoise: bool) -> str:
    """Name the denoiser one boolean selects, for every surface that reports it.

    The console header, the resource section and the machine report all state
    this fact, and before they shared an owner two of them disagreed on its
    spelling within one command's output.
    """
    return "reference" if reference_denoise else OPTIMIZED_DENOISE_MODE


def validate_variant_source(
    settings: Krea2Settings,
    *,
    declared_variant: str | None,
    variant_requires_explicit: bool,
) -> None:
    """Reuse the field-owned variant rule before opening other components."""
    VARIANT_SOURCE.run(
        RuleInputs(
            {"variant": settings.variant},
            {},
            {
                "declared_variant": declared_variant,
                "variant_requires_explicit": variant_requires_explicit,
            },
        )
    )


@config_record
@dataclass(frozen=True)
class Krea2Settings(EnvironmentSettings):
    """Select one Krea 2 family and independently replace its components."""

    environment_prefix: ClassVar[str] = "KINO_KREA2_"
    config_table: ClassVar[str] = "[model_settings]"

    # None preserves the path-free Turbo default while allowing an explicit
    # source root with model_index.json to declare Raw or Turbo itself.
    variant: str | None = field(
        default=None,
        metadata={
            "config_constraints": (ChoiceConstraint(KREA2_VARIANT_CHOICES),),
            "config_rules": (VARIANT_SOURCE,),
        },
    )
    # The baseline may be an ordinary local directory, including a directory
    # outside the Hugging Face cache. Component overrides below replace only
    # their named component and retain compatible baseline parts.
    source_path: Path | None = None
    tokenizer_path: Path | None = None
    text_encoder_path: Path | None = None
    transformer_path: Path | None = None
    vae_path: Path | None = None
    # None means stock Qwen unless an explicit VAE source is recognized as a
    # supported alternate decoder. A named alternate may resolve from HF_HOME.
    vae_decoder: str | None = field(
        default=None,
        metadata={
            "config_constraints": (ChoiceConstraint(KREA2_VAE_DECODER_CHOICES),),
            "config_rules": (VAE_SOURCE,),
        },
    )
    transformer_dtype: FloatDType = field(
        default=FloatDType.BFLOAT16,
        metadata={"config_constraints": (ChoiceConstraint(FLOAT_DTYPE_CHOICES),)},
    )
    reference_denoise: bool = field(
        default=False, metadata={"config_rules": (REFERENCE_PRECISION,)}
    )
    vae_dtype: FloatDType = field(
        default=FloatDType.BFLOAT16,
        metadata={"config_constraints": (ChoiceConstraint(FLOAT_DTYPE_CHOICES),)},
    )

    @property
    def denoise_mode(self) -> str:
        return denoise_mode(self.reference_denoise)

    @property
    def component_overrides(self) -> dict[str, Path]:
        """Return only explicitly selected logical-component replacements."""
        values = {
            "tokenizer": self.tokenizer_path,
            "text_encoder": self.text_encoder_path,
            "transformer": self.transformer_path,
            "vae": self.vae_path,
        }
        return {name: path for name, path in values.items() if path is not None}


__all__ = [
    "OPTIMIZED_DENOISE_MODE",
    "REFERENCE_FP16_ERROR",
    "Krea2Settings",
    "denoise_mode",
]
