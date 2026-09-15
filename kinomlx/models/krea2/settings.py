"""Data-only Krea 2 source, component, and weight dtype settings."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import ChoiceConstraint
from kinomlx.config_records import config_record
from kinomlx.config_rules import RecordRule, RuleInputs
from kinomlx.settings import EnvironmentSettings

from .catalog import KREA2_VAE_DECODER_CHOICES, KREA2_VARIANT_CHOICES

TRANSFORMER_DTYPE_CHOICES = ("bfloat16", "float16", "float32")
VAE_DTYPE_CHOICES = TRANSFORMER_DTYPE_CHOICES
OPTIMIZED_DENOISE_MODE = "compact_unmasked_pretransposed"
REFERENCE_FP16_ERROR = (
    "Krea 2 reference denoising does not support float16: padding can cause FP16 "
    "overflow and nonfinite outputs. Remove --reference-denoise or select "
    "--transformer-dtype bfloat16 (or float32)."
)


def _reference_precision(values: Mapping[str, object]) -> None:
    if values["reference_denoise"] and values["transformer_dtype"] == "float16":
        raise ValueError(REFERENCE_FP16_ERROR)


_REFERENCE_PRECISION = RecordRule(
    "reference_precision",
    frozenset({"reference_denoise", "transformer_dtype"}),
    _reference_precision,
)


def _variant_source(values: RuleInputs) -> None:
    declared = values.context["declared_variant"]
    if values.context["variant_requires_explicit"] and values["variant"] is None:
        raise ValueError(
            "component-only or index-free Krea 2 sources require an explicit Krea 2 variant"
        )
    if declared is not None and values["variant"] is not None and values["variant"] != declared:
        raise ValueError(
            f"Krea 2 source declares variant {declared!r}, contradicting "
            f"variant {values['variant']!r}"
        )


def validate_variant_source(
    settings: Krea2Settings,
    *,
    declared_variant: str | None,
    variant_requires_explicit: bool,
) -> None:
    """Reuse the field-owned variant rule before opening other components."""
    _variant_source(
        RuleInputs(
            {"variant": settings.variant},
            {},
            {
                "declared_variant": declared_variant,
                "variant_requires_explicit": variant_requires_explicit,
            },
        )
    )


VARIANT_SOURCE = RecordRule(
    "variant_source",
    frozenset({"variant"}),
    _variant_source,
    phase="capability",
    context_reads=frozenset({"declared_variant", "variant_requires_explicit"}),
)


def _vae_source(values: RuleInputs) -> None:
    detected = values.context["detected_vae_decoder"]
    if values["vae_decoder"] is not None and values["vae_decoder"] != detected:
        raise ValueError(
            f"Krea 2 vae_decoder={values['vae_decoder']!r} contradicts "
            f"the {detected!r} decoder detected in the selected VAE source"
        )


VAE_SOURCE = RecordRule(
    "vae_source",
    frozenset({"vae_decoder"}),
    _vae_source,
    phase="capability",
    context_reads=frozenset({"detected_vae_decoder"}),
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
    transformer_dtype: str = field(
        default="bfloat16",
        metadata={"config_constraints": (ChoiceConstraint(TRANSFORMER_DTYPE_CHOICES),)},
    )
    reference_denoise: bool = field(
        default=False, metadata={"config_rules": (_REFERENCE_PRECISION,)}
    )
    vae_dtype: str = field(
        default="bfloat16", metadata={"config_constraints": (ChoiceConstraint(VAE_DTYPE_CHOICES),)}
    )

    @property
    def denoise_mode(self) -> str:
        return "reference" if self.reference_denoise else OPTIMIZED_DENOISE_MODE

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
    "TRANSFORMER_DTYPE_CHOICES",
    "VAE_DTYPE_CHOICES",
    "Krea2Settings",
]
