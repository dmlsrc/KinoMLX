"""Pure final resource policies owned by Krea 2 Settings fields."""

from collections.abc import Mapping

from kinomlx.config_rules import RecordRule, RuleInputs
from kinomlx.modeling.dtypes import FloatDType

REFERENCE_FP16_ERROR = (
    "Krea 2 reference denoising does not support float16: padding can cause FP16 "
    "overflow and nonfinite outputs. Remove --reference-denoise or select "
    "--transformer-dtype bfloat16 (or float32)."
)


def _reference_precision(values: Mapping[str, object]) -> None:
    if values["reference_denoise"] and values["transformer_dtype"] == FloatDType.FLOAT16:
        raise ValueError(REFERENCE_FP16_ERROR)


REFERENCE_PRECISION = RecordRule(
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
