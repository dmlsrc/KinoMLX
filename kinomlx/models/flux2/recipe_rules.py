"""Pure Klein recipe defaults and field-owned profile relationships."""

from typing import cast

from kinomlx.config_rules import RecordRule, RuleInputs

from .catalog import Flux2Variant, profile


def recipe_values(
    variant: str, steps: int | None, guidance: float | None, sigmas: tuple[float, ...]
) -> tuple[int, float]:
    """Resolve profile defaults and the step count owned by an explicit schedule."""
    selected = profile(variant)
    return (
        len(sigmas) - 1 if sigmas else selected.steps if steps is None else steps,
        selected.guidance if guidance is None else guidance,
    )


def validate_cfg(variant: str, guidance: float, cfg_mode: str) -> None:
    """The base profile owns external guidance and its execution modes."""
    if variant != Flux2Variant.BASE:
        if guidance != 1:
            raise ValueError("External CFG is supported by the base recipe only")
        if cfg_mode != "sequential":
            raise ValueError("cfg_mode batched requires the base model")


def _recipe(values: RuleInputs) -> None:
    variant = cast(str, values["variant"])
    if values["recompute_reference_kv"] and not profile(variant).kv:
        raise ValueError("Reference K/V recomputation requires the KV recipe")
    _, guidance = recipe_values(
        variant,
        cast(int | None, values["steps"]),
        cast(float | None, values["guidance"]),
        cast(tuple[float, ...], values["sigmas"]),
    )
    validate_cfg(variant, guidance, cast(str, values["cfg_mode"]))


SAMPLING_RECIPE = RecordRule(
    "sampling_recipe",
    frozenset({"variant", "steps", "guidance", "sigmas", "cfg_mode", "recompute_reference_kv"}),
    _recipe,
)
