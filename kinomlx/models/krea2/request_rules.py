"""Pure coupled Krea request policies owned by participating fields."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kinomlx.adapters.selection import validate_adapter_gate
from kinomlx.config_rules import RecordRule, RuleInputs


def _vae_tiling(values: Mapping[str, object]) -> None:
    if cast(int, values["vae_tile_stride"]) >= cast(int, values["vae_tile_size"]):
        raise ValueError("vae_tile_stride must be smaller than vae_tile_size")


VAE_TILING = RecordRule("vae_tiling", frozenset({"vae_tile_stride", "vae_tile_size"}), _vae_tiling)


def _references(values: Mapping[str, object]) -> None:
    sources = cast(tuple[Path, ...], values["source_images"])
    if sources and values["system_prompt"] is not None:
        raise ValueError(
            "system_prompt is for regular text; use grounding_system_prompt for image editing"
        )
    boost = values["reference_boost_mask"] is not None or bool(values["reference_boost_regions"])
    if not sources and (boost or cast(str, values["grounding_system_prompt"]).strip()):
        raise ValueError(
            "grounding system prompt and reference boost selections require source images"
        )
    if boost and values["reference_strength"] == 1:
        raise ValueError("reference boost selections require --reference-strength other than 1")
    if values["scene_reference_strength"] != 1 and len(sources) != 2:
        raise ValueError("--scene-reference-strength requires two source images")
    if values["match_source_aspect"] is not None and not sources:
        raise ValueError("match_source_aspect requires source images")
    if not sources and (
        values["edit_adapter_path"] is not None
        or values["edit_adapter_strength"] != 1
        or values["edit_adapter_exclusions"]
    ):
        raise ValueError("edit adapter options require source images")
    if sources and values["text_conditioning"] is not None:
        raise ValueError("source images cannot be combined with a text-only restart")


def _profiles(values: RuleInputs) -> None:
    for profile in ("turbo_4step", "turbo_sda"):
        strength_field = f"{profile}_strength"
        exclusions_field = f"{profile}_exclusions"
        window_field = f"{profile}_sigma_window"
        steps_field = f"{profile}_step_strengths"
        window = values[window_field]
        array = values[steps_field]
        if not values[profile] and (values[strength_field] != 1 or values[exclusions_field]):
            raise ValueError(f"{strength_field} and {exclusions_field} require {profile}")
        validate_adapter_gate(
            has_window=bool(window),
            has_steps=bool(array),
            window_field=window_field,
            steps_field=steps_field,
        )
        if not values[profile] and (window or array):
            raise ValueError(f"{window_field} and {steps_field} require {profile}")
        if array and values.presence[strength_field]:
            raise ValueError(
                f"{steps_field} replaces {strength_field}; put the magnitude in the array"
            )


def _recipe_mode(values: RuleInputs) -> None:
    mode = values.context["recipe_mode"]
    sources = cast(tuple[Path, ...], values["source_images"])
    if mode == "text":
        if sources:
            raise ValueError("use generate_krea2_edit for source images")
    elif mode == "edit":
        if not 1 <= len(sources) <= 2:
            raise ValueError("Krea editing requires one or two source images")
        prompt = values["prompt"]
        negative = values["negative_prompt"]
        prompt_count = (
            1
            if prompt is None and values["prompt_unconditional"]
            else len(cast(tuple[str, ...], prompt or ()))
        )
        negative_count = 0 if negative is None else len(cast(tuple[str, ...], negative))
        if prompt_count != 1 or negative_count > 1:
            raise ValueError("Krea editing requires one prompt and at most one negative prompt")
        if values["text_conditioning"] is not None:
            raise ValueError(
                "editing requires image-grounded conditioning; text-only replay is incompatible"
            )
    elif mode != "decode":
        raise ValueError(f"unknown Krea recipe mode: {mode!r}")


def _negative_cfg(values: RuleInputs) -> None:
    negative = values["negative_prompt"]
    if (
        not values.context["cfg_enabled"]
        and negative is not None
        and any(cast(tuple[str, ...], negative))
    ):
        raise ValueError("negative_prompt is unused when Krea guidance is zero")


REFERENCES = RecordRule(
    "references",
    frozenset(
        {
            "source_images",
            "system_prompt",
            "grounding_system_prompt",
            "reference_boost_mask",
            "reference_boost_regions",
            "reference_strength",
            "scene_reference_strength",
            "match_source_aspect",
            "edit_adapter_path",
            "edit_adapter_strength",
            "edit_adapter_exclusions",
            "text_conditioning",
        }
    ),
    _references,
)
RECIPE_MODE = RecordRule(
    "recipe_mode",
    frozenset(
        {
            "source_images",
            "prompt",
            "prompt_unconditional",
            "negative_prompt",
            "text_conditioning",
        }
    ),
    _recipe_mode,
    phase="capability",
    context_reads=frozenset({"recipe_mode"}),
)
NEGATIVE_CFG = RecordRule(
    "negative_cfg",
    frozenset({"negative_prompt"}),
    _negative_cfg,
    phase="capability",
    context_reads=frozenset({"cfg_enabled"}),
)


def admit_recipe_mode(request: object, mode: str) -> None:
    """Run the field-owned mode decision before final sampler context exists."""
    RECIPE_MODE.run(
        RuleInputs(
            {name: getattr(request, name) for name in RECIPE_MODE.reads},
            {},
            {"recipe_mode": mode},
        )
    )


PROFILES = RecordRule(
    "profiles",
    frozenset(
        {
            "turbo_4step",
            "turbo_4step_strength",
            "turbo_4step_exclusions",
            "turbo_4step_sigma_window",
            "turbo_4step_step_strengths",
            "turbo_sda",
            "turbo_sda_strength",
            "turbo_sda_exclusions",
            "turbo_sda_sigma_window",
            "turbo_sda_step_strengths",
        }
    ),
    _profiles,
    presence_reads=frozenset({"turbo_4step_strength", "turbo_sda_strength"}),
)
