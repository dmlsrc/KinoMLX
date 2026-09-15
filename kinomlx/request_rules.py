"""Pure coupled policies shared by typed generation request fields."""

from collections.abc import Mapping
from typing import cast

from kinomlx.config_rules import RecordRule
from kinomlx.prompts import validate_prompt_unconditional
from kinomlx.seeds import resolve_seeds


def _unconditional_prompt(values: Mapping[str, object]) -> None:
    validate_prompt_unconditional(cast(bool, values["prompt_unconditional"]), values["prompt"])


def _image_seeds(values: Mapping[str, object]) -> None:
    resolve_seeds(
        cast(int, values["seed"]),
        cast(tuple[int, ...] | None, values["seeds"]),
        cast(int, values["images_per_prompt"]),
    )


def _saved_conditioning(values: Mapping[str, object]) -> None:
    if values["text_conditioning"] is None:
        return
    if values["source_images"] or values["reference_fit"] or values["match_source_aspect"]:
        raise ValueError(
            "Saved conditioning replaces source images and reference geometry controls"
        )
    prompt = values["prompt"]
    if isinstance(prompt, tuple) and len(prompt) > 1:
        raise ValueError("Saved conditioning replay requires one prompt")


UNCONDITIONAL_PROMPT = RecordRule(
    "unconditional_prompt", frozenset({"prompt_unconditional", "prompt"}), _unconditional_prompt
)
IMAGE_SEEDS = RecordRule(
    "image_seeds", frozenset({"seed", "seeds", "images_per_prompt"}), _image_seeds
)
SAVED_IMAGE_CONDITIONING = RecordRule(
    "saved_image_conditioning",
    frozenset(
        {"text_conditioning", "source_images", "reference_fit", "match_source_aspect", "prompt"}
    ),
    _saved_conditioning,
)


def _prompt(values: Mapping[str, object]) -> None:
    prompt = values["prompt"]
    unconditional = cast(bool, values["prompt_unconditional"])
    validate_prompt_unconditional(unconditional, prompt)
    if prompt is not None and not unconditional:
        rows = (prompt,) if isinstance(prompt, str) else cast(tuple[str, ...], prompt)
        if any(not row.strip() for row in rows):
            raise ValueError("prompt must not contain empty prompts")


IMAGE_PROMPT = RecordRule("prompt", frozenset({"prompt", "prompt_unconditional"}), _prompt)
