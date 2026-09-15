"""Klein request relationships and prepared recipe capability policy."""

from typing import cast

from kinomlx.adapters.selection import require_adapter_schedule_steps
from kinomlx.config_rules import RecordRule, RuleInputs
from kinomlx.prompts import normalize_prompts

from .adapters.inputs import Flux2AdapterConfig
from .catalog import profile
from .recipe_rules import recipe_values, validate_cfg


def negative_rows(
    prompt_count: int, negative_prompt: str | tuple[str, ...] | None
) -> tuple[str, ...]:
    """Broadcast one negative row or require one for each positive row."""
    rows = (
        ("",)
        if negative_prompt is None
        else normalize_prompts(negative_prompt, name="negative_prompt", allow_empty=True)
    )
    if len(rows) == 1:
        return rows * prompt_count
    if len(rows) != prompt_count:
        raise ValueError("negative_prompt requires one value or one per prompt")
    return rows


def _prompt_count(values: RuleInputs) -> int:
    prompt = cast(str | tuple[str, ...] | None, values["prompt"])
    if isinstance(prompt, tuple):
        return len(prompt)
    return int(prompt is not None or bool(values["prompt_unconditional"]))


def _negative_rows(values: RuleInputs) -> None:
    negative_rows(
        _prompt_count(values), cast(str | tuple[str, ...] | None, values["negative_prompt"])
    )


NEGATIVE_ROWS = RecordRule(
    "negative_rows",
    frozenset({"prompt", "prompt_unconditional", "negative_prompt"}),
    _negative_rows,
)


def _recipe_capabilities(values: RuleInputs) -> None:
    variant = cast(str, values.context["variant"])
    steps, guidance = recipe_values(
        variant,
        cast(int | None, values["steps"]),
        cast(float | None, values["guidance"]),
        cast(tuple[float, ...], values["sigmas"]),
    )
    validate_cfg(variant, guidance, cast(str, values["cfg_mode"]))
    if values["recompute_reference_kv"] and not profile(variant).kv:
        raise ValueError("recompute_reference_kv requires the KV variant")
    negatives = negative_rows(
        _prompt_count(values), cast(str | tuple[str, ...] | None, values["negative_prompt"])
    )
    if guidance == 1 and any(negatives):
        raise ValueError("Negative prompts require active base CFG")
    adapters = cast(tuple[Flux2AdapterConfig, ...], values["adapters"])
    require_adapter_schedule_steps(
        tuple(
            adapter.selection(location=f"[generate].adapters[{index}]")
            for index, adapter in enumerate(adapters)
        ),
        steps,
    )


RECIPE_CAPABILITIES = RecordRule(
    "recipe_capabilities",
    frozenset(
        {
            "steps",
            "guidance",
            "sigmas",
            "cfg_mode",
            "recompute_reference_kv",
            "prompt",
            "prompt_unconditional",
            "negative_prompt",
            "adapters",
        }
    ),
    _recipe_capabilities,
    phase="capability",
    context_reads=frozenset({"variant"}),
)
