"""Shared explicit empty-user-prompt contract for generation requests."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from kinomlx.config_constraints import LengthConstraint


def is_empty_prompt_batch(prompt: object) -> bool:
    """Recognize literal empty text, including expanded saved-conditioning rows."""
    if isinstance(prompt, str):
        return prompt == ""
    return isinstance(prompt, tuple | list) and bool(prompt) and all(row == "" for row in prompt)


def validate_prompt_unconditional(prompt_unconditional: bool, prompt: object) -> None:
    """Reject conflicting text while allowing resolved empty text in internal requests."""
    if type(prompt_unconditional) is not bool:
        raise ValueError("prompt_unconditional must be a boolean")
    if prompt_unconditional and prompt is not None and not is_empty_prompt_batch(prompt):
        raise ValueError("prompt_unconditional cannot be combined with prompt or prompt_file")


def normalize_prompts(
    value: str | Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    prompts: tuple[str, ...]
    if isinstance(value, str):
        prompts = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        prompts = tuple(value)
    else:
        raise TypeError(f"{name} must be a string or nonempty sequence of strings")
    if not prompts:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError(f"{name} must contain only strings")
    if not allow_empty and any(not prompt.strip() for prompt in prompts):
        raise ValueError(f"{name} must not contain empty prompts")
    return prompts


@dataclass(frozen=True)
class PromptInputConstraint(LengthConstraint):
    def validate(self, value: object) -> None:
        normalize_prompts(cast(str | Sequence[str], value), name="prompt", allow_empty=True)

    def describe(self) -> dict[str, object]:
        return {"kind": "prompt_rows", "minimum_rows": 1}
