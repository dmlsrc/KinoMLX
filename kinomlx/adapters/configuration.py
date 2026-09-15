"""Bounded, typed interpretation of data-only adapter configuration."""

import math
import re

from kinomlx._json import JsonObject


def number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"adapter {name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"adapter {name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"adapter {name} must be a finite number")
    return result


def integer(value: object, *, name: str, minimum: int = 1) -> int:
    result = number(value, name=name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f"adapter {name} must be an integer >= {minimum}")
    return int(result)


def boolean(config: JsonObject, name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"PEFT {name} must be a boolean")
    return value


def module_name(prefix: str) -> str:
    return prefix.removeprefix("base_model.model.")


def matches(pattern: str, name: str, *, suffix: bool) -> bool:
    # Checkpoint configuration is data, not an invitation to unbounded regex
    # backtracking. Flat patterns cover module paths, numeric classes, anchors,
    # and wildcards; repeated groups/backreferences must be exported literally.
    if (
        len(pattern) > 256
        or "(?" in pattern
        or re.search(r"\\[1-9]|\)[*+?{]", pattern)
        or sum(pattern.count(token) for token in ("*", "+", "{")) > 2
    ):
        raise ValueError("PEFT complex regex patterns require literal per-target keys")
    try:
        return re.fullmatch((r"(?:.*\.)?" if suffix else "") + pattern, name) is not None
    except re.error as exc:
        raise ValueError(f"invalid PEFT module pattern {pattern!r}") from exc


def pattern_value(config: JsonObject, field: str, prefix: str, default: object) -> object:
    patterns = config.get(field, {})
    if not isinstance(patterns, dict):
        raise ValueError(f"PEFT {field} must be an object")
    for pattern, value in patterns.items():
        if matches(pattern, module_name(prefix), suffix=True):
            return value
    return default
