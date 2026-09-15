"""Concrete configuration addresses with explicit nonnegative collection indices."""

import re

from kinomlx.config_values import ConfigError

type TargetPath = tuple[str | int, ...]

_MEMBER = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)((?:\[[0-9]+\])*)")
_INDEX = re.compile(r"\[([0-9]+)\]")


def parse_target_path(text: str) -> TargetPath:
    """Parse dotted members and bracket indices; no wildcard or slice syntax."""
    parts: list[str | int] = []
    for segment in text.strip().split("."):
        match = _MEMBER.fullmatch(segment.strip())
        if match is None:
            raise ConfigError(f"invalid configuration path {text!r}: expected members and [index]")
        parts.append(match.group(1))
        try:
            parts.extend(int(index.group(1)) for index in _INDEX.finditer(match.group(2)))
        except ValueError as exc:
            raise ConfigError(f"invalid configuration path {text!r}: index is too large") from exc
    return tuple(parts)


def format_target_path(path: TargetPath) -> str:
    """Render a concrete address without turning an index into a member name."""
    result = ""
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else ("." if result else "") + part
    return result


def field_path(path: TargetPath) -> tuple[str, ...]:
    """Require a declaration path where a consumer has no contextual item projection."""
    if any(isinstance(part, int) for part in path):
        raise ConfigError(
            f"{format_target_path(path)}: indexed overrides are not yet supported by this configuration consumer"
        )
    return tuple(part for part in path if isinstance(part, str))
