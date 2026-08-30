"""TOML-typed ``--set <key>[.<key>...]=<value>`` overrides."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from typing import cast

from .validate import ConfigError


def _parse_toml_value(raw: str) -> object:
    try:
        parsed = tomllib.loads(f"value = {raw}")
        return cast(object, parsed["value"])
    except tomllib.TOMLDecodeError:
        return raw


def parse_set_argument(argument: str) -> tuple[list[str], object]:
    """Parse one override into a dotted key path and a typed value."""
    key_part, separator, value_part = argument.partition("=")
    if not separator:
        raise ConfigError(f"--set {argument!r}: expected <key>[.<key>...]=<value>")
    path = [part.strip() for part in key_part.strip().split(".")]
    if not all(path):
        raise ConfigError(
            f"--set {argument!r}: key path must be <key>[.<key>...] (got {key_part.strip()!r})"
        )
    return path, _parse_toml_value(value_part)


def apply_set_overrides(
    config: Mapping[str, object],
    set_arguments: list[str] | None,
) -> dict[str, object]:
    """Apply overrides in order to a copy of ``config``."""
    result = dict(config)
    for argument in set_arguments or []:
        path, value = parse_set_argument(argument)
        node = result
        for index, key in enumerate(path[:-1]):
            existing = node.get(key)
            if existing is None:
                child: dict[str, object] = {}
                node[key] = child
                node = child
            elif isinstance(existing, dict):
                child = dict(cast(dict[str, object], existing))
                node[key] = child
                node = child
            else:
                location = ".".join(path[: index + 1])
                raise ConfigError(
                    f"--set {argument!r}: {location} is not a table "
                    f"(found {type(existing).__name__})"
                )
        node[path[-1]] = value
    return result
