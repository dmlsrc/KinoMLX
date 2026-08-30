"""Deterministic TOML serialization for resolved configuration."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from .validate import ConfigError

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_LITERAL_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_TOML_INT_MIN = -(2**63)
_TOML_INT_MAX = 2**63 - 1


def _key(value: str) -> str:
    return value if _BARE_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def _multiline_literal(value: str) -> str | None:
    if "'''" in value or _LITERAL_CONTROL.search(value) is not None:
        return None
    return "'''\n" + value + "\n'''"


def dump_toml_value(
    value: object,
    *,
    multiline: bool = False,
    stringify_large_int: bool = False,
) -> str:
    """Serialize one scalar, array, or inline-table TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | Path):
        if multiline and isinstance(value, str):
            rendered = _multiline_literal(value)
            if rendered is not None:
                return rendered
        return json.dumps(str(value), ensure_ascii=False)
    if isinstance(value, int):
        if not _TOML_INT_MIN <= value <= _TOML_INT_MAX:
            if stringify_large_int:
                return json.dumps(str(value))
            raise ConfigError(f"integer {value} is outside TOML's signed 64-bit range")
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return repr(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return (
            "["
            + ", ".join(
                dump_toml_value(item, stringify_large_int=stringify_large_int) for item in value
            )
            + "]"
        )
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{_key(str(key))} = {dump_toml_value(item, stringify_large_int=stringify_large_int)}"
            for key, item in value.items()
        )
        return "{ " + items + " }"
    raise ConfigError(f"cannot print config value {value!r} ({type(value).__name__}) as TOML")


def _mapping_has_content(value: Mapping[object, object]) -> bool:
    return any(
        _mapping_has_content(item) if isinstance(item, Mapping) else True for item in value.values()
    )


def dump_config(
    config: Mapping[str, object],
    *,
    multiline_paths: frozenset[tuple[str, ...]] = frozenset(),
    stringify_large_int_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> str:
    """Serialize nested mappings as deterministic, human-readable TOML tables."""
    blocks: list[list[str]] = []

    def assignment(path: tuple[str, ...], value: object) -> str:
        rendered = dump_toml_value(
            value,
            multiline=path in multiline_paths,
            stringify_large_int=path in stringify_large_int_paths,
        )
        return f"{_key(path[-1])} = {rendered}"

    root_values = [
        assignment((str(key),), value)
        for key, value in config.items()
        if not isinstance(value, Mapping)
    ]
    if root_values:
        blocks.append(root_values)

    def collect(path: tuple[str, ...], table: Mapping[object, object]) -> None:
        if not _mapping_has_content(table):
            return
        direct = [
            assignment((*path, str(key)), value)
            for key, value in table.items()
            if not isinstance(value, Mapping)
        ]
        if direct:
            header = ".".join(_key(part) for part in path)
            blocks.append([f"[{header}]", *direct])
        for key, value in table.items():
            if isinstance(value, Mapping):
                collect((*path, str(key)), value)

    for key, value in config.items():
        if isinstance(value, Mapping):
            collect((str(key),), value)
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


__all__ = ["dump_config", "dump_toml_value"]
