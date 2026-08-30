"""Small dependency-free schema helpers for dataclass-backed config."""

from __future__ import annotations

import dataclasses
import difflib
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, Union, cast, get_args, get_origin, get_type_hints

from kinomlx.errors import KinoMLXError


class ConfigError(KinoMLXError, ValueError):
    """A user-authored configuration error safe to render without traceback."""


class _KeywordConstructor[ResultT](Protocol):
    def __call__(self, **kwargs: object) -> ResultT: ...


def _unknown_key(name: str, allowed: set[str], location: str) -> ConfigError:
    known = sorted(allowed)
    suggestion = difflib.get_close_matches(name, known, n=1)
    hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
    return ConfigError(f"{location}: unknown field {name!r}{hint} (known: {', '.join(known)})")


def _config_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location}: expected table, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{location}: table keys must be strings")
    return cast(Mapping[str, object], value)


def validate_top_level(
    config: Mapping[str, object],
    *,
    tables: set[str] | frozenset[str],
    scalars: set[str] | frozenset[str] = frozenset(),
    source: str = "config",
) -> None:
    """Validate known root names and enforce table/scalar structure."""
    allowed = set(tables) | set(scalars)
    for key, value in config.items():
        if key not in allowed:
            raise _unknown_key(key, allowed, source)
        if key in tables and not isinstance(value, Mapping):
            raise ConfigError(f"{source}: [{key}] must be a table")
        if key in scalars and isinstance(value, Mapping):
            raise ConfigError(f"{source}: {key} must be a scalar")


def _coerce_value(value: object, annotation: object, location: str) -> object:
    if annotation is typing.Any:
        return value

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin in (types.UnionType, Union):
        if value is None and type(None) in arguments:
            return None
        concrete = [choice for choice in arguments if choice is not type(None)]
        if len(concrete) == 1:
            return _coerce_value(value, concrete[0], location)
        failures: list[ConfigError] = []
        for choice in concrete:
            try:
                return _coerce_value(value, choice, location)
            except ConfigError as exc:
                failures.append(exc)
        expected = " or ".join(_type_name(choice) for choice in concrete)
        raise ConfigError(f"{location}: expected {expected}, got {type(value).__name__}") from (
            failures[-1] if failures else None
        )

    if origin is Literal:
        if value not in arguments:
            choices = ", ".join(repr(choice) for choice in arguments)
            raise ConfigError(f"{location}: expected one of {choices}, got {value!r}")
        return value

    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return dataclass_from_mapping(annotation, value, table=location)

    if annotation is Path:
        if not isinstance(value, str | Path):
            raise ConfigError(f"{location}: expected path string, got {type(value).__name__}")
        return Path(value).expanduser()
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{location}: expected bool, got {type(value).__name__}")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{location}: expected int, got {type(value).__name__}")
        return value
    if annotation is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"{location}: expected float, got {type(value).__name__}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{location}: expected string, got {type(value).__name__}")
        return value

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{location}: expected array, got {type(value).__name__}")
        item_type = arguments[0] if arguments else object
        return [
            _coerce_value(item, item_type, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{location}: expected array, got {type(value).__name__}")
        if not arguments:
            return tuple(value)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _coerce_value(item, arguments[0], f"{location}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ConfigError(f"{location}: expected {len(arguments)} items, got {len(value)}")
        return tuple(
            _coerce_value(item, item_type, f"{location}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, arguments, strict=True))
        )

    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    raise ConfigError(f"{location}: unsupported schema type {_type_name(annotation)}")


def _type_name(annotation: object) -> str:
    name = getattr(annotation, "__name__", None)
    return name if isinstance(name, str) else str(annotation).replace("typing.", "")


def _dataclass_fields(cls: type[object]) -> tuple[dataclasses.Field[object], ...]:
    """Reflect a class after the caller has established the dataclass contract."""
    return cast(
        tuple[dataclasses.Field[object], ...],
        dataclasses.fields(cls),  # type: ignore[arg-type]
    )


def coerce_dataclass_fields(
    cls: type[object],
    mapping: object,
    *,
    table: str,
) -> dict[str, object]:
    """Validate and coerce only fields present in ``mapping``."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"expected dataclass type, got {cls.__name__}")
    config_mapping = _config_mapping(mapping, table)
    init_fields = {field.name: field for field in _dataclass_fields(cls) if field.init}
    allowed = set(init_fields)
    for key in config_mapping:
        if key not in allowed:
            raise _unknown_key(key, allowed, table)

    hints: dict[str, object] = get_type_hints(cls)
    return {
        key: _coerce_value(value, hints[key], f"{table}.{key}")
        for key, value in config_mapping.items()
    }


def dataclass_from_mapping[DataclassT](
    cls: type[DataclassT],
    mapping: object,
    *,
    table: str,
) -> DataclassT:
    """Build one dataclass from a validated config table."""
    values = coerce_dataclass_fields(cls, mapping, table=table)
    try:
        constructor = cast(_KeywordConstructor[DataclassT], cls)
        return constructor(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{table}: {exc}") from exc


def dataclass_to_mapping(value: object, *, omit_none: bool = True) -> dict[str, object]:
    """Convert nested dataclasses to ordered config mappings."""
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"expected dataclass instance, got {type(value).__name__}")
    result: dict[str, object] = {}
    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        if item is None and omit_none:
            continue
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            result[field.name] = dataclass_to_mapping(item, omit_none=omit_none)
        elif isinstance(item, list) and item and dataclasses.is_dataclass(item[0]):
            result[field.name] = [
                dataclass_to_mapping(element, omit_none=omit_none) for element in item
            ]
        else:
            result[field.name] = item
    return result
