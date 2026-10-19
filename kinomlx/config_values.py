"""Small dependency-free schema helpers for dataclass-backed config."""

import dataclasses
import difflib
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, Union, cast, get_args, get_origin, get_type_hints

from kinomlx._config_record_state import (
    construct_record,
    record_intent,
    supplied_fields,
    tracks_presence,
)
from kinomlx.config_constraints import canonicalize_field_value, validate_field_constraints
from kinomlx.config_transport import TRANSPORT_SCALARS, enum_vocabulary
from kinomlx.errors import KinoMLXError


class ConfigError(KinoMLXError, ValueError):
    """A user-authored configuration error safe to render without traceback."""


class _KeywordConstructor[ResultT](Protocol):
    def __call__(self, **kwargs: object) -> ResultT: ...


def unknown_field_error(name: str, allowed: set[str], location: str | None) -> ConfigError:
    known = sorted(allowed)
    suggestion = difflib.get_close_matches(name, known, n=1)
    hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
    prefix = "" if location is None else f"{location}: "
    return ConfigError(f"{prefix}unknown field {name!r}{hint} (known: {', '.join(known)})")


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
            raise unknown_field_error(key, allowed, source)
        if key in tables and not isinstance(value, Mapping):
            raise ConfigError(f"{source}: [{key}] must be a table")
        if key in scalars and isinstance(value, Mapping):
            raise ConfigError(f"{source}: {key} must be a scalar")


def coerce_config_value(value: object, annotation: object, location: str) -> object:
    if annotation is typing.Any:
        return value
    if isinstance(annotation, typing.TypeAliasType):
        return coerce_config_value(value, annotation.__value__, location)

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin in (types.UnionType, Union):
        if value is None and type(None) in arguments:
            return None
        concrete = [choice for choice in arguments if choice is not type(None)]
        if len(concrete) == 1:
            return coerce_config_value(value, concrete[0], location)
        failures: list[ConfigError] = []
        for choice in concrete:
            try:
                return coerce_config_value(value, choice, location)
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
        if value == "":
            raise ConfigError(f"{location}: path must not be empty")
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
        try:
            return float(value)
        except OverflowError as exc:
            raise ConfigError(f"{location}: number is too large for a float") from exc
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{location}: expected string, got {type(value).__name__}")
        return value

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{location}: expected array, got {type(value).__name__}")
        item_type = arguments[0] if arguments else object
        return [
            coerce_config_value(item, item_type, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is Sequence:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ConfigError(f"{location}: expected sequence, got {type(value).__name__}")
        item_type = arguments[0] if arguments else object
        return tuple(
            coerce_config_value(item, item_type, f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{location}: expected array, got {type(value).__name__}")
        if not arguments:
            return tuple(value)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                coerce_config_value(item, arguments[0], f"{location}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ConfigError(f"{location}: expected {len(arguments)} items, got {len(value)}")
        return tuple(
            coerce_config_value(item, item_type, f"{location}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, arguments, strict=True))
        )

    vocabulary = enum_vocabulary(annotation)
    if vocabulary is not None:
        if not isinstance(value, str):
            raise ConfigError(f"{location}: expected string, got {type(value).__name__}")
        try:
            return vocabulary(value)
        except ValueError as exc:
            allowed = ", ".join(repr(item.value) for item in vocabulary)
            raise ConfigError(f"{location}: expected one of {allowed}, got {value!r}") from exc
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
    init_fields = {field.name: field for field in _dataclass_fields(cls) if field.init}
    presence = supplied_fields(mapping)
    if isinstance(mapping, cls):
        # Typed inputs still pass the same field coercion and constructor checks.
        mapping = {name: getattr(mapping, name) for name in init_fields}
    config_mapping = _config_mapping(mapping, table)
    return coerce_record_fields(
        cls,
        config_mapping,
        presence=frozenset(config_mapping) if presence is None else presence,
        table=table,
    )


def coerce_record_fields(
    cls: type[object],
    config_mapping: Mapping[str, object],
    *,
    presence: frozenset[str],
    table: str | None,
    hints: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Coerce provided values using required supplied-field context.

    ``table`` is the TOML table the values were written at, or ``None`` for a
    record constructed directly, whose messages then name the field alone.

    A caller that already resolved this record's annotations passes them in;
    ``get_type_hints`` walks the module namespace on every call, and a tracked
    record constructor would otherwise repeat that walk for every instance.
    """
    init_fields = {field.name: field for field in _dataclass_fields(cls) if field.init}
    allowed = set(init_fields)
    for key in config_mapping:
        if key not in allowed:
            raise unknown_field_error(key, allowed, table)

    if hints is None:
        hints = get_type_hints(cls)
    values = {}
    for key, value in config_mapping.items():
        location = key if table is None else f"{table}.{key}"
        value = canonicalize_field_value(init_fields[key], value)
        values[key] = coerce_config_value(value, hints[key], location)
        try:
            validate_field_constraints(
                init_fields[key],
                hints[key],
                values[key],
                location,
                supplied=key in presence,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    return values


def dataclass_from_mapping[DataclassT](
    cls: type[DataclassT],
    mapping: object,
    *,
    table: str,
) -> DataclassT:
    """Build one dataclass from a validated config table."""
    values = coerce_dataclass_fields(cls, mapping, table=table)
    missing = [
        field.name
        for field in _dataclass_fields(cls)
        if field.init
        and field.name not in values
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    ]
    if missing:
        # The constructor would name a Python argument; the reader needs the
        # configuration path they have to write.
        raise ConfigError("; ".join(f"{table}.{name} is required" for name in missing))
    try:
        presence = supplied_fields(mapping)
        if isinstance(mapping, cls) and tracks_presence(cls):
            if presence is None:
                raise TypeError("typed configuration input has unknown constructor presence")
            return construct_record(cls, values, presence & frozenset(values))
        constructor = cast(_KeywordConstructor[DataclassT], cls)
        return constructor(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{table}: {exc}") from exc


def _record_value(
    value: object, *, omit_none: bool, preserve_nested_intent: bool, location: str
) -> object:
    """Project records recursively while preserving scalar and sequence transport shapes."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if preserve_nested_intent and tracks_presence(type(value)):
            return _record_value(
                {
                    key: item
                    for key, item in record_intent(value).items()
                    if item is not None or not omit_none
                },
                omit_none=omit_none,
                preserve_nested_intent=True,
                location=location,
            )
        return dataclass_to_mapping(
            value,
            omit_none=omit_none,
            preserve_nested_intent=preserve_nested_intent,
            location=location,
        )
    if isinstance(value, list):
        return [
            _record_value(
                item,
                omit_none=omit_none,
                preserve_nested_intent=preserve_nested_intent,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _record_value(
                item,
                omit_none=omit_none,
                preserve_nested_intent=preserve_nested_intent,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return {
            key: _record_value(
                item,
                omit_none=omit_none,
                preserve_nested_intent=preserve_nested_intent,
                location=f"{location}.{key}",
            )
            for key, item in value.items()
        }
    if value is None or isinstance(value, TRANSPORT_SCALARS):
        return value
    raise ConfigError(f"{location}: {type(value).__name__} is not a configuration transport type")


def dataclass_to_mapping(
    value: object,
    *,
    omit_none: bool = True,
    preserve_nested_intent: bool = False,
    location: str = "",
) -> dict[str, object]:
    """Project a record's values, optionally retaining intent inside nested typed inputs.

    The root remains a resolved value projection. A reloadable configuration can
    retain nested input presence so omitted controls do not become explicit defaults.
    Ordinary observation snapshots keep the full projection by default.
    """
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"expected dataclass instance, got {type(value).__name__}")
    result: dict[str, object] = {}
    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        if item is None and omit_none:
            continue
        result[field.name] = _record_value(
            item,
            omit_none=omit_none,
            preserve_nested_intent=preserve_nested_intent,
            location=f"{location}.{field.name}" if location else field.name,
        )
    return result
