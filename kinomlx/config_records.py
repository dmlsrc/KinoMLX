"""Typed construction and supplied presence for frozen configuration records."""

import dataclasses
import inspect
from functools import wraps
from typing import cast, get_type_hints

from kinomlx.config_constraints import field_constraints, validate_field_constraints
from kinomlx.config_record_state import (
    _CONSTRUCTOR,
    _PRESENCE,
    _Constructor,
    _Initializer,
    construct_record,
    derive_config,
    record_intent,
    replace_config,
    supplied_fields,
    tracks_presence,
)
from kinomlx.config_rules import record_rules
from kinomlx.config_values import coerce_record_fields


def config_record[RecordT](record: type[RecordT]) -> type[RecordT]:
    """Track supplied arguments without changing dataclass fields or its public signature.

    Apply after @dataclass(frozen=True). Normal defaults and post-init validation
    remain owned by that dataclass. Slotted records and InitVar/custom initializer
    shapes are deliberately unsupported by this initial mechanism.
    """
    if not dataclasses.is_dataclass(record) or not getattr(
        getattr(record, "__dataclass_params__", None), "frozen", False
    ):
        raise TypeError("config_record requires a frozen dataclass")
    if "__slots__" in record.__dict__:
        raise TypeError("config_record requires an instance dictionary")
    original = cast(_Initializer, record.__init__)
    signature = inspect.signature(original)
    parameters = tuple(signature.parameters)
    fields = frozenset(item.name for item in dataclasses.fields(record) if item.init)
    if not parameters or set(parameters[1:]) != fields:
        raise TypeError("config_record requires the dataclass field initializer")
    if _CONSTRUCTOR in record.__dict__:
        raise TypeError("configuration record is already decorated")
    record_rules(record)
    constrained = tuple(item for item in dataclasses.fields(record) if field_constraints(item))
    # A record that declares the TOML table it is written at reports a violation
    # there; one that does not names the field alone. Never the Python class.
    table: str | None = getattr(record, "config_table", None)
    # Resolved on the first construction rather than here: a record may annotate
    # a name this module defines further down, and get_type_hints walks the
    # module namespace. One walk per record then serves every instance.
    resolved: list[dict[str, object]] = []

    def annotations() -> dict[str, object]:
        if not resolved:
            resolved.append(get_type_hints(record))
        return resolved[0]

    @wraps(original)
    def validated(instance: object, *args: object, **kwargs: object) -> None:
        presence = supplied_fields(instance)
        if presence is None:
            raise TypeError(f"{type(instance).__name__} requires its own config_record declaration")
        bound = signature.bind(instance, *args, **kwargs)
        values = {name: value for name, value in bound.arguments.items() if name != parameters[0]}
        hints = annotations()
        coerced = coerce_record_fields(record, values, presence=presence, table=table, hints=hints)
        original(instance, **coerced)
        for item in constrained:
            value = getattr(instance, item.name)
            # A supplied value was already checked above. Re-check it only when
            # __post_init__ replaced it, which is the one case the first pass
            # could not have seen.
            if item.name in coerced and value is coerced[item.name]:
                continue
            validate_field_constraints(
                item,
                hints[item.name],
                value,
                item.name if table is None else f"{table}.{item.name}",
                supplied=item.name in presence,
            )

    setattr(record, _CONSTRUCTOR, _Constructor(validated, fields))

    @wraps(original)
    def initialize(instance: object, *args: object, **kwargs: object) -> None:
        bound = signature.bind(instance, *args, **kwargs)
        presence = frozenset(bound.arguments) - {parameters[0]}
        object.__setattr__(instance, _PRESENCE, presence)
        validated(instance, *args, **kwargs)

    setattr(record, "__init__", initialize)  # noqa: B010 - install the typed constructor wrapper
    return record


__all__ = [
    "config_record",
    "construct_record",
    "derive_config",
    "record_intent",
    "replace_config",
    "supplied_fields",
    "tracks_presence",
]
