"""Supplied-argument presence for ordinary frozen configuration dataclasses."""

import dataclasses
from collections.abc import Mapping
from typing import Protocol, cast

_CONSTRUCTOR = "__kinomlx_config_constructor__"
_PRESENCE = "__kinomlx_config_presence__"


class _Initializer(Protocol):
    def __call__(self, instance: object, *args: object, **kwargs: object) -> None: ...


@dataclasses.dataclass(frozen=True)
class _Constructor:
    initialize: _Initializer
    fields: frozenset[str]


def _constructor(record: type[object]) -> _Constructor:
    declaration = record.__dict__.get(_CONSTRUCTOR)
    if not isinstance(declaration, _Constructor):
        raise TypeError(f"{record.__name__} is not a presence-tracked configuration record")
    return declaration


def tracks_presence(record: type[object]) -> bool:
    """Whether this exact record declaration owns a tracked initializer."""
    return isinstance(record.__dict__.get(_CONSTRUCTOR), _Constructor)


def supplied_fields(value: object) -> frozenset[str] | None:
    """Return known constructor presence, or None for an ordinary untracked record."""
    if not tracks_presence(type(value)):
        return None
    presence = getattr(value, _PRESENCE, None)
    return cast(frozenset[str], presence) if isinstance(presence, frozenset) else None


def construct_record[RecordT](
    record: type[RecordT], values: Mapping[str, object], presence: frozenset[str]
) -> RecordT:
    """Construct already coerced values with known presence before post-init validation."""
    declaration = _constructor(record)
    if not presence <= values.keys() or not values.keys() <= declaration.fields:
        raise ValueError("configuration construction contains unknown fields")
    instance = record.__new__(record)
    object.__setattr__(instance, _PRESENCE, presence)
    declaration.initialize(instance, **values)
    return instance


def _copy_config[RecordT](
    value: RecordT, changes: Mapping[str, object], *, supplied: bool
) -> RecordT:
    declaration = _constructor(type(value))
    presence = supplied_fields(value)
    if presence is None:
        raise TypeError("configuration record has no supplied-field metadata")
    unknown = changes.keys() - declaration.fields
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    values = {name: getattr(value, name) for name in declaration.fields}
    values.update(changes)
    return construct_record(
        type(value), values, presence | frozenset(changes) if supplied else presence
    )


def replace_config[RecordT](value: RecordT, **changes: object) -> RecordT:
    """Copy a tracked record, preserving presence and marking only edited fields."""
    return _copy_config(value, changes, supplied=True)


def derive_config[RecordT](value: RecordT, **changes: object) -> RecordT:
    """Resolve effective values without inventing additional supplied fields.

    This keeps presence for subsequent validation and projection. Changed values
    are observations; the original source record remains the input-intent authority.
    """
    return _copy_config(value, changes, supplied=False)


def _intent_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return record_intent(value)
    if isinstance(value, tuple):
        return tuple(_intent_value(item) for item in value)
    if isinstance(value, list):
        return [_intent_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _intent_value(item) for key, item in value.items()}
    return value


def record_intent(value: object) -> dict[str, object]:
    """Project supplied values recursively, never guessing presence from defaults."""
    declaration = _constructor(type(value))
    presence = supplied_fields(value)
    if presence is None or not dataclasses.is_dataclass(value):
        raise TypeError("configuration record has no supplied-field metadata")
    return {
        item.name: _intent_value(getattr(value, item.name))
        for item in dataclasses.fields(value)
        if item.name in declaration.fields and item.name in presence
    }
