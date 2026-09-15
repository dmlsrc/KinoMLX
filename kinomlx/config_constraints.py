"""Pure local value constraints owned by typed configuration fields."""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import get_args


@dataclass(frozen=True)
class NumberConstraint:
    """Require a finite number within optional bounds, including a strict lower bound."""

    minimum: float | None = None
    maximum: float | None = None
    when_supplied: bool = False
    exclusive_minimum: bool = False

    def __post_init__(self) -> None:
        if type(self.exclusive_minimum) is not bool:
            raise TypeError("exclusive_minimum must be a boolean")
        if self.exclusive_minimum and self.minimum is None:
            raise ValueError("exclusive minimum requires a lower bound")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or (isinstance(bound, float) and not math.isfinite(bound))
            ):
                raise ValueError("numeric constraint bounds must be finite numbers")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("numeric constraint minimum exceeds maximum")
        if self.exclusive_minimum and self.minimum == self.maximum:
            raise ValueError("exclusive minimum leaves no admitted numeric values")

    def validate(self, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("must be finite")
        if self.exclusive_minimum and self.minimum is not None and value <= self.minimum:
            raise ValueError(f"must be greater than {self.minimum:g}")
        if self.minimum is not None and self.maximum is not None:
            if not self.minimum <= value <= self.maximum:
                raise ValueError(f"must be between {self.minimum:g} and {self.maximum:g}")
        elif self.minimum is not None and value < self.minimum:
            raise ValueError(f"must be greater than or equal to {self.minimum:g}")
        elif self.maximum is not None and value > self.maximum:
            raise ValueError(f"must be less than or equal to {self.maximum:g}")

    def describe(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": "number",
            "finite": True,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        if self.when_supplied:
            result["when_supplied"] = True
        if self.exclusive_minimum:
            result["exclusive_minimum"] = True
        return result


@dataclass(frozen=True)
class IntegerConstraint(NumberConstraint):
    """Require an integer, excluding booleans, within the owned numeric bounds."""

    def validate(self, value: object) -> None:
        if type(value) is not int:
            raise ValueError("must be an integer")
        super().validate(value)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "kind": "integer"}


@dataclass(frozen=True)
class ChoiceConstraint:
    """Own an ordered set of scalar string choices on the typed field."""

    choices: tuple[str, ...]
    when_supplied: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.choices, tuple)
            or not self.choices
            or any(not isinstance(choice, str) or not choice for choice in self.choices)
            or len(set(self.choices)) != len(self.choices)
        ):
            raise ValueError("choices must be unique nonempty strings")

    def validate(self, value: object) -> None:
        if not isinstance(value, str) or value not in self.choices:
            raise ValueError(f"must be one of: {', '.join(self.choices)}")

    def describe(self) -> dict[str, object]:
        return {
            "kind": "choices",
            "choices": list(self.choices),
            "when_supplied": self.when_supplied,
        }


@dataclass(frozen=True)
class LengthConstraint:
    """Bound the supplied length of text or a sequence."""

    minimum: int = 0
    when_supplied: bool = False
    maximum: int | None = None

    def __post_init__(self) -> None:
        if type(self.minimum) is not int or self.minimum < 0:
            raise ValueError("minimum length must be a non-negative integer")

        if self.maximum is not None and (
            type(self.maximum) is not int or self.maximum < self.minimum
        ):
            raise ValueError("maximum length must be an integer at least the minimum")

    def validate(self, value: object) -> None:
        if not isinstance(value, (str, list, tuple)):
            raise ValueError("must be text or an array")
        if len(value) < self.minimum:
            unit = "character" if isinstance(value, str) else "item"
            if self.minimum != 1:
                unit += "s"
            raise ValueError(f"must contain at least {self.minimum} {unit}")

        if self.maximum is not None and len(value) > self.maximum:
            raise ValueError(f"must contain at most {self.maximum} characters or items")

    def describe(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": "length",
            "minimum": self.minimum,
            "when_supplied": self.when_supplied,
        }
        if self.maximum is not None:
            result["maximum"] = self.maximum
        return result


@dataclass(frozen=True)
class NumberSequenceConstraint:
    """Apply a numeric rule to each sequence member, with an optional minimum length."""

    minimum_length: int = 0
    item: NumberConstraint = NumberConstraint()
    when_supplied: bool = False

    def __post_init__(self) -> None:
        LengthConstraint(self.minimum_length)
        if not isinstance(self.item, NumberConstraint):
            raise TypeError("number sequence item must be a numeric constraint")

    def validate(self, value: object) -> None:
        if not isinstance(value, (list, tuple)):
            raise ValueError("must be an array of numbers")
        LengthConstraint(self.minimum_length).validate(value)
        for index, item in enumerate(value):
            try:
                self.item.validate(item)
            except ValueError as exc:
                raise ValueError(f"item [{index}] {exc}") from exc

    def describe(self) -> dict[str, object]:
        return {
            "kind": "number_sequence",
            "minimum_length": self.minimum_length,
            "item": self.item.describe(),
            "when_supplied": self.when_supplied,
        }


@dataclass(frozen=True)
class ItemChoicesConstraint:
    """Require every sequence member to belong to an owned string vocabulary."""

    choices: frozenset[str]
    when_supplied: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.choices, frozenset) or not all(
            isinstance(choice, str) for choice in self.choices
        ):
            raise TypeError("item choices must be a frozen set of strings")

    def validate(self, value: object) -> None:
        if not isinstance(value, (list, tuple)):
            raise ValueError("must be an array of choices")
        for index, item in enumerate(value):
            if not isinstance(item, str) or item not in self.choices:
                raise ValueError(
                    f"item [{index}] must be one of {sorted(self.choices)}, got {item!r}"
                )

    def describe(self) -> dict[str, object]:
        return {
            "kind": "item_choices",
            "choices": sorted(self.choices),
            "when_supplied": self.when_supplied,
        }


@dataclass(frozen=True)
class PathSuffixConstraint:
    """Admit a path with a declared, case-insensitive file suffix without accessing it."""

    suffixes: frozenset[str]
    when_supplied: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.suffixes, frozenset)
            or not self.suffixes
            or any(
                not isinstance(suffix, str)
                or len(suffix) < 2
                or not suffix.startswith(".")
                or suffix != suffix.lower()
                or "/" in suffix
                or "\\" in suffix
                for suffix in self.suffixes
            )
        ):
            raise ValueError("path suffixes must be a nonempty frozen set of lowercase extensions")

    def validate(self, value: object) -> None:
        if not isinstance(value, Path):
            raise ValueError("must be a path")
        if value.suffix.lower() not in self.suffixes:
            raise ValueError(f"must end in {' or '.join(sorted(self.suffixes))}")

    def describe(self) -> dict[str, object]:
        return {
            "kind": "path_suffix",
            "suffixes": sorted(self.suffixes),
            "case_sensitive": False,
            "when_supplied": self.when_supplied,
        }


type LocalConstraint = (
    NumberConstraint
    | ChoiceConstraint
    | LengthConstraint
    | NumberSequenceConstraint
    | ItemChoicesConstraint
    | PathSuffixConstraint
)
_CONSTRAINT_TYPES = (
    NumberConstraint,
    ChoiceConstraint,
    LengthConstraint,
    NumberSequenceConstraint,
    ItemChoicesConstraint,
    PathSuffixConstraint,
)


def field_constraints(field: dataclasses.Field[object]) -> tuple[LocalConstraint, ...]:
    """Read the typed owner's executable metadata, rejecting unsupported declarations."""
    constraints: object = field.metadata.get("config_constraints", ())
    if not isinstance(constraints, tuple) or not all(
        isinstance(item, _CONSTRAINT_TYPES) for item in constraints
    ):
        raise TypeError(
            f"{field.name}: config_constraints must be a tuple of supported constraints"
        )
    return tuple(item for item in constraints if isinstance(item, _CONSTRAINT_TYPES))


def canonicalize_field_value(field: dataclasses.Field[object], value: object) -> object:
    """Let a field's constraints accept a second spelling before type coercion.

    A constraint that defines ``canonicalize`` (the audio onset trim accepting a
    bare millisecond count for its string field) rewrites the value into the
    form the field stores; every other value passes through unchanged. Both the
    layer admission in the registry and record construction call this, so the
    two spellings mean the same thing on every surface.
    """
    for constraint in field_constraints(field):
        canonicalize = getattr(constraint, "canonicalize", None)
        if canonicalize is not None:
            value = canonicalize(value)
    return value


def field_choices(record: type[object], name: str) -> tuple[str, ...]:
    """Read the scalar choices from their typed owner for surface bindings."""
    if not dataclasses.is_dataclass(record):
        raise TypeError("choice owner must be a dataclass")
    field = next((item for item in dataclasses.fields(record) if item.name == name), None)
    if field is None:
        raise ValueError(f"{record.__name__}: unknown choice field {name}")
    choices = [
        item.choices for item in field_constraints(field) if isinstance(item, ChoiceConstraint)
    ]
    if len(choices) != 1:
        raise ValueError(f"{record.__name__}.{name}: expected one scalar choice constraint")
    return choices[0]


def validate_choice_binding(field: dataclasses.Field[object], choices: tuple[str, ...]) -> None:
    """Reject a surface inventory that differs from the typed field's choices."""
    for constraint in field_constraints(field):
        if isinstance(constraint, ChoiceConstraint) and choices != constraint.choices:
            raise ValueError(f"{field.name}: scalar choice binding drift")


def validate_field_constraints(
    field: dataclasses.Field[object],
    annotation: object,
    value: object,
    location: str,
    *,
    supplied: bool = True,
) -> None:
    """Apply owned rules, retaining the distinction between omitted and supplied values.

    Ordinary rules respect nullable annotations. A when_supplied rule explicitly
    validates supplied values, including None, while leaving omitted defaults alone.
    """
    constraints = field_constraints(field)
    nullable = annotation is type(None) or type(None) in get_args(annotation)
    for constraint in constraints:
        if constraint.when_supplied and not supplied:
            continue
        if value is None and nullable and not constraint.when_supplied:
            continue
        try:
            constraint.validate(value)
        except ValueError as exc:
            raise ValueError(f"{location}: {exc}") from exc
