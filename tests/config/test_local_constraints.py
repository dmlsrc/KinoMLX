"""Typed field metadata supplies the same local rules to every admission boundary."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kinomlx.config import (
    ConfigError,
    ConfigFieldSpec,
    ConfigRecordSpec,
    ConfigTableSpec,
    dataclass_from_mapping,
)
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config_constraints import (
    ItemChoicesConstraint,
    LengthConstraint,
    NumberConstraint,
    NumberSequenceConstraint,
)
from kinomlx.config_records import config_record, record_intent, replace_config
from kinomlx.config_surfaces import STRUCTURED_SURFACES
from kinomlx.model_registry import config_registry
from kinomlx.models.ltx2.types import HDRReferenceConditioningConfig, ImageConditioningConfig


@pytest.mark.parametrize("record", [ImageConditioningConfig, HDRReferenceConditioningConfig])
@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_constructor_copy_and_mapping_share_owned_strength_constraints(record, strength):
    with pytest.raises(ValueError, match="strength"):
        record(Path("source"), strength=strength)
    with pytest.raises(ValueError, match="strength"):
        replace_config(record(Path("source")), strength=strength)
    with pytest.raises(ConfigError, match="strength"):
        dataclass_from_mapping(
            record, {"path": "source", "strength": strength}, table="conditioning"
        )


@pytest.mark.parametrize("path", ["generate.image.strength", "generate.hdr_reference.strength"])
@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_invalid_local_value_cannot_disappear_behind_a_later_override(path, value):
    schema = config_registry().model("ltx2")
    with pytest.raises(ConfigError, match="strength"):
        schema.parse_overrides([path + "=" + value, path + "=0.5"])


def test_image_frame_lower_bound_is_available_without_constructing_an_image():
    schema = config_registry().model("ltx2")
    with pytest.raises(ConfigError, match="frame_index"):
        schema.parse_overrides(["generate.image.frame_index=-1", "generate.image.frame_index=0"])
    with pytest.raises(ValueError, match="frame_index"):
        ImageConditioningConfig(Path("image.png"), frame_index=-1)


def test_nullable_constraint_keeps_explicit_none_distinct_from_omission():
    @config_record
    @dataclass(frozen=True)
    class OptionalValue:
        strength: float | None = field(
            default=None, metadata={"config_constraints": (NumberConstraint(0, 1),)}
        )

    assert record_intent(OptionalValue()) == {}
    assert record_intent(OptionalValue(strength=None)) == {"strength": None}
    assert record_intent(replace_config(OptionalValue(strength=0.0), strength=None)) == {
        "strength": None
    }
    assert (
        dataclass_from_mapping(OptionalValue, {"strength": None}, table="optional").strength is None
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        OptionalValue(strength=2.0)


def test_invalid_constraint_declarations_and_defaults_fail_before_invocation():
    with pytest.raises(ValueError, match="minimum exceeds"):
        NumberConstraint(1, 0)
    with pytest.raises(ValueError, match="finite"):
        NumberConstraint(maximum=float("inf"))

    @dataclass(frozen=True)
    class Malformed:
        value: float = field(default=1.0, metadata={"config_constraints": [NumberConstraint()]})

    with pytest.raises(TypeError, match="tuple of supported constraints"):
        config_record(Malformed)

    @config_record
    @dataclass(frozen=True)
    class BadDefault:
        value: float = field(
            default=-1.0, metadata={"config_constraints": (NumberConstraint(minimum=0),)}
        )

    member = ConfigFieldSpec(
        "value",
        "Set value",
        "non-negative number",
        1.0,
        "value",
        persistence="intent",
        surfaces=STRUCTURED_SURFACES,
    )
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ConfigRecordSpec(BadDefault, (member,))
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ConfigTableSpec(("example",), BadDefault, (member,))


def test_inspection_exposes_executable_constraints_from_the_typed_owner():
    schema = config_registry().model("ltx2")
    entry = configuration_inventory(schema, field="generate.image.strength")["fields"][0]
    assert entry["local_constraints"] == [
        {"kind": "number", "finite": True, "minimum": 0, "maximum": 1}
    ]


def test_integer_numbers_are_finite_without_conversion_to_float():
    NumberConstraint().validate(10**400)


@config_record
@dataclass(frozen=True)
class SuppliedControls:
    schedule: tuple[float, ...] = field(
        default=(),
        metadata={
            "config_constraints": (NumberSequenceConstraint(minimum_length=1, when_supplied=True),)
        },
    )
    window: str = field(
        default="",
        metadata={"config_constraints": (LengthConstraint(1, when_supplied=True),)},
    )
    stage: float | None = field(
        default=None,
        metadata={"config_constraints": (NumberConstraint(when_supplied=True),)},
    )
    exclude: tuple[str, ...] = field(
        default=(),
        metadata={"config_constraints": (ItemChoicesConstraint(frozenset({"a", "b"})),)},
    )


def supplied_control_fields():
    return tuple(
        ConfigFieldSpec(
            name,
            "Set " + name,
            "owned constraint",
            example,
            name,
            persistence="intent",
            surfaces=STRUCTURED_SURFACES,
        )
        for name, example in (
            ("schedule", (1.0,)),
            ("window", "1:2"),
            ("stage", 1.0),
            ("exclude", ("a",)),
        )
    )


def test_supplied_only_constraints_preserve_valid_absence_across_construction_and_copy():
    original = SuppliedControls()
    assert record_intent(original) == {}
    assert record_intent(replace_config(original, exclude=())) == {"exclude": ()}
    assert record_intent(dataclass_from_mapping(SuppliedControls, original, table="controls")) == {}
    assert record_intent(dataclass_from_mapping(SuppliedControls, {}, table="controls")) == {}
    ConfigRecordSpec(SuppliedControls, supplied_control_fields())
    ConfigTableSpec(("controls",), SuppliedControls, supplied_control_fields())


@pytest.mark.parametrize(
    "changes",
    [
        {"schedule": ()},
        {"schedule": (float("nan"),)},
        {"schedule": (True,)},
        {"window": ""},
        {"stage": None},
        {"stage": float("inf")},
        {"exclude": ("unknown",)},
    ],
)
def test_explicit_invalid_controls_fail_constructor_copy_and_mapping(changes):
    member = next(iter(changes))
    with pytest.raises(ValueError, match=member):
        SuppliedControls(**changes)
    with pytest.raises(ValueError, match=member):
        replace_config(SuppliedControls(), **changes)
    with pytest.raises(ConfigError, match=member):
        dataclass_from_mapping(SuppliedControls, changes, table="controls")


def test_finite_sequences_keep_zero_negative_order_and_repeated_categories():
    value = SuppliedControls(schedule=(1.0, 0.0, -0.5), window="1:2", stage=0.0, exclude=("a", "a"))
    assert record_intent(value) == {
        "schedule": (1.0, 0.0, -0.5),
        "window": "1:2",
        "stage": 0.0,
        "exclude": ("a", "a"),
    }
    restored = dataclass_from_mapping(SuppliedControls, value, table="controls")
    assert record_intent(restored) == record_intent(value)


def test_owned_constraint_descriptions_include_presence_and_element_rules():
    fields = supplied_control_fields()
    schema = ConfigRecordSpec(SuppliedControls, fields)
    constraint = schema.local_constraints(fields[0])[0].describe()
    assert constraint["kind"] == "number_sequence"
    assert constraint["when_supplied"] is True
    assert constraint["item"]["finite"] is True
    assert schema.local_constraints(fields[3])[0].describe()["choices"] == ["a", "b"]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LengthConstraint(-1),
        lambda: LengthConstraint(True),
        lambda: NumberSequenceConstraint(minimum_length=-1),
        lambda: ItemChoicesConstraint(["a"]),
    ],
)
def test_invalid_sequence_constraint_declarations_fail_registration(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def _counted_constraint_checks(monkeypatch) -> list[str]:
    """Record every field-constraint check both admission passes run."""
    import kinomlx.config_records as config_records
    import kinomlx.config_values as config_values

    locations: list[str] = []
    original = config_values.validate_field_constraints

    def counted(item, annotation, value, location, *, supplied=True):
        locations.append(location)
        original(item, annotation, value, location, supplied=supplied)

    monkeypatch.setattr(config_values, "validate_field_constraints", counted)
    monkeypatch.setattr(config_records, "validate_field_constraints", counted)
    return locations


def test_a_supplied_constrained_value_is_checked_once(monkeypatch):
    locations = _counted_constraint_checks(monkeypatch)

    ImageConditioningConfig(Path("source"), strength=0.5)

    # A record that declares no table names the field alone, never the class.
    assert locations.count("strength") == 1
    # An unsupplied constrained default is still checked, once.
    assert locations.count("frame_index") == 1


def test_a_value_the_record_replaces_after_construction_is_checked_again(monkeypatch):
    @config_record
    @dataclass(frozen=True)
    class Scaled:
        value: float = field(
            default=0.25, metadata={"config_constraints": (NumberConstraint(0, 1),)}
        )

        def __post_init__(self) -> None:
            object.__setattr__(self, "value", self.value * 4)

    locations = _counted_constraint_checks(monkeypatch)
    assert Scaled(0.25).value == 1.0
    assert locations.count("value") == 2

    with pytest.raises(ValueError, match="^value: "):
        Scaled(0.5)
