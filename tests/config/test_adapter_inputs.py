"""Canonical adapter input admission and intent preservation before public cutover."""

from pathlib import Path

import pytest

from kinomlx.config import ConfigError, ConfigFieldSpec, ConfigRecordSpec, dataclass_from_mapping
from kinomlx.config_records import record_intent, replace_config, supplied_fields
from kinomlx.config_surfaces import CONFIG_SURFACES
from kinomlx.models.flux2.adapters.inputs import Flux2AdapterConfig
from kinomlx.models.krea2.adapters.inputs import Krea2AdapterConfig
from kinomlx.models.ltx2.adapters.inputs import LTX2AdapterConfig

RECORDS = (Krea2AdapterConfig, Flux2AdapterConfig, LTX2AdapterConfig)
SCHEDULED = (Krea2AdapterConfig, Flux2AdapterConfig)


@pytest.mark.parametrize("record", RECORDS)
def test_common_intent_preserves_omission_and_literal_paths(record):
    original = record("missing.safetensors:0.5")
    assert original.path == Path("missing.safetensors:0.5")
    assert original.strength == 1.0
    assert original.exclude == ()
    assert record_intent(original) == {"path": original.path}
    for rebuilt in (
        replace_config(original),
        dataclass_from_mapping(record, original, table="generate.adapters[1]"),
        dataclass_from_mapping(record, record_intent(original), table="generate.adapters[1]"),
    ):
        assert record_intent(rebuilt) == record_intent(original)
    explicit = replace_config(original, strength=1, exclude=[])
    assert type(explicit.strength) is float
    assert explicit.exclude == ()
    assert supplied_fields(explicit) == {"path", "strength", "exclude"}


@pytest.mark.parametrize("record", RECORDS)
@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("path", ""),
        ("path", False),
        ("strength", True),
        ("strength", float("nan")),
        ("strength", float("inf")),
        ("strength", 10**400),
        ("exclude", ["not_a_category"]),
        ("exclude", "attn"),
    ],
)
def test_common_local_errors_apply_to_constructor_copy_and_mapping(record, member, value):
    values = {"path": "adapter.safetensors", member: value}
    with pytest.raises(ValueError, match=member):
        record(**values)
    with pytest.raises(ValueError, match=member):
        replace_config(record(Path("adapter.safetensors")), **{member: value})
    with pytest.raises(ConfigError, match=member):
        dataclass_from_mapping(record, values, table="generate.adapters[1]")


@pytest.mark.parametrize("record", SCHEDULED)
def test_schedule_conflicts_are_checked_after_surviving_intent_is_resolved(record):
    scheduled = record(Path("adapter"), step_strengths=[1, 0, -0.5])
    assert scheduled.selection().schedule.per_step == (1.0, 0.0, -0.5)
    assert "strength" not in record_intent(scheduled)
    conflict = replace_config(scheduled, strength=1.0)
    with pytest.raises(ValueError, match=r"generate.adapters\[1\].*strength.*step_strengths"):
        conflict.selection(location="generate.adapters[1]")
    conflict = replace_config(scheduled, sigma_window="0.5:1")
    with pytest.raises(ValueError, match="sigma_window.*step_strengths"):
        conflict.selection()
    # An atomic replacement may remove a cross-field conflict before resolution.
    replacement = record(conflict.path, sigma_window="0.5:1", strength=-0.5)
    assert replacement.selection().sigma_window is not None


@pytest.mark.parametrize("record", SCHEDULED)
@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("step_strengths", []),
        ("step_strengths", [1, True]),
        ("step_strengths", [1, float("inf")]),
        ("sigma_window", ""),
        ("sigma_window", "none"),
        ("sigma_window", "off"),
        ("sigma_window", "0.9:0.1"),
        ("sigma_window", "0.5@inf"),
        ("sigma_window", "0.2,,0.8"),
    ],
)
def test_schedule_members_are_locally_admitted_even_without_a_complete_item(record, member, value):
    schema = ConfigRecordSpec(
        record,
        fields=tuple(
            ConfigFieldSpec(
                name,
                "Adapter control.",
                "Owned constraints.",
                "",
                name,
                persistence="intent",
                surfaces=CONFIG_SURFACES,
            )
            for name in record.__dataclass_fields__
        ),
    )
    with pytest.raises(ValueError, match=member):
        record(Path("adapter"), **{member: value})
    with pytest.raises(ConfigError, match=member):
        schema.coerce_members({member: value}, location="generate.adapters[1]")


def test_ltx_stage_fallback_and_zero_are_independent_per_record():
    first = LTX2AdapterConfig(Path("same"), strength=-0.5, stage1_strength=0)
    second = LTX2AdapterConfig(Path("same"), stage2_strength=0.25)
    assert first.selection().strength_for_stage(1) == 0.0
    assert first.selection().strength_for_stage(2) == -0.5
    assert second.selection().strength_for_stage(1) == 1.0
    assert second.selection().strength_for_stage(2) == 0.25
    assert "stage2_strength" not in record_intent(first)
    assert "strength" not in record_intent(second)


@pytest.mark.parametrize("member", ["stage1_strength", "stage2_strength"])
@pytest.mark.parametrize("value", [None, True, float("inf"), float("nan")])
def test_explicit_invalid_stage_values_are_not_omission(member, value):
    with pytest.raises(ValueError, match=member):
        LTX2AdapterConfig(Path("adapter"), **{member: value})


@pytest.mark.parametrize(
    ("record", "member", "value"),
    [
        (Krea2AdapterConfig, "stage1_strength", 0.5),
        (Flux2AdapterConfig, "stage2_strength", 0.5),
        (LTX2AdapterConfig, "step_strengths", [1, 0]),
        (LTX2AdapterConfig, "sigma_window", "0.5"),
        (Flux2AdapterConfig, "exclude", ["text_fusion"]),
        (Krea2AdapterConfig, "exclude", ["double"]),
        (LTX2AdapterConfig, "exclude", ["double"]),
    ],
)
def test_model_capabilities_reject_unsupported_fields_and_categories(record, member, value):
    with pytest.raises(ConfigError, match=member):
        dataclass_from_mapping(
            record, {"path": "adapter", member: value}, table="generate.adapters[1]"
        )


def test_sigma_resolution_keeps_closed_first_match_bands_and_strength_fallback():
    item = Krea2AdapterConfig(Path("adapter"), strength=-0.5, sigma_window="0.5:0.75@0,0.25:1")
    window = item.selection().sigma_window
    assert window.resolve(
        [1, 0.75, 0.5, 0.25, 0], strength=item.strength, label="adapter"
    ).per_step == (-0.5, 0.0, 0.0, -0.5)
    with pytest.raises(ValueError, match="selects no step"):
        window.resolve([0.1, 0], strength=item.strength, label="adapter")
