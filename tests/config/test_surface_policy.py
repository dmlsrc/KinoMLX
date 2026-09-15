"""Required surface policies expose and enforce deliberate transport exclusions."""

import argparse
from dataclasses import replace

import pytest

from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.config import ConfigFieldSpec
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config_surfaces import CONFIG_SURFACES, STRUCTURED_SURFACES, SURFACES, SurfacePolicy
from kinomlx.model_registry import config_registry, model_choices
from tests.config.test_bindings import sensor_schema


@pytest.mark.parametrize("model", model_choices())
def test_every_field_has_complete_surface_classification(model):
    schema = config_registry().model(model)
    for field in configuration_inventory(schema)["fields"]:
        surfaces = field["surface_classification"]
        assert set(surfaces) == set(SURFACES)
        for classification in surfaces.values():
            assert classification["status"] in {"supported", "unavailable"}
            if classification["status"] == "unavailable":
                assert classification["reason"]
        if field["virtual"]:
            assert surfaces["api"]["status"] == "unavailable"
        else:
            assert surfaces["api"]["status"] == "supported"


def test_field_requires_an_explicit_surface_policy():
    with pytest.raises(TypeError, match="surfaces"):
        ConfigFieldSpec("count", "Sample count", "integer", 2, "count", persistence="intent")


@pytest.mark.parametrize(
    "exclusions", [(("cli", ""),), (("unknown", "reason"),), (("cli", "one"), ("cli", "two"))]
)
def test_invalid_or_duplicate_exclusions_fail(exclusions):
    with pytest.raises(ValueError, match="surface"):
        SurfacePolicy(exclusions)


def test_missing_binding_requires_an_explicit_exclusion():
    table = sensor_schema().tables[0]
    fields = (replace(table.fields[0], cli=None, surfaces=CONFIG_SURFACES), *table.fields[1:])
    with pytest.raises(ValueError, match="missing CLI binding or explicit exclusion"):
        replace(table, fields=fields)


def test_excluded_cli_field_retains_structured_inputs_and_environment(monkeypatch):
    schema = sensor_schema()
    table = schema.tables[0]
    field = replace(table.fields[0], cli=None, surfaces=STRUCTURED_SURFACES)
    schema = replace(schema, tables=(replace(table, fields=(field, *table.fields[1:])),))
    parser = argparse.ArgumentParser(add_help=False)
    add_config_arguments(parser, schema)
    assert all(action.dest != "count" for action in parser._actions)
    assert "count" not in schema.cli_destinations()
    assert schema.cli_layer(argparse.Namespace(count=7)) == {}
    schema.validate_layer({"capture": {"count": 7}})
    monkeypatch.setenv("KINO_SENSOR_COUNT", "7")
    assert schema.environment_config()["capture"]["count"] == 7


def test_excluded_environment_field_rejects_its_unregistered_derived_name(monkeypatch):
    schema = sensor_schema()
    table = schema.tables[0]
    policy = SurfacePolicy((("environment", "Choose this per invocation."),))
    field = replace(table.fields[0], surfaces=policy)
    schema = replace(schema, tables=(replace(table, fields=(field, *table.fields[1:])),))
    monkeypatch.setenv("KINO_SENSOR_COUNT", "bad-integer")
    assert ("capture", "count") not in schema.environment_variables()
    with pytest.raises(ValueError, match="unknown field 'KINO_SENSOR_COUNT'"):
        schema.environment_config()


def test_binding_cannot_coexist_with_cli_exclusion():
    table = sensor_schema().tables[0]
    field = replace(table.fields[0], surfaces=STRUCTURED_SURFACES)
    with pytest.raises(ValueError, match="excluded CLI surface has a binding"):
        replace(table, fields=(field, *table.fields[1:]))


def test_settings_cannot_disagree_with_its_owned_environment_surface():
    table = config_registry().model("krea2").table(("settings",))
    field = replace(table.fields[0], surfaces=SurfacePolicy((("environment", "Unavailable"),)))
    with pytest.raises(ValueError, match="Settings members retain their owned ENV surface"):
        replace(table, fields=(field, *table.fields[1:]))


def test_collection_member_cannot_claim_to_exclude_a_parent_transport():
    schema = config_registry().model("flux2")
    table = schema.table(("generate",))
    field = schema.field(("generate", "adapters"))
    member_schema = field.item_schema
    member = replace(member_schema.fields[0], surfaces=STRUCTURED_SURFACES)
    members = replace(member_schema, fields=(member, *member_schema.fields[1:]))
    field = replace(field, item_schema=members)
    with pytest.raises(ValueError, match="member excludes the collection's cli surface"):
        replace(
            table,
            fields=tuple(
                field if original.name == "adapters" else original for original in table.fields
            ),
        )
