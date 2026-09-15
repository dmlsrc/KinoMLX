"""Inspection is generated from typed and policy declarations."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kinomlx.cli.main import main
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config.stations import StationPolicy
from kinomlx.config_surfaces import STRUCTURED_SURFACES
from kinomlx.model_registry import config_registry


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_inspection_covers_every_registered_field_without_source_resolution(model, monkeypatch):
    import kinomlx.config.registry as registry_module
    import kinomlx.settings as settings_module

    schema = config_registry().model(model)

    def forbidden(*args, **kwargs):
        pytest.fail("inspection must not resolve external input")

    monkeypatch.setattr(registry_module, "read_environment", forbidden)
    monkeypatch.setattr(settings_module, "read_environment", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    result = configuration_inventory(schema)
    assert {field["path"] for field in result["fields"]} == {
        ".".join(path) for path, _ in schema.field_items()
    }
    json.dumps(result)


def test_inspection_has_real_boolean_flags_rules_and_station_projection():
    schema = config_registry().model("krea2")
    fields = {field["path"]: field for field in configuration_inventory(schema)["fields"]}
    assert fields["settings.verbose"]["cli"]["flags"] == ["--verbose", "--no-verbose"]
    assert fields["settings.verbose"]["environment"] == ["KINO_VERBOSE"]
    assert fields["generate.prompt"]["station_policies"] == [
        {
            "station": station,
            "mode": "locked",
            "projection": "prompts",
            "reason": policy.reason,
            "irrelevant_action": None,
        }
        for station, policy in zip(
            ("denoise", "decode"),
            schema.field(("generate", "prompt")).station_policies,
            strict=True,
        )
    ]
    assert fields["generate.prompt"]["station_coverage"] == "registered"
    assert fields["output.prefix"]["station_coverage"] == "registered"
    assert fields["generate.seed"]["persistence_policy"] == "intent"
    assert any(
        rule["phase"] == "source" for rule in fields["generate.prompt_file"]["normalization"]
    )


def test_inspection_states_when_a_model_has_no_restart_station():
    inventory = configuration_inventory(config_registry().model("gmnet"))
    assert inventory["restart"]["stations"] == []
    assert "no saved restart station" in inventory["restart"]["unavailable_reason"]
    coverage = {field["station_coverage"] for field in inventory["fields"]}
    assert coverage == {"no_restart", "source_virtual"}
    assert all(field["station_policies"] == [] for field in inventory["fields"])


def test_policy_change_appears_without_editing_inventory_consumer():
    schema = config_registry().model("krea2")
    tables = tuple(
        replace(
            table,
            fields=tuple(
                replace(
                    field,
                    station_policies=(
                        StationPolicy("denoise", "locked", reason="An edited policy still locks."),
                        StationPolicy("decode", "locked", reason="An edited policy still locks."),
                    ),
                )
                if field.name == "width"
                else field
                for field in table.fields
            ),
        )
        if table.path == ("generate",)
        else table
        for table in schema.tables
    )
    changed = configuration_inventory(replace(schema, tables=tables), field="generate.width")
    assert changed["fields"][0]["station_policies"][0]["mode"] == "locked"


def test_inspect_cli_filters_and_reports_unknown_field(capsys):
    assert main(["config", "inspect", "--model", "krea2", "--field", "generate.width"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == "krea2"
    assert [field["path"] for field in result["fields"]] == ["generate.width"]
    assert main(["config", "inspect", "--model", "krea2", "--field", "generate.typo"]) != 0
    assert "unknown configuration field" in capsys.readouterr().err


def test_inspection_distinguishes_required_none_and_factory_defaults():
    schema = config_registry().model("ltx2")
    fields = {item["path"]: item for item in configuration_inventory(schema)["fields"]}
    required = fields["generate.image.path"]
    optional = fields["settings.profile_signpost_log"]
    factory = fields["settings.app_cache_dir"]
    assert required["default_kind"] == "required"
    assert required["default_toml"] is None
    assert optional["default_kind"] == "value"
    assert optional["default_toml"] is None
    assert factory["default_kind"] == "factory"
    assert factory["default_toml"] is not None


def test_default_kind_reflects_owner_without_executing_factory():
    from dataclasses import dataclass, field

    from kinomlx.config import ConfigFieldSpec, ConfigTableSpec

    calls = []

    def factory():
        calls.append(True)
        return "default"

    @dataclass(frozen=True)
    class Example:
        value: str = field(default_factory=factory)

    member = ConfigFieldSpec(
        "value",
        "Choose value",
        "text",
        "sample",
        "value",
        persistence="intent",
        surfaces=STRUCTURED_SURFACES,
    )
    table = ConfigTableSpec(("example",), Example, (member,))
    calls.clear()
    assert table.default_kind(member) == "factory"
    assert calls == []
