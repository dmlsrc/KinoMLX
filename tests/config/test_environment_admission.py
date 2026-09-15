"""Only current schema bindings are accepted in a model's ENV namespace."""

import argparse
from dataclasses import dataclass, replace

import pytest

from kinomlx.config import CliBinding, ConfigError, ConfigFieldSpec, ConfigTableSpec
from kinomlx.config.assembly import collect_invocation_sources
from kinomlx.config_records import config_record
from kinomlx.config_surfaces import CONFIG_SURFACES
from kinomlx.model_registry import config_registry


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
@pytest.mark.parametrize("value", ["", "none", "[]"])
def test_unknown_model_environment_is_rejected_before_cli_precedence(model, value, monkeypatch):
    variable = f"KINO_{model.upper()}_NO_SUCH_FIELD"
    monkeypatch.setenv(variable, value)
    options = argparse.Namespace(config=None, set_overrides=["settings.verbose=false"])
    with pytest.raises(ConfigError, match=f"unknown field '{variable}'"):
        collect_invocation_sources(config_registry().model(model), options)


def test_current_model_namespace_is_checked_without_interpreting_other_namespaces(monkeypatch):
    monkeypatch.setenv("KINO_KREA2_NO_SUCH_FIELD", "bad")
    monkeypatch.setenv("KINO_WORKSPACE_VARIABLE", "workspace")
    config_registry().model("flux2").environment_state()


def test_unknown_environment_fails_before_prompt_file_is_opened(monkeypatch, tmp_path):
    monkeypatch.setenv("KINO_FLUX2_WIDHT", "512")
    options = argparse.Namespace(config=None, set_overrides=[], prompt_file=tmp_path / "missing")
    with pytest.raises(ConfigError, match="did you mean 'KINO_FLUX2_WIDTH'"):
        collect_invocation_sources(config_registry().model("flux2"), options)


def test_removed_adapter_field_has_no_special_interface_or_translation(monkeypatch):
    monkeypatch.setenv("KINO_KREA2_ADAPTER_PATHS", '["old.safetensors"]')
    schema = config_registry().model("krea2")
    with pytest.raises(ConfigError, match="unknown field 'KINO_KREA2_ADAPTER_PATHS'"):
        schema.environment_state()
    assert "KINO_KREA2_ADAPTER_PATHS" not in {
        name for variables in schema.environment_variables().values() for name in variables
    }


def test_registering_a_new_field_automatically_admits_its_environment(monkeypatch):
    @config_record
    @dataclass(frozen=True)
    class Sensor:
        gain: float = 1.0

    schema = config_registry().model("gmnet")
    extension = ConfigTableSpec(
        path=("sensor",),
        record=Sensor,
        cli_generated=True,
        fields=(
            ConfigFieldSpec(
                "gain",
                "Set sensor gain",
                "a number",
                0.5,
                "sensor_gain",
                persistence="intent",
                cli=CliBinding("Sensor"),
                surfaces=CONFIG_SURFACES,
            ),
        ),
    )
    monkeypatch.setenv("KINO_GMNET_GAIN", "0.5")
    with pytest.raises(ConfigError, match="unknown field"):
        schema.environment_state()
    extended = replace(schema, tables=(*schema.tables, extension))
    assert extended.environment_state().values["sensor"] == {"gain": 0.5}


def test_a_new_field_cannot_silently_claim_another_public_environment_name():
    @config_record
    @dataclass(frozen=True)
    class Sensor:
        variant: str = "hdr"

    schema = config_registry().model("gmnet")
    extension = ConfigTableSpec(
        path=("sensor",),
        record=Sensor,
        cli_generated=True,
        fields=(
            ConfigFieldSpec(
                "variant",
                "Set the sensor variant",
                "a declared name",
                "hdr",
                "sensor_variant",
                persistence="intent",
                cli=CliBinding("Sensor"),
                surfaces=CONFIG_SURFACES,
            ),
        ),
    )
    with pytest.raises(ValueError, match="KINO_GMNET_VARIANT is claimed by"):
        replace(schema, tables=(*schema.tables, extension))
