"""Automatic model namespaces, typed values, and complete invocation precedence."""

import tomllib
from dataclasses import dataclass

import pytest

from kinomlx.cli.main import main
from kinomlx.config import ConfigError, ConfigFieldSpec, ConfigTableSpec
from kinomlx.config_surfaces import STRUCTURED_SURFACES
from kinomlx.model_registry import config_registry


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2", "gmnet"])
def test_every_registered_table_field_has_a_model_owned_binding(model):
    schema = config_registry().model(model)
    bindings = schema.environment_variables()
    for table in schema.tables:
        for field in table.fields:
            names = bindings[(*table.path, field.name)]
            if table.path != ("settings",):
                assert names == (
                    f"KINO_{model.upper()}_{field.cli_dest.upper()}",
                    *(("KINO_OUTPUT_DIR",) if field.cli_dest == "output_dir" else ()),
                )
    assert bindings[("settings", "hf_home")] == ("HF_HOME",)


def test_new_plain_record_gets_environment_binding_without_metadata(monkeypatch):
    @dataclass
    class Request:
        count: int = 1

    table = ConfigTableSpec(
        path=("generate",),
        record=Request,
        fields=(
            ConfigFieldSpec(
                "count",
                "Set the count",
                "an integer",
                3,
                "count",
                persistence="intent",
                surfaces=STRUCTURED_SURFACES,
            ),
        ),
    )
    monkeypatch.setenv("KINO_EXAMPLE_COUNT", "7")
    assert table.environment_overrides(model="example") == {"count": 7}


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2"])
def test_environment_file_cli_and_set_resolve_in_order(model, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(f"KINO_{model.upper()}_PROMPT", "A painted landscape.")
    monkeypatch.setenv(f"KINO_{model.upper()}_SEED", "10")
    monkeypatch.setenv(f"KINO_{model.upper()}_OUTPUT_DIR", str(tmp_path))
    config = tmp_path / "run.toml"
    config.write_text("[generate]\nseed = 11\n")
    base = ["--model", model, "--print-config"]
    for args, expected in (
        ([], 10),
        (["--config", str(config)], 11),
        (["--config", str(config), "--seed", "12"], 12),
        (["--config", str(config), "--seed", "12", "--set", "generate.seed=13"], 13),
    ):
        assert main([*base, *args]) == 0
        parsed = tomllib.loads(capsys.readouterr().out)
        assert parsed["generate"]["seed"] == expected
        assert parsed["output"]["directory"] == str(tmp_path)


def test_environment_values_preserve_text_empty_strings_and_structured_adapters(monkeypatch):
    schema = config_registry().model("krea2")
    names = schema.environment_variables()
    values = {
        "system_prompt": "",
        "prompt": '{"subject": "a cat", "count": 2}',
        "adapters": '[{path="first.safetensors",strength=0.25},{path="second.safetensors",strength=0.5}]',
    }
    for field, value in values.items():
        monkeypatch.setenv(names[("generate", field)][0], value)
    result = schema.environment_config()["generate"]
    assert result["system_prompt"] == ""
    assert result["prompt"] == values["prompt"]
    assert result["adapters"] == (
        {"path": "first.safetensors", "strength": 0.25},
        {"path": "second.safetensors", "strength": 0.5},
    )
    monkeypatch.setenv(names[("generate", "prompt")][0], "123")
    assert schema.environment_config()["generate"]["prompt"] == "123"


def test_model_namespaces_isolate_shared_registration_names(monkeypatch):
    monkeypatch.setenv("KINO_LTX2_SEED", "3")
    monkeypatch.setenv("KINO_KREA2_SEED", "5")
    monkeypatch.setenv("KINO_FLUX2_SEED", "7")
    for model, expected in (("ltx2", 3), ("krea2", 5), ("flux2", 7)):
        assert config_registry().model(model).environment_config()["generate"]["seed"] == expected


def test_bad_environment_value_reports_model_binding(monkeypatch):
    monkeypatch.setenv("KINO_FLUX2_WIDTH", "wide")
    with pytest.raises(ConfigError, match="KINO_FLUX2_WIDTH"):
        config_registry().model("flux2").environment_config()


def test_environment_virtual_controls_follow_normal_layer_processing(monkeypatch, tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A painted landscape.")
    monkeypatch.setenv("KINO_LTX2_PROMPT_FILE", str(prompt))
    monkeypatch.setenv("KINO_LTX2_AUTO_DURATION", "true")
    assert main(["--model", "ltx2", "--model-generation", "2.5", "--print-config"]) == 0
    result = tomllib.loads(capsys.readouterr().out)
    assert result["generate"]["prompt"].rstrip("\n") == "A painted landscape."
    assert "frames" not in result["generate"]
    assert "prompt_file" not in result["generate"]


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2", "gmnet"])
def test_output_directory_shared_fallback(model, monkeypatch):
    schema = config_registry().model(model)
    variable = f"KINO_{model.upper()}_OUTPUT_DIR"
    monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("KINO_OUTPUT_DIR", raising=False)
    assert "directory" not in schema.environment_config().get("output", {})
    monkeypatch.setenv("KINO_OUTPUT_DIR", "shared-output")
    assert str(schema.environment_config()["output"]["directory"]) == "shared-output"
    monkeypatch.setenv(variable, "model-output")
    assert str(schema.environment_config()["output"]["directory"]) == "model-output"


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2", "gmnet"])
def test_empty_output_path_env_uses_fallback(model, monkeypatch, tmp_path):
    schema = config_registry().model(model)
    monkeypatch.setenv(f"KINO_{model.upper()}_OUTPUT_DIR", "")
    monkeypatch.setenv("KINO_OUTPUT_DIR", str(tmp_path))
    assert schema.environment_config()["output"]["directory"] == tmp_path
    monkeypatch.setenv("KINO_OUTPUT_DIR", "")
    assert "directory" not in schema.environment_config().get("output", {})


def test_empty_model_path_is_unset_in_registry_and_standalone_settings(monkeypatch):
    from kinomlx.models.ltx2.settings import LTX2Settings

    monkeypatch.setenv("KINO_LTX2_WEIGHTS_PATH", "")
    assert LTX2Settings.from_env().weights_path is None
    assert "weights_path" not in config_registry().model("ltx2").environment_config().get(
        "model_settings", {}
    )


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2", "gmnet"])
def test_invalid_shared_env_is_rendered_as_json(model, monkeypatch, capsys):
    import json

    monkeypatch.setenv("KINO_VERBOSE", "enabled")
    assert main(["--model", model, "--json", "--print-config"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert "verbose" in str(result)


def test_environment_error_has_one_location(monkeypatch):
    monkeypatch.setenv("KINO_KREA2_TURBO_4STEP", "maybe")
    with pytest.raises(ConfigError) as caught:
        config_registry().model("krea2").environment_config()
    assert str(caught.value).count("environment") == 1


def test_ltx_retention_cli_and_environment_parity(monkeypatch, capsys):
    monkeypatch.setenv("KINO_LTX2_TRANSFORMER_RETAIN", '["small", "audio-attention"]')
    args = ["--model", "ltx2", "--stream-transformer", "--print-config"]
    assert main(args) == 0
    env = tomllib.loads(capsys.readouterr().out)
    assert env["model_settings"]["transformer_retain"] == ["small", "audio-attention"]
    monkeypatch.delenv("KINO_LTX2_TRANSFORMER_RETAIN")
    assert main([*args, "--transformer-retain", "small,audio-attention"]) == 0
    cli = tomllib.loads(capsys.readouterr().out)
    assert (
        cli["model_settings"]["transformer_retain"] == env["model_settings"]["transformer_retain"]
    )
