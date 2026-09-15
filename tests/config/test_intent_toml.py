"""TOML projects canonical inputs and preserves meaningful clearing."""

import argparse
import tomllib

import pytest

from kinomlx.cli.config_output import handle_config_output
from kinomlx.config import ConfigError
from kinomlx.config.provenance import PreparedConfig
from kinomlx.model_registry import config_registry
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--duration", "6"], {"prompt": "test", "duration": 6.0}),
        (["--frames", "145"], {"prompt": "test", "frames": 145}),
        (["--auto-duration"], {"prompt": "test", "auto_duration": True}),
    ],
)
def test_duration_toml_preserves_the_selected_input(arguments, expected):
    invocation = assemble(build_parser().parse_args(["--prompt", "test", *arguments]))
    schema = config_registry().model("ltx2")
    text = schema.dump_intent(invocation.config_intent)
    decoded = tomllib.loads(text)
    decoded["generate"]["prompt"] = decoded["generate"]["prompt"].strip()
    assert decoded["generate"] == expected
    assert schema.prepare_layer(decoded)["generate"] == invocation.config_intent.values["generate"]


def test_seed_and_output_alternatives_recreate_clear_markers():
    schema = config_registry().model("flux2")
    original = schema.prepare_layer({"generate": {"seed": 0}, "output": {"prefix": "sample"}})
    text = schema.dump_intent(PreparedConfig(original, {}))
    replay = schema.prepare_layer(tomllib.loads(text))
    assert replay["generate"] == {"seed": 0, "seeds": None}
    assert replay["output"] == {"prefix": "sample", "path": None}


@pytest.mark.parametrize(
    "values",
    [
        {"generate": {"seeds": None}},
        {"output": {"path": None}},
        {"generate": {"adapters": [{"path": "adapter.safetensors", "strength": None}]}},
    ],
)
def test_unrepresentable_clears_fail_instead_of_disappearing(values):
    with pytest.raises(ConfigError, match="TOML cannot preserve|strength: expected float"):
        config_registry().model("flux2").dump_intent(PreparedConfig(values, {}))


def test_explicit_empty_collection_and_default_strength_survive():
    schema = config_registry().model("flux2")
    for adapters in ([], [{"path": "adapter.safetensors", "strength": 1.0}]):
        values = {"generate": {"adapters": adapters}}
        assert (
            tomllib.loads(schema.dump_intent(PreparedConfig(values, {})))["generate"]
            == values["generate"]
        )


def test_unrepresentable_clear_does_not_create_a_file(tmp_path, capsys):
    destination = tmp_path / "config.toml"
    result = handle_config_output(
        argparse.Namespace(print_config=True, save_config=destination, only_non_defaults=True),
        model="flux2",
        intent=PreparedConfig({"output": {"path": None}}, {}),
        json_output=False,
    )
    assert result == 2
    assert not destination.exists()
    assert not capsys.readouterr().out


@pytest.mark.parametrize("minimal", [False, True])
def test_full_and_minimal_toml_have_identical_inputs(minimal, capsys):
    invocation = assemble(build_parser().parse_args(["--prompt", "test", "--duration", "6"]))
    options = argparse.Namespace(print_config=True, save_config=None, only_non_defaults=minimal)
    assert (
        handle_config_output(
            options, model="ltx2", intent=invocation.config_intent, json_output=False
        )
        == 0
    )
    text = capsys.readouterr().out
    generation = tomllib.loads(text)["generate"]
    assert generation["duration"] == 6.0
    assert "frames" not in generation
    assert "width" not in generation
    assert ("# Built-in defaults" in text) is not minimal
