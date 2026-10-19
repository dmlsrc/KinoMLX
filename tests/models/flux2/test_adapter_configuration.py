"""Canonical adapter records through the public Klein invocation and restart boundaries."""

import json
import tomllib
from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_records import record_intent
from kinomlx.models.flux2 import Flux2AdapterConfig, Flux2Request, command
from kinomlx.models.flux2.arguments import build_parser
from kinomlx.models.flux2.config import assemble
from kinomlx.models.flux2.config_schema import CONFIG_SCHEMA
from tests.models.flux2.test_cli import _fake_generation


def invocation(*arguments):
    return assemble(build_parser().parse_args(["--prompt", "a painted bowl", *arguments]))


def test_public_api_accepts_ordered_typed_records_with_independent_strength_modes():
    request = Flux2Request(
        adapters=(
            Flux2AdapterConfig(Path("same"), strength=-0.5),
            Flux2AdapterConfig(Path("same"), step_strengths=(1, 0)),
        ),
        steps=2,
    )
    request.validate("distilled")
    first, second = request.resolved_adapters()
    assert first.strength == -0.5
    assert second.schedule.per_step == (1.0, 0.0)
    assert "strength" not in record_intent(request.adapters[1])


@pytest.mark.parametrize("flag", ["--adapter", "--lora"])
def test_public_cli_shorthand_and_local_modifiers_preserve_omission(flag):
    result = invocation(flag, "a.safetensors:0.8", flag, "b.safetensors")
    assert result.resolved_config["generate"]["adapters"] == (
        {"path": Path("a.safetensors"), "strength": 0.8},
        {"path": Path("b.safetensors")},
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--adapter-strength", "0.5"],
        ["--adapter", "a.safetensors:0.5", "--lora-strength", "0.5"],
        ["--adapter", "a.safetensors:0.5:0.8"],
    ],
)
def test_public_cli_rejects_orphans_duplicates_and_unsupported_shorthand(arguments):
    with pytest.raises(SystemExit):
        invocation(*arguments)


def test_cli_replaces_lower_collection_and_indexed_override_patches_only_one_item(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KINO_FLUX2_ADAPTERS", '[{path="environment",strength=0.2}]')
    config = tmp_path / "config.toml"
    config.write_text('[generate]\nadapters=[{path="file",strength=0.3}]\n')
    result = invocation(
        "--config",
        str(config),
        "--adapter",
        "first",
        "--adapter",
        "second",
        "--set",
        "generate.adapters[1].strength=0.25",
    )
    assert result.resolved_config["generate"]["adapters"] == (
        {"path": Path("first")},
        {"path": Path("second"), "strength": 0.25},
    )
    cleared = invocation("--config", str(config), "--set", "generate.adapters=[]")
    assert cleared.generate.adapters == ()


def test_structured_paths_are_literal_and_do_not_run_cli_shorthand(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[generate]\nadapters=[{path="style.safetensors:0.5"}]\n')
    result = invocation("--config", str(config))
    assert result.generate.adapters[0].path == Path("style.safetensors:0.5")
    assert result.generate.adapters[0].strength == 1.0


@pytest.mark.parametrize(
    "field",
    [
        "adapter_paths",
        "adapter_strengths",
        "adapter_exclusions",
        "adapter_step_strengths",
        "adapter_sigma_windows",
    ],
)
def test_retired_array_inputs_fail_even_when_a_higher_layer_replaces_adapters(field, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(f"[generate]\n{field}=[]\n")
    with pytest.raises(ConfigError, match=field):
        invocation("--config", str(config), "--adapter", "new")
    with pytest.raises(TypeError, match=field):
        Flux2Request(**{field: ()})


def test_printed_and_saved_configuration_reloads_a_schedule_without_added_controls(tmp_path):
    original = invocation("--steps", "2", "--adapter", "a", "--adapter-step-strengths", "1,0")
    document = CONFIG_SCHEMA.dump_config(original.resolved_config)
    assert tomllib.loads(document)["generate"]["adapters"] == [
        {"path": "a", "step_strengths": [1.0, 0.0]}
    ]
    config = tmp_path / "saved.toml"
    config.write_text(document)
    reloaded = assemble(build_parser().parse_args(["--config", str(config)]))
    assert reloaded.generate.resolved_adapters() == original.generate.resolved_adapters()
    assert record_intent(reloaded.generate.adapters[0]) == record_intent(
        original.generate.adapters[0]
    )


def test_final_step_count_is_checked_before_any_adapter_weight_load():
    with pytest.raises(ValueError, match="resolved recipe runs 3 steps"):
        invocation(
            "--steps",
            "2",
            "--adapter",
            "a",
            "--adapter-step-strengths",
            "1,0",
            "--set",
            "generate.steps=3",
        )


def test_saved_adapter_index_edits_obey_denoise_and_decode_stage_policies(tmp_path, monkeypatch):
    _fake_generation(monkeypatch)
    assert (
        command.run_flux2_command(
            [
                "--prompt",
                "a painted bowl",
                "--width",
                "64",
                "--height",
                "64",
                "--save-all-sidecars",
                "--output",
                str(tmp_path / "first.png"),
                "--quiet",
            ]
        )
        == 0
    )
    run = tmp_path / "first_run.json"
    payload = json.loads(run.read_text())
    payload["configuration"]["values"]["generate"]["adapters"] = [
        {"path": "same.safetensors", "strength": 0.5},
        {"path": "same.safetensors"},
    ]
    run.write_text(json.dumps(payload))
    arguments = ["--restart", str(run), "--set", "generate.adapters[1].strength=0.25"]
    resumed = assemble(build_parser().parse_args([*arguments, "--restart-from", "denoise"]))
    assert [item.strength for item in resumed.generate.resolved_adapters()] == [0.5, 0.25]
    assert record_intent(resumed.generate.adapters[0]) == {
        "path": Path("same.safetensors"),
        "strength": 0.5,
    }
    with pytest.raises(ConfigError, match="adapters.*earlier stage"):
        assemble(build_parser().parse_args([*arguments, "--restart-from", "decode"]))
