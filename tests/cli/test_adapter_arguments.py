"""Canonical adapter behavior at each public model CLI boundary."""

import importlib
from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.models.ltx2.adapters.selection import lora_profile_for_stage


def assemble(model, *args):
    arguments = importlib.import_module(f"kinomlx.models.{model}.arguments")
    config = importlib.import_module(f"kinomlx.models.{model}.config")
    return config.assemble(arguments.build_parser().parse_args(["--prompt", "A landscape.", *args]))


def effective(invocation):
    return invocation.resolved_config["generate"]


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2"])
@pytest.mark.parametrize("flag", ["--adapter", "--lora"])
def test_common_shorthand_and_default_presence_match_across_models(model, flag):
    generation = effective(
        assemble(model, flag, "style.safetensors:0.8", flag, "detail.safetensors")
    )
    assert generation["adapters"] == (
        {"path": Path("style.safetensors"), "strength": 0.8},
        {"path": Path("detail.safetensors")},
    )


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2"])
@pytest.mark.parametrize("suffix", ["nan", "inf", "1e999", "0.1:0.2:0.3"])
def test_invalid_numeric_shorthand_fails_at_the_public_cli(model, suffix):
    with pytest.raises(SystemExit):
        assemble(model, "--adapter", "style.safetensors:" + suffix)


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2"])
def test_duplicate_strength_cannot_override_shorthand_on_the_same_item(model):
    with pytest.raises(SystemExit):
        assemble(model, "--adapter", "style.safetensors:0.8", "--adapter-strength", "0.8")


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2"])
def test_cli_collection_replaces_the_lower_collection_without_broadcasting(model, tmp_path):
    config = tmp_path / "input.toml"
    config.write_text('[generate]\nadapters=[{path="old.safetensors",strength=0.2}]\n')
    generation = effective(
        assemble(model, "--config", str(config), "--adapter", "new.safetensors:0.8")
    )
    assert generation["adapters"] == ({"path": Path("new.safetensors"), "strength": 0.8},)


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2"])
def test_structured_colon_paths_stay_literal_and_old_arrays_are_rejected(model, tmp_path):
    config = tmp_path / "input.toml"
    config.write_text('[generate]\nadapters=[{path="style.safetensors:0.5"}]\n')
    assert effective(assemble(model, "--config", str(config)))["adapters"] == (
        {"path": Path("style.safetensors:0.5")},
    )
    config.write_text("[generate]\nadapter_paths=[]\n")
    with pytest.raises(ConfigError, match="adapter_paths"):
        assemble(model, "--config", str(config), "--adapter", "replacement")


def test_ltx_stage_strengths_filter_zero_before_loading():
    invocation = assemble(
        "ltx2",
        "--adapter",
        "first.safetensors:0.25:0.5",
        "--adapter",
        "second.safetensors:0:0.8",
        "--adapter",
        "third.safetensors:0.7:0",
        "--adapter",
        "both.safetensors:0.3",
        "--adapter",
        "neither.safetensors:0:0",
    )
    selections = invocation.request.resolved_adapters()
    first = lora_profile_for_stage(selections, 1)
    second = lora_profile_for_stage(selections, 2)
    assert [(p.path.name, p.strength) for p in first] == [
        ("first.safetensors", 0.25),
        ("third.safetensors", 0.7),
        ("both.safetensors", 0.3),
    ]
    assert [(p.path.name, p.strength) for p in second] == [
        ("first.safetensors", 0.5),
        ("second.safetensors", 0.8),
        ("both.safetensors", 0.3),
    ]
