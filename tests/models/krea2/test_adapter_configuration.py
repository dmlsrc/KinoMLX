"""Canonical user adapter records compose with Krea's independently enabled profiles."""

from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_records import record_intent
from kinomlx.models.krea2 import Krea2AdapterConfig, Krea2Request
from kinomlx.models.krea2.adapters import sda_profile, turbo_profile
from kinomlx.models.krea2.adapters.profiles import resolve_profile_adapters
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.config_schema import CONFIG_SCHEMA


def invocation(*arguments):
    return assemble(build_parser().parse_args(["--prompt", "a painted bowl", *arguments]))


def test_user_records_preserve_independent_modes_through_api_and_saved_configuration(tmp_path):
    request = Krea2Request(
        prompt="a painted bowl",
        adapters=(
            Krea2AdapterConfig(Path("same"), strength=-0.5),
            Krea2AdapterConfig(Path("same"), step_strengths=(1, 0)),
        ),
    )
    first, second = request.resolved_adapters()
    assert first.strength == -0.5
    assert second.schedule.per_step == (1.0, 0.0)
    original = invocation(
        "--steps",
        "2",
        "--adapter",
        "same.safetensors:0.5",
        "--adapter",
        "same.safetensors",
        "--adapter-step-strengths",
        "1,0",
    )
    config = tmp_path / "saved.toml"
    config.write_text(CONFIG_SCHEMA.dump_config(original.resolved_config))
    reloaded = assemble(build_parser().parse_args(["--config", str(config)]))
    assert reloaded.request.resolved_adapters() == original.request.resolved_adapters()
    assert "strength" not in record_intent(reloaded.request.adapters[1])
    assert "sigma_window" not in record_intent(reloaded.request.adapters[1])


@pytest.mark.parametrize(
    "retired",
    [
        "adapter_paths",
        "adapter_strengths",
        "adapter_exclusions",
        "adapter_step_strengths",
        "adapter_sigma_windows",
    ],
)
def test_retired_fields_are_rejected_before_a_higher_cli_collection_can_hide_them(
    retired, tmp_path
):
    config = tmp_path / "input.toml"
    config.write_text(f"[generate]\n{retired}=[]\n")
    with pytest.raises(ConfigError, match=retired):
        invocation("--config", str(config), "--adapter", "replacement")
    with pytest.raises(TypeError, match=retired):
        Krea2Request(prompt="scene", **{retired: ()})


@pytest.mark.parametrize("clear", [False, True])
def test_user_collection_clear_does_not_disable_profiles_or_change_prepend_order(
    clear, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        turbo_profile, "resolve_cached_adapter", lambda *args, **kwargs: (tmp_path / "four", {})
    )
    monkeypatch.setattr(
        sda_profile, "resolve_cached_adapter", lambda *args, **kwargs: (tmp_path / "sda", {})
    )
    arguments = ["--turbo-4step", "--turbo-sda", "--adapter", "user.safetensors:0.5"]
    if clear:
        arguments += ["--set", "generate.adapters=[]"]
    request = invocation(*arguments).request
    selections, four, sda = resolve_profile_adapters(
        tmp_path,
        four_step=request.four_step_profile(),
        sda=request.sda_profile(),
        additional=request.resolved_adapters(),
    )
    assert [item.path.name for item in selections] == (
        ["four", "sda"] if clear else ["four", "sda", "user.safetensors"]
    )
    assert four["profile"] == "turbo-4step"
    assert sda["profile"] == "turbo-sda"
    assert selections[1].sigma_window is not None


def test_cli_collection_replaces_file_and_env_before_indexed_leaf_patch(tmp_path, monkeypatch):
    monkeypatch.setenv("KINO_KREA2_ADAPTERS", '[{path="environment",strength=0.1}]')
    config = tmp_path / "input.toml"
    config.write_text('[generate]\nadapters=[{path="file",strength=0.2}]\n')
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
    assert [record_intent(item) for item in result.request.adapters] == [
        {"path": Path("first")},
        {"path": Path("second"), "strength": 0.25},
    ]
