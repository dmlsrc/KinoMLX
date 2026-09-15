"""LTX record roundtrips and effective completed-stage restart contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

from kinomlx.config import ConfigError, dataclass_from_mapping
from kinomlx.config.assembly import collect_invocation_sources
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config_records import record_intent
from kinomlx.models.ltx2 import DistilledRequest, LTX2AdapterConfig
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble, effective_generate_config
from kinomlx.models.ltx2.config_schema import CONFIG_SCHEMA


def test_public_stage_shorthand_roundtrips_without_manufacturing_fallbacks(tmp_path):
    original = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "scene",
                "--adapter",
                "first.safetensors:0:0.5",
                "--adapter",
                "second.safetensors:0.8",
                "--adapter",
                "third.safetensors",
            ]
        )
    )
    config = tmp_path / "saved.toml"
    config.write_text(CONFIG_SCHEMA.dump_config(original.resolved_config))
    reloaded = assemble(build_parser().parse_args(["--config", str(config)]))
    assert [record_intent(item) for item in reloaded.request.adapters] == [
        {"path": Path("first.safetensors"), "stage1_strength": 0.0, "stage2_strength": 0.5},
        {"path": Path("second.safetensors"), "strength": 0.8},
        {"path": Path("third.safetensors")},
    ]
    assert reloaded.request.resolved_adapters() == original.request.resolved_adapters()


def test_per_seed_projection_matches_the_assembled_sidecar_table():
    """The command re-projects each seed's request; that must not manufacture defaults."""
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "scene",
                "--duration",
                "2",
                "--seed",
                "42",
                "--adapter",
                "first.safetensors:0.25:0.5",
                "--adapter",
                "second.safetensors",
                "--adapter-exclude",
                "audio",
            ]
        )
    )
    per_seed = effective_generate_config(replace(invocation.request, seed=42, seeds=None))
    assert per_seed == invocation.resolved_config["generate"]
    assert per_seed["adapters"] == (
        {"path": Path("first.safetensors"), "stage1_strength": 0.25, "stage2_strength": 0.5},
        {"path": Path("second.safetensors"), "exclude": ("audio",)},
    )
    assert "duration" not in per_seed
    assert per_seed["frames"] == invocation.request.frames
    automatic = effective_generate_config(DistilledRequest(prompt="scene", frames=None))
    assert automatic["auto_duration"] is True


@pytest.mark.parametrize(
    ("arguments", "stage2_locked", "decode_locked"),
    [
        (["generate.adapters[0].stage2_strength=0.25"], False, True),
        (["generate.adapters[0].stage1_strength=0.5"], False, False),
        (["generate.adapters[0].strength=0.25"], True, True),
        (['generate.adapters[0]={path="old",strength=0.25,stage1_strength=0.5}'], False, True),
        (['generate.adapters[0]={path="other",strength=0.5}'], True, True),
        (['generate.adapters=[{path="old",strength=0.25}]'], True, True),
        (["generate.adapters=[]"], True, True),
    ],
)
def test_indexed_and_whole_replacements_cannot_bypass_completed_stage_locks(
    arguments, stage2_locked, decode_locked
):
    base = {"generate": {"adapters": [{"path": "old", "strength": 0.5}]}}
    tokens = [token for assignment in arguments for token in ("--set", assignment)]
    inputs = collect_invocation_sources(CONFIG_SCHEMA, build_parser().parse_args(tokens)).prepare(
        PreparedConfig(base, {})
    )
    current = dataclass_from_mapping(DistilledRequest, inputs.merge()["generate"], table="generate")
    parent = dataclass_from_mapping(DistilledRequest, base["generate"], table="generate")
    expected = (("generate", "adapters"),)
    assert CONFIG_SCHEMA.locked_changes(("generate",), current, parent, station="stage-2") == (
        expected if stage2_locked else ()
    )
    assert CONFIG_SCHEMA.locked_changes(("generate",), current, parent, station="decode") == (
        expected if decode_locked else ()
    )


def test_stage2_only_record_does_not_change_the_completed_first_stage():
    parent = DistilledRequest(adapters=(LTX2AdapterConfig(Path("first"), strength=0.5),))
    current = DistilledRequest(
        adapters=(
            *parent.adapters,
            LTX2AdapterConfig(Path("second"), stage1_strength=0, stage2_strength=-0.5),
        )
    )
    assert CONFIG_SCHEMA.locked_changes(("generate",), current, parent, station="stage-2") == ()
    assert CONFIG_SCHEMA.locked_changes(("generate",), current, parent, station="decode") == (
        ("generate", "adapters"),
    )


@pytest.mark.parametrize(
    "retired",
    [
        "adapter_paths",
        "adapter_strengths",
        "adapter_stage1_strengths",
        "adapter_stage2_strengths",
        "adapter_exclusions",
    ],
)
def test_retired_stage_arrays_fail_in_structured_and_python_inputs(retired, tmp_path):
    config = tmp_path / "input.toml"
    config.write_text(f"[generate]\n{retired}=[]\n")
    with pytest.raises(ConfigError, match=retired):
        assemble(build_parser().parse_args(["--config", str(config), "--adapter", "replacement"]))
    with pytest.raises(TypeError, match=retired):
        DistilledRequest(**{retired: ()})
