"""GMNet output admission and final selection share owned field policies."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_constraints import PathSuffixConstraint
from kinomlx.config_records import config_record, replace_config
from kinomlx.config_rules import validate_record
from kinomlx.models.gmnet.arguments import build_parser
from kinomlx.models.gmnet.config import assemble
from kinomlx.models.gmnet.output import plan_gmnet_output
from kinomlx.models.gmnet.types import GMNetOutputConfig, GMNetRequest


@config_record
@dataclass(frozen=True)
class TextArtifact:
    path: Path = field(
        metadata={"config_constraints": (PathSuffixConstraint(frozenset({".txt"})),)}
    )


def test_neutral_suffix_owner_coerces_paths_without_requiring_existing_files():
    assert TextArtifact("missing.TXT").path == Path("missing.TXT")
    with pytest.raises(ValueError, match="path.*must end in"):
        TextArtifact("missing.csv")


@pytest.mark.parametrize(
    "suffixes",
    [
        frozenset(),
        {".txt"},
        frozenset({"txt"}),
        frozenset({".TXT"}),
        frozenset({1}),
        frozenset({".a/b"}),
    ],
)
def test_invalid_suffix_declarations_fail_registration(suffixes):
    with pytest.raises(ValueError, match="suffixes"):
        PathSuffixConstraint(suffixes)


@pytest.mark.parametrize(
    ("values", "repair", "message"),
    [
        (
            {"path": "output.exr", "exr": False},
            {"exr": True},
            "an exact .exr path requires EXR output",
        ),
        (
            {"path": "output.heic", "heic": False},
            {"heic": True},
            "an exact .heic path requires HEIC output",
        ),
        (
            {"exr": False, "heic": False},
            {"save_gain_map": True},
            "nothing to write: EXR, HEIC, and gain-map outputs are disabled",
        ),
    ],
)
def test_coupled_output_selection_can_be_repaired_after_local_admission(values, repair, message):
    config = GMNetOutputConfig(**values)
    with pytest.raises(ValueError, match=rf"\[output\]: {re.escape(message)}"):
        validate_record(config, location="[output]")
    repaired = replace_config(config, **repair)
    validate_record(repaired, location="[output]")
    plan = plan_gmnet_output(GMNetRequest(Path("input.png")), repaired, base=Path("result"))
    assert (
        plan.artifacts.exr is not None
        or plan.artifacts.heic is not None
        or plan.artifacts.gain_map is not None
    )


def test_higher_layer_repairs_output_selection(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[output]\npath = "result.exr"\nexr = false\n')
    options = build_parser().parse_args(
        ["--config", str(config), "--set", "output.exr=true", "--print-config"]
    )
    assert assemble(options).output.exr is True


def test_malformed_lower_suffix_cannot_be_hidden_by_cli(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[output]\npath = "result.png"\n')
    options = build_parser().parse_args(
        ["--config", str(config), "--set", 'output.path="result.exr"', "--print-config"]
    )
    with pytest.raises(ConfigError, match="path.*must end in"):
        assemble(options)


def test_print_config_rejects_surviving_empty_output_plan():
    options = build_parser().parse_args(
        ["--set", "output.exr=false", "--set", "output.heic=false", "--print-config"]
    )
    with pytest.raises(ConfigError, match="nothing to write"):
        assemble(options)


def test_explicit_gain_map_off_beats_save_all_in_final_output_policy():
    config = GMNetOutputConfig(exr=False, heic=False, save_gain_map=False, save_all_sidecars=True)
    with pytest.raises(ValueError, match="nothing to write"):
        validate_record(config, location="[output]")
    validate_record(replace_config(config, save_gain_map=None), location="[output]")
