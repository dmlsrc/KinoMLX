"""Video output fields own choices, numeric limits and onset grammar."""

from dataclasses import fields
from pathlib import Path

import pytest

from kinomlx.audio.trim_mode import NAMED_TRIM_MODES, AudioOnsetTrimConstraint, parse_trim_mode
from kinomlx.config import ConfigError
from kinomlx.config_constraints import field_choices
from kinomlx.config_rules import validate_capabilities
from kinomlx.debug.metadata import SidecarPaths
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.artifacts import LTX2ArtifactConfig
from kinomlx.models.ltx2.config import assemble
from kinomlx.models.ltx2.config_schema import CONFIG_SCHEMA
from kinomlx.models.ltx2.output_config import (
    INDEPENDENT_SIDECAR_FIELDS,
    SAVE_ALL_SIDECAR_FIELDS,
    OutputConfig,
    sidecar_selection,
)


@pytest.mark.parametrize(
    "name", ["vsr_spatial_mode", "vsr_temporal_mode", "cut_detect_mode", "audio_codec"]
)
def test_output_choice_owner_drives_schema_and_api_admission(name):
    assert CONFIG_SCHEMA.field(("output", name)).choices == field_choices(OutputConfig, name)
    with pytest.raises(ValueError, match=name):
        OutputConfig(**{name: "unknown"})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("target_fps", 0),
        ("target_fps", float("nan")),
        ("cut_detect_threshold", -1),
        ("cut_detect_threshold", float("inf")),
        ("encode_quality", -0.1),
        ("encode_quality", 1.1),
    ],
)
def test_output_numeric_limits_are_constructor_admission(name, value):
    with pytest.raises(ValueError, match=name):
        OutputConfig(**{name: value})


@pytest.mark.parametrize(
    ("name", "literal", "override"),
    [
        ("audio_onset_trim", '"nan"', '"off"'),
        ("target_fps", "0", "24"),
        ("audio_codec", '"unknown"', '"alac"'),
    ],
)
def test_malformed_lower_output_value_cannot_be_hidden(tmp_path, name, literal, override):
    config = tmp_path / "config.toml"
    config.write_text(f"[output]\n{name} = {literal}\n")
    options = build_parser().parse_args(
        ["--config", str(config), "--set", f"output.{name}={override}", "--print-config"]
    )
    with pytest.raises(ConfigError, match=name):
        assemble(options)


def test_onset_constraint_uses_the_parser_mode_inventory():
    constraint = AudioOnsetTrimConstraint()
    assert constraint.describe()["named_modes"] == list(NAMED_TRIM_MODES)
    for name, expected in NAMED_TRIM_MODES.items():
        value = f" {name.upper()} "
        constraint.validate(value)
        assert parse_trim_mode(OutputConfig(audio_onset_trim=value).audio_onset_trim) == expected
    constraint.validate("80.5")
    assert parse_trim_mode("80.5") == ("force", 80.5)


def test_output_numeric_endpoints_remain_valid():
    assert OutputConfig(encode_quality=0, cut_detect_threshold=0).encode_quality == 0
    assert OutputConfig(encode_quality=1, target_fps=0.5).encode_quality == 1


@pytest.mark.parametrize("name", ["clip.mp4", "clip.MP4", "clip.Mp4"])
def test_an_exact_video_path_accepts_mp4_in_any_case(name):
    assert OutputConfig(path=Path(name)).path == Path(name)


@pytest.mark.parametrize("name", ["clip.mov", "clip.mkv", "clip"])
def test_an_exact_video_path_refuses_a_suffix_the_encoder_never_writes(name):
    with pytest.raises(ValueError, match=r"must end in \.mp4"):
        OutputConfig(path=Path(name))


def test_cli_refuses_a_non_mp4_output_path_before_generation(tmp_path):
    options = build_parser().parse_args(
        ["--prompt", "x", "--output", str(tmp_path / "clip.mov"), "--print-config"]
    )
    with pytest.raises(ConfigError, match=r"\[output\]\.path: must end in \.mp4"):
        assemble(options)


def test_save_all_selects_every_unset_member_of_its_group():
    selection = sidecar_selection(OutputConfig(save_all_sidecars=True))
    assert (
        selection.run_log,
        selection.console_log,
        selection.effective_config,
        selection.audio_sidecar,
    ) == (True, True, True, True)


def test_an_explicit_opt_out_survives_save_all():
    selection = sidecar_selection(
        OutputConfig(save_all_sidecars=True, save_console_log=False, save_audio_sidecar=False)
    )
    assert selection.console_log is False
    assert selection.audio_sidecar is False
    assert selection.run_log is True


def test_an_excluded_frame_sequence_needs_its_own_request():
    assert sidecar_selection(OutputConfig(save_all_sidecars=True)).vae_frames is False
    assert sidecar_selection(OutputConfig(save_all_sidecars=True)).hdr_heic_frames is False
    requested = sidecar_selection(OutputConfig(save_vae_frames=True, save_hdr_heic_frames=True))
    assert (requested.vae_frames, requested.hdr_heic_frames) == (True, True)


def test_an_api_caller_gets_the_execution_sidecars_save_all_names(tmp_path):
    config = OutputConfig(path=tmp_path / "clip.mp4", save_all_sidecars=True)
    paths = SidecarPaths.for_output(config.path)
    assert sorted(paths.selected_execution_paths(config)) == [
        "effective_config",
        "execution_log",
        "run_log",
    ]


def test_the_resolver_and_the_schema_declare_the_same_save_all_split():
    implication = next(
        item
        for item in CONFIG_SCHEMA.implications
        if item.trigger == ("output", "save_all_sidecars")
    )
    implied = {path[-1] for path, _value in implication.implied if path[0] == "output"}
    assert implied - {"vsr_save_original"} == set(SAVE_ALL_SIDECAR_FIELDS)
    assert {path[-1] for path in implication.excluded} == set(INDEPENDENT_SIDECAR_FIELDS)


def test_hdr_still_refuses_an_original_video_request_outside_save_all():
    with pytest.raises(ValueError, match="vsr_save_original"):
        validate_capabilities(
            OutputConfig(vsr_save_original=True), {"hdr": True, "fps": 24.0}, location="[output]"
        )
    validate_capabilities(
        OutputConfig(save_all_sidecars=True), {"hdr": True, "fps": 24.0}, location="[output]"
    )


def test_a_bare_millisecond_count_is_the_string_the_field_stores(tmp_path):
    options = build_parser().parse_args(
        ["--prompt", "scene", "--set", "output.audio_onset_trim=120", "--print-config"]
    )
    assert assemble(options).output.audio_onset_trim == "120"
    config = tmp_path / "trim.toml"
    config.write_text('[generate]\nprompt = "scene"\n[output]\naudio_onset_trim = 120\n')
    assert (
        assemble(
            build_parser().parse_args(["--config", str(config), "--print-config"])
        ).output.audio_onset_trim
        == "120"
    )
    assert OutputConfig(audio_onset_trim=120).audio_onset_trim == "120"
    assert OutputConfig(audio_onset_trim=120.0).audio_onset_trim == "120"
    assert OutputConfig(audio_onset_trim=12.5).audio_onset_trim == "12.5"
    assert parse_trim_mode(OutputConfig(audio_onset_trim=0).audio_onset_trim) == ("off", 0.0)


@pytest.mark.parametrize(
    ("value", "reason"),
    [(-5, "finite and non-negative"), (True, "expected string, got bool")],
)
def test_a_number_the_grammar_refuses_is_named_by_the_grammar(value, reason):
    with pytest.raises(ValueError, match=reason):
        OutputConfig(audio_onset_trim=value)


def test_the_schema_declares_the_artifacts_that_inherit_save_all():
    implication = next(
        item
        for item in CONFIG_SCHEMA.implications
        if item.trigger == ("output", "save_all_sidecars")
    )
    artifact_toggles = {
        field.name for field in fields(LTX2ArtifactConfig) if field.name.startswith("save_")
    }
    assert {path for path, _ in implication.implied}.isdisjoint(implication.inherited)
    assert {path[-1] for path in implication.inherited} == artifact_toggles
    assert all(path[0] == "model_artifacts" for path in implication.inherited)
    assert (
        "Also enables while unset: Save stage-1 and final latents; "
        "Save encoded media conditions; Save prompt tensors and their mask."
    ) in CONFIG_SCHEMA.implication_help(("output", "save_all_sidecars"))
