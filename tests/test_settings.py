"""Behavioral tests for ``kinomlx.settings``.

Covers the env-resolution + CLI-verbatim contracts that are easy to
regress when adding fields or adjusting the resolver.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path

import pytest

from kinomlx.cli.config_records import OutputConfig
from kinomlx.models.gmnet.types import GMNetOutputConfig
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import (
    EnvironmentSettings,
    Settings,
    _resolve_env_entry,
    add_argparse_args,
    add_settings_argparse_args,
    settings_from_args,
)

# Env vars that the resolver / Settings.from_env can read.  We wipe them
# all at the start of each test so behavior is deterministic regardless
# of the shell environment pytest was launched from.
_ENV_KEYS = [
    "HF_HOME",
    "KINO_CACHE_DIR",
    "KINO_CACHE_MODE",
    "KINO_DURATION_HEAD_PATH",
    "KINO_FAST_MODE",
    "KINO_GEMMA_PATH",
    "KINO_JSON",
    "KINO_MLX_CACHE_LIMIT_GB",
    "KINO_OUTPUT_DIR",
    "KINO_PROFILE_SIGNPOST_LOG",
    "KINO_PROFILE_SIGNPOSTS",
    "KINO_QUIET",
    "KINO_TEXT_ENCODER_PATH",
    "KINO_TEMPORAL_LATENT_UPSCALER_PATH",
    "KINO_TRANSFORMER_PATH",
    "KINO_TRANSFORMER_COMPILE_GROUP_SIZE",
    "KINO_TRANSFORMER_CACHE_QUANTIZE",
    "KINO_TRANSFORMER_DTYPE",
    "KINO_STEEL_ATTENTION",
    "KINO_STEEL_ATTENTION_D64",
    "KINO_STEEL_ATTENTION_PROBE",
    "KINO_STREAM_TRANSFORMER",
    "KINO_COMPILE_ATTENTION",
    "KINO_TRANSFORMER_RESIDENT_BLOCKS",
    "KINO_COMPILE_BLOCK_GROUPS",
    "KINO_VIDEO_FF_LAYOUT_SPECS",
    "KINO_VIDEO_FF_LAYOUT_LAYERS",
    "KINO_VIDEO_ATTN_LAYOUT_SPECS",
    "KINO_VIDEO_ATTN_LAYOUT_LAYERS",
    "KINO_AUDIO_LAYOUT_MIRROR",
    "KINO_AUDIO_VAE_PATH",
    "KINO_AUDIO_FF_LAYOUT_SPECS",
    "KINO_AUDIO_FF_LAYOUT_LAYERS",
    "KINO_AUDIO_ATTN_LAYOUT_SPECS",
    "KINO_AUDIO_ATTN_LAYOUT_LAYERS",
    "KINO_ADALN_PRETRANSPOSE",
    "KINO_VIDEO_FF_QUANTIZE_SPECS",
    "KINO_VIDEO_FF_QUANTIZE_LAYERS",
    "KINO_VIDEO_FF_QUANTIZE_GROUP_SIZE",
    "KINO_VIDEO_FF_QUANTIZE_BITS",
    "KINO_VIDEO_VAE_PATH",
    "KINO_SPATIAL_UPSCALER_PATH",
    "KINO_VERBOSE",
    "KINO_WEIGHTS_PATH",
    "KINOMLX_TEST_ENV_A",
    "KINOMLX_TEST_ENV_B",
    "KINOMLX_TEST_ENV_MISSING",
]


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    """Wipe Settings-relevant env vars; restore after the test."""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# _resolve_env_entry
# ---------------------------------------------------------------------------


def test_bare_string_is_literal_not_env_lookup() -> None:
    """A string with no ``{{}}`` resolves to itself, never to the env."""
    os.environ["KINOMLX_TEST_ENV_A"] = "/from/env"
    # A bare name is literal and must not pick up its environment value.
    assert _resolve_env_entry("KINOMLX_TEST_ENV_A") == "KINOMLX_TEST_ENV_A"
    assert _resolve_env_entry("/literal/path") == "/literal/path"
    assert _resolve_env_entry("KINO_CACHE_DIR") == "KINO_CACHE_DIR"


def test_template_substitution_resolves_from_env() -> None:
    os.environ["KINOMLX_TEST_ENV_A"] = "/foo/val"
    os.environ["KINOMLX_TEST_ENV_B"] = "barv"
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_A}}") == "/foo/val"
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_A}}/sub") == "/foo/val/sub"
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_A}}-{{KINOMLX_TEST_ENV_B}}") == "/foo/val-barv"


def test_template_returns_none_when_any_var_unset() -> None:
    os.environ["KINOMLX_TEST_ENV_A"] = "/foo/val"
    # KINOMLX_TEST_ENV_MISSING is unset, so the template is unresolvable.
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_MISSING}}") is None
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_A}}/{{KINOMLX_TEST_ENV_MISSING}}") is None


def test_template_returns_none_when_var_is_empty() -> None:
    os.environ["KINOMLX_TEST_ENV_A"] = ""
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_A}}") is None
    assert _resolve_env_entry("{{KINOMLX_TEST_ENV_A}}/sub") is None


def test_environment_metadata_rejects_a_malformed_variable_template() -> None:
    @dataclass(frozen=True)
    class _MalformedSettings(EnvironmentSettings):
        value: str = field(default="", metadata={"env": "{{BAD-NAME}}"})

    with pytest.raises(TypeError, match="malformed environment template"):
        _MalformedSettings.environment_sources_for_field("value")


# ---------------------------------------------------------------------------
# Settings.from_env behavior matrix
# ---------------------------------------------------------------------------


def test_from_env_no_env_uses_field_defaults() -> None:
    infrastructure = Settings.from_env()
    model = LTX2Settings.from_env()
    assert infrastructure.cache_dir == Path("~/.cache/kinomlx").expanduser()
    assert infrastructure.profile_signposts is False
    assert infrastructure.profile_signpost_log is None
    assert model.fast_mode is True
    assert model.transformer_dtype == "bfloat16"
    assert model.steel_attention is True
    assert model.steel_attention_d64 is True
    assert model.steel_attention_probe is False
    assert model.compile_attention is True
    assert model.weights_path is None


def test_from_env_cache_dir_ignores_hf_home() -> None:
    os.environ["HF_HOME"] = "/tmp/hfroot"
    s = Settings.from_env()
    assert s.cache_dir == Path("~/.cache/kinomlx").expanduser()
    assert s.hf_home == Path("/tmp/hfroot")


def test_from_env_cache_dir_uses_kino_var_independently_of_hf_home() -> None:
    os.environ["KINO_CACHE_DIR"] = "/tmp/explicit"
    os.environ["HF_HOME"] = "/tmp/hfroot"
    s = Settings.from_env()
    assert s.cache_dir == Path("/tmp/explicit")
    assert s.hf_home == Path("/tmp/hfroot")


def test_output_directory_environment_is_product_specific(monkeypatch) -> None:
    monkeypatch.setenv("KINO_OUTPUT_DIR", "/tmp/kinomlx")
    expected = {"directory": Path("/tmp/kinomlx")}
    assert OutputConfig.overrides_from_env_fields("directory") == expected
    assert GMNetOutputConfig.overrides_from_env_fields("directory") == expected
    monkeypatch.delenv("KINO_OUTPUT_DIR")
    assert OutputConfig.overrides_from_env_fields("directory") == {}
    assert GMNetOutputConfig.overrides_from_env_fields("directory") == {}


def test_from_env_bare_settings_reads_no_env() -> None:
    """Direct infrastructure and model records do not touch env."""
    os.environ["KINO_CACHE_DIR"] = "/should/be/ignored"
    os.environ["HF_HOME"] = "/should/also/be/ignored"
    os.environ["KINO_FAST_MODE"] = "0"
    assert Settings().cache_dir == Path("~/.cache/kinomlx").expanduser()
    assert LTX2Settings().fast_mode is True


def test_runtime_light_layout_default_matches_cache_policy() -> None:
    from kinomlx.models.ltx2.cache.policy import DEFAULT_VIDEO_FF_LAYOUT_SPECS

    expected = tuple(f"{target}:{layout}" for target, layout in DEFAULT_VIDEO_FF_LAYOUT_SPECS)
    assert LTX2Settings().video_ff_layout_specs == expected


def test_from_env_parses_typed_values() -> None:
    os.environ["KINO_LTX_GENERATION"] = "2.5"
    os.environ["KINO_VIDEO_VAE"] = "diffusion"
    os.environ["KINO_FAST_MODE"] = "0"
    os.environ["KINO_TRANSFORMER_RESIDENT_BLOCKS"] = "4"
    os.environ["KINO_TRANSFORMER_DTYPE"] = "float16"
    os.environ["KINO_STEEL_ATTENTION"] = "0"
    os.environ["KINO_STEEL_ATTENTION_D64"] = "0"
    os.environ["KINO_STEEL_ATTENTION_PROBE"] = "1"
    os.environ["KINO_COMPILE_ATTENTION"] = "0"
    os.environ["KINO_MLX_CACHE_LIMIT_GB"] = "2.5"
    os.environ["KINO_VIDEO_FF_LAYOUT_SPECS"] = "project_in:pretranspose,project_out:pretranspose"
    os.environ["KINO_VIDEO_FF_LAYOUT_LAYERS"] = "0,7,47"
    infrastructure = Settings.from_env()
    model = LTX2Settings.from_env()
    assert model.model_generation == "2.5"
    assert model.video_vae == "diffusion"
    assert model.fast_mode is False
    assert model.transformer_resident_blocks == 4
    assert model.transformer_dtype == "float16"
    assert model.steel_attention is False
    assert model.steel_attention_d64 is False
    assert model.steel_attention_probe is True
    assert model.compile_attention is False
    assert infrastructure.mlx_cache_limit_gb == pytest.approx(2.5)
    assert model.video_ff_layout_specs == (
        "project_in:pretranspose",
        "project_out:pretranspose",
    )
    assert model.video_ff_layout_layers == (0, 7, 47)


def test_from_env_parses_generation_neutral_component_paths() -> None:
    values = {
        "KINO_TRANSFORMER_PATH": "/models/transformer.safetensors",
        "KINO_TEXT_ENCODER_PATH": "/models/text.safetensors",
        "KINO_VIDEO_VAE_PATH": "/models/video-vae.safetensors",
        "KINO_AUDIO_VAE_PATH": "/models/audio-vae.safetensors",
        "KINO_SPATIAL_UPSCALER_PATH": "/models/spatial.safetensors",
        "KINO_TEMPORAL_LATENT_UPSCALER_PATH": "/models/temporal.safetensors",
        "KINO_DURATION_HEAD_PATH": "/models/duration.safetensors",
    }
    os.environ.update(values)

    settings = LTX2Settings.from_env()

    assert settings.uses_split_checkpoint
    for environment_name, value in values.items():
        field_name = environment_name.removeprefix("KINO_").lower()
        assert getattr(settings, field_name) == Path(value)


def test_pack_root_and_transformer_override_are_valid_config_intent() -> None:
    settings = LTX2Settings(
        weights_path=Path("ltx25-pack"),
        transformer_path=Path("transformer.safetensors"),
    )

    settings.validate()

    assert settings.uses_split_checkpoint


def test_gemma_and_split_text_encoder_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        LTX2Settings(
            gemma_path=Path("gemma"),
            text_encoder_path=Path("text.safetensors"),
        ).validate()


def test_video_vae_override_does_not_select_split_pack() -> None:
    override = Path("video-vae.safetensors")
    settings = LTX2Settings(
        weights_path=Path("monolith.safetensors"),
        video_vae_path=override,
    )

    settings.validate()

    assert not settings.uses_split_checkpoint
    assert not LTX2Settings(video_vae_path=override).uses_split_checkpoint


def test_monolithic_primary_accepts_independent_component_overrides() -> None:
    settings = LTX2Settings(
        model_generation="2.5",
        weights_path=Path("monolith.safetensors"),
        text_encoder_path=Path("text.safetensors"),
        video_vae_path=Path("video.safetensors"),
        audio_vae_path=Path("audio.safetensors"),
        temporal_latent_upscaler_path=Path("temporal.safetensors"),
        duration_head_path=Path("duration.safetensors"),
    )

    settings.validate()

    assert not settings.uses_split_checkpoint


def test_model_generation_rejects_unknown_binder() -> None:
    with pytest.raises(ValueError, match="model_generation must be one of: 2.3, 2.5"):
        LTX2Settings(model_generation="future").validate()


def test_video_vae_selector_is_generation_neutral() -> None:
    assert LTX2Settings().video_vae == "conv"
    LTX2Settings(model_generation="2.3", video_vae="diffusion").validate()
    LTX2Settings(model_generation="2.5", video_vae="diffusion").validate()

    with pytest.raises(ValueError, match="video_vae must be one of: conv, diffusion"):
        LTX2Settings(video_vae="future").validate()


def test_stream_transformer_environment_and_preset_resolution() -> None:
    os.environ["KINO_STREAM_TRANSFORMER"] = "1"
    unresolved = LTX2Settings.from_env()
    assert unresolved.stream_transformer is True
    assert unresolved.transformer_resident_blocks is None
    resolved = unresolved.resolve_presets()
    assert resolved.transformer_resident_blocks == 16
    assert resolved.transformer_compile_group_size == 4


def test_stream_transformer_preset_keeps_explicit_low_level_values() -> None:
    resolved = LTX2Settings(
        stream_transformer=True,
        transformer_resident_blocks=3,
        transformer_compile_group_size=2,
    ).resolve_presets()
    assert resolved.transformer_resident_blocks == 3
    assert resolved.transformer_compile_group_size == 2
    resolved.validate()


def test_steel_probe_requires_steel_backend() -> None:
    with pytest.raises(ValueError, match="steel_attention_probe requires"):
        LTX2Settings(steel_attention=False, steel_attention_probe=True).validate()


def test_signpost_log_requires_signposts() -> None:
    with pytest.raises(ValueError, match="profile_signpost_log requires"):
        Settings(profile_signpost_log=Path("trace.log")).validate()


def test_signpost_settings_resolve_from_environment() -> None:
    os.environ["KINO_PROFILE_SIGNPOSTS"] = "1"
    os.environ["KINO_PROFILE_SIGNPOST_LOG"] = "/tmp/kinomlx-signposts.log"
    settings = Settings.from_env()
    assert settings.profile_signposts is True
    assert settings.profile_signpost_log == Path("/tmp/kinomlx-signposts.log")


def test_from_env_names_the_setting_with_an_invalid_typed_value() -> None:
    os.environ["KINO_TRANSFORMER_RESIDENT_BLOCKS"] = "many"
    with pytest.raises(ValueError, match="transformer_resident_blocks: invalid value"):
        LTX2Settings.from_env()


def test_from_env_fields_bootstraps_one_setting_independently() -> None:
    os.environ["KINO_JSON"] = "1"
    os.environ["KINO_TRANSFORMER_RESIDENT_BLOCKS"] = "many"
    settings = Settings.from_env_fields("json_output")
    assert settings.json_output is True
    with pytest.raises(ValueError, match="transformer_resident_blocks: invalid value"):
        LTX2Settings.from_env()


def test_from_env_fields_rejects_unknown_field_names() -> None:
    with pytest.raises(KeyError, match="unknown Settings fields"):
        Settings.from_env_fields("not_a_setting")


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "anything"])
def test_from_env_accepts_conventional_truthy_booleans(raw: str) -> None:
    os.environ["KINO_FAST_MODE"] = raw
    assert LTX2Settings.from_env().fast_mode is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
def test_from_env_accepts_conventional_falsy_booleans(raw: str) -> None:
    os.environ["KINO_FAST_MODE"] = raw
    assert LTX2Settings.from_env().fast_mode is False


def test_from_env_empty_boolean_is_unresolved_not_truthy() -> None:
    os.environ["KINO_QUIET"] = ""
    assert Settings.from_env().quiet is False


# ---------------------------------------------------------------------------
# CLI is verbatim - {{VAR}} is NOT substituted from --flag values
# ---------------------------------------------------------------------------


def test_cli_does_not_substitute_template_syntax_in_paths() -> None:
    """``--cache-dir '{{KINOMLX_TEST_ENV_A}}/foo'`` stays literal."""
    os.environ["KINOMLX_TEST_ENV_A"] = "/some/cache-root"
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    args = parser.parse_args(["--cache-dir", "{{KINOMLX_TEST_ENV_A}}/foo"])
    final = settings_from_args(args, Settings.from_env())
    # The literal braces must survive into the resolved Path.
    assert final.cache_dir == Path("{{KINOMLX_TEST_ENV_A}}/foo")


def test_cli_overrides_env_resolved_values() -> None:
    os.environ["KINO_CACHE_DIR"] = "/from/env"
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    args = parser.parse_args(["--cache-dir", "/from/cli"])
    final = settings_from_args(args, Settings.from_env())
    assert final.cache_dir == Path("/from/cli")


def test_cli_boolean_pairs_work() -> None:
    parser = argparse.ArgumentParser()
    add_settings_argparse_args(parser, LTX2Settings, title="LTX-2 settings")
    base = LTX2Settings.from_env()
    assert base.fast_mode is True
    args = parser.parse_args(["--no-fast-mode"])
    final = base.with_overrides(fast_mode=args.fast_mode)
    assert final.fast_mode is False


def test_cli_parses_cache_selector_tuples_and_off() -> None:
    parser = argparse.ArgumentParser()
    add_settings_argparse_args(parser, LTX2Settings, title="LTX-2 settings")
    args = parser.parse_args(
        [
            "--video-ff-layout-specs",
            "project_in:pretranspose,project_out:pretranspose",
            "--video-ff-layout-layers",
            "0,2,47",
            "--audio-ff-layout-specs",
            "off",
            "--no-audio-layout-mirror",
        ]
    )
    final = LTX2Settings().with_overrides(
        **{field.name: getattr(args, field.name, None) for field in fields(LTX2Settings)}
    )
    assert final.video_ff_layout_specs == (
        "project_in:pretranspose",
        "project_out:pretranspose",
    )
    assert final.video_ff_layout_layers == (0, 2, 47)
    assert final.audio_ff_layout_specs == ()
    assert final.audio_layout_mirror is False
    final.validate()


def test_settings_validate_cache_policy_conflicts() -> None:
    with pytest.raises(ValueError, match="audio_layout_mirror"):
        LTX2Settings(audio_ff_layout_specs=("project_out:pretranspose",)).validate()
    with pytest.raises(ValueError, match="cannot be combined"):
        LTX2Settings(
            transformer_cache_quantize="mxfp8-blocks",
            video_ff_quantize_specs=("project_out:mxfp8",),
        ).validate()
    with pytest.raises(ValueError, match="between 0 and 47"):
        LTX2Settings(video_ff_layout_layers=(48,)).validate()


def test_every_settings_field_has_a_cli_flag() -> None:
    """``add_argparse_args`` exposes a ``--kebab-case`` flag for every field.

    The CLI override surface is generated from the dataclass, so adding a field
    grows the CLI for free - but only as long as the generator handles the
    field's type. This guards that completeness: a field with no matching flag
    (or a type the generator silently skips) fails here.
    """
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    dests = {action.dest for action in parser._actions}
    missing = [f.name for f in fields(Settings) if f.name not in dests]
    assert not missing, f"Settings fields with no --kebab-case CLI flag: {missing}"


def test_every_ltx2_setting_has_a_model_contributed_cli_flag() -> None:
    parser = argparse.ArgumentParser()
    add_settings_argparse_args(parser, LTX2Settings, title="LTX-2 settings")
    destinations = {action.dest for action in parser._actions}
    missing = [field.name for field in fields(LTX2Settings) if field.name not in destinations]
    assert not missing, f"LTX2Settings fields with no CLI flag: {missing}"


def test_json_setting_uses_short_cli_vocabulary() -> None:
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    assert settings_from_args(parser.parse_args(["--json"]), Settings()).json_output
    assert not settings_from_args(parser.parse_args(["--no-json"]), Settings()).json_output


def test_argparse_generation_can_skip_an_owned_setting() -> None:
    parser = argparse.ArgumentParser()
    add_argparse_args(parser, skip={"cache_dir"})
    dests = {action.dest for action in parser._actions}
    assert "cache_dir" not in dests


def test_gemma_variant_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="gemma_variant must be one of: qat, plain"):
        LTX2Settings(gemma_variant="quantized").validate()
