"""Behavioral tests for ``kinomlx.settings``.

Covers the env-resolution + CLI-verbatim contracts that are easy to
regress when adding fields or adjusting the resolver.
"""

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from unittest.mock import patch

import pytest

from kinomlx.config_records import config_record, record_intent, supplied_fields
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import (
    EnvironmentSettings,
    Settings,
    _resolve_env_entry,
    add_argparse_args,
    add_settings_argparse_args,
    settings_from_args,
)


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    """Isolate registered settings and restore even direct writes made by tests."""
    with patch.dict(os.environ):
        for name in config_registry().environment_variables():
            os.environ.pop(name, None)
        for name in tuple(os.environ):
            if name.startswith("KINOMLX_TEST_ENV_"):
                os.environ.pop(name)
        yield


# ---------------------------------------------------------------------------
# _resolve_env_entry
# ---------------------------------------------------------------------------


def test_bare_string_is_literal_not_env_lookup() -> None:
    """A string with no ``{{}}`` resolves to itself, never to the env."""
    os.environ["KINOMLX_TEST_ENV_A"] = "/from/env"
    # A bare name is literal and must not pick up its environment value.
    assert _resolve_env_entry("KINOMLX_TEST_ENV_A") == "KINOMLX_TEST_ENV_A"
    assert _resolve_env_entry("/literal/path") == "/literal/path"
    assert _resolve_env_entry("KINO_APP_CACHE_DIR") == "KINO_APP_CACHE_DIR"


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
    assert infrastructure.app_cache_dir == Path("~/Library/Caches/KinoMLX").expanduser()
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
    assert s.app_cache_dir == Path("~/Library/Caches/KinoMLX").expanduser()
    assert s.hf_home == Path("/tmp/hfroot")


def test_from_env_cache_dir_uses_kino_var_independently_of_hf_home() -> None:
    os.environ["KINO_APP_CACHE_DIR"] = "/tmp/explicit"
    os.environ["HF_HOME"] = "/tmp/hfroot"
    s = Settings.from_env()
    assert s.app_cache_dir == Path("/tmp/explicit")
    assert s.hf_home == Path("/tmp/hfroot")


def test_from_env_bare_settings_reads_no_env() -> None:
    """Direct infrastructure and model records do not touch env."""
    os.environ["KINO_APP_CACHE_DIR"] = "/should/be/ignored"
    os.environ["HF_HOME"] = "/should/also/be/ignored"
    os.environ["KINO_LTX2_FAST_MODE"] = "0"
    assert Settings().app_cache_dir == Path("~/Library/Caches/KinoMLX").expanduser()
    assert LTX2Settings().fast_mode is True


def test_runtime_light_layout_default_matches_weight_policy() -> None:
    from kinomlx.models.ltx2.weights.policy import DEFAULT_VIDEO_FF_LAYOUT_SPECS

    expected = tuple(f"{target}:{layout}" for target, layout in DEFAULT_VIDEO_FF_LAYOUT_SPECS)
    assert LTX2Settings().video_ff_layout_specs == expected


def test_from_env_parses_typed_values() -> None:
    os.environ["KINO_LTX2_MODEL_GENERATION"] = "2.5"
    os.environ["KINO_LTX2_VIDEO_VAE"] = "diffusion"
    os.environ["KINO_LTX2_FAST_MODE"] = "0"
    os.environ["KINO_LTX2_TRANSFORMER_RESIDENT_BLOCKS"] = "4"
    os.environ["KINO_LTX2_TRANSFORMER_DTYPE"] = "float16"
    os.environ["KINO_LTX2_STEEL_ATTENTION"] = "0"
    os.environ["KINO_LTX2_STEEL_ATTENTION_D64"] = "0"
    os.environ["KINO_LTX2_STEEL_ATTENTION_PROBE"] = "1"
    os.environ["KINO_LTX2_COMPILE_ATTENTION"] = "0"
    os.environ["KINO_MLX_CACHE_LIMIT_GB"] = "2.5"
    os.environ["KINO_MLX_WIRED_LIMIT_GB"] = "48"
    os.environ["KINO_LTX2_VIDEO_FF_LAYOUT_SPECS"] = (
        "project_in:pretranspose,project_out:pretranspose"
    )
    os.environ["KINO_LTX2_VIDEO_FF_LAYOUT_LAYERS"] = "0,7,47"
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
    assert infrastructure.mlx_wired_limit_gb == pytest.approx(48.0)
    assert Settings().mlx_wired_limit_gb is None
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mlx_wired_limit_gb"):
            Settings(mlx_wired_limit_gb=value).validate()
    Settings(mlx_wired_limit_gb=0.0).validate()
    assert model.video_ff_layout_specs == (
        "project_in:pretranspose",
        "project_out:pretranspose",
    )
    assert model.video_ff_layout_layers == (0, 7, 47)


def test_from_env_parses_generation_neutral_component_paths() -> None:
    values = {
        "KINO_LTX2_TRANSFORMER_PATH": "/models/transformer.safetensors",
        "KINO_LTX2_TEXT_ENCODER_PATH": "/models/text.safetensors",
        "KINO_LTX2_VIDEO_VAE_PATH": "/models/video-vae.safetensors",
        "KINO_LTX2_AUDIO_VAE_PATH": "/models/audio-vae.safetensors",
        "KINO_LTX2_SPATIAL_UPSCALER_PATH": "/models/spatial.safetensors",
        "KINO_LTX2_TEMPORAL_LATENT_UPSCALER_PATH": "/models/temporal.safetensors",
        "KINO_LTX2_DURATION_HEAD_PATH": "/models/duration.safetensors",
    }
    os.environ.update(values)

    settings = LTX2Settings.from_env()

    assert settings.uses_split_checkpoint
    for environment_name, value in values.items():
        field_name = environment_name.removeprefix("KINO_LTX2_").lower()
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
    with pytest.raises(ValueError, match="model_generation: must be one of: 2.3, 2.5"):
        LTX2Settings(model_generation="future").validate()


def test_video_vae_selector_is_generation_neutral() -> None:
    assert LTX2Settings().video_vae == "conv"
    LTX2Settings(model_generation="2.3", video_vae="diffusion").validate()
    LTX2Settings(model_generation="2.5", video_vae="diffusion").validate()

    with pytest.raises(ValueError, match="video_vae: must be one of: conv, diffusion"):
        LTX2Settings(video_vae="future").validate()


def test_stream_transformer_environment_and_preset_resolution() -> None:
    os.environ["KINO_LTX2_STREAM_TRANSFORMER"] = "1"
    unresolved = LTX2Settings.from_env()
    assert unresolved.stream_transformer is True
    assert unresolved.transformer_resident_blocks is None
    resolved = unresolved.resolve_presets()
    assert resolved.transformer_resident_blocks == 8
    assert resolved.transformer_compile_group_size is None


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
    os.environ["KINO_LTX2_TRANSFORMER_RESIDENT_BLOCKS"] = "many"
    with pytest.raises(ValueError, match="transformer_resident_blocks: invalid value"):
        LTX2Settings.from_env()


def test_from_env_fields_bootstraps_one_setting_independently() -> None:
    os.environ["KINO_JSON"] = "1"
    os.environ["KINO_LTX2_TRANSFORMER_RESIDENT_BLOCKS"] = "many"
    settings = Settings.from_env_fields("json_output")
    assert settings.json_output is True
    with pytest.raises(ValueError, match="transformer_resident_blocks: invalid value"):
        LTX2Settings.from_env()


def test_from_env_fields_rejects_unknown_field_names() -> None:
    with pytest.raises(KeyError, match="unknown Settings fields"):
        Settings.from_env_fields("not_a_setting")


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_from_env_accepts_conventional_truthy_booleans(raw: str) -> None:
    os.environ["KINO_LTX2_FAST_MODE"] = raw
    assert LTX2Settings.from_env().fast_mode is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
def test_from_env_accepts_conventional_falsy_booleans(raw: str) -> None:
    os.environ["KINO_LTX2_FAST_MODE"] = raw
    assert LTX2Settings.from_env().fast_mode is False


def test_from_env_empty_boolean_is_explicit_false() -> None:
    os.environ["KINO_QUIET"] = ""
    assert Settings.from_env().quiet is False


# ---------------------------------------------------------------------------
# CLI is verbatim - {{VAR}} is NOT substituted from --flag values
# ---------------------------------------------------------------------------


def test_cli_does_not_substitute_template_syntax_in_paths() -> None:
    """``--app-cache-dir '{{KINOMLX_TEST_ENV_A}}/foo'`` stays literal."""
    os.environ["KINOMLX_TEST_ENV_A"] = "/some/cache-root"
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    args = parser.parse_args(["--app-cache-dir", "{{KINOMLX_TEST_ENV_A}}/foo"])
    final = settings_from_args(args, Settings.from_env())
    # The literal braces must survive into the resolved Path.
    assert final.app_cache_dir == Path("{{KINOMLX_TEST_ENV_A}}/foo")


def test_cli_overrides_env_resolved_values() -> None:
    os.environ["KINO_APP_CACHE_DIR"] = "/from/env"
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    args = parser.parse_args(["--app-cache-dir", "/from/cli"])
    final = settings_from_args(args, Settings.from_env())
    assert final.app_cache_dir == Path("/from/cli")


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


def test_settings_validate_weight_policy_conflicts() -> None:
    with pytest.raises(ValueError, match="audio_layout_mirror"):
        LTX2Settings(audio_ff_layout_specs=("project_out:pretranspose",)).validate()
    with pytest.raises(ValueError, match="cannot be combined"):
        LTX2Settings(
            transformer_quantize="mxfp8-blocks",
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
    add_argparse_args(parser, skip={"app_cache_dir"})
    dests = {action.dest for action in parser._actions}
    assert "app_cache_dir" not in dests


def test_gemma_variant_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="gemma_variant: must be one of: qat, plain"):
        LTX2Settings(gemma_variant="quantized").validate()


def test_adapter_weight_check_shared_environment_default_and_override(monkeypatch):
    monkeypatch.delenv("KINO_CHECK_ADAPTER_WEIGHTS", raising=False)
    assert Settings.from_env().check_adapter_weights is True
    monkeypatch.setenv("KINO_CHECK_ADAPTER_WEIGHTS", "false")
    assert Settings.from_env().check_adapter_weights is False
    assert (
        Settings.from_env().with_overrides(check_adapter_weights=True).check_adapter_weights is True
    )


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_adapter_weight_check_cli_is_identical_between_models(model):
    import importlib

    module = importlib.import_module(f"kinomlx.models.{model}.arguments")
    parser = module.build_parser()
    assert parser.parse_args(["--no-check-adapter-weights"]).check_adapter_weights is False
    assert parser.parse_args(["--check-adapter-weights"]).check_adapter_weights is True


def test_application_cache_override_and_xdg_fallback(monkeypatch, tmp_path):
    from kinomlx.settings import Settings

    monkeypatch.delenv("KINO_APP_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert Settings.from_env_fields("app_cache_dir").app_cache_dir == tmp_path / "xdg" / "KinoMLX"
    monkeypatch.setenv("KINO_APP_CACHE_DIR", str(tmp_path / "explicit"))
    assert Settings.from_env_fields("app_cache_dir").app_cache_dir == tmp_path / "explicit"


@pytest.mark.parametrize("layers", [(0, 0), (-1,), (48,)])
def test_quantization_exclusions_reject_invalid_layers(layers):
    with pytest.raises(ValueError, match="transformer_quantize_exclude_layers"):
        LTX2Settings(
            transformer_quantize="mxfp8-blocks", transformer_quantize_exclude_layers=layers
        ).validate()


def test_quantization_exclusions_require_quantization():
    with pytest.raises(ValueError, match="requires whole-block quantization"):
        LTX2Settings(transformer_quantize_exclude_layers=(0,)).validate()


def test_quantization_exclusions_parse_cli_and_environment(monkeypatch):
    from kinomlx.models.ltx2.arguments import build_parser

    args = build_parser().parse_args(
        [
            "--transformer-quantize",
            "mxfp8-blocks",
            "--transformer-quantize-exclude-layers",
            "0,1,2,3,47",
        ]
    )
    assert tuple(args.transformer_quantize_exclude_layers) == (0, 1, 2, 3, 47)
    monkeypatch.setenv("KINO_LTX2_TRANSFORMER_QUANTIZE", "mxfp8-blocks")
    monkeypatch.setenv("KINO_LTX2_TRANSFORMER_QUANTIZE_EXCLUDE_LAYERS", "0,1,46,47")
    assert LTX2Settings.from_env().transformer_quantize_exclude_layers == (0, 1, 46, 47)


def test_environment_boolean_rejects_misspellings(monkeypatch) -> None:
    monkeypatch.setenv("KINO_LTX2_FAST_MODE", "anything")
    with pytest.raises(ValueError, match="fast_mode"):
        LTX2Settings.from_env()


def test_adapter_accumulation_choices_validation_and_auto_resolution() -> None:
    with pytest.raises(ValueError, match="adapter_accumulation: must be one of"):
        LTX2Settings(adapter_accumulation="half").validate()
    with pytest.raises(ValueError, match="requires transformer_dtype=float16"):
        LTX2Settings(adapter_accumulation="float16").validate()
    LTX2Settings(adapter_accumulation="float16", transformer_dtype="float16").validate()

    def resolved(**overrides: object) -> str:
        return LTX2Settings(**overrides).resolved_adapter_accumulation()

    assert resolved() == "float32"
    assert resolved(transformer_dtype="float16") == "float32"
    assert resolved(stream_transformer=True) == "float32"
    assert resolved(stream_transformer=True, transformer_dtype="float16") == "float16"
    assert resolved(transformer_resident_blocks=8, transformer_dtype="float16") == "float16"
    assert (
        resolved(
            stream_transformer=True, transformer_dtype="float16", adapter_accumulation="float32"
        )
        == "float32"
    )
    assert resolved(transformer_dtype="float16", adapter_accumulation="float16") == "float16"


def test_environment_value_resolves_each_selected_source_once(monkeypatch):
    from kinomlx import settings

    reads = []

    def read(name):
        reads.append(name)
        return {"KINO_APP_CACHE_DIR": None, "XDG_CACHE_HOME": "/cache"}.get(name)

    monkeypatch.setattr(settings, "read_environment", read)
    selected = Settings.values_from_env_fields("app_cache_dir")["app_cache_dir"]
    assert reads == ["KINO_APP_CACHE_DIR", "XDG_CACHE_HOME"]
    assert selected.value == Path("/cache/KinoMLX")
    assert selected.source == "{{XDG_CACHE_HOME}}/KinoMLX"
    assert selected.explicit


def test_settings_environment_presence_excludes_literal_fallbacks():
    settings = Settings.from_env()
    assert supplied_fields(settings) == frozenset()
    assert record_intent(settings) == {}
    assert settings.app_cache_dir == Path("~/Library/Caches/KinoMLX").expanduser()


def test_settings_environment_presence_retains_explicit_default_and_zero():
    os.environ["KINO_VERBOSE"] = "false"
    os.environ["KINO_MLX_CACHE_LIMIT_GB"] = "0"
    settings = Settings.from_env()
    assert record_intent(settings) == {"verbose": False, "mlx_cache_limit_gb": 0.0}
    changed = settings.with_overrides(quiet=False, profile_signpost_log=None)
    assert record_intent(changed) == {
        "verbose": False,
        "quiet": False,
        "mlx_cache_limit_gb": 0.0,
    }
    assert record_intent(settings.with_overrides()) == record_intent(settings)


def test_settings_environment_template_fallback_is_explicit():
    os.environ["XDG_CACHE_HOME"] = "cache-root"
    settings = Settings.from_env_fields("app_cache_dir")
    assert record_intent(settings) == {"app_cache_dir": Path("cache-root/KinoMLX")}
    os.environ["KINO_APP_CACHE_DIR"] = "preferred-cache"
    assert record_intent(Settings.from_env_fields("app_cache_dir")) == {
        "app_cache_dir": Path("preferred-cache")
    }


def test_settings_cli_bridge_preserves_sparse_presence():
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    base = Settings(verbose=False)
    settings = settings_from_args(parser.parse_args(["--quiet", "--mlx-cache-limit-gb", "0"]), base)
    assert record_intent(settings) == {"verbose": False, "quiet": True, "mlx_cache_limit_gb": 0.0}
    assert record_intent(settings_from_args(parser.parse_args([]), base)) == {"verbose": False}


def test_environment_presence_is_available_during_post_init():
    @config_record
    @dataclass(frozen=True)
    class Tracked(EnvironmentSettings):
        enabled: bool = field(default=True, metadata={"env": "false"})

        def __post_init__(self):
            if "enabled" in supplied_fields(self):
                raise ValueError("explicit enabled is not allowed")

    assert Tracked.from_env().enabled is False
    assert record_intent(Tracked.from_env()) == {}
    with pytest.raises(ValueError, match="explicit enabled"):
        Tracked(enabled=False)


@pytest.mark.parametrize(
    ("record", "variable", "field_name", "raw", "expected"),
    [
        (
            Flux2Settings,
            "KINO_FLUX2_TRANSFORMER_DTYPE",
            "transformer_dtype",
            "bfloat16",
            "bfloat16",
        ),
        (
            Krea2Settings,
            "KINO_KREA2_TRANSFORMER_DTYPE",
            "transformer_dtype",
            "bfloat16",
            "bfloat16",
        ),
        (GMNetSettings, "KINO_GMNET_VARIANT", "variant", "realworld", "realworld"),
    ],
)
def test_model_settings_share_presence_across_constructors_env_and_copies(
    record, variable, field_name, raw, expected
):
    from kinomlx.config import dataclass_from_mapping

    assert record_intent(record.from_env()) == {}
    os.environ[variable] = raw
    loaded = record.from_env()
    assert record_intent(loaded) == {field_name: expected}
    assert record_intent(record(**{field_name: expected})) == {field_name: expected}
    copied = loaded.with_overrides(**{field_name: expected})
    assert record_intent(copied) == {field_name: expected}
    assert record_intent(dataclass_from_mapping(record, copied, table="model_settings")) == {
        field_name: expected
    }


@pytest.mark.parametrize("model", ["krea2", "flux2", "gmnet"])
def test_model_settings_inspection_reports_constructor_tracking(model):
    from kinomlx.config.inspection import configuration_inventory

    items = configuration_inventory(config_registry().model(model))["fields"]
    settings = [item for item in items if item["path"].startswith("model_settings.")]
    assert settings
    assert all(item["constructor_presence"] == "tracked" for item in settings)
