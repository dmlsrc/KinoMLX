"""Typed CLI/TOML resolution and precedence tests."""

from __future__ import annotations

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from kinomlx.cli._registry import CONFIG_CONTROL_DESTINATIONS, config_registry
from kinomlx.cli.args import build_parser
from kinomlx.cli.config import (
    assemble,
    build_timestamped_output_path,
    resolve_for_execution,
    validate_for_execution,
)
from kinomlx.config import (
    ConfigError,
    ConfigGroup,
    dump_config,
    load_config,
    normalize_output_selection,
)
from kinomlx.models.ltx2.artifacts import (
    FINAL_LATENTS,
    STAGE_1_CONDITIONING,
    STAGE_1_LATENTS,
    STAGE_2_CONDITIONING,
    TEXT_CONDITIONING,
    requested_artifacts,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.settings import Settings


def test_config_path_expansion_failure_is_typed(monkeypatch) -> None:
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda _path: (_ for _ in ()).throw(RuntimeError("home is unavailable")),
    )

    with pytest.raises(ConfigError, match="cannot resolve config path"):
        load_config("~/run.toml")


def test_output_normalization_emits_an_explicit_exact_path_tombstone() -> None:
    normalized = normalize_output_selection({"output": {"directory": "renders", "prefix": "clip"}})

    assert normalized["output"] == {
        "directory": "renders",
        "prefix": "clip",
        "path": None,
    }


def test_same_layer_exact_output_still_wins_over_generated_names() -> None:
    normalized = normalize_output_selection(
        {
            "output": {
                "path": "exact.mp4",
                "directory": "renders",
                "prefix": "clip",
            }
        }
    )

    assert normalized["output"]["path"] == "exact.mp4"


def test_global_registry_owns_every_public_ltx2_config_destination() -> None:
    parser_destinations = {
        action.dest
        for action in build_parser()._actions
        if action.dest not in CONFIG_CONTROL_DESTINATIONS
    }
    assert parser_destinations == config_registry().model("ltx2").cli_destinations()


def test_flags_toml_and_set_share_one_precedence_path(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
model = "ltx2"

[generate]
prompt = "from config"
seed = 1
width = 1024
height = 576

[output]
path = "config.mp4"

[model_settings]
fast_mode = false
""".strip(),
        encoding="utf-8",
    )
    options = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--prompt",
            "from cli",
            "--seed",
            "2",
            "--output",
            "cli.mp4",
            "--fast-mode",
            "--set",
            "generate.seed=3",
            "--set",
            "model_settings.fast_mode=false",
        ]
    )
    invocation = assemble(options, base_settings=Settings())
    assert invocation.request.prompt == "from cli"
    assert invocation.request.seed == 3
    assert invocation.output.path == Path("cli.mp4")
    assert invocation.model_settings.fast_mode is False


def test_prompt_multiline_dump_round_trips_through_resolution(tmp_path: Path) -> None:
    schema = config_registry().model("ltx2")
    rendered = schema.dump_config(
        {"model": "ltx2", "generate": {"prompt": "coastal sunrise", "seed": 7}}
    )
    assert "prompt = '''\ncoastal sunrise\n'''" in rendered
    raw = tomllib.loads(rendered)
    assert raw["generate"]["prompt"] == "coastal sunrise\n"

    config = tmp_path / "prompt.toml"
    config.write_text(rendered, encoding="utf-8")
    invocation = assemble(
        build_parser().parse_args(["--config", str(config), "--print-config"]),
        base_settings=Settings(),
    )
    assert invocation.request.prompt == "coastal sunrise"
    assert invocation.resolved_config["generate"]["prompt"] == "coastal sunrise"


def test_prompt_multiline_dump_falls_back_for_literal_delimiter_and_controls() -> None:
    schema = config_registry().model("ltx2")
    for prompt in ("contains ''' delimiter", "contains \x01 control"):
        rendered = schema.dump_config({"model": "ltx2", "generate": {"prompt": prompt}})
        assert "prompt = '''" not in rendered
        assert tomllib.loads(rendered)["generate"]["prompt"] == prompt


def test_uint64_seed_is_stringified_for_valid_toml_and_restored(tmp_path: Path) -> None:
    maximum = 2**64 - 1
    schema = config_registry().model("ltx2")
    rendered = schema.dump_config(
        {"model": "ltx2", "generate": {"prompt": "test", "seed": maximum}}
    )
    assert f'seed = "{maximum}"' in rendered
    raw = tomllib.loads(rendered)
    assert raw["generate"]["seed"] == str(maximum)
    assert schema.normalize_config(raw)["generate"]["seed"] == maximum

    config = tmp_path / "maximum-seed.toml"
    config.write_text(rendered, encoding="utf-8")
    invocation = assemble(
        build_parser().parse_args(["--config", str(config), "--print-config"]),
        base_settings=Settings(),
    )
    assert invocation.request.seed == maximum


def test_registry_semantic_groups_point_only_at_typed_ltx2_fields() -> None:
    schema = config_registry().model("ltx2")
    model_fields = LTX2Settings.__dataclass_fields__
    for group in (
        ConfigGroup.MODEL_SOURCE,
        ConfigGroup.MODEL_MONOLITHIC_SOURCE,
        ConfigGroup.MODEL_SPLIT_SOURCE,
        ConfigGroup.MODEL_GEMMA_SOURCE,
        ConfigGroup.MODEL_TEXT_ENCODER_SOURCE,
        ConfigGroup.MODEL_GENERATION_SELECTOR,
        ConfigGroup.MODEL_VIDEO_VAE_SELECTOR,
        ConfigGroup.MODEL_VIDEO_VAE_SOURCE,
    ):
        assert schema.cli_destinations_in_group(group)
        assert set(schema.cli_destinations_in_group(group)) <= set(model_fields)
    request_fields = DistilledRequest.__dataclass_fields__
    for group in (
        ConfigGroup.RESTART_DECODE_LOCKED,
        ConfigGroup.RESTART_STAGE2_LOCKED,
    ):
        assert schema.cli_destinations_in_group(group)
        assert set(schema.cli_destinations_in_group(group)) <= set(request_fields)


def test_hdr_reference_cli_builds_one_typed_generation_request(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mp4"
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--hdr",
                "ACESCG",
                "--hdr-reference",
                str(reference),
                "--hdr-reference-strength",
                "0.8",
            ]
        ),
        base_settings=Settings(),
    )
    assert invocation.request.hdr == "ACESCG"
    assert invocation.request.hdr_reference is not None
    assert invocation.request.hdr_reference.path == reference
    assert invocation.request.hdr_reference.strength == 0.8


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--vsr-spatial-mode", "balanced"), "vsr_spatial_mode"),
        (("--target-fps", "48"), "target_fps"),
        (("--save-vae-frames",), "lossless EXR"),
        (("--vsr-save-original",), "vsr_save_original"),
    ],
)
def test_hdr_rejects_unvalidated_sdr_postprocessing_before_execution(
    tmp_path: Path,
    extra: tuple[str, ...],
    message: str,
) -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--hdr",
                "ACESCG",
                "--output",
                str(tmp_path / "out.mp4"),
                *extra,
            ]
        ),
        base_settings=Settings(),
    )
    with pytest.raises(ConfigError, match=message):
        validate_for_execution(invocation)


def test_hdr_save_all_sidecars_ignores_inapplicable_vsr_original(tmp_path: Path) -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--hdr",
                "ACESCG",
                "--output",
                str(tmp_path / "out.mp4"),
                "--save-all-sidecars",
            ]
        ),
        base_settings=Settings(),
    )

    assert invocation.output.save_all_sidecars
    assert invocation.output.vsr_save_original
    validate_for_execution(invocation)


def test_hdr_heic_frame_dump_requires_hdr_generation(tmp_path: Path) -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--output",
                str(tmp_path / "out.mp4"),
                "--save-hdr-heic-frames",
            ]
        ),
        base_settings=Settings(),
    )
    with pytest.raises(ConfigError, match="requires HDR generation"):
        validate_for_execution(invocation)


def test_split_component_paths_share_toml_cli_set_and_print_config(
    tmp_path: Path,
) -> None:
    config = tmp_path / "split.toml"
    config.write_text(
        """
[model_settings]
transformer_path = "from-toml-transformer.safetensors"
video_vae_path = "from-toml-video-vae.safetensors"
spatial_upscaler_path = "from-toml-spatial.safetensors"
""".strip(),
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(
            [
                "--config",
                str(config),
                "--text-encoder-path",
                "from-cli-text.safetensors",
                "--set",
                'model_settings.duration_head_path="from-set-duration.safetensors"',
                "--print-config",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(
            weights_path=Path("from-environment-monolith.safetensors"),
            gemma_path=Path("from-environment-gemma"),
        ),
    )

    assert invocation.model_settings.weights_path is None
    assert invocation.model_settings.gemma_path is None
    assert invocation.model_settings.transformer_path == Path("from-toml-transformer.safetensors")
    assert invocation.model_settings.text_encoder_path == Path("from-cli-text.safetensors")
    assert invocation.model_settings.video_vae_path == Path("from-toml-video-vae.safetensors")
    assert invocation.model_settings.duration_head_path == Path("from-set-duration.safetensors")
    assert invocation.model_settings.uses_split_checkpoint
    printed = dump_config(invocation.resolved_config)
    assert 'transformer_path = "from-toml-transformer.safetensors"' in printed
    assert 'text_encoder_path = "from-cli-text.safetensors"' in printed
    assert 'duration_head_path = "from-set-duration.safetensors"' in printed


def test_higher_precedence_primary_keeps_independent_component_overrides() -> None:
    invocation = assemble(
        build_parser().parse_args(
            ["--weights-path", "from-cli-monolith.safetensors", "--print-config"]
        ),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(
            transformer_path=Path("from-environment-transformer.safetensors"),
            text_encoder_path=Path("from-environment-text.safetensors"),
            video_vae_path=Path("shared-video-vae.safetensors"),
        ),
    )

    assert invocation.model_settings.weights_path == Path("from-cli-monolith.safetensors")
    assert invocation.model_settings.transformer_path is None
    assert invocation.model_settings.text_encoder_path == Path("from-environment-text.safetensors")
    assert invocation.model_settings.video_vae_path == Path("shared-video-vae.safetensors")


def test_pack_root_and_transformer_override_survive_cli_assembly(tmp_path: Path) -> None:
    pack = tmp_path / "ltx25"
    pack.mkdir()
    transformer = tmp_path / "dev-transformer.safetensors"

    invocation = assemble(
        build_parser().parse_args(
            [
                "--weights-path",
                str(pack),
                "--transformer-path",
                str(transformer),
                "--print-config",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(),
    )

    assert invocation.model_settings.weights_path == pack
    assert invocation.model_settings.transformer_path == transformer


def test_cli_transformer_override_preserves_environment_pack_root(tmp_path: Path) -> None:
    pack = tmp_path / "ltx25"
    pack.mkdir()
    transformer = tmp_path / "dev-transformer.safetensors"

    invocation = assemble(
        build_parser().parse_args(["--transformer-path", str(transformer), "--print-config"]),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(weights_path=pack),
    )

    assert invocation.model_settings.weights_path == pack
    assert invocation.model_settings.transformer_path == transformer


def test_generation_selector_cli_vocabulary_matches_toml_field() -> None:
    action = next(item for item in build_parser()._actions if item.dest == "model_generation")
    field = config_registry().model("ltx2").field(("model_settings", "model_generation"))

    assert action.option_strings == ["--model-generation", "--ltx-generation"]
    assert action.dest == field.name == field.cli_dest == "model_generation"


@pytest.mark.parametrize("flag", ["--model-generation", "--ltx-generation"])
def test_generation_selector_round_trips_without_checkpoint_paths(flag: str) -> None:
    invocation = assemble(
        build_parser().parse_args([flag, "2.5", "--print-config"]),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(),
    )

    assert invocation.model_settings.model_generation == "2.5"
    assert invocation.model_settings.weights_path is None
    assert invocation.model_settings.transformer_path is None
    assert 'model_generation = "2.5"' in dump_config(invocation.resolved_config)


def test_generation_selector_clears_lower_precedence_checkpoint_paths() -> None:
    invocation = assemble(
        build_parser().parse_args(["--ltx-generation", "2.5", "--print-config"]),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(
            weights_path=Path("from-environment-monolith.safetensors"),
            gemma_path=Path("from-environment-gemma"),
            video_vae_path=Path("from-environment-video-vae.safetensors"),
            audio_vae_path=Path("from-environment-audio-vae.safetensors"),
            spatial_upscaler_path=Path("from-environment-spatial-upscaler.safetensors"),
            temporal_latent_upscaler_path=Path("from-environment-temporal-upscaler.safetensors"),
            duration_head_path=Path("from-environment-duration-head.safetensors"),
        ),
    )

    assert invocation.model_settings.model_generation == "2.5"
    for field_name in (
        "weights_path",
        "gemma_path",
        "spatial_upscaler_path",
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "temporal_latent_upscaler_path",
        "duration_head_path",
    ):
        assert getattr(invocation.model_settings, field_name) is None


def test_generation_selector_preserves_same_layer_checkpoint_paths() -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--ltx-generation",
                "2.5",
                "--transformer-path",
                "custom-transformer.safetensors",
                "--video-vae-path",
                "custom-video-vae.safetensors",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(
            weights_path=Path("from-environment-monolith.safetensors"),
            video_vae_path=Path("from-environment-video-vae.safetensors"),
        ),
    )

    assert invocation.model_settings.model_generation == "2.5"
    assert invocation.model_settings.transformer_path == Path("custom-transformer.safetensors")
    assert invocation.model_settings.video_vae_path == Path("custom-video-vae.safetensors")
    assert invocation.model_settings.weights_path is None


def test_video_vae_flag_selects_cached_diffusion_variant() -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--ltx-generation",
                "2.5",
                "--video-vae",
                "diffusion",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(),
    )

    assert invocation.model_settings.video_vae == "diffusion"
    assert invocation.model_settings.video_vae_path is None


def test_sampler_flag_resolves_on_the_public_request() -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--ltx-generation",
                "2.5",
                "--sampler",
                "deterministic",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(),
    )

    assert invocation.request.sampler == "deterministic"
    assert invocation.resolved_config["generate"]["sampler"] == "deterministic"


def test_noise_backend_flag_resolves_on_the_public_request() -> None:
    invocation = assemble(
        build_parser().parse_args(["--noise-backend", "torch-mps", "--print-config"]),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(),
    )

    assert invocation.request.noise_backend == "torch-mps"
    assert invocation.resolved_config["generate"]["noise_backend"] == "torch-mps"


def test_reference_aligned_audio_flag_resolves_on_the_public_request() -> None:
    invocation = assemble(
        build_parser().parse_args(["--reference-aligned-audio", "--print-config"]),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(),
    )

    assert invocation.request.reference_aligned_audio is True
    assert invocation.resolved_config["generate"]["reference_aligned_audio"] is True


def test_config_settings_override_environment_derived_base(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text('[settings]\ncache_dir = "from-config"\n', encoding="utf-8")
    options = build_parser().parse_args(["--config", str(config), "--print-config"])
    invocation = assemble(
        options,
        base_settings=Settings(cache_dir=Path("from-env")),
    )
    assert invocation.settings.cache_dir == Path("from-config")


def test_nested_image_config_can_be_partly_overridden_from_cli(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[generate]
prompt = "test"

[generate.image]
path = "first.png"
frame_index = 0
strength = 0.5
""".strip(),
        encoding="utf-8",
    )
    options = build_parser().parse_args(
        ["--config", str(config), "--image-strength", "0.8", "--print-config"]
    )
    invocation = assemble(options, base_settings=Settings())
    assert invocation.request.image is not None
    assert invocation.request.image.path == Path("first.png")
    assert invocation.request.image.strength == pytest.approx(0.8)


def test_duration_and_frames_obey_cross_surface_precedence(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nduration = 30.0\n',
        encoding="utf-8",
    )
    frames_win = assemble(
        build_parser().parse_args(["--config", str(config), "--frames", "121", "--print-config"]),
        base_settings=Settings(),
    )
    assert frames_win.request.duration is None
    assert frames_win.request.frames == 121

    with pytest.raises(ConfigError, match="frames and generate.duration"):
        assemble(
            build_parser().parse_args(
                [
                    "--config",
                    str(config),
                    "--frames",
                    "121",
                    "--duration",
                    "20",
                    "--print-config",
                ]
            ),
            base_settings=Settings(),
        )


def test_auto_duration_clears_lower_precedence_frames_and_duration(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nduration = 5.0\n',
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(["--config", str(config), "--auto-duration", "--print-config"]),
        base_settings=Settings(),
    )
    assert invocation.request.duration is None
    assert invocation.request.frames is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--auto-duration", "--frames", "121"],
        ["--frames", "121", "--auto-duration"],
        ["--auto-duration", "--duration", "5.0"],
    ],
)
def test_auto_duration_rejects_same_cli_layer_frames_or_duration(
    arguments: list[str],
) -> None:
    with pytest.raises(
        ConfigError,
        match=r"auto_duration cannot be combined with generate\.(frames|duration)",
    ):
        assemble(
            build_parser().parse_args([*arguments, "--print-config"]),
            base_settings=Settings(),
        )


def test_auto_duration_rejects_same_toml_layer_frames(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nauto_duration = true\nframes = 121\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="same configuration layer"):
        assemble(
            build_parser().parse_args(["--config", str(config), "--print-config"]),
            base_settings=Settings(),
        )


def test_auto_duration_rejects_non_boolean_config_values(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nauto_duration = "yes"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="auto_duration must be a boolean"):
        assemble(
            build_parser().parse_args(["--config", str(config), "--print-config"]),
            base_settings=Settings(),
        )


def test_partial_lora_override_flags_are_not_public_cli_options() -> None:
    options = {option for action in build_parser()._actions for option in action.option_strings}
    assert "--lora-allow-partial" not in options
    assert "--no-lora-allow-partial" not in options


def test_text_conditioning_and_vae_tiling_cli_land_in_request(tmp_path: Path) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    invocation = assemble(
        build_parser().parse_args(
            [
                "--text-conditioning",
                str(sidecar),
                "--vae-decode-dtype",
                "bfloat16",
                "--vae-tiling",
                "custom",
                "--vae-temporal-tile-frames",
                "64",
                "--vae-temporal-overlap-frames",
                "8",
                "--vae-spatial-tile-pixels",
                "512",
                "--vae-spatial-overlap-pixels",
                "64",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
    )
    assert invocation.request.text_conditioning == sidecar
    assert invocation.request.vae_decode_dtype == "bfloat16"
    assert invocation.request.vae_tiling.mode == "custom"
    assert invocation.request.vae_tiling.temporal_tile_frames == 64
    assert invocation.request.vae_tiling.spatial_tile_pixels == 512


def test_nested_vae_tiling_toml_and_set_share_the_schema(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[generate]
prompt = "test"

[generate.vae_tiling]
mode = "custom"
temporal_tile_frames = 64
temporal_overlap_frames = 8
""".strip(),
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(
            [
                "--config",
                str(config),
                "--set",
                "generate.vae_tiling.temporal_tile_frames=128",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
    )
    assert invocation.request.vae_tiling.mode == "custom"
    assert invocation.request.vae_tiling.temporal_tile_frames == 128
    assert invocation.request.vae_tiling.temporal_overlap_frames == 8


def test_mode_less_vae_geometry_selects_custom_policy(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[generate.vae_tiling]
temporal_tile_frames = 64
temporal_overlap_frames = 8
""".strip(),
        encoding="utf-8",
    )

    invocation = assemble(
        build_parser().parse_args(["--config", str(config), "--print-config"]),
        base_settings=Settings(),
    )
    assert invocation.request.vae_tiling.mode == "custom"
    assert invocation.request.vae_tiling.temporal_tile_frames == 64
    assert invocation.request.vae_tiling.temporal_overlap_frames == 8


@pytest.mark.parametrize("mode", ["auto", "single"])
def test_higher_precedence_inactive_vae_policy_clears_lower_geometry(
    tmp_path: Path,
    mode: str,
) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[generate.vae_tiling]
temporal_tile_frames = 64
temporal_overlap_frames = 8
""".strip(),
        encoding="utf-8",
    )

    invocation = assemble(
        build_parser().parse_args(
            ["--config", str(config), "--vae-tiling", mode, "--print-config"]
        ),
        base_settings=Settings(),
    )
    assert invocation.request.vae_tiling.mode == mode
    assert invocation.request.vae_tiling.temporal_tile_frames is None
    assert invocation.request.vae_tiling.temporal_overlap_frames == 24


def test_audio_onset_policy_resolves_from_cli_toml_and_set(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text('[output]\naudio_onset_trim = "off"\n', encoding="utf-8")
    invocation = assemble(
        build_parser().parse_args(
            [
                "--config",
                str(config),
                "--audio-onset-trim",
                "80",
                "--set",
                'output.audio_onset_trim="120"',
                "--print-config",
            ]
        ),
        base_settings=Settings(),
    )
    assert invocation.output.audio_onset_trim == "120"


def test_cut_detection_policy_resolves_from_cli_toml_and_set(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[output]\ncut_detect_mode = "off"\ncut_detect_threshold = 0.2\n',
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(
            [
                "--config",
                str(config),
                "--cut-detect-mode",
                "simple",
                "--set",
                'output.cut_detect_mode="hist"',
                "--set",
                "output.cut_detect_threshold=0.6",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
    )
    assert invocation.output.cut_detect_mode == "hist"
    assert invocation.output.cut_detect_threshold == pytest.approx(0.6)


def test_unknown_field_is_rejected_before_model_load(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text('[generate]\npromp = "typo"\n', encoding="utf-8")
    options = build_parser().parse_args(["--config", str(config)])
    with pytest.raises(ConfigError, match="did you mean 'prompt'"):
        assemble(options, base_settings=Settings())


def test_settings_table_rejects_string_for_boolean(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text('[settings]\nquiet = "yes"\n', encoding="utf-8")
    options = build_parser().parse_args(["--config", str(config)])
    with pytest.raises(ConfigError, match=r"\[settings\]\.quiet: expected bool"):
        assemble(options, base_settings=Settings())


def test_nested_image_schema_preserves_specific_validation_error(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate.image]\npath = "frame.png"\nstrength = 2.0\n',
        encoding="utf-8",
    )
    options = build_parser().parse_args(["--config", str(config)])
    with pytest.raises(ConfigError, match="image strength must be between 0 and 1"):
        assemble(options, base_settings=Settings())


def test_invalid_geometry_and_settings_are_typed_config_errors(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        "[generate]\nwidth = 1000\n[settings]\nmlx_cache_limit_gb = -1\n",
        encoding="utf-8",
    )
    options = build_parser().parse_args(["--config", str(config)])
    with pytest.raises(ConfigError, match="mlx_cache_limit_gb"):
        assemble(options, base_settings=Settings())


def test_invalid_transformer_dtype_is_a_typed_config_error(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[model_settings]\ntransformer_dtype = "float8"\n',
        encoding="utf-8",
    )
    options = build_parser().parse_args(["--config", str(config)])
    with pytest.raises(ConfigError, match="transformer_dtype"):
        assemble(options, base_settings=Settings())


def test_stale_model_fields_in_infrastructure_settings_fail_loudly(
    tmp_path: Path,
) -> None:
    config = tmp_path / "stale.toml"
    config.write_text(
        '[settings]\ntransformer_dtype = "float16"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"\[settings\].*unknown field 'transformer_dtype'"):
        assemble(
            build_parser().parse_args(["--config", str(config), "--print-config"]),
            base_settings=Settings(),
            base_model_settings=LTX2Settings(),
        )
    with pytest.raises(ConfigError, match=r"\[settings\].*unknown field 'stream_transformer'"):
        assemble(
            build_parser().parse_args(
                ["--set", "settings.stream_transformer=true", "--print-config"]
            ),
            base_settings=Settings(),
            base_model_settings=LTX2Settings(),
        )
    with pytest.raises(ConfigError, match=r"\[output\].*unknown field 'save_latents'"):
        assemble(
            build_parser().parse_args(["--set", "output.save_latents=true", "--print-config"]),
            base_settings=Settings(),
            base_model_settings=LTX2Settings(),
        )


@pytest.mark.parametrize(
    "override",
    [
        "generate.fps=nan",
        "generate.fps=inf",
        "output.target_fps=nan",
        "output.encode_quality=nan",
        "output.cut_detect_threshold=nan",
        "settings.mlx_cache_limit_gb=nan",
    ],
)
def test_non_finite_numeric_config_is_rejected(override: str) -> None:
    options = build_parser().parse_args(["--set", override, "--print-config"])
    with pytest.raises(ConfigError, match="finite|between"):
        assemble(options, base_settings=Settings())


def test_execution_requires_prompt_but_not_an_explicit_output_path() -> None:
    invocation = assemble(
        build_parser().parse_args(["--print-config"]),
        base_settings=Settings(),
    )
    with pytest.raises(ConfigError, match="prompt or text_conditioning is required"):
        validate_for_execution(invocation)
    prompt_only = assemble(
        build_parser().parse_args(["--prompt", "test"]),
        base_settings=Settings(),
    )
    validate_for_execution(prompt_only)


def test_custom_vae_default_temporal_geometry_is_validated_at_assembly() -> None:
    options = build_parser().parse_args(
        [
            "--prompt",
            "test",
            "--vae-tiling",
            "custom",
            "--vae-temporal-overlap-frames",
            "256",
        ]
    )
    with pytest.raises(ConfigError, match="overlap must be smaller"):
        assemble(options, base_settings=Settings())


def test_output_flags_resolve_exact_or_timestamped_paths(tmp_path: Path) -> None:
    exact = tmp_path / "exact.mp4"
    exact.write_bytes(b"existing")
    exact_invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--output",
                str(exact),
                "--output-dir",
                str(tmp_path / "ignored"),
                "--output-prefix",
                "ignored",
            ]
        ),
        base_settings=Settings(),
    )
    with pytest.raises(ConfigError, match="output.*already exists"):
        resolve_for_execution(exact_invocation)
    assert exact.read_bytes() == b"existing"

    exact.unlink()
    resolved_exact = resolve_for_execution(exact_invocation)
    assert resolved_exact.output.path == exact
    assert not resolved_exact.generated_output
    assert not exact.exists()

    generated_invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--output-dir",
                str(tmp_path),
                "--output-prefix",
                "kitten test!",
            ]
        ),
        base_settings=Settings(),
    )
    resolved = resolve_for_execution(
        generated_invocation,
        now=datetime(2026, 8, 17, 19, 4, 5),
    )
    expected = tmp_path / "kitten_test_20260817_190405.mp4"
    assert resolved.output.path == expected
    assert resolved.generated_output
    assert expected.read_bytes() == b""
    assert resolved.resolved_config["output"]["path"] == expected

    collision = resolve_for_execution(
        generated_invocation,
        now=datetime(2026, 8, 17, 19, 4, 5),
    )
    second = tmp_path / "kitten_test_20260817_190405_2.mp4"
    assert collision.output.path == second
    assert collision.generated_output
    assert second.read_bytes() == b""
    assert collision.resolved_config["output"]["path"] == second


def test_vae_frame_dump_never_reuses_an_existing_path(tmp_path: Path) -> None:
    exact = tmp_path / "exact.mp4"
    (tmp_path / "exact_vae_frames").mkdir()
    exact_invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--output",
                str(exact),
                "--save-vae-frames",
            ]
        ),
        base_settings=Settings(),
    )
    with pytest.raises(ConfigError, match="would overwrite existing path"):
        resolve_for_execution(exact_invocation)

    base_frames = tmp_path / "result_20260817_190405_vae_frames"
    base_frames.mkdir()
    generated_invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--output-dir",
                str(tmp_path),
                "--output-prefix",
                "result",
                "--save-vae-frames",
            ]
        ),
        base_settings=Settings(),
    )
    resolved = resolve_for_execution(
        generated_invocation,
        now=datetime(2026, 8, 17, 19, 4, 5),
    )
    assert resolved.output.path == tmp_path / "result_20260817_190405_2.mp4"


def test_hdr_heic_frame_dump_never_reuses_an_existing_path(tmp_path: Path) -> None:
    exact = tmp_path / "exact.mp4"
    (tmp_path / "exact_heic").mkdir()
    exact_invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--hdr",
                "ACESCG",
                "--output",
                str(exact),
                "--save-hdr-heic-frames",
            ]
        ),
        base_settings=Settings(),
    )
    with pytest.raises(ConfigError, match="would overwrite existing path"):
        resolve_for_execution(exact_invocation)

    (tmp_path / "result_20260817_190405_heic").mkdir()
    generated_invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--hdr",
                "ACESCG",
                "--output-dir",
                str(tmp_path),
                "--output-prefix",
                "result",
                "--save-hdr-heic-frames",
            ]
        ),
        base_settings=Settings(),
    )
    resolved = resolve_for_execution(
        generated_invocation,
        now=datetime(2026, 8, 17, 19, 4, 5),
    )
    assert resolved.output.path == tmp_path / "result_20260817_190405_2.mp4"


def test_timestamped_output_defaults_match_cli_contract(monkeypatch) -> None:
    monkeypatch.delenv("KINO_OUTPUT_DIR", raising=False)
    invocation = assemble(
        build_parser().parse_args(["--prompt", "test"]),
        base_settings=Settings(),
    )
    assert invocation.output.directory == Path("outputs")
    assert invocation.output.prefix == "kinomlx"
    assert build_timestamped_output_path(
        invocation.output.directory,
        invocation.output.prefix,
        now=datetime(2026, 8, 17, 1, 2, 3),
    ) == Path("outputs/kinomlx_20260817_010203.mp4")


def test_concurrent_timestamped_output_reservations_are_unique(tmp_path: Path) -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--prompt",
                "test",
                "--output-dir",
                str(tmp_path),
                "--output-prefix",
                "parallel",
            ]
        ),
        base_settings=Settings(),
    )
    timestamp = datetime(2026, 8, 17, 19, 4, 5)

    with ThreadPoolExecutor(max_workers=4) as executor:
        resolved = tuple(
            executor.map(
                lambda _index: resolve_for_execution(invocation, now=timestamp),
                range(4),
            )
        )

    paths = {item.output.path for item in resolved}
    assert paths == {
        tmp_path / "parallel_20260817_190405.mp4",
        tmp_path / "parallel_20260817_190405_2.mp4",
        tmp_path / "parallel_20260817_190405_3.mp4",
        tmp_path / "parallel_20260817_190405_4.mp4",
    }
    assert all(item.generated_output for item in resolved)
    assert all(path.read_bytes() == b"" for path in paths)


def test_output_directory_environment_and_cli_precedence(monkeypatch) -> None:
    monkeypatch.setenv("KINO_OUTPUT_DIR", "/from/kino")
    from_environment = assemble(
        build_parser().parse_args(["--print-config"]),
        base_settings=Settings(),
    )
    assert from_environment.output.directory == Path("/from/kino")

    from_cli = assemble(
        build_parser().parse_args(["--output-dir", "/from/cli", "--print-config"]),
        base_settings=Settings(),
    )
    assert from_cli.output.directory == Path("/from/cli")


def test_higher_precedence_prefix_replaces_config_exact_path(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        '[generate]\nprompt = "test"\n[output]\npath = "from-config.mp4"\n',
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(
            [
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path),
                "--output-prefix",
                "from-cli",
            ]
        ),
        base_settings=Settings(),
    )
    resolved = resolve_for_execution(
        invocation,
        now=datetime(2026, 8, 17, 19, 30, 0),
    )
    assert resolved.output.path == tmp_path / "from-cli_20260817_193000.mp4"


def test_timestamped_output_directory_and_prefix_round_trip_through_toml(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run.toml"
    directory = tmp_path / "renders"
    config.write_text(
        "[generate]\n"
        'prompt = "test"\n'
        "[output]\n"
        f"directory = {json.dumps(str(directory))}\n"
        'prefix = "clipname"\n',
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(["--config", str(config)]),
        base_settings=Settings(),
    )
    resolved = resolve_for_execution(
        invocation,
        now=datetime(2026, 8, 17, 20, 0, 0),
    )
    assert resolved.output.path == directory / "clipname_20260817_200000.mp4"


def test_parser_does_not_accept_abbreviated_flags() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--prom", "test"])


def test_lora_cli_arrays_and_prompt_padding_land_in_pipeline_config() -> None:
    options = build_parser().parse_args(
        [
            "--prompt",
            "test",
            "--lora",
            "style.safetensors",
            "--lora",
            "motion.safetensors",
            "--lora-strength",
            "0.5",
            "--lora-stage1-strength",
            "0.25",
            "--lora-stage2-strength",
            "1.0",
            "--lora-exclude",
            "audio,cross",
            "--compact-prompt",
            "--print-config",
        ]
    )
    invocation = assemble(options, base_settings=Settings())
    assert invocation.request.lora_paths == (
        Path("style.safetensors"),
        Path("motion.safetensors"),
    )
    assert invocation.request.lora_strengths == (0.5,)
    assert invocation.request.lora_stage1_strengths == (0.25,)
    assert invocation.request.lora_stage2_strengths == (1.0,)
    assert invocation.request.lora_exclusions == ("audio,cross",)
    assert invocation.request.pad_prompt_to_max is False


@pytest.mark.parametrize(
    "flag",
    ["--stage2-lora-fuse-mode", "--lora-stage2-fuse-mode"],
)
def test_removed_stage_transition_cli_aliases_are_rejected(flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, "delta", "--print-config"])


def test_removed_stage_transition_toml_and_set_fields_are_unknown(tmp_path: Path) -> None:
    config = tmp_path / "stale.toml"
    config.write_text(
        '[generate]\nprompt = "test"\nstage2_lora_fuse_mode = "delta"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown field 'stage2_lora_fuse_mode'"):
        assemble(
            build_parser().parse_args(["--config", str(config), "--print-config"]),
            base_settings=Settings(),
        )
    with pytest.raises(ConfigError, match="unknown field 'stage2_lora_fuse_mode'"):
        assemble(
            build_parser().parse_args(
                ["--set", 'generate.stage2_lora_fuse_mode="delta"', "--print-config"]
            ),
            base_settings=Settings(),
        )


def test_toml_cache_arrays_are_strictly_typed(tmp_path: Path) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        """
[model_settings]
video_ff_layout_specs = ["project_in:pretranspose", "project_out:pretranspose"]
video_ff_layout_layers = [0, 7, 47]
audio_layout_mirror = false
audio_ff_layout_specs = []
""".strip(),
        encoding="utf-8",
    )
    invocation = assemble(
        build_parser().parse_args(["--config", str(config), "--print-config"]),
        base_settings=Settings(),
    )
    assert invocation.model_settings.video_ff_layout_layers == (0, 7, 47)
    assert invocation.model_settings.audio_ff_layout_specs == ()
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[model_settings]\nvideo_ff_layout_layers = "0,7,47"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="expected array"):
        assemble(
            build_parser().parse_args(["--config", str(bad), "--print-config"]),
            base_settings=Settings(),
        )


def test_lora_streaming_conflict_is_rejected_during_assembly() -> None:
    options = build_parser().parse_args(
        [
            "--lora",
            "adapter.safetensors",
            "--transformer-resident-blocks",
            "4",
            "--print-config",
        ]
    )
    with pytest.raises(ConfigError, match="block streaming"):
        assemble(options, base_settings=Settings())


def test_stream_transformer_expands_to_r16_g4_and_preserves_overrides() -> None:
    preset = assemble(
        build_parser().parse_args(["--stream-transformer", "--print-config"]),
        base_settings=Settings(),
    ).model_settings
    assert preset.stream_transformer is True
    assert preset.transformer_resident_blocks == 16
    assert preset.transformer_compile_group_size == 4

    explicit = assemble(
        build_parser().parse_args(
            [
                "--stream-transformer",
                "--transformer-resident-blocks",
                "8",
                "--transformer-compile-group-size",
                "2",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
    ).model_settings
    assert explicit.transformer_resident_blocks == 8
    assert explicit.transformer_compile_group_size == 2


def test_cli_can_disable_environment_streaming_preset() -> None:
    invocation = assemble(
        build_parser().parse_args(["--no-stream-transformer", "--print-config"]),
        base_settings=Settings(),
        base_model_settings=LTX2Settings(stream_transformer=True),
    )
    assert invocation.model_settings.stream_transformer is False
    assert invocation.model_settings.transformer_resident_blocks is None


def test_save_all_sidecars_expands_every_output_category() -> None:
    options = build_parser().parse_args(["--save-all-sidecars", "--print-config"])
    invocation = assemble(options, base_settings=Settings())
    output = invocation.output
    assert output.save_all_sidecars
    assert requested_artifacts(invocation.model_artifacts, save_all=True) == {
        STAGE_1_LATENTS,
        FINAL_LATENTS,
        TEXT_CONDITIONING,
    }
    assert output.save_run_log
    assert output.save_console_log
    assert output.save_effective_config
    assert output.save_audio_sidecar
    assert output.vsr_save_original
    assert output.save_vae_frames is False
    assert output.save_hdr_heic_frames is False


def test_save_all_sidecars_includes_both_encoded_media_conditioning_stages() -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--image",
                "condition.png",
                "--save-all-sidecars",
                "--print-config",
            ]
        ),
        base_settings=Settings(),
    )

    assert requested_artifacts(
        invocation.model_artifacts,
        save_all=True,
        has_media_conditioning=True,
    ) == {
        STAGE_1_LATENTS,
        FINAL_LATENTS,
        TEXT_CONDITIONING,
        STAGE_1_CONDITIONING,
        STAGE_2_CONDITIONING,
    }


def test_explicit_media_conditioning_sidecar_requires_a_media_condition() -> None:
    invocation = assemble(
        build_parser().parse_args(
            ["--prompt", "test", "--save-media-conditioning", "--print-config"]
        ),
        base_settings=Settings(),
    )

    with pytest.raises(ConfigError, match="requires --image or --hdr-reference"):
        validate_for_execution(invocation)


def test_save_vae_frames_is_a_separate_large_output() -> None:
    options = build_parser().parse_args(["--save-vae-frames", "--print-config"])
    output = assemble(options, base_settings=Settings()).output
    assert output.save_vae_frames
    assert output.save_all_sidecars is False


def test_hdr_heic_frames_are_a_separate_large_output() -> None:
    output = assemble(
        build_parser().parse_args(["--save-hdr-heic-frames", "--print-config"]),
        base_settings=Settings(),
    ).output
    assert output.save_hdr_heic_frames
    assert output.save_all_sidecars is False


def test_save_all_sidecars_preserves_cli_category_opt_outs() -> None:
    options = build_parser().parse_args(
        [
            "--save-all-sidecars",
            "--image",
            "condition.png",
            "--no-save-text-conditioning",
            "--no-save-media-conditioning",
            "--no-save-effective-config",
            "--no-vsr-save-original",
            "--print-config",
        ]
    )
    invocation = assemble(options, base_settings=Settings())
    output = invocation.output
    assert output.save_all_sidecars
    assert requested_artifacts(
        invocation.model_artifacts,
        save_all=True,
        has_media_conditioning=True,
    ) == {
        STAGE_1_LATENTS,
        FINAL_LATENTS,
    }
    assert output.save_run_log
    assert output.save_effective_config is False
    assert output.vsr_save_original is False


def test_set_override_can_compose_down_save_all_sidecars() -> None:
    options = build_parser().parse_args(
        [
            "--save-all-sidecars",
            "--set",
            "model_artifacts.save_latents=false",
            "--print-config",
        ]
    )
    invocation = assemble(options, base_settings=Settings())
    assert invocation.output.save_all_sidecars
    assert requested_artifacts(invocation.model_artifacts, save_all=True) == {TEXT_CONDITIONING}


def test_set_save_all_expands_its_own_precedence_layer() -> None:
    options = build_parser().parse_args(
        [
            "--set",
            "output.save_all_sidecars=true",
            "--set",
            "output.save_console_log=false",
            "--print-config",
        ]
    )
    output = assemble(options, base_settings=Settings()).output
    assert output.save_all_sidecars
    assert output.save_console_log is False


def test_sidecar_cli_aliases_share_model_artifact_config() -> None:
    options = build_parser().parse_args(
        [
            "--save-text-embeddings",
            "--save-metadata",
            "--save-debug-sidecars",
            "--print-config",
        ]
    )
    invocation = assemble(options, base_settings=Settings())
    assert invocation.model_artifacts.save_text_conditioning
    assert invocation.output.save_run_log
    assert invocation.output.save_all_sidecars
