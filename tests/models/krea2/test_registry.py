"""Krea 2 command registry and configuration surface."""

import subprocess
import sys
from pathlib import Path

import pytest

from kinomlx.config import ConfigError
from kinomlx.model_catalog import model_choices
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.settings import Settings


def test_krea2_is_a_complete_model_config_contribution(tmp_path: Path) -> None:
    assert "krea2" in model_choices()
    schema = config_registry().model("krea2")
    parser = build_parser()
    options = parser.parse_args(
        [
            "--model",
            "krea2",
            "--variant",
            "raw",
            "--source-path",
            str(tmp_path / "source"),
            "--transformer-path",
            str(tmp_path / "transformer.safetensors"),
            "--set",
            'model_settings.vae_path="from-set/vae"',
            "--print-config",
        ]
    )
    invocation = assemble(
        options,
        base_settings=Settings(
            hf_home=tmp_path / "hf",
        ),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.model_settings.variant == "raw"
    assert invocation.model_settings.source_path == tmp_path / "source"
    assert invocation.model_settings.transformer_path == tmp_path / "transformer.safetensors"
    assert invocation.model_settings.vae_path == Path("from-set/vae")
    assert set(schema.cli_destinations()) == {
        action.dest
        for action in parser._actions
        if action.dest
        not in {
            "config",
            "help",
            "only_non_defaults",
            "print_config",
            "save_config",
            "set_overrides",
        }
    }


def test_krea2_variant_defaults_and_native_noise_are_resolved_as_data() -> None:
    turbo = assemble(
        build_parser().parse_args(["--model", "krea2", "--prompt", "test"]),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )
    raw = assemble(
        build_parser().parse_args(["--model", "krea2", "--variant", "raw", "--prompt", "test"]),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert turbo.request is not None
    assert (turbo.request.steps, turbo.request.guidance, turbo.request.mu) == (
        8,
        0.0,
        1.15,
    )
    assert turbo.request.noise_backend == "mlx"
    assert raw.request is not None
    assert (raw.request.steps, raw.request.guidance, raw.request.mu) == (52, 3.5, None)
    assert turbo.resolved_config["generate"] == {
        "turbo_4step": False,
        "turbo_4step_exclusions": "",
        "turbo_4step_strength": 1.0,
        "turbo_4step_step_strengths": (),
        "turbo_4step_sigma_window": "",
        "turbo_sda": False,
        "turbo_sda_exclusions": "",
        "turbo_sda_strength": 1.0,
        "turbo_sda_step_strengths": (),
        "turbo_sda_sigma_window": "",
        "prompt": ("test",),
        "prompt_unconditional": False,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "seed_mode": "sequential",
        "steps": 8,
        "guidance": 0.0,
        "mu": 1.15,
        "images_per_prompt": 1,
        "noise_backend": "mlx",
        "vae_tiling": "auto",
        "vae_tile_size": 256,
        "vae_tile_stride": 192,
        "adapters": (),
        "sigmas": (),
        "source_images": (),
        "edit_adapter_strength": 1.0,
        "edit_adapter_exclusions": "",
        "reference_fit": "fit",
        "grounding_px": 768,
        "grounding_system_prompt": "",
        "reference_boost_regions": (),
        "reference_strength": 1.0,
        "scene_reference_strength": 1.0,
    }
    assert raw.resolved_config["generate"] == {
        "turbo_4step": False,
        "turbo_4step_exclusions": "",
        "turbo_4step_strength": 1.0,
        "turbo_4step_step_strengths": (),
        "turbo_4step_sigma_window": "",
        "turbo_sda": False,
        "turbo_sda_exclusions": "",
        "turbo_sda_strength": 1.0,
        "turbo_sda_step_strengths": (),
        "turbo_sda_sigma_window": "",
        "prompt": ("test",),
        "prompt_unconditional": False,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "seed_mode": "sequential",
        "steps": 52,
        "guidance": 3.5,
        "images_per_prompt": 1,
        "noise_backend": "mlx",
        "vae_tiling": "auto",
        "vae_tile_size": 256,
        "vae_tile_stride": 192,
        "adapters": (),
        "sigmas": (),
        "source_images": (),
        "edit_adapter_strength": 1.0,
        "edit_adapter_exclusions": "",
        "reference_fit": "fit",
        "grounding_px": 768,
        "grounding_system_prompt": "",
        "reference_boost_regions": (),
        "reference_strength": 1.0,
        "scene_reference_strength": 1.0,
    }


def test_krea2_repeatable_prompts_overrides_and_save_all_share_one_schema() -> None:
    invocation = assemble(
        build_parser().parse_args(
            [
                "--model",
                "krea2",
                "--prompt",
                "one",
                "--prompt",
                "two",
                "--negative-prompt",
                "none",
                "--images-per-prompt",
                "2",
                "--steps",
                "3",
                "--guidance",
                "1.25",
                "--mu",
                "0.75",
                "--noise-backend",
                "torch-mps",
                "--save-all-sidecars",
                "--no-save-console-log",
            ]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.request is not None
    assert invocation.request.prompts == ("one", "two")
    assert invocation.request.expanded_negative_prompts() == (
        "none",
        "none",
        "none",
        "none",
    )
    assert (
        invocation.request.steps,
        invocation.request.guidance,
        invocation.request.mu,
        invocation.request.noise_backend,
    ) == (3, 1.25, 0.75, "torch-mps")
    assert invocation.output.save_initial_noise is True
    assert invocation.output.save_final_latents is True
    assert invocation.output.save_text_conditioning is True
    assert invocation.output.save_run_log is True
    assert invocation.output.save_console_log is False
    assert invocation.output.save_effective_config is True


@pytest.mark.parametrize("mode", ["auto", "single", "custom"])
def test_krea2_explicit_vae_modes_share_cli_and_toml_values(mode) -> None:
    invocation = assemble(
        build_parser().parse_args(["--model", "krea2", "--prompt", "test", "--vae-tiling", mode]),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )
    assert invocation.request is not None
    assert invocation.request.vae_tiling == mode
    assert invocation.resolved_config["generate"]["vae_tiling"] == mode


def test_krea2_vae_decoder_selector_round_trips_cli_and_config() -> None:
    invocation = assemble(
        build_parser().parse_args(
            ["--model", "krea2", "--prompt", "test", "--vae-decoder", "upscale2x"]
        ),
        base_settings=Settings(),
        base_model_settings=Krea2Settings(),
    )

    assert invocation.model_settings.vae_decoder == "upscale2x"
    assert invocation.resolved_config["model_settings"]["vae_decoder"] == "upscale2x"


def test_reference_denoising_round_trips_config_and_cli_overrides(tmp_path: Path) -> None:
    config = tmp_path / "reference.toml"
    config.write_text("[model_settings]\nreference_denoise = true\n", encoding="utf-8")
    parser = build_parser()
    for arguments, expected in (
        (["--reference-denoise"], True),
        (["--config", str(config)], True),
        (["--config", str(config), "--no-reference-denoise"], False),
        (["--no-reference-denoise", "--set", "model_settings.reference_denoise=true"], True),
    ):
        invocation = assemble(
            parser.parse_args(["--model", "krea2", "--prompt", "test", *arguments]),
            base_settings=Settings(
                hf_home=tmp_path / "hf",
            ),
            base_model_settings=Krea2Settings(),
        )
        assert invocation.model_settings.reference_denoise is expected
        assert invocation.resolved_config["model_settings"]["reference_denoise"] is expected


def test_cli_rejects_reference_fp16_before_missing_source_resolution(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="padding can cause FP16 overflow"):
        assemble(
            build_parser().parse_args(
                [
                    "--model",
                    "krea2",
                    "--prompt",
                    "test",
                    "--reference-denoise",
                    "--transformer-dtype",
                    "float16",
                    "--source-path",
                    str(tmp_path / "missing"),
                ]
            ),
            base_settings=Settings(
                hf_home=tmp_path / "hf",
            ),
            base_model_settings=Krea2Settings(),
        )
    assert not (tmp_path / "cache").exists()


def test_krea2_help_and_registry_avoid_optional_external_dependencies() -> None:
    probe = r"""
import sys

BLOCKED = ("tokenizers", "transformers", "diffusers", "torch", "PIL")

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED):
            raise ImportError(f"blocked runtime dependency: {name}")
        return None

sys.meta_path.insert(0, Blocker())
from kinomlx.model_registry import config_registry
from kinomlx.models.krea2.arguments import build_parser

help_text = build_parser().format_help()
assert "Krea 2" in help_text
assert "--adapter, --lora" in help_text
assert "--adapter-strength, --lora-strength" in help_text
assert "--adapter-exclude, --lora-exclude" in help_text
assert "krea2" in config_registry().models()
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in BLOCKED
)
print("dependency-minimal krea2 help ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "dependency-minimal krea2 help ok" in result.stdout


def test_missing_snapshot_guidance_avoids_optional_external_dependencies() -> None:
    probe = r"""
import sys
import tempfile
from pathlib import Path

BLOCKED = ("tokenizers", "transformers", "diffusers", "torch", "PIL")

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED):
            raise ImportError(f"blocked runtime dependency: {name}")
        return None

sys.meta_path.insert(0, Blocker())
from kinomlx.models.krea2.resources import prepare_resources
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.settings import Settings

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    try:
        prepare_resources(
            Krea2Settings(),
            infrastructure=Settings(hf_home=root / "hf"),
        )
    except FileNotFoundError as exc:
        assert "hf download krea/Krea-2-Turbo" in str(exc)
    else:
        raise AssertionError("missing Krea 2 cache unexpectedly resolved")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in BLOCKED
)
print("missing-source guidance stayed dependency-minimal")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "missing-source guidance stayed dependency-minimal" in result.stdout
