"""Composable, abbreviation-free KinoMLX argument parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from kinomlx.settings import add_argparse_args

from ._registry import (
    add_model_arguments,
    config_registry,
    modal_model_summary,
    model_choices,
    validate_model_parser,
)
from .common import add_invocation_arguments


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser without importing a model runtime."""
    config_schema = config_registry().model("ltx2")
    parser = argparse.ArgumentParser(
        prog="kinomlx",
        description="Generate distilled LTX-2 video on Apple Silicon (--model ltx2).",
        epilog=modal_model_summary(),
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help=config_schema.cli_help("model"),
    )
    help_for = config_schema.cli_help

    restart = parser.add_argument_group("Restart")
    restart.add_argument(
        "--restart",
        dest="restart_run",
        type=Path,
        default=None,
        metavar="RUN_JSON",
        help=help_for("restart_run"),
    )
    restart.add_argument(
        "--restart-from",
        dest="restart_phase",
        choices=config_schema.field(("restart", "phase")).choices,
        default=None,
        help=help_for("restart_phase"),
    )
    restart.add_argument(
        "--latent-stage",
        choices=config_schema.field(("restart", "latent_stage")).choices,
        default=None,
        help=help_for("latent_stage"),
    )
    restart.add_argument(
        "--restart-latents",
        type=Path,
        default=None,
        metavar="SAFETENSORS",
        help=help_for("restart_latents"),
    )

    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help=help_for("output_path"),
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=help_for("output_dir"),
    )
    output.add_argument(
        "--output-prefix",
        default=None,
        help=help_for("output_prefix"),
    )
    output.add_argument(
        "--vsr-spatial-mode",
        choices=config_schema.field(("output", "vsr_spatial_mode")).choices,
        default=None,
        help=help_for("vsr_spatial_mode"),
    )
    output.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=help_for("target_fps"),
    )
    output.add_argument(
        "--vsr-temporal-mode",
        choices=config_schema.field(("output", "vsr_temporal_mode")).choices,
        default=None,
        help=help_for("vsr_temporal_mode"),
    )
    output.add_argument(
        "--cut-detect-mode",
        choices=config_schema.field(("output", "cut_detect_mode")).choices,
        default=None,
        help=help_for("cut_detect_mode"),
    )
    output.add_argument(
        "--cut-detect-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=help_for("cut_detect_threshold"),
    )
    output.add_argument(
        "--vsr-save-original",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("vsr_save_original"),
    )
    output.add_argument(
        "--encode-quality",
        type=float,
        default=None,
        help=help_for("encode_quality"),
    )
    output.add_argument(
        "--audio-codec",
        choices=config_schema.field(("output", "audio_codec")).choices,
        default=None,
        help=help_for("audio_codec"),
    )
    output.add_argument(
        "--audio-onset-trim",
        default=None,
        metavar="AUTO|OFF|MS",
        help=help_for("audio_onset_trim"),
    )
    output.add_argument(
        "--save-run-log",
        "--save-metadata",
        dest="save_run_log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_run_log"),
    )
    output.add_argument(
        "--save-console-log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_console_log"),
    )
    output.add_argument(
        "--save-effective-config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_effective_config"),
    )
    output.add_argument(
        "--save-audio-sidecar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_audio_sidecar"),
    )
    output.add_argument(
        "--save-vae-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_vae_frames"),
    )
    output.add_argument(
        "--save-hdr-heic-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_hdr_heic_frames"),
    )
    output.add_argument(
        "--save-all-sidecars",
        "--save-debug-sidecars",
        dest="save_all_sidecars",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=config_schema.implication_help(("output", "save_all_sidecars")),
    )

    add_argparse_args(
        parser,
        choices_by_field=config_schema.table_choices(("settings",)),
        help_by_field=config_schema.table_help(("settings",)),
        negative_help_by_field=config_schema.table_help(
            ("settings",),
            negative=True,
        ),
    )
    add_model_arguments(parser)
    validate_model_parser(parser, "ltx2")
    return parser


def build_root_parser() -> argparse.ArgumentParser:
    """Build the compact model-neutral discovery surface."""
    parser = argparse.ArgumentParser(
        prog="kinomlx",
        description="Native multimodal MLX inference on Apple Silicon.",
        epilog=(
            "Models:\n"
            "  ltx2   distilled LTX-2.3/LTX-2.5 text/image-to-video\n"
            "  gmnet  SDR-to-HDR still expansion (EXR / PQ HEIC)\n\n"
            "Utilities:\n"
            "  config init              write an annotated model-specific config\n"
            "  weights convert          generic checkpoint re-serialization\n"
            "  weights convert gmnet    validated GMNet conversion\n\n"
            "Use 'kinomlx --model <name> --help' for model-specific arguments, "
            "'kinomlx config --help' for config utilities, or "
            "'kinomlx weights --help' for conversion help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help="Model family (default execution model: ltx2).",
    )
    return parser


__all__ = ["build_parser", "build_root_parser"]
