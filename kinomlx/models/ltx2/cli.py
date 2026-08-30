"""Data-only LTX-2 CLI arguments; importing this module does not load MLX."""

from __future__ import annotations

import argparse
from pathlib import Path

from kinomlx.config import ModelConfigSpec
from kinomlx.settings import EnvironmentSettings, add_settings_argparse_args


def add_arguments(
    parser: argparse.ArgumentParser,
    settings_type: type[EnvironmentSettings],
    config_schema: ModelConfigSpec,
) -> None:
    """Add distilled LTX-2 generation arguments to ``parser``."""
    help_for = config_schema.cli_help
    group = parser.add_argument_group("LTX-2 generation")
    group.add_argument("--prompt", default=None, help=help_for("prompt"))
    group.add_argument("--width", type=int, default=None, help=help_for("width"))
    group.add_argument("--height", type=int, default=None, help=help_for("height"))
    group.add_argument("--frames", type=int, default=None, help=help_for("frames"))
    group.add_argument(
        "--auto-duration",
        action="store_true",
        default=None,
        help=help_for("auto_duration"),
    )
    group.add_argument(
        "--duration",
        type=float,
        default=None,
        help=help_for("duration"),
    )
    group.add_argument("--fps", type=float, default=None, help=help_for("fps"))
    group.add_argument("--seed", type=int, default=None, help=help_for("seed"))
    group.add_argument(
        "--noise-backend",
        choices=config_schema.field(("generate", "noise_backend")).choices,
        default=None,
        help=help_for("noise_backend"),
    )
    group.add_argument(
        "--sampler",
        choices=config_schema.field(("generate", "sampler")).choices,
        default=None,
        help=help_for("sampler"),
    )
    group.add_argument(
        "--hdr",
        choices=config_schema.field(("generate", "hdr")).choices,
        default=None,
        help=help_for("hdr"),
    )
    group.add_argument(
        "--generate-audio",
        dest="generate_audio",
        action="store_true",
        default=None,
        help=help_for("generate_audio"),
    )
    group.add_argument(
        "--no-generate-audio",
        dest="generate_audio",
        action="store_false",
        default=None,
        help=help_for("generate_audio", negative=True),
    )
    group.add_argument(
        "--reference-aligned-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("reference_aligned_audio"),
    )
    group.add_argument(
        "--image",
        type=Path,
        default=None,
        help=help_for("image"),
    )
    group.add_argument(
        "--image-frame-index",
        type=int,
        default=None,
        help=help_for("image_frame_index"),
    )
    group.add_argument(
        "--image-strength",
        type=float,
        default=None,
        help=help_for("image_strength"),
    )
    group.add_argument(
        "--hdr-reference",
        type=Path,
        default=None,
        help=help_for("hdr_reference"),
    )
    group.add_argument(
        "--hdr-reference-strength",
        type=float,
        default=None,
        help=help_for("hdr_reference_strength"),
    )
    group.add_argument(
        "--generated-keyframes",
        type=int,
        default=None,
        help=help_for("generated_keyframes"),
    )
    group.add_argument(
        "--text-conditioning",
        type=Path,
        default=None,
        help=help_for("text_conditioning"),
    )
    group.add_argument(
        "--vae-decode-dtype",
        choices=config_schema.field(("generate", "vae_decode_dtype")).choices,
        default=None,
        help=help_for("vae_decode_dtype"),
    )
    group.add_argument(
        "--vae-tiling",
        choices=config_schema.field(("generate", "vae_tiling", "mode")).choices,
        default=None,
        help=help_for("vae_tiling"),
    )
    group.add_argument(
        "--vae-temporal-tile-frames",
        type=int,
        default=None,
        help=help_for("vae_temporal_tile_frames"),
    )
    group.add_argument(
        "--vae-temporal-overlap-frames",
        type=int,
        default=None,
        help=help_for("vae_temporal_overlap_frames"),
    )
    group.add_argument(
        "--vae-spatial-tile-pixels",
        type=int,
        default=None,
        help=help_for("vae_spatial_tile_pixels"),
    )
    group.add_argument(
        "--vae-spatial-overlap-pixels",
        type=int,
        default=None,
        help=help_for("vae_spatial_overlap_pixels"),
    )
    group.add_argument(
        "--reference-prompt-padding",
        dest="pad_prompt_to_max",
        action="store_true",
        default=None,
        help=help_for("pad_prompt_to_max"),
    )
    group.add_argument(
        "--compact-prompt",
        dest="pad_prompt_to_max",
        action="store_false",
        default=None,
        help=help_for("pad_prompt_to_max", negative=True),
    )
    group.add_argument(
        "--lora",
        dest="lora_paths",
        type=Path,
        action="append",
        default=None,
        help=help_for("lora_paths"),
    )
    group.add_argument(
        "--lora-strength",
        dest="lora_strengths",
        type=float,
        action="append",
        default=None,
        help=help_for("lora_strengths"),
    )
    group.add_argument(
        "--lora-stage1-strength",
        "--lora-strength-stage-1",
        dest="lora_stage1_strengths",
        type=float,
        action="append",
        default=None,
        help=help_for("lora_stage1_strengths"),
    )
    group.add_argument(
        "--lora-stage2-strength",
        "--lora-strength-stage-2",
        dest="lora_stage2_strengths",
        type=float,
        action="append",
        default=None,
        help=help_for("lora_stage2_strengths"),
    )
    group.add_argument(
        "--lora-exclude",
        dest="lora_exclusions",
        action="append",
        default=None,
        help=help_for("lora_exclusions"),
    )
    artifacts = parser.add_argument_group("LTX-2 artifacts")
    artifacts.add_argument(
        "--video-vae",
        choices=config_schema.field(("model_settings", "video_vae")).choices,
        default=None,
        help=help_for("video_vae"),
    )
    artifacts.add_argument(
        "--save-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_latents"),
    )
    artifacts.add_argument(
        "--save-text-conditioning",
        "--save-text-embeddings",
        dest="save_text_conditioning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_text_conditioning"),
    )
    artifacts.add_argument(
        "--save-media-conditioning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_for("save_media_conditioning"),
    )
    add_settings_argparse_args(
        parser,
        settings_type,
        title="LTX-2 runtime settings (override env vars)",
        skip={"video_vae"},
        choices_by_field=config_schema.table_choices(("model_settings",)),
        help_by_field=config_schema.table_help(("model_settings",)),
        negative_help_by_field=config_schema.table_help(
            ("model_settings",),
            negative=True,
        ),
    )


__all__ = ["add_arguments"]
