"""Data-only GMNet contribution to the global configuration registry."""

from __future__ import annotations

from pathlib import Path

from kinomlx.cli.config_schema import settings_table
from kinomlx.config import (
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigTableSpec,
    ModelConfigSpec,
)
from kinomlx.models.gmnet.catalog import GMNetVariant
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.models.gmnet.types import GMNetExpandConfig, GMNetOutputConfig

F = ConfigFieldSpec


CONFIG_SCHEMA = ModelConfigSpec(
    model="gmnet",
    title="KinoMLX GMNet starter configuration",
    introduction=(
        "The input image below is the only non-default starter value.",
        (
            "Every other setting is commented out so KinoMLX can use its built-in "
            "default or an environment variable. Uncomment only the settings you "
            "want to override."
        ),
    ),
    root_fields=(
        F(
            name="model",
            purpose="Select the invocation schema",
            valid="",
            choices=("gmnet",),
            example="gmnet",
            starter="gmnet",
            cli_dest="model",
            virtual=True,
            default="gmnet",
        ),
    ),
    tables=(
        ConfigTableSpec(
            path=("expand",),
            record=GMNetExpandConfig,
            default_present=False,
            fields=(
                F(
                    name="image",
                    purpose="Set the display-referred SDR still to expand",
                    valid="a readable PNG, JPEG, HEIC, or TIFF path",
                    example=Path("input.png"),
                    starter=Path("input.png"),
                    cli_dest="image",
                ),
            ),
        ),
        ConfigTableSpec(
            path=("output",),
            record=GMNetOutputConfig,
            fields=(
                F(
                    name="path",
                    purpose="Set the exact primary output artifact",
                    valid="an EXR or HEIC path",
                    example=Path("expanded.exr"),
                    cli_dest="output_path",
                ),
                F(
                    name="directory",
                    purpose="Set the output directory",
                    valid="a directory path",
                    example=Path("outputs"),
                    cli_dest="output_dir",
                    default_text='KINO_OUTPUT_DIR, then "outputs"',
                ),
                F(
                    name="prefix",
                    purpose="Set the output filename prefix",
                    valid="a filename-safe string",
                    example="expanded",
                    cli_dest="output_prefix",
                    default_text="the input image stem",
                ),
                F(
                    name="exr",
                    purpose="Write a half-float scene-linear Rec.709 EXR",
                    valid="true or false",
                    example=True,
                    cli_dest="exr",
                    default_text="selected from the exact path suffix, or true",
                ),
                F(
                    name="heic",
                    purpose="Write a user-viewable 10-bit BT.2100 PQ HEIC",
                    valid="true or false",
                    example=True,
                    cli_dest="heic",
                    default_text="selected from the exact path suffix, or true",
                ),
                F(
                    name="save_gain_map",
                    purpose="Save the normalized gain map as safetensors",
                    valid="true or false",
                    example=False,
                    cli_dest="save_gain_map",
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                ),
                F(
                    name="save_run_log",
                    purpose="Save the resolved invocation and timing as JSON",
                    valid="true or false",
                    example=False,
                    cli_dest="save_run_log",
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                ),
                F(
                    name="save_console_log",
                    purpose="Save a human-readable execution log",
                    valid="true or false",
                    example=False,
                    cli_dest="save_console_log",
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                ),
                F(
                    name="save_effective_config",
                    purpose="Save the effective invocation as a TOML sidecar",
                    valid="true or false",
                    example=False,
                    cli_dest="save_effective_config",
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                ),
                F(
                    name="save_all_sidecars",
                    purpose="Enable every applicable GMNet sidecar",
                    valid="true or false",
                    example=False,
                    cli_dest="save_all_sidecars",
                ),
                F(
                    name="force",
                    purpose="Replace a complete existing artifact bundle",
                    valid="true or false",
                    example=False,
                    cli_dest="force",
                ),
            ),
        ),
        settings_table(),
        ConfigTableSpec(
            path=("model_settings",),
            record=GMNetSettings,
            fields=(
                F(
                    name="variant",
                    purpose="Select the published GMNet behavior",
                    valid="",
                    choices=tuple(item.value for item in GMNetVariant),
                    choice_help=(
                        ("realworld", "Use the photographed-pair checkpoint"),
                        ("synthetic", "Use the HDR-video-frame checkpoint"),
                    ),
                    example=GMNetVariant.REALWORLD.value,
                    cli_dest="variant",
                ),
                F(
                    name="weights_path",
                    purpose="Override the converted GMNet weights",
                    valid="a compatible safetensors path",
                    example=Path("gmnet_realworld.safetensors"),
                    cli_dest="weights_path",
                ),
            ),
        ),
    ),
    implications=(
        ConfigImplication(
            trigger=("output", "save_all_sidecars"),
            trigger_value=True,
            implied=(
                (("output", "save_gain_map"), True),
                (("output", "save_run_log"), True),
                (("output", "save_console_log"), True),
                (("output", "save_effective_config"), True),
            ),
            expand_in_layers=True,
            coverage_group=ConfigGroup.SAVE_ALL_CANDIDATE,
        ),
    ),
)


__all__ = ["CONFIG_SCHEMA"]
