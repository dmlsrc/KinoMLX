"""Data-only GMNet contribution to the global configuration registry."""

from pathlib import Path

from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.cli.config_schema import settings_table
from kinomlx.cli.resource_operation import operation_table
from kinomlx.config import (
    CliBinding,
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigTableSpec,
    ModelConfigSpec,
)
from kinomlx.config.layer_rules import OUTPUT_SELECTION
from kinomlx.config_bindings import CliAction
from kinomlx.config_constraints import field_choices
from kinomlx.config_surfaces import CONFIG_SURFACES, MODEL_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.models.gmnet.catalog import GMNetVariant
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.models.gmnet.types import GMNetExpandConfig, GMNetOutputConfig
from kinomlx.settings import Settings

from .pipelines import expand_gmnet
from .resources import prepare_resources
from .runner import API_BOUNDARIES as RUNNER_API_BOUNDARIES

F = ConfigFieldSpec


DIRECT_API_BOUNDARIES = (
    APIBoundary(
        "gmnet",
        prepare_resources,
        (
            configuration("settings", GMNetSettings, "model_settings"),
            configuration("infrastructure", Settings, "settings"),
            runtime("reporter", "progress reporter"),
        ),
    ),
    APIBoundary(
        "gmnet",
        expand_gmnet,
        (
            configuration("request", GMNetExpandConfig, "expand", projection=True),
            runtime("resources", "prepared GMNet checkpoint and memory policy"),
            runtime("components", "component provider"),
            runtime("reporter", "progress reporter"),
        ),
    ),
)


CONFIG_SCHEMA = ModelConfigSpec(
    model="gmnet",
    title="KinoMLX GMNet starter configuration",
    stages=("expand",),
    restart_stages=(),
    no_restart_reason="GMNet exposes one still-expansion pass and has no saved restart stage.",
    api_boundaries=(*RUNNER_API_BOUNDARIES, *DIRECT_API_BOUNDARIES),
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
            persistence="intent",
            name="model",
            cli=MODEL_CONTROL.binding,
            purpose="Select the invocation schema",
            valid="",
            choices=("gmnet",),
            example="gmnet",
            starter="gmnet",
            cli_dest="model",
            virtual=True,
            default="gmnet",
            surfaces=MODEL_SURFACES,
        ),
    ),
    tables=(
        ConfigTableSpec(
            path=("expand",),
            cli_generated=True,
            record=GMNetExpandConfig,
            default_present=False,
            fields=(
                F(
                    persistence="intent",
                    name="image",
                    purpose="Set the display-referred SDR still to expand",
                    valid="a readable PNG, JPEG, HEIC, or TIFF path",
                    example=Path("input.png"),
                    starter=Path("input.png"),
                    cli_dest="image",
                    cli=CliBinding("GMNet input"),
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        operation_table(cli_group="GMNet execution", stages=()),
        ConfigTableSpec(
            path=("output",),
            cli_generated=True,
            record=GMNetOutputConfig,
            fields=(
                F(
                    persistence="intent",
                    name="path",
                    layer_transforms=(OUTPUT_SELECTION,),
                    purpose="Set the exact primary output artifact",
                    valid="an EXR or HEIC path",
                    example=Path("expanded.exr"),
                    cli_dest="output_path",
                    environment_key="output_path",
                    cli=CliBinding("Output", flags=("--output",)),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="directory",
                    purpose="Set the directory for timestamped output",
                    valid="a directory path",
                    example=Path("outputs"),
                    cli_dest="output_dir",
                    environment_key="output_dir",
                    cli=CliBinding("Output"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="prefix",
                    purpose="Set the filename prefix before the run timestamp",
                    valid="a filename-safe string",
                    example="expanded",
                    cli_dest="output_prefix",
                    environment_key="output_prefix",
                    cli=CliBinding("Output"),
                    default_text="the input image stem",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="exr",
                    purpose="Write a half-float scene-linear Rec.709 EXR",
                    valid="true or false",
                    example=True,
                    cli_dest="exr",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="selected from the exact path suffix, or true",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="heic",
                    purpose="Write a user-viewable 10-bit BT.2100 PQ HEIC",
                    valid="true or false",
                    example=True,
                    cli_dest="heic",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="selected from the exact path suffix, or true",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="save_gain_map",
                    purpose="Save the normalized gain map as safetensors",
                    valid="true or false",
                    example=False,
                    cli_dest="save_gain_map",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="save_run_log",
                    purpose="Save the resolved invocation and timing as JSON",
                    valid="true or false",
                    example=False,
                    cli_dest="save_run_log",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="save_console_log",
                    purpose="Save a human-readable execution log",
                    valid="true or false",
                    example=False,
                    cli_dest="save_console_log",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="save_effective_config",
                    purpose="Save the effective invocation as a TOML sidecar",
                    valid="true or false",
                    example=False,
                    cli_dest="save_effective_config",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="save_all_sidecars",
                    purpose="Enable every applicable GMNet sidecar",
                    valid="true or false",
                    example=False,
                    cli_dest="save_all_sidecars",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="force",
                    purpose="Replace the existing bundle selected by an exact output path",
                    valid="true or false",
                    example=False,
                    cli_dest="force",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        settings_table(stages=(), adapter_weight_check_stages=frozenset()),
        ConfigTableSpec(
            path=("model_settings",),
            cli_generated=True,
            cli_group="GMNet settings (override env vars)",
            record=GMNetSettings,
            fields=(
                F(
                    persistence="intent",
                    name="variant",
                    purpose="Select the published GMNet behavior",
                    valid="",
                    choices=field_choices(GMNetSettings, "variant"),
                    choice_help=(
                        ("realworld", "Use the photographed-pair checkpoint"),
                        ("synthetic", "Use the HDR-video-frame checkpoint"),
                    ),
                    example=GMNetVariant.REALWORLD.value,
                    cli_dest="variant",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    persistence="intent",
                    name="weights_path",
                    purpose="Override the converted GMNet weights",
                    valid="a compatible safetensors path",
                    example=Path("gmnet_realworld.safetensors"),
                    cli_dest="weights_path",
                    surfaces=CONFIG_SURFACES,
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
