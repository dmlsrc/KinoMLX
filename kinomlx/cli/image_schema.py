"""Common image output schema; tensor meaning remains model-owned."""

from collections.abc import Mapping
from pathlib import Path

from kinomlx.config import CliBinding, ConfigGroup, ConfigTableSpec
from kinomlx.config import ConfigFieldSpec as F
from kinomlx.config.layer_rules import OUTPUT_SELECTION
from kinomlx.config.stages import IrrelevantAction, OutputPolicy, StageMode, StagePolicy
from kinomlx.config_bindings import CliAction, argument_value
from kinomlx.config_surfaces import CONFIG_SURFACES
from kinomlx.io.output_config import ImageOutputConfig
from kinomlx.samplers.sigmas import parse_sigmas

SIGMAS_CLI = CliBinding("Generation", parse=argument_value(parse_sigmas, "sigmas"))
SOURCE_IMAGE_CLI = CliBinding("Image editing", flags=("--source-image",), action=CliAction.APPEND)
SOURCE_ASPECT_CLI = CliBinding("Image editing", nargs="?", const="long")
RESTART_RUN_CLI = CliBinding("Restart", flags=("--restart",))
RESTART_STAGE_CLI = CliBinding("Restart", flags=("--restart-from",))


def _mutable(stages: tuple[str, ...], reason: str) -> tuple[StagePolicy, ...]:
    return tuple(StagePolicy(stage, StageMode.MUTABLE, reason=reason) for stage in stages)


def _artifact(stages: tuple[str, ...], availability: OutputPolicy) -> tuple[StagePolicy, ...]:
    return tuple(
        StagePolicy(
            stage,
            StageMode.MUTABLE if stage in availability.produced_at else StageMode.IRRELEVANT,
            reason=(
                "This artifact is produced by the selected restart stage."
                if stage in availability.produced_at
                else "This artifact is not produced here; explicit requests remain invalid."
            ),
            irrelevant_action=None
            if stage in availability.produced_at
            else IrrelevantAction.IGNORE,
        )
        for stage in stages
    )


def image_output_table(
    model: str,
    *,
    noise_help: str,
    latents_help: str,
    conditioning_help: str,
    output_policies: Mapping[str, OutputPolicy],
    stages: tuple[str, ...],
) -> ConfigTableSpec:
    if not stages:
        raise ValueError("image output table needs its model's restart stages")
    return ConfigTableSpec(
        path=("output",),
        cli_generated=True,
        record=ImageOutputConfig,
        stages=stages,
        stage_record=ImageOutputConfig,
        fields=(
            F(
                persistence="intent",
                name="path",
                layer_transforms=(OUTPUT_SELECTION,),
                purpose="Set the single-image filename or multi-image naming base, including .png/.tif/.tiff",
                valid="a PNG or TIFF path",
                example=Path("image.png"),
                cli_dest="output_path",
                environment_key="output_path",
                cli=CliBinding("Output", flags=("--output",)),
                surfaces=CONFIG_SURFACES,
                stage_policies=_mutable(
                    stages, "A restart writes a new output bundle at the requested path."
                ),
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
                stage_policies=_mutable(
                    stages, "A restart may place its new output bundle in another directory."
                ),
            ),
            F(
                persistence="intent",
                name="prefix",
                purpose="Set the filename prefix before the run timestamp",
                valid="one filename-safe component",
                example="image-draft",
                cli_dest="output_prefix",
                environment_key="output_prefix",
                cli=CliBinding("Output"),
                default_text=f"{model}-<variant>-seed<seed>",
                surfaces=CONFIG_SURFACES,
                stage_policies=_mutable(
                    stages,
                    "A restart may name its new output bundle independently of the source run.",
                ),
            ),
            F(
                persistence="intent",
                name="save_initial_noise",
                output_policy=output_policies["save_initial_noise"],
                purpose=noise_help,
                valid="true or false",
                example=False,
                cli_dest="save_initial_noise",
                cli=CliBinding("Output", action=CliAction.BOOLEAN),
                default_text="unset, which inherits the save-all setting",
                groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                surfaces=CONFIG_SURFACES,
                stage_policies=_artifact(stages, output_policies["save_initial_noise"]),
            ),
            F(
                persistence="intent",
                name="save_final_latents",
                output_policy=output_policies["save_final_latents"],
                purpose=latents_help,
                valid="true or false",
                example=False,
                cli_dest="save_final_latents",
                cli=CliBinding("Output", action=CliAction.BOOLEAN),
                default_text="unset, which inherits the save-all setting",
                groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                surfaces=CONFIG_SURFACES,
                stage_policies=_artifact(stages, output_policies["save_final_latents"]),
            ),
            F(
                persistence="intent",
                name="save_text_conditioning",
                output_policy=output_policies["save_text_conditioning"],
                purpose=conditioning_help,
                valid="true or false",
                example=False,
                cli_dest="save_text_conditioning",
                cli=CliBinding(
                    "Output",
                    flags=("--save-text-conditioning", "--save-text-embeddings"),
                    action=CliAction.BOOLEAN,
                ),
                default_text="unset, which inherits the save-all setting",
                groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                surfaces=CONFIG_SURFACES,
                stage_policies=_artifact(stages, output_policies["save_text_conditioning"]),
            ),
            F(
                persistence="intent",
                name="save_run_log",
                purpose="Save resolved generation metadata and timing as JSON",
                valid="true or false",
                example=False,
                cli_dest="save_run_log",
                cli=CliBinding("Output", action=CliAction.BOOLEAN),
                default_text="unset, which inherits the save-all setting",
                groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                surfaces=CONFIG_SURFACES,
                stage_policies=_mutable(stages, "The new run may save its own execution record."),
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
                stage_policies=_mutable(stages, "The new run may save its own console log."),
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
                stage_policies=_mutable(
                    stages, "The new run may save its own effective configuration."
                ),
            ),
            F(
                persistence="intent",
                name="save_all_sidecars",
                purpose="Enable every applicable image tensor and execution sidecar",
                valid="true or false",
                example=False,
                cli_dest="save_all_sidecars",
                cli=CliBinding("Output", action=CliAction.BOOLEAN),
                surfaces=CONFIG_SURFACES,
                stage_policies=_mutable(
                    stages, "The new run may select its available sidecars as a group."
                ),
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
                stage_policies=_mutable(
                    stages,
                    "Replacement applies only to the new output bundle, with source protection retained.",
                ),
            ),
        ),
    )
