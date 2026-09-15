"""Klein resource-discovery contribution to the model-owned configuration registry."""

from dataclasses import dataclass, replace
from pathlib import Path

from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.cli.config_schema import settings_table
from kinomlx.cli.image_schema import (
    RESTART_PHASE_CLI,
    RESTART_RUN_CLI,
    SIGMAS_CLI,
    SOURCE_ASPECT_CLI,
    SOURCE_IMAGE_CLI,
    image_output_table,
)
from kinomlx.cli.prompt_arguments import prompt_unconditional_field
from kinomlx.cli.seed_arguments import seed_fields
from kinomlx.config import (
    CliBinding,
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigTableSpec,
    ModelConfigSpec,
)
from kinomlx.config.source_rules import PROMPT_FILE
from kinomlx.config.stations import StationPolicy
from kinomlx.config_constraints import field_choices
from kinomlx.config_records import config_record
from kinomlx.config_surfaces import CONFIG_SURFACES, MODEL_SURFACES, SOURCE_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.settings import Settings

from .adapters.config_schema import ADAPTERS
from .output_policies import OUTPUT_POLICIES
from .pipeline import generate_flux2, restart_flux2
from .resources import inspect_resources
from .runner import API_BOUNDARIES as RUNNER_API_BOUNDARIES
from .settings import Flux2Settings
from .types import Flux2Request, Flux2RestartConfig

_RESTART_STATIONS = field_choices(Flux2RestartConfig, "phase")


def _locked(reason: str) -> tuple[StationPolicy, ...]:
    return tuple(StationPolicy(station, "locked", reason=reason) for station in _RESTART_STATIONS)


def _denoise(reason: str) -> tuple[StationPolicy, ...]:
    return (
        StationPolicy("denoise", "mutable", reason=reason),
        StationPolicy("decode", "locked", reason=reason),
    )


def _mutable(reason: str) -> tuple[StationPolicy, ...]:
    return tuple(StationPolicy(station, "mutable", reason=reason) for station in _RESTART_STATIONS)


def _recorded_resource(reason: str) -> tuple[StationPolicy, ...]:
    return tuple(
        StationPolicy(station, "irrelevant", reason=reason, irrelevant_action="reject")
        for station in _RESTART_STATIONS
    )


DIRECT_API_BOUNDARIES = (
    APIBoundary(
        "flux2",
        inspect_resources,
        (
            configuration("settings", Flux2Settings, "model_settings"),
            configuration("infrastructure", Settings, "settings"),
        ),
    ),
    APIBoundary(
        "flux2",
        generate_flux2,
        (
            configuration("request", Flux2Request, "generate"),
            runtime("resources", "prepared Klein components and memory policy"),
            runtime("reporter", "progress reporter"),
            runtime("conditioning", "retained prepared conditioning"),
            runtime("on_image", "image callback"),
        ),
    ),
    APIBoundary(
        "flux2",
        restart_flux2,
        (
            configuration("request", Flux2Request, "generate"),
            runtime("resources", "recorded station components and memory policy"),
            runtime("restart", "prepared station handoff"),
            runtime("reporter", "progress reporter"),
            runtime("conditioning", "retained prepared conditioning"),
        ),
    ),
)

_CHOICE_HELP = {
    "mlx": "Use native MLX normal noise",
    "torch-mps": "Match the PyTorch MPS normal-noise stream; a PyTorch CPU generator differs",
    "fit": "Fit the whole source into the output grid with centered reference positions",
    "crop": "Fill the output grid with a centered source crop",
    "long": "Pin the longer source axis",
    "short": "Pin the shorter source axis",
    "width": "Pin output width",
    "height": "Pin output height",
    "sequential": "Evaluate positive and negative branches separately to limit memory",
    "batched": "Evaluate both CFG branches together with higher activation memory",
}


@config_record
@dataclass(frozen=True)
class Flux2Operation:
    dry_run: bool = False


CONFIG_SCHEMA = ModelConfigSpec(
    model="flux2",
    title="KinoMLX FLUX.2 Klein image generation and restart",
    introduction=("Generate Klein base, distilled, and KV still images from local resources.",),
    restart_stations=_RESTART_STATIONS,
    api_boundaries=(*RUNNER_API_BOUNDARIES, *DIRECT_API_BOUNDARIES),
    root_fields=(
        ConfigFieldSpec(
            persistence="intent",
            name="model",
            cli=MODEL_CONTROL.binding,
            purpose="Select the invocation schema",
            valid="",
            choices=("flux2",),
            example="flux2",
            starter="flux2",
            cli_dest="model",
            virtual=True,
            default="flux2",
            surfaces=MODEL_SURFACES,
        ),
    ),
    tables=(
        ConfigTableSpec(
            path=("generate",),
            cli_generated=True,
            record=Flux2Request,
            stations=_RESTART_STATIONS,
            station_record=Flux2Request,
            fields=(
                *(
                    replace(
                        field,
                        station_policies={
                            "seed": _denoise(
                                "Denoise can draw new noise; decode consumes the saved latent image."
                            ),
                            "seeds": _locked(
                                "A restart selects one recorded image, not a new seed collection."
                            ),
                            "seed_mode": _locked(
                                "The recorded single-image station has no new multi-seed execution mode."
                            ),
                        }[field.name],
                    )
                    for field in seed_fields(image=True)
                ),
                replace(
                    prompt_unconditional_field(),
                    station_policies=_locked(
                        "Saved conditioning fixes whether the selected prompt was unconditional."
                    ),
                ),
                replace(
                    ADAPTERS,
                    station_policies=_denoise(
                        "Denoise can apply a new adapter stack; decode consumes final latents."
                    ),
                ),
                ConfigFieldSpec(
                    persistence="source",
                    name="prompt_file",
                    purpose="Load one complete UTF-8 prompt",
                    valid="a nonempty UTF-8 file, exclusive with prompt in the same layer",
                    example=Path("prompt.txt"),
                    cli_dest="prompt_file",
                    input_annotation=Path | None,
                    layer_transforms=(PROMPT_FILE,),
                    cli=CliBinding("Generation", parse=Path),
                    virtual=True,
                    default=None,
                    surfaces=SOURCE_SURFACES,
                ),
                ConfigFieldSpec(
                    persistence="intent",
                    name="prompt",
                    station_policies=_locked(
                        "The saved conditioning fixes the selected prompt row."
                    ),
                    purpose="Set one prompt or an ordered prompt array",
                    valid="nonempty text",
                    example="A cinematic portrait in afternoon light",
                    # The generated template prints a command that runs it, so
                    # Klein activates the one generation input that command needs.
                    starter="A cinematic portrait in afternoon light",
                    cli_dest="prompt",
                    cli=CliBinding("Generation", action="append"),
                    surfaces=CONFIG_SURFACES,
                ),
                *(
                    ConfigFieldSpec(
                        persistence="intent",
                        cli=cli,
                        name=name,
                        station_policies=station_policies,
                        purpose=purpose,
                        valid=valid,
                        example=example,
                        cli_dest=name,
                        choices=choices,
                        choice_help=tuple((choice, _CHOICE_HELP[choice]) for choice in choices),
                        surfaces=CONFIG_SURFACES,
                    )
                    for name, station_policies, purpose, valid, example, choices, cli in (
                        (
                            "negative_prompt",
                            _locked(
                                "The saved conditioning fixes the selected negative prompt row."
                            ),
                            "Set base CFG negative text",
                            "one string or one per prompt; base with guidance >1 only",
                            "blurry",
                            (),
                            CliBinding("Generation", action="append"),
                        ),
                        (
                            "width",
                            _locked(
                                "The saved conditioning and latents fix the output grid width."
                            ),
                            "Set output width before optional source-aspect matching",
                            "a multiple of 16 >=64",
                            1024,
                            (),
                            CliBinding("Generation"),
                        ),
                        (
                            "height",
                            _locked(
                                "The saved conditioning and latents fix the output grid height."
                            ),
                            "Set output height before optional source-aspect matching",
                            "a multiple of 16 >=64",
                            1024,
                            (),
                            CliBinding("Generation"),
                        ),
                        (
                            "recompute_reference_kv",
                            (
                                StationPolicy(
                                    "denoise",
                                    "mutable",
                                    reason="Denoise reruns the sampler and its reference extraction.",
                                ),
                                StationPolicy(
                                    "decode",
                                    "irrelevant",
                                    reason="Decode does not sample.",
                                    irrelevant_action="ignore",
                                ),
                            ),
                            "Extract the reference K/V again at every denoise step",
                            "boolean; KV variant only; costs one extra reference forward per step and "
                            "is required when a per-step adapter schedule or sigma window edits with "
                            "references",
                            True,
                            (),
                            CliBinding("Generation", action="boolean"),
                        ),
                        (
                            "steps",
                            _denoise(
                                "Denoise can choose a new recipe length; decode uses saved latents."
                            ),
                            "Override the selected profile's denoise steps",
                            "a positive integer",
                            4,
                            (),
                            CliBinding("Generation"),
                        ),
                        (
                            "guidance",
                            _denoise(
                                "Denoise can choose fresh CFG guidance; decode uses saved latents."
                            ),
                            "Set external CFG for base",
                            "at least 1 for base; exactly 1 for distilled and KV",
                            4.0,
                            (),
                            CliBinding("Generation"),
                        ),
                        (
                            "images_per_prompt",
                            _locked("A saved-station restart resumes exactly one selected image."),
                            "Generate sequential images for each prompt",
                            "a positive integer",
                            1,
                            (),
                            CliBinding("Generation"),
                        ),
                        (
                            "noise_backend",
                            _denoise(
                                "Denoise can choose a new noise stream; decode uses saved latents."
                            ),
                            "Select the initial-noise implementation",
                            "",
                            "mlx",
                            field_choices(Flux2Request, "noise_backend"),
                            CliBinding("Generation"),
                        ),
                        (
                            "sigmas",
                            _denoise(
                                "Denoise can choose a new schedule; decode uses saved latents."
                            ),
                            "Set the denoise schedule explicitly",
                            "numbers between 0 and 1, strictly decreasing, one more than the steps they drive",
                            (1.0, 0.75, 0.5, 0.0),
                            (),
                            SIGMAS_CLI,
                        ),
                        (
                            "text_conditioning",
                            _denoise(
                                "Denoise can select compatible saved conditioning; decode uses latents."
                            ),
                            "Reuse complete saved text and reference conditioning",
                            "a Klein conditioning safetensors artifact",
                            Path("previous_text.safetensors"),
                            (),
                            CliBinding("Generation"),
                        ),
                        (
                            "source_images",
                            _locked(
                                "Saved reference conditioning already contains the source image context."
                            ),
                            "Select ordered reference images",
                            "local image paths",
                            (Path("source.png"),),
                            (),
                            SOURCE_IMAGE_CLI,
                        ),
                        (
                            "reference_fit",
                            _locked(
                                "Saved reference geometry fixes source fitting before this station."
                            ),
                            "Set reference fitting against the output grid",
                            "",
                            "fit",
                            field_choices(Flux2Request, "reference_fit"),
                            CliBinding("Image editing"),
                        ),
                        (
                            "match_source_aspect",
                            _locked("Saved reference geometry fixes the selected output aspect."),
                            "Set output aspect from the first source image",
                            "",
                            "long",
                            field_choices(Flux2Request, "match_source_aspect"),
                            SOURCE_ASPECT_CLI,
                        ),
                        (
                            "cfg_mode",
                            _denoise(
                                "Denoise can choose CFG branch execution; decode uses saved latents."
                            ),
                            "Choose base CFG branch execution",
                            "",
                            "sequential",
                            field_choices(Flux2Request, "cfg_mode"),
                            CliBinding("Klein execution"),
                        ),
                    )
                ),
            ),
        ),
        image_output_table(
            "flux2",
            output_policies=OUTPUT_POLICIES,
            stations=_RESTART_STATIONS,
            noise_help="Save packed initial noise and the Euler schedule",
            latents_help="Save packed final VAE latents and their output grid",
            conditioning_help="Save complete text, negative text, and ordered reference conditioning",
        ),
        ConfigTableSpec(
            path=("restart",),
            cli_generated=True,
            record=Flux2RestartConfig,
            stations=_RESTART_STATIONS,
            station_record=Flux2RestartConfig,
            default_present=False,
            fields=(
                ConfigFieldSpec(
                    persistence="intent",
                    name="run",
                    purpose="Load a completed Klein run record",
                    valid="a Klein run JSON path",
                    example=Path("previous_run.json"),
                    cli_dest="restart_run",
                    environment_key="restart_run",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=RESTART_RUN_CLI,
                    notes=("Omit the whole [restart] table to start a fresh run.",),
                    surfaces=CONFIG_SURFACES,
                    station_policies=_mutable(
                        "This selector identifies the completed source run for the new restart."
                    ),
                ),
                ConfigFieldSpec(
                    persistence="intent",
                    name="phase",
                    purpose="Select the first station to repeat",
                    valid="",
                    choices=field_choices(Flux2RestartConfig, "phase"),
                    choice_help=(
                        ("decode", "Decode saved final latents"),
                        (
                            "denoise",
                            "Denoise from complete saved conditioning with fresh reference KV",
                        ),
                    ),
                    example="decode",
                    cli_dest="restart_phase",
                    environment_key="restart_phase",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=RESTART_PHASE_CLI,
                    surfaces=CONFIG_SURFACES,
                    station_policies=_mutable(
                        "This selector chooses which saved station the new invocation repeats."
                    ),
                ),
                ConfigFieldSpec(
                    persistence="intent",
                    name="latents",
                    purpose="Substitute compatible final latents for decode",
                    valid="a Klein final-latent artifact",
                    example=Path("final_latents.safetensors"),
                    cli_dest="restart_latents",
                    environment_key="restart_latents",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=CliBinding("Restart"),
                    default_text="the final latents artifact recorded in the run log",
                    surfaces=CONFIG_SURFACES,
                    station_policies=_mutable(
                        "This optional selector chooses compatible final latents for decode."
                    ),
                ),
                ConfigFieldSpec(
                    persistence="intent",
                    name="index",
                    purpose="Select which image of a multi-image source run to resume",
                    valid="a one-based output index within the source run's image count",
                    example=2,
                    cli_dest="restart_index",
                    environment_key="restart_index",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=CliBinding("Restart"),
                    default_text="the only image, which a single-image source run implies",
                    surfaces=CONFIG_SURFACES,
                    station_policies=_mutable(
                        "This selector chooses one recorded image from a multi-image source run."
                    ),
                ),
            ),
        ),
        settings_table(
            stations=_RESTART_STATIONS,
            adapter_weight_check_stations=frozenset({"denoise"}),
        ),
        ConfigTableSpec(
            path=("model_settings",),
            cli_generated=True,
            cli_group="Klein settings (override env vars)",
            record=Flux2Settings,
            stations=_RESTART_STATIONS,
            station_record=Flux2Settings,
            fields=(
                ConfigFieldSpec(
                    persistence="intent",
                    name="variant",
                    purpose="Select the Klein checkpoint profile",
                    valid="",
                    choices=field_choices(Flux2Settings, "variant"),
                    choice_help=(
                        ("base", "Undistilled model with CFG"),
                        ("distilled", "Four-step distilled model"),
                        ("kv", "Distilled model with generation-local reference KV caching"),
                    ),
                    example="distilled",
                    cli_dest="variant",
                    station_policies=_locked(
                        "The checkpoint variant determines the recorded resource profile and recipe."
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                ConfigFieldSpec(
                    persistence="intent",
                    name="transformer_dtype",
                    purpose="Select the prepared transformer weight and execution dtype",
                    valid="",
                    choices=field_choices(Flux2Settings, "transformer_dtype"),
                    choice_help=(
                        ("bfloat16", "Preserve the bundled BF16 transformer precision"),
                        ("float16", "Opt into FP16 execution for potential speed improvements"),
                    ),
                    example="bfloat16",
                    cli_dest="transformer_dtype",
                    station_policies=_recorded_resource(
                        "Replay reopens the recorded component dtype; use fresh generation to change it."
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                *(
                    ConfigFieldSpec(
                        persistence="intent",
                        name=name,
                        purpose="Set the model source directory"
                        if name == "source_path"
                        else f"Set the {name.removesuffix('_path').replace('_', ' ')} source",
                        valid="an existing local source path",
                        example=Path("local") / name.removesuffix("_path"),
                        cli_dest=name,
                        station_policies=_recorded_resource(reason),
                        surfaces=CONFIG_SURFACES,
                    )
                    for name, reason in (
                        (
                            "source_path",
                            "Replay reopens recorded component roots; use fresh generation to change the source.",
                        ),
                        (
                            "tokenizer_path",
                            "Replay uses recorded conditioning or latents, not a new tokenizer; use fresh generation to change it.",
                        ),
                        (
                            "text_encoder_path",
                            "Replay uses recorded conditioning or latents, not a new text encoder; use fresh generation to change it.",
                        ),
                        (
                            "transformer_path",
                            "Replay reopens the recorded transformer receipt for denoise; use fresh generation to change it.",
                        ),
                        (
                            "vae_path",
                            "Replay reopens the recorded VAE receipt; use fresh generation to change it.",
                        ),
                    )
                ),
            ),
        ),
        ConfigTableSpec(
            path=("operation",),
            cli_generated=True,
            record=Flux2Operation,
            stations=_RESTART_STATIONS,
            station_record=Flux2Operation,
            fields=(
                ConfigFieldSpec(
                    persistence="intent",
                    name="dry_run",
                    purpose="Describe validated local resources",
                    valid="true or false",
                    example=True,
                    cli_dest="dry_run",
                    cli=CliBinding("Klein execution", action="store_true"),
                    surfaces=CONFIG_SURFACES,
                    station_policies=_locked(
                        "A saved-station restart executes a generation stage, so resource-only dry-run stays false."
                    ),
                ),
            ),
        ),
    ),
    implications=(
        ConfigImplication(
            trigger=("output", "save_all_sidecars"),
            trigger_value=True,
            implied=tuple(
                (("output", name), True)
                for name in (
                    "save_initial_noise",
                    "save_final_latents",
                    "save_text_conditioning",
                    "save_run_log",
                    "save_console_log",
                    "save_effective_config",
                )
            ),
            expand_in_layers=True,
            coverage_group=ConfigGroup.SAVE_ALL_CANDIDATE,
        ),
    ),
)
