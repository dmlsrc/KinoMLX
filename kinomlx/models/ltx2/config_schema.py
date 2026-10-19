"""Data-only LTX-2 contribution to the global configuration registry."""

from dataclasses import replace
from pathlib import Path

from kinomlx.api_boundaries import APIBoundary, configuration, runtime
from kinomlx.cli.config_schema import settings_table
from kinomlx.cli.prompt_arguments import prompt_unconditional_field
from kinomlx.cli.resource_operation import operation_table
from kinomlx.cli.seed_arguments import seed_fields
from kinomlx.config import (
    CliBinding,
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigTableSpec,
    ModelConfigSpec,
)
from kinomlx.config.layer_rules import OUTPUT_SELECTION
from kinomlx.config.source_rules import PROMPT_FILE
from kinomlx.config.stages import IrrelevantAction, StageMode, StagePolicy
from kinomlx.config_bindings import CliAction, argument_value
from kinomlx.config_constraints import field_choices
from kinomlx.config_surfaces import CONFIG_SURFACES, MODEL_SURFACES, SOURCE_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.samplers.sigmas import parse_sigmas
from kinomlx.settings import Settings

from .adapters.config_schema import ADAPTERS
from .artifacts import LTX2ArtifactConfig
from .config_groups import LTX2ConfigGroup
from .image_arguments import IMAGES
from .layer_rules import (
    DURATION_SELECTION,
    DURATION_SERIALIZATION,
    VAE_TILING_EFFECTIVE,
    VAE_TILING_INPUT,
    VAE_TILING_SELECTION,
    source_selection_rule,
)
from .output_config import (
    OutputConfig,
)
from .output_policies import OUTPUT_POLICIES
from .pipelines.distilled import generate_ltx2
from .pipelines.restart import restart_ltx2
from .resources import prepare_resources
from .restart import RestartConfig
from .restart_rules import RESTARTS, STAGES, LTX2Stage
from .runner import API_BOUNDARIES as RUNNER_API_BOUNDARIES
from .settings import (
    STREAM_TRANSFORMER_RESIDENT_BLOCKS,
    LTX2Settings,
)
from .sparse_attention import canonical_layer_ranges, canonical_step_schedule
from .types import (
    LTX2HDRReferenceConditioningConfig,
    LTX2Request,
    LTX2VideoVAETilingConfig,
)
from .vocabularies import GemmaVariant, LTX2ModelGeneration

F = ConfigFieldSpec
_LOCKED = (
    StagePolicy(LTX2Stage.DECODE, StageMode.LOCKED),
    StagePolicy(LTX2Stage.STAGE_2, StageMode.LOCKED),
)
_DECODE_LOCKED = (
    StagePolicy(LTX2Stage.DECODE, StageMode.LOCKED),
    StagePolicy(LTX2Stage.STAGE_2, StageMode.MUTABLE),
)
_MUTABLE = (
    StagePolicy(LTX2Stage.DECODE, StageMode.MUTABLE),
    StagePolicy(LTX2Stage.STAGE_2, StageMode.MUTABLE),
)


def _policy(
    stage_2: StageMode,
    decode: StageMode,
    *,
    stage_2_reason: str,
    decode_reason: str,
    stage_2_action: IrrelevantAction | None = None,
    decode_action: IrrelevantAction | None = None,
) -> tuple[StagePolicy, ...]:
    """Give one LTX field explicit decisions at both supported restart stages."""
    return (
        StagePolicy(
            LTX2Stage.STAGE_2, stage_2, reason=stage_2_reason, irrelevant_action=stage_2_action
        ),
        StagePolicy(
            LTX2Stage.DECODE, decode, reason=decode_reason, irrelevant_action=decode_action
        ),
    )


_MODEL_SOURCE_POLICIES = {
    "model_generation": _policy(
        StageMode.LOCKED,
        StageMode.LOCKED,
        stage_2_reason="The saved stage-1 latent fixes checkpoint generation and geometry.",
        decode_reason="The saved latent fixes checkpoint generation and geometry.",
    ),
    "weights_path": _policy(
        StageMode.MUTABLE,
        StageMode.MUTABLE,
        stage_2_reason="A compatible baseline can supply the repeated transformer and decoder.",
        decode_reason="A compatible baseline can supply the decoder; resources inspect its full inventory.",
    ),
    "gemma_variant": _policy(
        StageMode.MUTABLE,
        StageMode.LOCKED,
        stage_2_reason="Gemma discovery can select stage-2 text projection and connector identity.",
        decode_reason="Gemma discovery remains part of the inspected inventory; decode does not encode text.",
    ),
    "spatial_upscaler_path": _policy(
        StageMode.MUTABLE,
        StageMode.LOCKED,
        stage_2_reason="Stage-2 explicitly loads the spatial latent upscaler.",
        decode_reason="The saved latent needs no upscaling, but this source remains in resource inspection.",
    ),
    "transformer_path": _policy(
        StageMode.MUTABLE,
        StageMode.LOCKED,
        stage_2_reason="Stage-2 loads a transformer for its repeated denoise.",
        decode_reason="Decode skips transformer execution but resource preparation inspects this source.",
    ),
    "text_encoder_path": _policy(
        StageMode.MUTABLE,
        StageMode.LOCKED,
        stage_2_reason="The text source can supply projection or connector identity for saved text.",
        decode_reason="The source remains in the inspected inventory; decode does not encode text.",
    ),
    "video_vae": _policy(
        StageMode.MUTABLE,
        StageMode.MUTABLE,
        stage_2_reason="Stage-2 media conditions and terminal video decode use this VAE variant.",
        decode_reason="Terminal video decode uses this VAE variant.",
    ),
    "video_vae_path": _policy(
        StageMode.MUTABLE,
        StageMode.MUTABLE,
        stage_2_reason="Stage-2 media conditions and terminal video decode use this VAE source.",
        decode_reason="Terminal video decode uses this VAE source.",
    ),
    "audio_vae_path": _policy(
        StageMode.MUTABLE,
        StageMode.MUTABLE,
        stage_2_reason="Stage-2 and its terminal decode can use this audio VAE source.",
        decode_reason="Terminal audio decode can use this audio VAE source.",
    ),
    "temporal_latent_upscaler_path": _policy(
        StageMode.LOCKED,
        StageMode.LOCKED,
        stage_2_reason="The restart uses recorded geometry and this source remains in inspected capabilities.",
        decode_reason="Decode uses recorded geometry and this source remains in inspected capabilities.",
    ),
    "duration_head_path": _policy(
        StageMode.LOCKED,
        StageMode.LOCKED,
        stage_2_reason="The restart uses recorded frame count and this source remains in inspected capabilities.",
        decode_reason="Decode uses recorded frame count and this source remains in inspected capabilities.",
    ),
}

_TRANSFORMER_SETTINGS = frozenset(
    {
        "fast_mode",
        "transformer_dtype",
        "adapter_accumulation",
        "compile_attention",
        "dense_attention_kernel",
        "video_ff_layout_specs",
        "video_ff_layout_layers",
        "video_attn_layout_specs",
        "video_attn_layout_layers",
        "audio_layout_mirror",
        "audio_ff_layout_specs",
        "audio_ff_layout_layers",
        "audio_attn_layout_specs",
        "audio_attn_layout_layers",
        "adaln_pretranspose",
        "transformer_quantize",
        "transformer_quantize_exclude_layers",
        "video_ff_quantize_specs",
        "video_ff_quantize_layers",
        "video_ff_quantize_group_size",
        "video_ff_quantize_bits",
        "stream_transformer",
        "transformer_retain",
        "transformer_resident_blocks",
        "transformer_compile_group_size",
        "compile_block_group_size",
    }
)

_OUTPUT_REASONS = {
    "path": "the fresh video destination",
    "directory": "the fresh output directory",
    "prefix": "the fresh output stem",
    "vsr_spatial_mode": "post-decode spatial VSR, subject to HDR restrictions",
    "target_fps": "post-decode frame conversion, subject to HDR restrictions",
    "vsr_temporal_mode": "post-decode temporal VSR",
    "cut_detect_mode": "post-decode cut detection",
    "cut_detect_threshold": "the post-decode cut threshold",
    "vsr_save_original": "the post-decode original-video sidecar, subject to HDR restrictions",
    "encode_quality": "terminal video encoding quality",
    "audio_codec": "terminal audio encoding",
    "audio_onset_trim": "terminal audio trimming",
    "save_run_log": "the current run record",
    "save_console_log": "the current execution log",
    "save_effective_config": "the current effective-configuration sidecar",
    "save_audio_sidecar": "the current audio sidecar",
    "save_hdr_heic_frames": "HDR-only terminal HEIC frames",
    "save_vae_frames": "SDR-only terminal VAE diagnostic frames",
    "save_all_sidecars": "applicable current-stage sidecars",
}


def _artifact_policy(
    field: str,
    *,
    stage_2_reason: str,
    decode_reason: str,
) -> tuple[StagePolicy, ...]:
    """Derive one artifact field's stage modes from where its group is produced.

    A stage that produces the group may be asked for it; one that does not
    cannot, and its explicit request is ignored rather than refused as a change.
    Only the reasons are written by hand, because no synthesized sentence is
    true of every artifact group.
    """
    produced = OUTPUT_POLICIES[field].produced_at
    return tuple(
        StagePolicy(
            stage,
            StageMode.MUTABLE if stage in produced else StageMode.IRRELEVANT,
            reason=reason,
            irrelevant_action=None if stage in produced else IrrelevantAction.IGNORE,
        )
        for stage, reason in (("stage-2", stage_2_reason), ("decode", decode_reason))
    )


_ARTIFACT_POLICIES = {
    "save_latents": _artifact_policy(
        "save_latents",
        stage_2_reason="Stage-2 emits new final latents and re-emits the stage-1 latent it read.",
        decode_reason="Decode re-emits the one latent product it read, so its bundle restarts too.",
    ),
    "save_media_conditioning": _artifact_policy(
        "save_media_conditioning",
        stage_2_reason="Stage-2 can emit stage-2 media conditioning when media input exists.",
        decode_reason="Decode emits no media conditioning; artifact availability rejects explicit true.",
    ),
    "save_text_conditioning": _artifact_policy(
        "save_text_conditioning",
        stage_2_reason="Stage-2 re-emits the saved text it consumed, so its bundle restarts too.",
        decode_reason="Decode neither consumes nor creates text-conditioning artifacts.",
    ),
}

_RESTART_REASONS = {
    "run": "Select the current parent run before loading its manifest.",
    "stage": "Select the first repeated stage before saved-base assembly.",
    "latent_stage": "Select the current source latent stage; stage-2 requires stage-1.",
    "latents": "Select a substitute current-run latent artifact for input validation.",
}


def _completed_table(table: ConfigTableSpec) -> ConfigTableSpec:
    """Attach the model's exact stage decisions to its existing field owners."""
    if table.path == ("settings",):
        # The shared builder already bound this model's stages at declaration.
        return table
    fields = []
    for field in table.fields:
        if table.path == ("model_settings",):
            if field.name in _MODEL_SOURCE_POLICIES:
                policies = _MODEL_SOURCE_POLICIES[field.name]
            elif field.name in _TRANSFORMER_SETTINGS:
                policies = _policy(
                    StageMode.MUTABLE,
                    StageMode.LOCKED,
                    stage_2_reason=f"The repeated stage-2 transformer uses {field.name}.",
                    decode_reason=(
                        f"Decode skips transformer execution, but {field.name} remains part of "
                        "the inspected resource policy."
                    ),
                )
            else:
                raise ValueError(f"unclassified LTX model setting {field.name}")
        elif table.path == ("output",):
            use = _OUTPUT_REASONS.get(field.name)
            if use is None:
                raise ValueError(f"unclassified LTX output field {field.name}")
            policies = _policy(
                StageMode.MUTABLE,
                StageMode.MUTABLE,
                stage_2_reason=f"The stage-2 run selects {use} after terminal decode.",
                decode_reason=f"The decode run selects {use} after terminal decode.",
            )
        elif table.path == ("model_artifacts",):
            policies = _ARTIFACT_POLICIES[field.name]
        elif table.path == ("restart",):
            reason = _RESTART_REASONS[field.name]
            policies = _policy(
                StageMode.MUTABLE,
                StageMode.MUTABLE,
                stage_2_reason=reason,
                decode_reason=reason,
            )
        else:
            # Keep the established generation projections; supply their missing
            # independent reason so final inspection can explain each decision.
            # A field that already states its own reason keeps it, because a
            # synthesized sentence cannot be true of every generation field.
            policies = tuple(
                policy
                if policy.reason
                else replace(
                    policy,
                    reason=(
                        "The saved latent already embodies this generation or conditioning value."
                        if policy.mode == StageMode.LOCKED
                        else "The repeated stage-2 path consumes this value before terminal output."
                        if policy.stage == LTX2Stage.STAGE_2
                        else "The terminal decode path consumes this value."
                    ),
                )
                for policy in field.stage_policies
            )
        fields.append(replace(field, stage_policies=policies))
    if not table.stages:
        return replace(
            table,
            fields=tuple(fields),
            stages=RESTARTS,
            stage_record=table.record,
        )
    return replace(table, fields=tuple(fields))


def _completed_tables(*tables: ConfigTableSpec) -> tuple[ConfigTableSpec, ...]:
    """Complete every declared table so the contribution is valid when constructed."""
    return tuple(_completed_table(table) for table in tables)


CONFIG_SCHEMA = ModelConfigSpec(
    model="ltx2",
    stages=STAGES,
    restart_stages=RESTARTS,
    title="KinoMLX LTX-2 starter configuration",
    introduction=(
        "The prompt below is the only non-default starter value.",
        (
            "Every other setting is commented out so KinoMLX can use its built-in "
            "default or an environment variable. Uncomment only the settings you "
            "want to override."
        ),
    ),
    semantic_groups=frozenset(LTX2ConfigGroup),
    root_fields=(
        F(
            persistence="intent",
            name="model",
            cli=MODEL_CONTROL.binding,
            purpose="Select the invocation schema",
            valid="",
            choices=("ltx2",),
            example="ltx2",
            starter="ltx2",
            cli_dest="model",
            virtual=True,
            default="ltx2",
            surfaces=MODEL_SURFACES,
        ),
    ),
    tables=_completed_tables(
        ConfigTableSpec(
            path=("generate",),
            cli_generated=True,
            record=LTX2Request,
            stages=RESTARTS,
            stage_record=LTX2Request,
            nested=("hdr_reference", "vae_tiling"),
            fields=(
                *(
                    replace(field, stage_policies={"seed": _MUTABLE, "seeds": _MUTABLE}[field.name])
                    for field in seed_fields(image=False)
                ),
                replace(prompt_unconditional_field(), stage_policies=_LOCKED),
                F(
                    persistence="source",
                    name="prompt_file",
                    purpose="Load one complete prompt from a UTF-8 text file",
                    valid="a path to a nonempty UTF-8 file; exclusive with prompt in the same layer",
                    example=Path("prompt.txt"),
                    cli_dest="prompt_file",
                    input_annotation=Path | None,
                    layer_transforms=(PROMPT_FILE,),
                    cli=CliBinding("LTX-2 generation", parse=Path),
                    virtual=True,
                    default=None,
                    surfaces=SOURCE_SURFACES,
                ),
                F(
                    "prompt",
                    "Describe the video to generate",
                    "any string",
                    "Describe your video here.",
                    persistence="intent",
                    cli_dest="prompt",
                    cli=CliBinding("LTX-2 generation"),
                    starter="Describe your video here.",
                    multiline=True,
                    strip=True,
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "width",
                    "Set the output width in pixels",
                    "a positive multiple of 64",
                    1024,
                    persistence="intent",
                    cli_dest="width",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "height",
                    "Set the output height in pixels",
                    "a positive multiple of 64",
                    576,
                    persistence="intent",
                    cli_dest="height",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "frames",
                    "Set the output frame count",
                    "a positive integer matching 8*k + 1",
                    121,
                    persistence="intent",
                    cli_dest="frames",
                    cli=CliBinding("LTX-2 generation"),
                    notes=("Do not combine this with duration or auto_duration in the same layer",),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "duration",
                    "Set the duration instead of setting the frame count",
                    "a finite number of seconds greater than 0",
                    5.0,
                    persistence="intent",
                    cli_dest="duration",
                    cli=CliBinding("LTX-2 generation", parse=float),
                    notes=("Do not combine this with frames or auto_duration in the same layer",),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "auto_duration",
                    "Let the checkpoint duration head choose the duration",
                    "true or false",
                    True,
                    persistence="source",
                    cli_dest="auto_duration",
                    layer_transforms=(DURATION_SELECTION, DURATION_SERIALIZATION),
                    cli=CliBinding("LTX-2 generation", action=CliAction.STORE_TRUE),
                    virtual=True,
                    default=False,
                    notes=("Do not combine this with frames or duration in the same layer",),
                    surfaces=SOURCE_SURFACES,
                ),
                F(
                    "fps",
                    "Set the generation frame rate",
                    "a finite number greater than 0",
                    24.0,
                    persistence="intent",
                    cli_dest="fps",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "noise_backend",
                    "Select the normal-noise stream",
                    "",
                    "mlx",
                    persistence="intent",
                    cli_dest="noise_backend",
                    cli=CliBinding("LTX-2 generation"),
                    choices=field_choices(LTX2Request, "noise_backend"),
                    choice_help=(
                        ("mlx", "Use KinoMLX's native MLX normal-noise stream"),
                        ("torch-mps", "Match the pinned PyTorch MPS normal-noise stream"),
                    ),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_1_sampler",
                    "Select the stage-1 sampler",
                    "",
                    "auto",
                    persistence="intent",
                    cli_dest="stage_1_sampler",
                    cli=CliBinding(
                        "LTX-2 generation", flags=("--stage1-sampler", "--sampler-stage-1")
                    ),
                    choices=field_choices(LTX2Request, "stage_1_sampler"),
                    choice_help=(
                        ("auto", "Use the checkpoint generation's published stage-1 sampler"),
                        ("deterministic", "Use deterministic Euler for stage 1"),
                        ("ancestral", "Use RF ancestral Euler for stage 1"),
                    ),
                    notes=(
                        "auto is deterministic on LTX-2.3 and ancestral on LTX-2.5. Either "
                        "explicit value runs on both generations, for experiments.",
                    ),
                    stage_policies=_policy(
                        StageMode.LOCKED,
                        StageMode.LOCKED,
                        stage_2_reason="The saved stage-1 latent embodies the stage-1 sampler.",
                        decode_reason="The saved final latent embodies both stages' samplers.",
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_2_sampler",
                    "Select the stage-2 sampler",
                    "",
                    "auto",
                    persistence="intent",
                    cli_dest="stage_2_sampler",
                    cli=CliBinding(
                        "LTX-2 generation", flags=("--stage2-sampler", "--sampler-stage-2")
                    ),
                    choices=field_choices(LTX2Request, "stage_2_sampler"),
                    choice_help=(
                        ("auto", "Use the checkpoint generation's published stage-2 sampler"),
                        ("deterministic", "Use deterministic Euler for stage 2"),
                        ("ancestral", "Use RF ancestral Euler for stage 2"),
                    ),
                    notes=(
                        "auto is deterministic on LTX-2.3 and ancestral on LTX-2.5. Either "
                        "explicit value runs on both generations, for experiments.",
                    ),
                    stage_policies=_policy(
                        StageMode.MUTABLE,
                        StageMode.LOCKED,
                        stage_2_reason=(
                            "A stage-2 restart reruns stage 2 from the saved stage-1 latent."
                        ),
                        decode_reason="The saved final latent embodies both stages' samplers.",
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_1_sigmas",
                    "Set the stage-1 denoise schedule explicitly",
                    "numbers between 0 and 1, strictly decreasing, one more than the steps they drive",
                    (1.0, 0.975, 0.725, 0.0),
                    persistence="intent",
                    cli_dest="stage_1_sigmas",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--sigmas-stage-1", "--stage1-sigmas"),
                        parse=argument_value(parse_sigmas, "stage 1 sigmas"),
                    ),
                    notes=(
                        "Each stage has its own schedule; overriding one leaves the other "
                        "on its pinned distilled values.",
                    ),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_2_sigmas",
                    "Set the stage-2 denoise schedule explicitly",
                    "numbers between 0 and 1, strictly decreasing, one more than the steps they drive",
                    (0.909375, 0.725, 0.421875, 0.0),
                    persistence="intent",
                    cli_dest="stage_2_sigmas",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--sigmas-stage-2", "--stage2-sigmas"),
                        parse=argument_value(parse_sigmas, "stage 2 sigmas"),
                    ),
                    notes=(
                        "Each stage has its own schedule; overriding one leaves the other "
                        "on its pinned distilled values.",
                    ),
                    stage_policies=_policy(
                        StageMode.MUTABLE,
                        StageMode.LOCKED,
                        stage_2_reason=(
                            "A stage-2 restart reruns the stage-2 schedule from the saved "
                            "stage-1 latent."
                        ),
                        decode_reason="The saved final latent already embodies the stage-2 schedule.",
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_1_sparse_attention",
                    "Set stage-1 sparse video self-attention per step",
                    (
                        'a Sol-Attn tau or "dense" per step: "dense", "1,1.25,1.5" (one entry '
                        'per step, or one for every step), or "5-8=1.5" (steps left out stay '
                        "dense)"
                    ),
                    "5-8=1.5",
                    persistence="intent",
                    cli_dest="stage_1_sparse_attention",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--stage1-sparse-attention", "--sparse-attention-stage-1"),
                        parse=argument_value(canonical_step_schedule, "stage 1 sparse attention"),
                    ),
                    notes=(
                        "Give each step a Sol-Attn tau or dense: 1,1.25,1.5 lists every step, "
                        "1.5 sets all of them, and 5-8=1.5 sets steps 5 to 8 and leaves the "
                        "rest dense. A larger tau computes fewer key blocks exactly: faster, "
                        "and further from dense attention. Only video self-attention is made "
                        "sparse.",
                    ),
                    stage_policies=(
                        StagePolicy(
                            LTX2Stage.STAGE_2,
                            StageMode.LOCKED,
                            "sparse_attention_stage1_profile",
                            reason="The saved stage-1 latent embodies stage-1 sparse attention.",
                        ),
                        StagePolicy(
                            LTX2Stage.DECODE,
                            StageMode.LOCKED,
                            "sparse_attention_stage_profiles",
                            reason="The saved final latent embodies both stages' attention.",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_2_sparse_attention",
                    "Set stage-2 sparse video self-attention per step",
                    (
                        '"auto" for Sol-Engine\'s schedule, or a Sol-Attn tau or "dense" per '
                        'step: "dense", "1,1.25,1.5" (one entry per step, or one for every '
                        'step), or "2-3=1.5" (steps left out stay dense)'
                    ),
                    "auto",
                    persistence="intent",
                    cli_dest="stage_2_sparse_attention",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--stage2-sparse-attention", "--sparse-attention-stage-2"),
                        parse=argument_value(canonical_step_schedule, "stage 2 sparse attention"),
                    ),
                    notes=(
                        "auto takes Sol-Engine's published stage-2 schedule, tau 1, 1.25 and "
                        "1.5 on the distilled stage's three steps. Or give each step a Sol-Attn "
                        "tau or dense: 1,1.25,1.5 lists every step, 1.5 sets all of them, and "
                        "2-3=1.5 sets steps 2 and 3 and leaves the rest dense. A larger tau "
                        "computes fewer key blocks exactly: faster, and further from dense "
                        "attention. Only video self-attention is made sparse.",
                    ),
                    stage_policies=(
                        StagePolicy(
                            LTX2Stage.STAGE_2,
                            StageMode.MUTABLE,
                            reason="A stage-2 restart reruns stage 2 from the saved stage-1 latent.",
                        ),
                        StagePolicy(
                            LTX2Stage.DECODE,
                            StageMode.LOCKED,
                            "sparse_attention_stage_profiles",
                            reason="The saved final latent embodies both stages' attention.",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_1_dense_layers",
                    "Keep these layers dense on sparse stage-1 steps",
                    'transformer layer numbers and ranges from 0 to 47, like "0,46-47", or "none"',
                    "0,47",
                    persistence="intent",
                    cli_dest="stage_1_dense_layers",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--stage1-dense-layers", "--dense-layers-stage-1"),
                        parse=argument_value(canonical_layer_ranges, "stage 1 dense layers"),
                    ),
                    notes=("Layer numbers and ranges such as 0,46-47, or none.",),
                    stage_policies=(
                        StagePolicy(
                            LTX2Stage.STAGE_2,
                            StageMode.LOCKED,
                            "sparse_attention_stage1_profile",
                            reason="The saved stage-1 latent embodies stage-1 sparse attention.",
                        ),
                        StagePolicy(
                            LTX2Stage.DECODE,
                            StageMode.LOCKED,
                            "sparse_attention_stage_profiles",
                            reason="The saved final latent embodies both stages' attention.",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage_2_dense_layers",
                    "Keep these layers dense on sparse stage-2 steps",
                    'transformer layer numbers and ranges from 0 to 47, like "0,46-47", or "none"',
                    "0,47",
                    persistence="intent",
                    cli_dest="stage_2_dense_layers",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--stage2-dense-layers", "--dense-layers-stage-2"),
                        parse=argument_value(canonical_layer_ranges, "stage 2 dense layers"),
                    ),
                    notes=("Layer numbers and ranges such as 0,46-47, or none.",),
                    stage_policies=(
                        StagePolicy(
                            LTX2Stage.STAGE_2,
                            StageMode.MUTABLE,
                            reason="A stage-2 restart reruns stage 2 from the saved stage-1 latent.",
                        ),
                        StagePolicy(
                            LTX2Stage.DECODE,
                            StageMode.LOCKED,
                            "sparse_attention_stage_profiles",
                            reason="The saved final latent embodies both stages' attention.",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "hdr",
                    "Select an HDR authoring profile",
                    "",
                    "SRGB_LINEAR",
                    persistence="intent",
                    cli_dest="hdr",
                    cli=CliBinding("LTX-2 generation"),
                    choices=field_choices(LTX2Request, "hdr"),
                    choice_help=(
                        ("SRGB_LINEAR", "Write scene-linear Rec.709 EXR masters"),
                        ("ACESCG", "Write scene-linear ACEScg EXR masters"),
                        ("ACESCCT", "Write ACEScct working-space EXR masters"),
                    ),
                    default_text="unset, which produces SDR output",
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "generate_audio",
                    "Generate the joint audio branch",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="generate_audio",
                    cli=CliBinding("LTX-2 generation", action=CliAction.PAIRED_BOOLEAN),
                    negative_purpose="Disable the joint audio branch",
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "reference_aligned_audio",
                    "Use the reference implementation's rounded audio length",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="reference_aligned_audio",
                    cli=CliBinding("LTX-2 generation", action=CliAction.BOOLEAN),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "generated_keyframes",
                    "Reserve generated keyframe slots in both stages",
                    "a non-negative integer",
                    0,
                    persistence="intent",
                    cli_dest="generated_keyframes",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "text_conditioning",
                    "Load saved prompt tensors instead of running Gemma",
                    "a path to a compatible safetensors file",
                    Path("prompt_text.safetensors"),
                    persistence="intent",
                    cli_dest="text_conditioning",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_DECODE_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vae_decode_dtype",
                    "Select the VAE decode precision",
                    "",
                    "auto",
                    persistence="intent",
                    cli_dest="vae_decode_dtype",
                    cli=CliBinding("LTX-2 generation"),
                    choices=field_choices(LTX2Request, "vae_decode_dtype"),
                    choice_help=(
                        (
                            "auto",
                            "Use float16, and bfloat16 again if the first frames overflow",
                        ),
                        ("bfloat16", "Force bfloat16 VAE decode compute"),
                        ("float16", "Force float16 VAE decode compute"),
                        ("float32", "Force float32 VAE decode compute"),
                    ),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vae_cache_limit_gb",
                    "Override the automatic VAE allocator cache limit",
                    "a finite number greater than or equal to 0; unset selects cache with tiling",
                    16.0,
                    persistence="intent",
                    cli_dest="vae_cache_limit_gb",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vae_sharpen",
                    "Sharpen the diffusion VAE decode by unsharp-masking its final-stage input",
                    "a number from 0 to 10; 0 turns sharpening off",
                    2.0,
                    persistence="intent",
                    cli_dest="vae_sharpen",
                    cli=CliBinding("LTX-2 generation"),
                    notes=(
                        "Applies to the diffusion video VAE only and is not part of the "
                        "reference decoder; about 1 is subtle and 3 to 4 is strong.",
                    ),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vae_sharpen_radius",
                    "Set the blur radius of VAE sharpening in output pixels",
                    "a number greater than 0 and at most 8",
                    2.0,
                    persistence="intent",
                    cli_dest="vae_sharpen_radius",
                    cli=CliBinding("LTX-2 generation"),
                    notes=(
                        "The Gaussian's standard deviation; larger radii also sharpen coarser "
                        "shapes and can outline them.",
                    ),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "pad_prompt_to_max",
                    "Pad Gemma input to 1,024 tokens",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="pad_prompt_to_max",
                    cli=CliBinding(
                        "LTX-2 generation",
                        flags=("--reference-prompt-padding",),
                        action=CliAction.PAIRED_BOOLEAN,
                        negative_flags=("--compact-prompt",),
                    ),
                    negative_purpose="Use variable-length Gemma input without maximum padding",
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                replace(
                    ADAPTERS,
                    stage_policies=(
                        StagePolicy(LTX2Stage.STAGE_2, StageMode.LOCKED, "adapter_stage1_profile"),
                        StagePolicy(LTX2Stage.DECODE, StageMode.LOCKED, "adapter_stage_profiles"),
                    ),
                ),
                replace(
                    IMAGES,
                    stage_policies=(
                        StagePolicy(LTX2Stage.STAGE_2, StageMode.LOCKED, "image_stage1_profile"),
                        StagePolicy(LTX2Stage.DECODE, StageMode.LOCKED, "image_stage_profiles"),
                    ),
                ),
                F(
                    "first_frame_mode",
                    "Choose how the frame-0 image condition is applied",
                    "",
                    "anchor",
                    persistence="intent",
                    cli_dest="first_frame_mode",
                    cli=CliBinding("LTX-2 generation"),
                    choices=field_choices(LTX2Request, "first_frame_mode"),
                    choice_help=(
                        ("anchor", "Replace the first frame with the image and hold it fixed"),
                        ("guide", "Guide a denoised first frame toward the image"),
                    ),
                    notes=(
                        "Only the frame-0 condition reads this; every other condition is "
                        "guidance either way.",
                    ),
                    stage_policies=_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        ConfigTableSpec(
            path=("generate", "hdr_reference"),
            cli_generated=True,
            record=LTX2HDRReferenceConditioningConfig,
            stages=RESTARTS,
            stage_record=LTX2HDRReferenceConditioningConfig,
            starter_enabled=False,
            default_present=False,
            introduction=(
                "Optional LTX-2.3 SDR reference conditioning. Uncomment this table "
                "header and the fields you want to set."
            ),
            fields=(
                F(
                    "path",
                    "Set the SDR reference video",
                    "a readable video path",
                    Path("reference.mov"),
                    persistence="intent",
                    cli_dest="hdr_reference",
                    environment_key="hdr_reference",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_DECODE_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "strength",
                    "Set the SDR reference strength",
                    "a number from 0 through 1",
                    1.0,
                    persistence="intent",
                    cli_dest="hdr_reference_strength",
                    environment_key="hdr_reference_strength",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_DECODE_LOCKED,
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        ConfigTableSpec(
            path=("generate", "vae_tiling"),
            cli_generated=True,
            record=LTX2VideoVAETilingConfig,
            stages=RESTARTS,
            stage_record=LTX2VideoVAETilingConfig,
            starter_enabled=False,
            introduction=(
                "Optional VAE decode tiling. Uncomment this table header and the "
                "fields you want to set."
            ),
            fields=(
                F(
                    "mode",
                    "Select the decode tiling policy",
                    "",
                    "custom",
                    persistence="intent",
                    cli_dest="vae_tiling",
                    environment_key="vae_tiling",
                    layer_transforms=(
                        VAE_TILING_INPUT,
                        VAE_TILING_SELECTION,
                        VAE_TILING_EFFECTIVE,
                    ),
                    cli=CliBinding("LTX-2 generation"),
                    choices=field_choices(LTX2VideoVAETilingConfig, "mode"),
                    choice_help=(
                        ("auto", "Let the decoder choose a memory-aware tiling plan"),
                        ("single", "Decode the complete latent in one operation"),
                        ("custom", "Use the temporal and spatial tile fields below"),
                    ),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "temporal_tile_frames",
                    "Set the temporal tile size",
                    "0, or a multiple of 8 that is at least 16",
                    32,
                    persistence="intent",
                    cli_dest="vae_temporal_tile_frames",
                    environment_key="vae_temporal_tile_frames",
                    cli=CliBinding("LTX-2 generation"),
                    default_text="unset, or 256 when the mode is custom",
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "temporal_overlap_frames",
                    "Set the temporal overlap",
                    "a non-negative multiple of 8 smaller than the temporal tile",
                    24,
                    persistence="intent",
                    cli_dest="vae_temporal_overlap_frames",
                    environment_key="vae_temporal_overlap_frames",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "spatial_tile_pixels",
                    "Set the spatial tile size",
                    "0, or a multiple of 32 that is at least 64",
                    512,
                    persistence="intent",
                    cli_dest="vae_spatial_tile_pixels",
                    environment_key="vae_spatial_tile_pixels",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "spatial_overlap_pixels",
                    "Set the spatial overlap",
                    "a non-negative multiple of 32 smaller than the spatial tile",
                    64,
                    persistence="intent",
                    cli_dest="vae_spatial_overlap_pixels",
                    environment_key="vae_spatial_overlap_pixels",
                    cli=CliBinding("LTX-2 generation"),
                    stage_policies=_MUTABLE,
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        ConfigTableSpec(
            path=("output",),
            cli_generated=True,
            record=OutputConfig,
            fields=(
                F(
                    "path",
                    "Write to an exact video path, including .mp4",
                    "an MP4 path",
                    Path("video.mp4"),
                    persistence="intent",
                    cli_dest="output_path",
                    environment_key="output_path",
                    layer_transforms=(OUTPUT_SELECTION,),
                    cli=CliBinding("Output", flags=("--output",)),
                    default_text="unset, which creates a timestamped output",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "directory",
                    "Choose the directory for timestamped output",
                    "a directory path",
                    Path("outputs"),
                    persistence="intent",
                    cli_dest="output_dir",
                    environment_key="output_dir",
                    cli=CliBinding("Output"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "prefix",
                    "Set the timestamped filename prefix",
                    "non-empty filename text",
                    "ltx2-distilled-seed42",
                    persistence="intent",
                    cli_dest="output_prefix",
                    environment_key="output_prefix",
                    cli=CliBinding("Output"),
                    default_text="ltx2-distilled-seed<seed>",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vsr_spatial_mode",
                    "Select the spatial VSR mode",
                    "",
                    "off",
                    persistence="intent",
                    cli_dest="vsr_spatial_mode",
                    cli=CliBinding("Output"),
                    choices=field_choices(OutputConfig, "vsr_spatial_mode"),
                    choice_help=(
                        ("off", "Disable spatial super-resolution"),
                        ("fast", "Use the low-latency spatial processor"),
                        ("balanced", "Use temporally aware balanced-quality VSR"),
                        ("image", "Use the highest-quality per-frame spatial processor"),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "target_fps",
                    "Set the post-generation frame rate",
                    "a finite number greater than 0",
                    48.0,
                    persistence="intent",
                    cli_dest="target_fps",
                    cli=CliBinding("Output"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vsr_temporal_mode",
                    "Select the temporal interpolation quality",
                    "",
                    "normal",
                    persistence="intent",
                    cli_dest="vsr_temporal_mode",
                    cli=CliBinding("Output"),
                    choices=field_choices(OutputConfig, "vsr_temporal_mode"),
                    choice_help=(
                        ("normal", "Use normal-quality frame interpolation"),
                        ("high", "Use high-quality frame interpolation"),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "cut_detect_mode",
                    "Select hard-cut protection",
                    "",
                    "simple",
                    persistence="intent",
                    cli_dest="cut_detect_mode",
                    cli=CliBinding("Output"),
                    choices=field_choices(OutputConfig, "cut_detect_mode"),
                    choice_help=(
                        ("off", "Disable hard-cut protection"),
                        ("simple", "Use the default lightweight cut detector"),
                        ("hist", "Use histogram detection for motion-heavy footage"),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "cut_detect_threshold",
                    "Override the cut detector threshold",
                    "a finite number greater than or equal to 0",
                    0.35,
                    persistence="intent",
                    cli_dest="cut_detect_threshold",
                    cli=CliBinding("Output", metavar="FLOAT"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "vsr_save_original",
                    "Keep the preprocessed video beside the VSR output",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="vsr_save_original",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "encode_quality",
                    "Set VideoToolbox encoding quality",
                    "a number from 0 through 1",
                    0.65,
                    persistence="intent",
                    cli_dest="encode_quality",
                    cli=CliBinding("Output"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_codec",
                    "Select the encoded audio codec",
                    "",
                    "alac",
                    persistence="intent",
                    cli_dest="audio_codec",
                    cli=CliBinding("Output"),
                    choices=field_choices(OutputConfig, "audio_codec"),
                    choice_help=(
                        ("alac", "Encode lossless Apple Lossless audio"),
                        ("aac", "Encode compact lossy AAC audio"),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_onset_trim",
                    "Configure audio onset repair",
                    '"auto", "off", or a non-negative millisecond count, bare or quoted '
                    '(in TOML, audio_onset_trim = 120 or "120"; saved as the string)',
                    "auto",
                    persistence="intent",
                    cli_dest="audio_onset_trim",
                    cli=CliBinding("Output", metavar="AUTO|OFF|MS"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_run_log",
                    "Save the resolved invocation and timing as JSON",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_run_log",
                    cli=CliBinding(
                        "Output",
                        flags=("--save-run-log", "--save-metadata"),
                        action=CliAction.BOOLEAN,
                    ),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_console_log",
                    "Save a human-readable execution log",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_console_log",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_effective_config",
                    "Save the effective invocation as a TOML sidecar",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_effective_config",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_audio_sidecar",
                    "Save generated audio as a WAV sidecar",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_audio_sidecar",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_hdr_heic_frames",
                    "Save HDR frames as PQ HEIC files",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_hdr_heic_frames",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which stays off because save-all excludes this sequence",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_vae_frames",
                    "Save normalized pre-postprocess PNG frames",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_vae_frames",
                    cli=CliBinding("Output", action=CliAction.BOOLEAN),
                    default_text="unset, which stays off because save-all excludes this sequence",
                    groups=frozenset({ConfigGroup.SAVE_ALL_CANDIDATE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_all_sidecars",
                    "Enable every applicable reproducibility sidecar",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="save_all_sidecars",
                    cli=CliBinding(
                        "Output",
                        flags=("--save-all-sidecars", "--save-debug-sidecars"),
                        action=CliAction.BOOLEAN,
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        settings_table(
            stages=RESTARTS,
            adapter_weight_check_stages=frozenset({"stage-2"}),
        ),
        ConfigTableSpec(
            path=("model_settings",),
            cli_generated=True,
            cli_group="LTX-2 runtime settings (override env vars)",
            record=LTX2Settings,
            fields=(
                F(
                    "model_generation",
                    "Select the checkpoint generation",
                    "",
                    LTX2ModelGeneration.LTX23,
                    persistence="intent",
                    cli_dest="model_generation",
                    choices=field_choices(LTX2Settings, "model_generation"),
                    choice_help=(
                        (
                            LTX2ModelGeneration.LTX23,
                            "Discover and run the LTX-2.3 checkpoint family",
                        ),
                        (
                            LTX2ModelGeneration.LTX25,
                            "Discover and run the LTX-2.5 checkpoint family",
                        ),
                    ),
                    default_text="unset, which uses 2.3 discovery",
                    groups=frozenset({LTX2ConfigGroup.MODEL_GENERATION_SELECTOR}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "weights_path",
                    "Load a monolithic baseline or use an LTX-2.5 pack as the component root",
                    "a compatible safetensors file or local LTX-2.5 pack directory",
                    Path("ltx-2.3.safetensors"),
                    persistence="intent",
                    cli_dest="weights_path",
                    default_text="unset or KINO_LTX2_WEIGHTS_PATH",
                    groups=frozenset(
                        {
                            LTX2ConfigGroup.MODEL_SOURCE,
                            LTX2ConfigGroup.MODEL_MONOLITHIC_SOURCE,
                        }
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "gemma_variant",
                    "Select the LTX-2.3 Gemma source",
                    "",
                    GemmaVariant.QAT,
                    persistence="intent",
                    cli_dest="gemma_variant",
                    choices=field_choices(LTX2Settings, "gemma_variant"),
                    choice_help=(
                        (
                            GemmaVariant.QAT,
                            "Use the QAT-unquantized Gemma source paired with LTX-2.3",
                        ),
                        (
                            GemmaVariant.PLAIN,
                            "Use the plain instruction-tuned Gemma source",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "spatial_upscaler_path",
                    "Load the spatial latent upscaler",
                    "a path to a compatible safetensors file",
                    Path("spatial-upscaler.safetensors"),
                    persistence="intent",
                    cli_dest="spatial_upscaler_path",
                    default_text="auto-discovered",
                    groups=frozenset({LTX2ConfigGroup.MODEL_SOURCE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_path",
                    "Load a componentized transformer or override the baseline transformer",
                    "a directory or safetensors path",
                    Path("transformer/"),
                    persistence="intent",
                    cli_dest="transformer_path",
                    groups=frozenset(
                        {
                            LTX2ConfigGroup.MODEL_SOURCE,
                            LTX2ConfigGroup.MODEL_SPLIT_SOURCE,
                        }
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "text_encoder_path",
                    "Load the text encoder from a local path",
                    "LTX-2.3: a Gemma 3 directory holding config.json, tokenizer.json, and the "
                    "model safetensors; LTX-2.5: a prepared text-encoder safetensors file",
                    Path("gemma-3-12b-it/"),
                    persistence="intent",
                    cli_dest="text_encoder_path",
                    default_text="auto-discovered",
                    groups=frozenset({LTX2ConfigGroup.MODEL_SOURCE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_vae",
                    "Select the named video decoder",
                    "",
                    "conv",
                    persistence="intent",
                    cli_dest="video_vae",
                    cli=CliBinding("LTX-2 artifacts"),
                    choices=field_choices(LTX2Settings, "video_vae"),
                    choice_help=(
                        ("conv", "Use the convolutional video VAE"),
                        ("diffusion", "Use the compatible LTX-2.5 diffusion decoder"),
                    ),
                    groups=frozenset({LTX2ConfigGroup.MODEL_VIDEO_VAE_SELECTOR}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_vae_path",
                    "Load an explicit video VAE",
                    "a directory or safetensors path",
                    Path("video-vae/"),
                    persistence="intent",
                    cli_dest="video_vae_path",
                    default_text="auto-discovered",
                    groups=frozenset(
                        {
                            LTX2ConfigGroup.MODEL_SOURCE,
                            LTX2ConfigGroup.MODEL_VIDEO_VAE_SOURCE,
                        }
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_vae_path",
                    "Load an explicit audio VAE and vocoder",
                    "a directory or safetensors path",
                    Path("audio-vae/"),
                    persistence="intent",
                    cli_dest="audio_vae_path",
                    default_text="auto-discovered",
                    groups=frozenset({LTX2ConfigGroup.MODEL_SOURCE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "temporal_latent_upscaler_path",
                    "Load the temporal x2 latent upscaler",
                    "a path to a compatible safetensors file",
                    Path("temporal-upscaler.safetensors"),
                    persistence="intent",
                    cli_dest="temporal_latent_upscaler_path",
                    default_text="auto-discovered",
                    groups=frozenset({LTX2ConfigGroup.MODEL_SOURCE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "duration_head_path",
                    "Load the automatic-duration head",
                    "a path to a compatible safetensors file",
                    Path("duration-head.safetensors"),
                    persistence="intent",
                    cli_dest="duration_head_path",
                    default_text="auto-discovered",
                    groups=frozenset({LTX2ConfigGroup.MODEL_SOURCE}),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "fast_mode",
                    "Build one full lazy transformer graph",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="fast_mode",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_dtype",
                    "Select transformer storage and compute precision",
                    "",
                    "float16",
                    persistence="intent",
                    cli_dest="transformer_dtype",
                    choices=field_choices(LTX2Settings, "transformer_dtype"),
                    choice_help=(
                        ("bfloat16", "Use bfloat16 transformer storage and compute"),
                        ("float16", "Use float16 transformer storage and compute"),
                        ("float32", "Use float32 transformer storage and compute"),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "adapter_accumulation",
                    "Select the precision adapter fusion accumulates in",
                    "",
                    "auto",
                    persistence="intent",
                    cli_dest="adapter_accumulation",
                    choices=field_choices(LTX2Settings, "adapter_accumulation"),
                    choice_help=(
                        ("auto", "float16 for a streamed float16 transformer, float32 otherwise"),
                        ("float32", "Accumulate every adapter term in float32 and cast once"),
                        (
                            "float16",
                            "Accumulate plain LoRA terms in float16; requires transformer_dtype "
                            "float16",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "dense_attention_kernel",
                    "Run unmasked attention through KinoMLX's FP16 dense attention kernel",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="dense_attention_kernel",
                    notes=(
                        "Video self- and text attention take it always; the 64-wide audio, "
                        "audio text and audio-video calls take it with at least 512 tokens on "
                        "both sides, where it is faster than MLX SDPA. In FP16 it matches MLX "
                        "SDPA bit for bit on the video calls; elsewhere it differs by rounding. "
                        "False runs every call on MLX SDPA.",
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "compile_attention",
                    "Compile eligible attention calls",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="compile_attention",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_ff_layout_specs",
                    "Configure video feed-forward weight layouts",
                    ('unique "project_in:pretranspose" or "project_out:pretranspose" entries'),
                    ("project_out:pretranspose",),
                    persistence="intent",
                    cli_dest="video_ff_layout_specs",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_ff_layout_layers",
                    "Choose video layers for feed-forward weight layouts",
                    "unique integers from 0 through 47",
                    (),
                    persistence="intent",
                    cli_dest="video_ff_layout_layers",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_attn_layout_specs",
                    "Configure video attention weight layouts",
                    'unique attention-target entries ending in ":pretranspose"',
                    (),
                    persistence="intent",
                    cli_dest="video_attn_layout_specs",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_attn_layout_layers",
                    "Choose video layers for attention weight layouts",
                    "unique integers from 0 through 47",
                    (),
                    persistence="intent",
                    cli_dest="video_attn_layout_layers",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_layout_mirror",
                    "Mirror video weight layouts to audio blocks",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="audio_layout_mirror",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_ff_layout_specs",
                    "Configure audio feed-forward weight layouts",
                    'unique feed-forward target entries ending in ":pretranspose"',
                    (),
                    persistence="intent",
                    cli_dest="audio_ff_layout_specs",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_ff_layout_layers",
                    "Choose audio layers for feed-forward weight layouts",
                    "unique integers from 0 through 47",
                    (),
                    persistence="intent",
                    cli_dest="audio_ff_layout_layers",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_attn_layout_specs",
                    "Configure audio attention weight layouts",
                    'unique attention-target entries ending in ":pretranspose"',
                    (),
                    persistence="intent",
                    cli_dest="audio_attn_layout_specs",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "audio_attn_layout_layers",
                    "Choose audio layers for attention weight layouts",
                    "unique integers from 0 through 47",
                    (),
                    persistence="intent",
                    cli_dest="audio_attn_layout_layers",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "adaln_pretranspose",
                    "Pretranspose AdaLN weights",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="adaln_pretranspose",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_quantize",
                    "Select transformer quantization",
                    "",
                    "off",
                    persistence="intent",
                    cli_dest="transformer_quantize",
                    choices=field_choices(LTX2Settings, "transformer_quantize"),
                    choice_help=(
                        ("off", "Keep transformer weights in their source precision"),
                        ("mxfp8-blocks", "Quantize transformer block weights to MXFP8"),
                        (
                            "mxfp8-blocks-pretranspose",
                            "Quantize and pretranspose transformer block weights",
                        ),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_quantize_exclude_layers",
                    "Keep selected blocks at the transformer dtype instead of quantizing",
                    "unique integers from 0 through 47; requires whole-block quantization",
                    (),
                    persistence="intent",
                    cli_dest="transformer_quantize_exclude_layers",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_ff_quantize_specs",
                    "Configure targeted video feed-forward quantization",
                    ("unique feed-forward targets using affine, mxfp4, mxfp8, or nvfp4"),
                    (),
                    persistence="intent",
                    cli_dest="video_ff_quantize_specs",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_ff_quantize_layers",
                    "Choose video layers for targeted quantization",
                    "unique integers from 0 through 47",
                    (),
                    persistence="intent",
                    cli_dest="video_ff_quantize_layers",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_ff_quantize_group_size",
                    "Set the targeted quantization group size",
                    "a positive integer when quantization specs are present",
                    64,
                    persistence="intent",
                    cli_dest="video_ff_quantize_group_size",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "video_ff_quantize_bits",
                    "Set the affine target bit count",
                    "a positive integer when quantization specs are present",
                    4,
                    persistence="intent",
                    cli_dest="video_ff_quantize_bits",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stream_transformer",
                    "Use the bounded-residency transformer preset",
                    "true or false",
                    False,
                    persistence="intent",
                    cli_dest="stream_transformer",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_retain",
                    "Retain selected block tensors across streamed denoising steps",
                    "small, audio-attention, video-attention, audio-ff, video-ff, or block-relative tensor globs",
                    (),
                    persistence="intent",
                    cli_dest="transformer_retain",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_resident_blocks",
                    "Set the number of resident transformer blocks",
                    "an integer from 1 through 48",
                    STREAM_TRANSFORMER_RESIDENT_BLOCKS,
                    persistence="intent",
                    cli_dest="transformer_resident_blocks",
                    default_text=(f"unset, or {STREAM_TRANSFORMER_RESIDENT_BLOCKS} when streaming"),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "transformer_compile_group_size",
                    "Set the compiled transformer group size",
                    "a positive integer no greater than the resident block count",
                    4,
                    persistence="intent",
                    cli_dest="transformer_compile_group_size",
                    default_text="unset (streamed groups run uncompiled)",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "compile_block_group_size",
                    "Compile the resident transformer in groups of this many blocks",
                    "an integer from 1 through 48",
                    48,
                    persistence="intent",
                    cli_dest="compile_block_group_size",
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        ConfigTableSpec(
            path=("model_artifacts",),
            cli_generated=True,
            record=LTX2ArtifactConfig,
            fields=(
                F(
                    "save_latents",
                    "Save stage-1 and final latents",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="save_latents",
                    output_policy=OUTPUT_POLICIES["save_latents"],
                    cli=CliBinding("LTX-2 artifacts", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_media_conditioning",
                    "Save encoded media conditions",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="save_media_conditioning",
                    output_policy=OUTPUT_POLICIES["save_media_conditioning"],
                    cli=CliBinding("LTX-2 artifacts", action=CliAction.BOOLEAN),
                    default_text="unset, which inherits the save-all setting",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "save_text_conditioning",
                    "Save prompt tensors and their mask",
                    "true or false",
                    True,
                    persistence="intent",
                    cli_dest="save_text_conditioning",
                    output_policy=OUTPUT_POLICIES["save_text_conditioning"],
                    cli=CliBinding(
                        "LTX-2 artifacts",
                        flags=("--save-text-conditioning", "--save-text-embeddings"),
                        action=CliAction.BOOLEAN,
                    ),
                    default_text="unset, which inherits the save-all setting",
                    surfaces=CONFIG_SURFACES,
                ),
            ),
        ),
        operation_table(cli_group="LTX-2 execution", stages=RESTARTS),
        ConfigTableSpec(
            path=("restart",),
            cli_generated=True,
            record=RestartConfig,
            default_present=False,
            fields=(
                F(
                    "run",
                    "Load a prior run record",
                    "a completed KinoMLX run-log path",
                    Path("previous_run.json"),
                    persistence="intent",
                    cli_dest="restart_run",
                    environment_key="restart_run",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=CliBinding("Restart", flags=("--restart",), metavar="RUN_JSON"),
                    notes=("Omit the whole [restart] table to start a fresh run.",),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "stage",
                    "Select the first stage to repeat",
                    "",
                    "decode",
                    persistence="intent",
                    cli_dest="restart_stage",
                    environment_key="restart_stage",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=CliBinding("Restart", flags=("--restart-from",)),
                    choices=field_choices(RestartConfig, "stage"),
                    choice_help=(
                        ("decode", "Decode an already generated latent"),
                        ("stage-2", "Resume at the second denoising stage"),
                    ),
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "latent_stage",
                    "Select which stage's saved latent to read",
                    "",
                    "stage-2",
                    persistence="intent",
                    cli_dest="latent_stage",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=CliBinding("Restart"),
                    choices=field_choices(RestartConfig, "latent_stage"),
                    choice_help=(
                        ("stage-1", "Read the saved stage-1 latent"),
                        ("stage-2", "Read the saved final latent, which stage 2 wrote"),
                    ),
                    default_text="the latest latent written before the restart stage",
                    surfaces=CONFIG_SURFACES,
                ),
                F(
                    "latents",
                    "Substitute a latent artifact",
                    "a path to a compatible safetensors file",
                    Path("edited.safetensors"),
                    persistence="intent",
                    cli_dest="restart_latents",
                    environment_key="restart_latents",
                    groups=frozenset({ConfigGroup.SOURCE_SELECTOR}),
                    cli=CliBinding("Restart", metavar="SAFETENSORS"),
                    default_text="the artifact recorded in the run log",
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
                (("output", "save_run_log"), True),
                (("output", "save_console_log"), True),
                (("output", "save_effective_config"), True),
                (("output", "save_audio_sidecar"), True),
                (("output", "vsr_save_original"), True),
            ),
            expand_in_layers=True,
            excluded=(
                ("output", "save_hdr_heic_frames"),
                ("output", "save_vae_frames"),
            ),
            coverage_group=ConfigGroup.SAVE_ALL_CANDIDATE,
            inherited=(
                ("model_artifacts", "save_latents"),
                ("model_artifacts", "save_media_conditioning"),
                ("model_artifacts", "save_text_conditioning"),
            ),
        ),
        ConfigImplication(
            trigger=("model_settings", "stream_transformer"),
            trigger_value=True,
            implied=(
                (
                    ("model_settings", "transformer_resident_blocks"),
                    STREAM_TRANSFORMER_RESIDENT_BLOCKS,
                ),
            ),
        ),
    ),
)


CONFIG_SCHEMA = replace(
    CONFIG_SCHEMA,
    api_boundaries=(
        *RUNNER_API_BOUNDARIES,
        APIBoundary(
            "ltx2",
            prepare_resources,
            (
                configuration("settings", LTX2Settings, "model_settings"),
                configuration("infrastructure", Settings, "settings"),
            ),
        ),
        APIBoundary(
            "ltx2",
            generate_ltx2,
            (
                configuration("request", LTX2Request, "generate"),
                runtime("resources", "prepared immutable component inventory"),
                runtime("components", "component provider"),
                runtime("text_conditioner", "text-conditioning provider"),
                runtime("reporter", "progress reporter"),
                runtime("artifact_sink", "artifact writer"),
                runtime("conditioning", "prepared shared stage conditioning"),
            ),
        ),
        APIBoundary(
            "ltx2",
            restart_ltx2,
            (
                configuration("request", LTX2Request, "generate"),
                runtime("resources", "prepared immutable component inventory"),
                runtime("restart", "typed prepared stage handoff"),
                runtime("components", "component provider"),
                runtime("text_conditioner", "text-conditioning provider"),
                runtime("reporter", "progress reporter"),
                runtime("artifact_sink", "artifact writer"),
            ),
        ),
    ),
)


CONFIG_SCHEMA = replace(
    CONFIG_SCHEMA,
    layer_transforms=(*CONFIG_SCHEMA.layer_transforms, source_selection_rule(CONFIG_SCHEMA)),
)

__all__ = ["CONFIG_SCHEMA"]
