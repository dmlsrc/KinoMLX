"""One public two-stage distilled recipe for prepared LTX-2 resources."""

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import mlx.core as mx

from kinomlx import __version__
from kinomlx.adapters.lora import LoRAProfile
from kinomlx.artifacts import ArtifactSink, NullArtifactSink
from kinomlx.components import ComponentLease
from kinomlx.config_rules import validate_capabilities
from kinomlx.media.frames import VideoFrameStream
from kinomlx.memory_policy import memory_managed
from kinomlx.models.ltx2.keyframe_positions import generated_keyframe_indices
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.samplers.noise import NoiseStreamState, noise_compatibility_profile
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend, VideoPixelShape

from ..adapters.selection import lora_profile_for_stage
from ..artifacts import (
    distilled_stage_latents_artifact,
    media_conditioning_artifact,
    text_conditioning_artifact,
)
from ..components import (
    LTX2Components,
    LTX2ComponentsProvider,
    LTX2DurationPredictorPort,
    LTX2TransformerPort,
)
from ..conditioning import (
    EncodedCondition,
    HDRReferenceConditionSource,
    ImageConditionSource,
    RawConditionSource,
    VideoConditionByReferenceLatent,
    prepare_conditions,
)
from ..conditioning.tools import AudioLatentTools, VideoLatentTools
from ..decode import (
    VideoDecoderProvider,
    decode_audio_mel,
    decode_ltx23_sdr_frames,
    decode_ltx_hdr_working_frames,
    video_decode_diagnostics,
    vocode_audio,
)
from ..denoise import DenoiseStepKind, LTX2TransitionNoiseProvider, denoise_loop
from ..frame_position import is_first_position, resolve_frame_positions
from ..generated_keyframes import (
    GeneratedKeyframeLayout,
    append_generated_keyframe_slots,
    extract_generated_keyframes,
)
from ..generation import LTX2GenerationOutput
from ..hdr_profile import resolve_hdr_recipe, validate_hdr_adapter_placement
from ..noising import GaussianNoiser
from ..precision import LTX2DTypePolicy, resolve_video_vae_decode_precision
from ..resources import LTX2Resources
from ..sigmas import stage_sigmas
from ..signals import ltx23_sdr_signal, ltx_hdr_working_signal
from ..sparse_attention import SparseAttentionSchedule, SparseAttentionStep, stage_steps
from ..state import (
    apply_encoded_conditions,
    create_audio_latent_tools,
    create_video_latent_tools,
    init_audio_latent_state,
    init_video_latent_state,
)
from ..text_conditioning import (
    LTX2EncodedTextConditioning,
    LTX2TextConditioner,
    LTX2TextConditionerPort,
)
from ..transition_noise import SeededGaussianNoise
from ..types import AudioLatentShape, LatentState, LTX2Request, video_latent_shape_from_pixel
from ..upscaler import upsample_video
from ..upscaler.ports import LTX2SpatialUpscalerPort
from ..vocabularies import LTX2FirstFrameMode, LTX2StageSampler


@dataclass(frozen=True)
class StageLatents:
    """Materialized video and optional audio latents crossing a stage.

    ``generated_keyframes`` holds the stage's denoised keyframe slots, one latent frame
    each, which seed the next stage's slots.
    """

    video: mx.array
    audio: mx.array | None = None
    initial_noise_state: NoiseStreamState | None = None
    generated_keyframes: mx.array | None = None


@dataclass(frozen=True)
class PreparedStage:
    """Patchified latent state ready for one denoise schedule."""

    video_state: LatentState
    video_tools: VideoLatentTools
    audio_state: LatentState | None
    audio_tools: AudioLatentTools | None
    video_keyframes_mask: mx.array | None = None
    generated_keyframe_layout: GeneratedKeyframeLayout | None = None


# Each ancestral stage draws its transition noise from its own stream, seeded at the
# request seed plus that stage's offset as ltx-pipelines does, so neither stage reuses
# the initial noise or the other stage's draws.
ANCESTRAL_SEED_OFFSETS = (10_000, 20_000)


@dataclass(frozen=True)
class DistilledStageSampler:
    """One stage's requested and effective sampler, and its ancestral noise seed."""

    stage: int
    requested: LTX2StageSampler
    effective: LTX2StageSampler
    noise_seed: int | None

    @property
    def step_kind(self) -> DenoiseStepKind:
        """The denoise step this stage runs."""
        if self.effective == LTX2StageSampler.ANCESTRAL:
            return "ancestral-rf"
        return "deterministic-euler"

    def transition_noise(
        self,
        *,
        transitions: int,
        backend: NoiseBackend,
    ) -> SeededGaussianNoise | None:
        """Open this stage's ancestral noise stream; a deterministic stage has none."""
        if self.noise_seed is None:
            return None
        return SeededGaussianNoise(
            self.noise_seed,
            stage=self.stage,
            transitions=transitions,
            backend=backend,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.requested.value,
            "effective": self.effective.value,
            "noise_seed": self.noise_seed,
        }


@dataclass(frozen=True)
class DistilledSamplerPlan:
    """Both stages' resolved samplers."""

    stage_1: DistilledStageSampler
    stage_2: DistilledStageSampler

    def metadata(
        self,
        noise_states: tuple[NoiseStreamState | None, NoiseStreamState | None] = (None, None),
    ) -> dict[str, dict[str, object]]:
        """Record both stages, with where each ancestral stream stopped when it ran."""
        return {
            f"stage_{stage.stage}": {
                **stage.to_metadata(),
                "noise_state": None if state is None else state.to_metadata(),
            }
            for stage, state in zip((self.stage_1, self.stage_2), noise_states, strict=True)
        }


@dataclass(frozen=True)
class PreparedDistilledRecipe:
    """Load-free request facts resolved before any heavyweight stage."""

    # Sources stay unset while the frame count does: a named frame position has
    # no absolute index until the duration head has run.
    sources: tuple[RawConditionSource, ...] | None
    frame_count: int | None
    sampler: DistilledSamplerPlan


@dataclass
class DistilledConditioning:
    """Seed-independent products owned by one multi-seed request."""

    request: LTX2Request | None = None
    text: LTX2EncodedTextConditioning | None = None
    frames: int | None = None
    # Keyed by stage and geometry: a prepared condition now carries the strength
    # its own stage applies, so geometry alone no longer identifies the product.
    stages: dict[tuple[int, VideoPixelShape], tuple[EncodedCondition, ...]] = field(
        default_factory=dict
    )

    def bind(self, request: LTX2Request) -> None:
        identity = replace(request, seed=0, seeds=None)
        if self.request is not None and self.request != identity:
            raise ValueError("Shared conditioning requires identical non-seed request settings")
        self.request = identity

    def close(self) -> None:
        self.text = None
        self.stages.clear()
        self.request = None
        self.frames = None


def resolve_adapter_profiles(
    request: LTX2Request,
) -> tuple[LoRAProfile, LoRAProfile]:
    """Resolve ordered, zero-free LoRA profiles for the selected adapters."""
    selections = request.resolved_adapters()
    return (
        lora_profile_for_stage(selections, 1),
        lora_profile_for_stage(selections, 2),
    )


def condition_sources(request: LTX2Request, *, frames: int) -> tuple[RawConditionSource, ...]:
    """Resolve caller-owned condition intent without loading an encoder.

    Every named frame position resolves here, against the frame count the run
    settled on, so nothing downstream of the request sees a name.
    """
    positions = resolve_frame_positions(
        tuple(image.frame_index for image in request.images), frames
    )
    sources: list[RawConditionSource] = []
    for image, frame_index in zip(request.images, positions, strict=True):
        hdr_authoring = request.hdr if image.path.suffix.lower() == ".exr" else None
        sources.append(
            ImageConditionSource.from_config(
                image,
                frame_index=frame_index,
                hdr_authoring=hdr_authoring,
            )
        )
    if request.hdr_reference is not None:
        sources.append(HDRReferenceConditionSource.from_config(request.hdr_reference))
    return tuple(sources)


def requested_condition_families(request: LTX2Request) -> tuple[str, ...]:
    """Classify each image condition's family before any frame count exists.

    ``last`` on a one-frame run would be frame 0 and is classified as a keyframe
    here. No supported checkpoint can tell the two apart, because every one that
    declares either family declares both.
    """
    return tuple(
        "image" if is_first_position(image.frame_index) else "keyframe" for image in request.images
    )


def _requested_frame_count(request: LTX2Request) -> int | None:
    return request.frames


def effective_stage_samplers(
    requested: tuple[LTX2StageSampler, LTX2StageSampler],
    published: tuple[LTX2StageSampler, LTX2StageSampler],
) -> tuple[LTX2StageSampler, LTX2StageSampler]:
    """Resolve each stage's ``auto`` to the sampler the checkpoint generation publishes."""
    effective = []
    for stage, (wanted, default) in enumerate(zip(requested, published, strict=True), start=1):
        if default == LTX2StageSampler.AUTO:
            raise ValueError(f"the checkpoint publishes no concrete stage-{stage} sampler")
        effective.append(default if wanted == LTX2StageSampler.AUTO else LTX2StageSampler(wanted))
    return (effective[0], effective[1])


def resolve_stage_samplers(
    requested: tuple[LTX2StageSampler, LTX2StageSampler],
    published: tuple[LTX2StageSampler, LTX2StageSampler],
    *,
    seed: int,
) -> DistilledSamplerPlan:
    """Resolve both stages' samplers and derive each ancestral stage's noise seed."""
    stages = []
    for stage, (wanted, effective) in enumerate(
        zip(requested, effective_stage_samplers(requested, published), strict=True), start=1
    ):
        noise_seed = None
        if effective == LTX2StageSampler.ANCESTRAL:
            offset = ANCESTRAL_SEED_OFFSETS[stage - 1]
            noise_seed = seed + offset
            if noise_seed >= 2**64:
                raise ValueError(
                    f"the stage-{stage} ancestral noise seed is seed + {offset}, which must "
                    "fit an unsigned 64-bit integer"
                )
        stages.append(
            DistilledStageSampler(
                stage=stage,
                requested=LTX2StageSampler(wanted),
                effective=effective,
                noise_seed=noise_seed,
            )
        )
    return DistilledSamplerPlan(stage_1=stages[0], stage_2=stages[1])


def stage_sampler_choices(
    request: LTX2Request,
    published: tuple[LTX2StageSampler, LTX2StageSampler],
) -> dict[str, dict[str, str]]:
    """Each stage's requested and effective sampler, keyed as the run record keys them."""
    requested = request.stage_samplers()
    effective = effective_stage_samplers(requested, published)
    return {
        f"stage_{stage}": {"requested": wanted.value, "effective": chosen.value}
        for stage, (wanted, chosen) in enumerate(zip(requested, effective, strict=True), start=1)
    }


def sampling_facts(
    request: LTX2Request,
    published: tuple[LTX2StageSampler, LTX2StageSampler],
) -> dict[str, object]:
    """Resolve the sampler facts a run announces and records; nothing here needs weights."""
    first, second = stage_sigmas(request.stage_1_sigmas, request.stage_2_sigmas)
    return {
        "samplers": stage_sampler_choices(request, published),
        "noise_backend": request.noise_backend,
        "noise_compatibility_profile": noise_compatibility_profile(request.noise_backend),
        "stage_1_sigmas": list(first),
        "stage_2_sigmas": list(second),
        "stage_1_steps": len(first) - 1,
        "stage_2_steps": len(second) - 1,
        "sigma_source": (
            "explicit" if request.stage_1_sigmas or request.stage_2_sigmas else "distilled-literal"
        ),
    }


def resolve_distilled_sampler_plan(
    request: LTX2Request,
    resources: LTX2Resources,
) -> DistilledSamplerPlan:
    """Resolve each stage against the samplers the checkpoint generation publishes."""
    return resolve_stage_samplers(
        request.stage_samplers(),
        resources.capabilities.stage_samplers,
        seed=request.seed,
    )


def validate_distilled_request(
    request: LTX2Request,
    resources: LTX2Resources,
) -> None:
    """Fail before model loading when resources cannot run this request."""
    capabilities = resources.capabilities
    validate_capabilities(
        request,
        {
            "model_generation": capabilities.model_generation,
            "video_vae_kind": capabilities.video_vae_kind,
            "recipe_families": capabilities.recipe_families,
            "duration_available": capabilities.duration_available,
            "generated_keyframes_available": capabilities.generated_keyframes,
            "generates_audio": capabilities.generates_audio,
            "condition_families": capabilities.condition_families,
            "requested_condition_families": requested_condition_families(request),
        },
        location="[generate]",
    )
    resolve_hdr_recipe(request, resources)
    if resources.dtype_policy.transformer == mx.float32 and any(
        schedule.active for schedule in request.sparse_attention_schedules()
    ):
        raise ValueError("sparse attention needs a bfloat16 or float16 transformer_dtype")


def prepare_distilled_recipe(
    request: LTX2Request,
    resources: LTX2Resources,
) -> PreparedDistilledRecipe:
    """Validate one request and prepare its stage policies."""
    request.validate_for_generation()
    validate_distilled_request(request, resources)
    frame_count = _requested_frame_count(request)
    # An auto-duration run has no frame count to resolve positions against yet,
    # so its sources are built after the duration head runs.
    sources = None if frame_count is None else condition_sources(request, frames=frame_count)
    sampler = resolve_distilled_sampler_plan(request, resources)
    return PreparedDistilledRecipe(
        sources=sources,
        frame_count=frame_count,
        sampler=sampler,
    )


def resolve_auto_frame_count(
    request: LTX2Request,
    resources: LTX2Resources,
    text: LTX2EncodedTextConditioning,
    predictor: LTX2DurationPredictorPort,
) -> int:
    """Resolve one omitted frame count from connector outputs and validate dependents."""
    frames = predictor.predict_num_frames(
        text.video_encoding,
        text.audio_encoding,
        frame_rate=request.fps,
        temporal_compression_ratio=resources.capabilities.video_compression.time,
    )
    resolve_frame_positions(
        tuple(image.frame_index for image in request.images), frames, predicted=True
    )
    if request.generated_keyframes:
        generated_keyframe_indices(frames, request.generated_keyframes)
    return frames


def prepare_text_conditioning(
    request: LTX2Request,
    resources: LTX2Resources,
    *,
    text_conditioner: LTX2TextConditionerPort | None,
    reporter: Reporter,
    artifacts: ArtifactSink,
    emit_artifact: bool = True,
) -> LTX2EncodedTextConditioning:
    selected = text_conditioner if text_conditioner is not None else LTX2TextConditioner()
    text = selected(
        request,
        resources,
        reporter=reporter,
    )
    if emit_artifact:
        artifacts.save(
            text_conditioning_artifact(
                prompt=text.prompt,
                video_encoding=text.video_encoding,
                audio_encoding=text.audio_encoding,
                attention_mask=text.attention_mask,
                provenance=text.provenance.to_metadata(),
            )
        )
    return text


def _materialize_state(state: LatentState) -> None:
    mx.eval(
        state.latent,
        state.denoise_mask,
        state.positions,
        state.clean_latent,
    )


def prepare_stage(
    request: LTX2Request,
    geometry: VideoPixelShape,
    *,
    dtype_policy: LTX2DTypePolicy,
    conditions: Sequence[EncodedCondition] = (),
    initial_latents: StageLatents | None = None,
    generated_keyframe_count: int = 0,
) -> PreparedStage:
    """Construct and materialize one conditioned stage state.

    Generated keyframe slots follow the conditions in the token order, and a prior stage's
    slots in ``initial_latents`` seed them.
    """
    video_tools = create_video_latent_tools(
        video_latent_shape_from_pixel(geometry),
        fps=request.fps,
    )
    video_state = init_video_latent_state(
        video_tools,
        dtype=dtype_policy.latent,
        initial_latent=(None if initial_latents is None else initial_latents.video),
    )
    video_state = apply_encoded_conditions(video_state, conditions, video_tools)
    generated_layout = None
    video_keyframes_mask = None
    if generated_keyframe_count:
        initial_slots = None if initial_latents is None else initial_latents.generated_keyframes
        if initial_latents is not None and initial_slots is None:
            raise RuntimeError("the prior stage returned no generated keyframe slots")
        video_state, generated_layout, video_keyframes_mask = append_generated_keyframe_slots(
            video_state,
            video_tools,
            pixel_frames=geometry.frames,
            count=generated_keyframe_count,
            initial=initial_slots,
        )
    _materialize_state(video_state)

    audio_state = None
    audio_tools = None
    if request.generate_audio:
        initial_audio = None if initial_latents is None else initial_latents.audio
        if initial_latents is not None and initial_audio is None:
            raise RuntimeError("the prior stage returned no audio latent")
        audio_tools = create_audio_latent_tools(
            AudioLatentShape.from_video(
                geometry,
                fps=request.fps,
                reference_aligned=request.reference_aligned_audio,
            )
        )
        audio_state = init_audio_latent_state(
            audio_tools,
            dtype=dtype_policy.latent,
            initial_latent=initial_audio,
        )
        _materialize_state(audio_state)
    return PreparedStage(
        video_state=video_state,
        video_tools=video_tools,
        audio_state=audio_state,
        audio_tools=audio_tools,
        video_keyframes_mask=video_keyframes_mask,
        generated_keyframe_layout=generated_layout,
    )


def run_stage(
    stage: PreparedStage,
    transformer: LTX2TransformerPort,
    text: LTX2EncodedTextConditioning,
    sigmas: Sequence[float],
    *,
    noiser: GaussianNoiser | None = None,
    seed: int | None = None,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    reporter: Reporter | None = None,
    phase: str = "denoise",
    step_kind: DenoiseStepKind = "deterministic-euler",
    stage_number: int | None = None,
    noise_provider: LTX2TransitionNoiseProvider | None = None,
    sparse_attention: SparseAttentionSchedule | None = None,
    transformer_dtype: mx.Dtype | None = None,
) -> StageLatents:
    """Noise, denoise, clear conditioning tokens, and materialize a stage."""
    if noiser is not None and seed is not None:
        raise ValueError("pass either noiser or seed, not both")
    stage_noiser = (
        noiser
        if noiser is not None
        else GaussianNoiser(0 if seed is None else seed, backend=noise_backend)
    )
    video_state = stage_noiser(stage.video_state, scale=float(sigmas[0]))
    audio_state = stage.audio_state
    if audio_state is not None:
        audio_state = stage_noiser(audio_state, scale=float(sigmas[0]))
    context_mask: mx.array | None
    if text.attention_mask.size and bool(mx.all(mx.equal(text.attention_mask, 1)).item()):
        # The encoded-text contract defines this as a binary validity mask.
        # Once every key is proven valid, omitting its additive zero form is
        # exact and lets text cross-attention take the dense attention kernel.
        context_mask = None
    else:
        context_mask = text.attention_mask
    sparse_steps: tuple[SparseAttentionStep | None, ...] = ()
    if sparse_attention is not None and sparse_attention.active:
        target = stage.video_tools.target_shape
        patch_time, patch_height, patch_width = stage.video_tools.patchifier.patch_size
        sparse_steps = stage_steps(
            sparse_attention,
            grid=(
                target.frames // patch_time,
                target.height // patch_height,
                target.width // patch_width,
            ),
            tokens=video_state.latent.shape[1],
        )
    video_state, audio_state = denoise_loop(
        video_state,
        audio_state,
        sigmas,
        transformer=transformer,
        video_context=text.video_encoding,
        audio_context=(text.audio_encoding if audio_state is not None else None),
        video_context_mask=context_mask,
        audio_context_mask=(context_mask if audio_state is not None else None),
        video_keyframes_mask=stage.video_keyframes_mask,
        reporter=reporter,
        phase=phase,
        step_kind=step_kind,
        stage=stage_number,
        noise_provider=noise_provider,
        video_sparse_attention=sparse_steps,
        transformer_dtype=transformer_dtype,
    )
    generated_keyframes = (
        None
        if stage.generated_keyframe_layout is None
        else extract_generated_keyframes(
            video_state.latent, stage.video_tools, stage.generated_keyframe_layout
        )
    )
    video_state = stage.video_tools.unpatchify(stage.video_tools.clear_conditioning(video_state))
    video = video_state.latent
    audio = None
    if audio_state is not None:
        if stage.audio_tools is None:
            raise RuntimeError("audio state has no matching latent tools")
        audio_state = stage.audio_tools.unpatchify(
            stage.audio_tools.clear_conditioning(audio_state)
        )
        audio = audio_state.latent
    mx.eval(
        video,
        *(() if audio is None else (audio,)),
        *(() if generated_keyframes is None else (generated_keyframes,)),
    )
    return StageLatents(
        video=video,
        audio=audio,
        initial_noise_state=stage_noiser.state,
        generated_keyframes=generated_keyframes,
    )


def upscale_between_stages(
    stage: StageLatents,
    upscaler: LTX2SpatialUpscalerPort,
    *,
    reporter: Reporter | None = None,
) -> StageLatents:
    """Upscale video latents and keyframe slots while carrying the materialized audio.

    The slot planes go through the upsampler as one clip, as ltx-pipelines upsamples them.
    """
    video = upsample_video(stage.video, upscaler, reporter=reporter)
    mx.eval(video)
    generated_keyframes = (
        None
        if stage.generated_keyframes is None
        else upsample_video(stage.generated_keyframes, upscaler)
    )
    return StageLatents(
        video=video,
        audio=stage.audio,
        initial_noise_state=stage.initial_noise_state,
        generated_keyframes=generated_keyframes,
    )


def prepare_stage_conditions(
    resources: LTX2Resources,
    factory: LTX2ComponentsProvider,
    sources: Sequence[RawConditionSource],
    geometry: VideoPixelShape,
    *,
    stage: int,
    first_frame_mode: LTX2FirstFrameMode,
    reporter: Reporter,
) -> tuple[EncodedCondition, ...]:
    """Encode one stage's conditions inside one bounded video-encoder lease."""
    if not sources:
        return ()
    uses_exr = any(
        isinstance(source, ImageConditionSource) and source.hdr_authoring is not None
        for source in sources
    )
    encoder_resources = (
        replace(resources, dtype_policy=replace(resources.dtype_policy, video_vae=mx.float32))
        if uses_exr
        else resources
    )
    with factory.video_encoder(encoder_resources) as encoder:
        return prepare_conditions(
            sources,
            geometry,
            encoder,
            resources.capabilities,
            stage=stage,
            compute_dtype=encoder_resources.dtype_policy.video_vae,
            first_frame_mode=first_frame_mode,
            reporter=reporter,
        )


def hdr_reference_receipt(
    request: LTX2Request,
    geometry: VideoPixelShape,
    conditions: Sequence[EncodedCondition],
    *,
    stage: int,
) -> dict[str, object] | None:
    references = tuple(
        item for item in conditions if isinstance(item, VideoConditionByReferenceLatent)
    )
    if not references:
        return None
    if len(references) != 1 or request.hdr_reference is None:
        raise RuntimeError("HDR reference condition receipt does not match the request")
    reference = references[0]
    base_shape = video_latent_shape_from_pixel(geometry)
    base_tokens = base_shape.frames * base_shape.height * base_shape.width
    reference_tokens = reference.token_count
    total_tokens = base_tokens + reference_tokens
    return {
        "stage": stage,
        "path": str(request.hdr_reference.path),
        "strength": request.hdr_reference.strength,
        "latent_shape": reference.latent_shape.to_tuple(),
        "base_tokens": base_tokens,
        "reference_tokens": reference_tokens,
        "total_tokens": total_tokens,
        "sequence_length_ratio": total_tokens / base_tokens,
        "self_attention_pair_ratio": (total_tokens / base_tokens) ** 2,
    }


def release_stage_temporaries() -> None:
    """Release materialized stage products no longer needed downstream."""
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def build_generation_output(
    request: LTX2Request,
    resources: LTX2Resources,
    *,
    frames: VideoFrameStream,
    waveform: mx.array | None,
    sample_rate: int | None,
    frame_count: int | None = None,
    geometry: VideoPixelShape | None = None,
    generated_keyframe_indices: tuple[int, ...] = (),
    adapter_receipts: Sequence[dict[str, object]] = (),
    text_conditioning_receipt: Mapping[str, object] | None = None,
    execution_mode: str | None = None,
    source_latent_stage: str | None = None,
    sampler_plan: DistilledSamplerPlan | None = None,
    initial_noise_state: NoiseStreamState | None = None,
    ancestral_noise_states: tuple[NoiseStreamState | None, NoiseStreamState | None] = (
        None,
        None,
    ),
    hdr_reference_receipts: Sequence[Mapping[str, object]] = (),
) -> LTX2GenerationOutput:
    """Validate terminal products and attach one reproducibility manifest."""
    if request.generate_audio:
        if waveform is None or sample_rate is None:
            frames.close()
            raise RuntimeError("audio generation returned no waveform or sample rate")
        if (
            waveform.ndim != 3
            or waveform.shape[0] != 1
            or waveform.shape[1] != 2
            or waveform.shape[2] < 1
        ):
            frames.close()
            raise RuntimeError(
                f"decoded audio must have shape (1, 2, samples), got {tuple(waveform.shape)}"
            )
        if sample_rate <= 0:
            frames.close()
            raise RuntimeError("decoded audio sample rate must be positive")
    elif waveform is not None or sample_rate is not None:
        frames.close()
        raise RuntimeError("audio output was returned while audio generation was disabled")

    resolved_frame_count = request.frames if frame_count is None else frame_count
    if resolved_frame_count is None:
        frames.close()
        raise ValueError("generation output requires an explicit frame count")
    resolved_geometry = geometry or VideoPixelShape(
        batch=1,
        frames=resolved_frame_count,
        height=request.height,
        width=request.width,
    )

    component_inventory = {
        component.kind.value: {
            "source_path": str(component.source_path),
            "source_fingerprint": component.source_fingerprint,
        }
        for component in resources.components
    }
    if sampler_plan is None:
        sampler_plan = resolve_distilled_sampler_plan(request, resources)
    audio_latent_shape = (
        AudioLatentShape.from_video(
            resolved_geometry,
            fps=request.fps,
            reference_aligned=request.reference_aligned_audio,
        )
        if request.generate_audio
        else None
    )
    adapter_warnings: list[str] = []
    for receipt in adapter_receipts:
        raw_warnings = receipt.get("warnings", ())
        if not isinstance(raw_warnings, (list, tuple)) or any(
            not isinstance(item, str) for item in raw_warnings
        ):
            frames.close()
            raise RuntimeError("LTX-2 adapter receipt contains malformed warnings")
        label = Path(str(receipt.get("path", "adapter"))).name
        adapter_warnings.extend(f"{label}: {warning}" for warning in raw_warnings)
    metadata: dict[str, object] = {
        "kinomlx_version": __version__,
        "model_generation": resources.capabilities.model_generation,
        "model_version": resources.checkpoint.model_version,
        "seed": request.seed,
        "noise_backend": request.noise_backend,
        "noise_compatibility_profile": noise_compatibility_profile(request.noise_backend),
        "initial_noise_state": (
            None if initial_noise_state is None else initial_noise_state.to_metadata()
        ),
        "samplers": sampler_plan.metadata(ancestral_noise_states),
        "stage_1_sigmas": list(stage_sigmas(request.stage_1_sigmas, request.stage_2_sigmas)[0]),
        "stage_2_sigmas": list(stage_sigmas(request.stage_1_sigmas, request.stage_2_sigmas)[1]),
        "sparse_attention": {
            f"stage_{stage}": schedule.to_metadata()
            for stage, schedule in enumerate(request.sparse_attention_schedules(), start=1)
        },
        "generated_keyframe_indices": generated_keyframe_indices,
        "adapter_receipts": tuple(adapter_receipts),
        "check_adapter_weights": resources.execution_policy.check_adapter_weights,
        "warnings": tuple(dict.fromkeys(adapter_warnings)),
        "video_shape": (
            resolved_geometry.batch,
            3,
            resolved_geometry.frames,
            resolved_geometry.height,
            resolved_geometry.width,
        ),
        "video_signal": frames.spec,
        "audio_latent_length_policy": (
            "reference-round" if request.reference_aligned_audio else "coverage-ceil"
        ),
        "audio_latent_shape": (
            None if audio_latent_shape is None else audio_latent_shape.to_tuple()
        ),
        "audio_shape": None if waveform is None else tuple(waveform.shape),
        "audio_sample_rate": sample_rate,
        "checkpoint_path": str(resources.checkpoint.source_path),
        "checkpoint_fingerprint": resources.checkpoint.source_fingerprint,
        "components": component_inventory,
        "dtype_policy": resources.dtype_policy.to_metadata(),
    }
    hdr_recipe = resolve_hdr_recipe(request, resources)
    if hdr_recipe is not None:
        metadata["hdr"] = {
            "authoring": request.hdr,
            "working_transfer": frames.spec.transfer.value,
            "working_primaries": frames.spec.primaries.value,
            "decode_dtype": frames.spec.dtype,
            "recipe": hdr_recipe.to_metadata(),
        }
    if hdr_reference_receipts:
        metadata["hdr_reference_conditioning"] = tuple(
            dict(receipt) for receipt in hdr_reference_receipts
        )
    if execution_mode is not None:
        metadata["execution_mode"] = execution_mode
    if source_latent_stage is not None:
        metadata["source_latent_stage"] = source_latent_stage
    if text_conditioning_receipt is not None:
        metadata["text_conditioning_replay"] = dict(text_conditioning_receipt)
    return LTX2GenerationOutput(
        frames=frames,
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
        diagnostics_provider=lambda: video_decode_diagnostics(frames),
        metadata=metadata,
    )


def _with_video_vae_dtype(resources: LTX2Resources, dtype: mx.Dtype) -> LTX2Resources:
    if dtype == resources.dtype_policy.video_vae:
        return resources
    return replace(resources, dtype_policy=replace(resources.dtype_policy, video_vae=dtype))


def decode_stage_latents(
    request: LTX2Request,
    resources: LTX2Resources,
    latents: StageLatents,
    *,
    geometry: VideoPixelShape,
    components: LTX2ComponentsProvider,
    reporter: Reporter,
    generated_keyframe_indices: tuple[int, ...] = (),
    adapter_receipts: Sequence[dict[str, object]] = (),
    text_conditioning_receipt: Mapping[str, object] | None = None,
    execution_mode: str | None = None,
    source_latent_stage: str | None = None,
    sampler_plan: DistilledSamplerPlan | None = None,
    ancestral_noise_states: tuple[NoiseStreamState | None, NoiseStreamState | None] = (
        None,
        None,
    ),
    hdr_reference_receipts: Sequence[Mapping[str, object]] = (),
) -> LTX2GenerationOutput:
    """Decode one materialized stage product through the common terminal boundary."""
    waveform = None
    sample_rate = None
    if latents.audio is not None:
        with components.audio_decoder(resources) as audio_decoder:
            mel = decode_audio_mel(
                latents.audio,
                audio_decoder,
                reporter=reporter,
            )
        with components.vocoder(resources) as vocoder:
            waveform = vocode_audio(mel, vocoder, reporter=reporter)
            sample_rate = int(vocoder.output_sample_rate)
        del mel

    video_latent = latents.video
    initial_noise_state = latents.initial_noise_state
    del latents
    release_stage_temporaries()
    precision = resolve_video_vae_decode_precision(request.vae_decode_dtype)
    terminal_resources = _with_video_vae_dtype(resources, precision.dtype)
    overflow_fallback: VideoDecoderProvider | None = None
    if precision.overflow_fallback is not None:
        overflow_fallback = partial(
            components.video_decoder,
            _with_video_vae_dtype(resources, precision.overflow_fallback),
        )
    if request.hdr is None:
        signal = ltx23_sdr_signal(
            width=geometry.width,
            height=geometry.height,
            fps=request.fps,
        )
        frames = decode_ltx23_sdr_frames(
            video_latent,
            lambda: components.video_decoder(terminal_resources),
            spec=signal,
            frame_count=geometry.frames,
            tiling_config=request.vae_tiling.to_runtime_config(),
            auto_tiling=request.vae_tiling.mode == "auto",
            vae_cache_limit_gb=request.vae_cache_limit_gb,
            tiling_mode=request.vae_tiling.mode,
            reporter=reporter,
            decoder_seed=request.seed,
            noise_backend=request.noise_backend,
            overflow_fallback=overflow_fallback,
            context_sharpening=request.context_sharpening(),
        )
    else:
        hdr_recipe = resolve_hdr_recipe(request, resources)
        if hdr_recipe is None:
            raise RuntimeError("HDR decode requires resolved recipe facts")
        signal = ltx_hdr_working_signal(
            transfer=hdr_recipe.working_transfer,
            width=geometry.width,
            height=geometry.height,
            fps=request.fps,
        )
        frames = decode_ltx_hdr_working_frames(
            video_latent,
            lambda: components.video_decoder(terminal_resources),
            spec=signal,
            frame_count=geometry.frames,
            tiling_config=request.vae_tiling.to_runtime_config(),
            auto_tiling=request.vae_tiling.mode == "auto",
            vae_cache_limit_gb=request.vae_cache_limit_gb,
            tiling_mode=request.vae_tiling.mode,
            reporter=reporter,
            decoder_seed=request.seed,
            noise_backend=request.noise_backend,
            overflow_fallback=overflow_fallback,
            context_sharpening=request.context_sharpening(),
        )
    return build_generation_output(
        request,
        terminal_resources,
        frames=frames,
        waveform=waveform,
        sample_rate=sample_rate,
        frame_count=geometry.frames,
        geometry=geometry,
        generated_keyframe_indices=generated_keyframe_indices,
        adapter_receipts=adapter_receipts,
        text_conditioning_receipt=text_conditioning_receipt,
        execution_mode=execution_mode,
        source_latent_stage=source_latent_stage,
        sampler_plan=sampler_plan,
        initial_noise_state=initial_noise_state,
        ancestral_noise_states=ancestral_noise_states,
        hdr_reference_receipts=hdr_reference_receipts,
    )


@memory_managed
def generate_ltx2(
    request: LTX2Request,
    resources: LTX2Resources,
    *,
    components: LTX2ComponentsProvider | None = None,
    text_conditioner: LTX2TextConditionerPort | None = None,
    reporter: Reporter | None = None,
    artifact_sink: ArtifactSink | None = None,
    conditioning: DistilledConditioning | None = None,
) -> LTX2GenerationOutput:
    """Generate through explicit, bounded component leases."""
    sink = reporter if reporter is not None else NullReporter()
    artifacts = artifact_sink if artifact_sink is not None else NullArtifactSink()
    factory = components if components is not None else LTX2Components(reporter=sink)
    if len(request.selected_seeds()) != 1:
        raise ValueError("Use LTX2Runner.generate for multiple seeds")
    if request.seeds is not None:
        request = replace(request, seed=request.selected_seeds()[0], seeds=None)
    shared = conditioning if conditioning is not None else DistilledConditioning()
    shared.bind(request)
    prepared = prepare_distilled_recipe(request, resources)
    hdr_recipe = resolve_hdr_recipe(request, resources)
    sources = prepared.sources
    frame_count = prepared.frame_count
    sampler = prepared.sampler
    if shared.text is None:
        shared.text = prepare_text_conditioning(
            request,
            resources,
            text_conditioner=text_conditioner,
            reporter=sink,
            artifacts=artifacts,
        )
    else:
        artifacts.save(
            text_conditioning_artifact(
                prompt=shared.text.prompt,
                video_encoding=shared.text.video_encoding,
                audio_encoding=shared.text.audio_encoding,
                attention_mask=shared.text.attention_mask,
                provenance=shared.text.provenance.to_metadata(),
            )
        )
    text = shared.text
    if shared.frames is not None:
        frame_count = shared.frames
    text_conditioning_receipt = text.replay_receipt
    if frame_count is None:
        with factory.duration_predictor(resources) as duration_predictor:
            frame_count = resolve_auto_frame_count(
                request,
                resources,
                text,
                duration_predictor,
            )
    shared.frames = frame_count
    if sources is None:
        # The predicted count is the one every named position resolves against.
        sources = condition_sources(request, frames=frame_count)
    stage_1_profile, stage_2_profile = resolve_adapter_profiles(request)
    stage_1_geometry = VideoPixelShape(
        batch=1,
        frames=frame_count,
        height=request.height // 2,
        width=request.width // 2,
    )
    if (1, stage_1_geometry) not in shared.stages:
        shared.stages[1, stage_1_geometry] = prepare_stage_conditions(
            resources,
            factory,
            sources,
            stage_1_geometry,
            stage=1,
            first_frame_mode=request.first_frame_mode,
            reporter=sink,
        )
    stage_1_conditions = shared.stages[1, stage_1_geometry]
    if stage_1_conditions:
        artifacts.save(
            media_conditioning_artifact(
                1,
                sources=sources,
                conditions=stage_1_conditions,
                geometry=stage_1_geometry,
                fps=request.fps,
                first_frame_mode=request.first_frame_mode,
            )
        )
    hdr_reference_receipts: list[dict[str, object]] = []
    stage_1_reference_receipt = hdr_reference_receipt(
        request,
        stage_1_geometry,
        stage_1_conditions,
        stage=1,
    )
    if stage_1_reference_receipt is not None:
        hdr_reference_receipts.append(stage_1_reference_receipt)
    stage_1 = prepare_stage(
        request,
        stage_1_geometry,
        dtype_policy=resources.dtype_policy,
        conditions=stage_1_conditions,
        generated_keyframe_count=request.generated_keyframes,
    )
    generated_indices = (
        ()
        if stage_1.generated_keyframe_layout is None
        else stage_1.generated_keyframe_layout.frame_indices
    )
    del stage_1_conditions

    transformer: ComponentLease[LTX2TransformerPort] | None = None
    adapter_receipts: list[dict[str, object]] = []
    final_latents: StageLatents
    noiser = GaussianNoiser(request.seed, backend=request.noise_backend)
    stage_1_schedule, stage_2_schedule = stage_sigmas(
        request.stage_1_sigmas, request.stage_2_sigmas
    )
    stage_1_noise = sampler.stage_1.transition_noise(
        transitions=len(stage_1_schedule) - 1, backend=request.noise_backend
    )
    stage_2_noise = sampler.stage_2.transition_noise(
        transitions=len(stage_2_schedule) - 1, backend=request.noise_backend
    )
    try:
        with reported_phase(sink, "load stage 1 transformer"):
            transformer = factory.transformer(
                resources,
                stage_1_profile,
            )
            validate_hdr_adapter_placement(
                hdr_recipe,
                transformer.value.adapter_receipts,
                stage=1,
            )
            stages = (1, 2) if stage_1_profile == stage_2_profile else (1,)
            for receipt in transformer.value.adapter_receipts:
                metadata = receipt.to_metadata()
                metadata["stages"] = stages
                adapter_receipts.append(metadata)
        stage_1_latents = run_stage(
            stage_1,
            transformer.value,
            text,
            stage_1_schedule,
            noiser=noiser,
            reporter=sink,
            phase="distilled stage 1",
            step_kind=sampler.stage_1.step_kind,
            stage_number=1,
            noise_provider=stage_1_noise,
            sparse_attention=request.sparse_attention_schedules()[0],
            transformer_dtype=resources.dtype_policy.transformer,
        )
        del stage_1
        artifacts.save(
            distilled_stage_latents_artifact(
                1,
                video_latent=stage_1_latents.video,
                audio_latent=stage_1_latents.audio,
                final=False,
                noise_state=stage_1_latents.initial_noise_state,
                generated_keyframes=stage_1_latents.generated_keyframes,
            )
        )

        if stage_1_profile != stage_2_profile:
            transformer.close()
            transformer = None

        with factory.spatial_upscaler(resources) as upscaler:
            upscaled = upscale_between_stages(
                stage_1_latents,
                upscaler,
                reporter=sink,
            )
        del stage_1_latents
        release_stage_temporaries()

        stage_2_geometry = VideoPixelShape(
            batch=1,
            frames=frame_count,
            height=request.height,
            width=request.width,
        )
        if (2, stage_2_geometry) not in shared.stages:
            shared.stages[2, stage_2_geometry] = prepare_stage_conditions(
                resources,
                factory,
                sources,
                stage_2_geometry,
                stage=2,
                first_frame_mode=request.first_frame_mode,
                reporter=sink,
            )
        stage_2_conditions = shared.stages[2, stage_2_geometry]
        if stage_2_conditions:
            artifacts.save(
                media_conditioning_artifact(
                    2,
                    sources=sources,
                    conditions=stage_2_conditions,
                    geometry=stage_2_geometry,
                    fps=request.fps,
                    first_frame_mode=request.first_frame_mode,
                )
            )
        stage_2_reference_receipt = hdr_reference_receipt(
            request,
            stage_2_geometry,
            stage_2_conditions,
            stage=2,
        )
        if stage_2_reference_receipt is not None:
            hdr_reference_receipts.append(stage_2_reference_receipt)
        stage_2 = prepare_stage(
            request,
            stage_2_geometry,
            dtype_policy=resources.dtype_policy,
            conditions=stage_2_conditions,
            initial_latents=upscaled,
            generated_keyframe_count=request.generated_keyframes,
        )
        del stage_2_conditions, upscaled

        if transformer is None:
            with reported_phase(sink, "load stage 2 transformer"):
                transformer = factory.transformer(
                    resources,
                    stage_2_profile,
                )
                validate_hdr_adapter_placement(
                    hdr_recipe,
                    transformer.value.adapter_receipts,
                    stage=2,
                )
                for receipt in transformer.value.adapter_receipts:
                    metadata = receipt.to_metadata()
                    metadata["stages"] = (2,)
                    adapter_receipts.append(metadata)
        final_latents = run_stage(
            stage_2,
            transformer.value,
            text,
            stage_2_schedule,
            noiser=noiser,
            reporter=sink,
            phase="distilled stage 2",
            step_kind=sampler.stage_2.step_kind,
            stage_number=2,
            noise_provider=stage_2_noise,
            sparse_attention=request.sparse_attention_schedules()[1],
            transformer_dtype=resources.dtype_policy.transformer,
        )
        del stage_2
        artifacts.save(
            distilled_stage_latents_artifact(
                2,
                video_latent=final_latents.video,
                audio_latent=final_latents.audio,
                final=True,
                noise_state=final_latents.initial_noise_state,
                generated_keyframes=final_latents.generated_keyframes,
            )
        )
    finally:
        if transformer is not None:
            transformer.close()

    del text
    release_stage_temporaries()

    return decode_stage_latents(
        request,
        resources,
        final_latents,
        geometry=VideoPixelShape(
            batch=1,
            frames=frame_count,
            height=request.height,
            width=request.width,
        ),
        components=factory,
        reporter=sink,
        generated_keyframe_indices=generated_indices,
        adapter_receipts=adapter_receipts,
        text_conditioning_receipt=text_conditioning_receipt,
        sampler_plan=sampler,
        ancestral_noise_states=(
            None if stage_1_noise is None else stage_1_noise.state,
            None if stage_2_noise is None else stage_2_noise.state,
        ),
        hdr_reference_receipts=hdr_reference_receipts,
    )


__all__ = [
    "ANCESTRAL_SEED_OFFSETS",
    "DistilledSamplerPlan",
    "DistilledStageSampler",
    "PreparedDistilledRecipe",
    "PreparedStage",
    "StageLatents",
    "build_generation_output",
    "condition_sources",
    "decode_stage_latents",
    "effective_stage_samplers",
    "generate_ltx2",
    "hdr_reference_receipt",
    "prepare_distilled_recipe",
    "prepare_stage",
    "prepare_stage_conditions",
    "prepare_text_conditioning",
    "release_stage_temporaries",
    "resolve_adapter_profiles",
    "resolve_distilled_sampler_plan",
    "resolve_stage_samplers",
    "run_stage",
    "sampling_facts",
    "stage_sampler_choices",
    "upscale_between_stages",
    "validate_distilled_request",
]
