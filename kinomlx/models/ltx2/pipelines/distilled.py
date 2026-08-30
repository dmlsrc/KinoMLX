"""One public two-stage distilled recipe for prepared LTX-2 resources."""

from __future__ import annotations

import gc
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx import __version__
from kinomlx.artifacts import ArtifactSink, NullArtifactSink
from kinomlx.components import ComponentLease
from kinomlx.lora.loading import LoRAConfig, LoRAProfile, lora_configs_for_stage
from kinomlx.media.frames import VideoFrameStream
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.samplers.noise import NoiseStreamState, noise_compatibility_profile
from kinomlx.samplers.noisers import GaussianNoiser, SeededGaussianNoise
from kinomlx.types import DEFAULT_NOISE_BACKEND, LatentState, NoiseBackend, VideoPixelShape

from ..artifacts import (
    distilled_stage_latents_artifact,
    media_conditioning_artifact,
    text_conditioning_artifact,
)
from ..components import (
    DistilledComponents,
    DurationPredictorPort,
    NativeLTX2Components,
    SpatialUpscalerPort,
    TransformerPort,
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
    decode_audio_mel,
    decode_ltx23_sdr_frames,
    decode_ltx_hdr_working_frames,
    video_decode_diagnostics,
    vocode_audio,
)
from ..denoise import DenoiseStepKind, TransitionNoiseProvider, denoise_loop
from ..generated_keyframes import (
    GeneratedKeyframeLayout,
    append_generated_keyframe_slots,
    generated_keyframe_indices,
)
from ..hdr_profile import resolve_hdr_recipe, validate_hdr_adapter_placement
from ..precision import LTX2DTypePolicy, resolve_video_vae_decode_dtype
from ..resources import LTX2Resources
from ..runner import GenerationOutput
from ..sigmas import DISTILLED_STAGE_1_SIGMAS, DISTILLED_STAGE_2_SIGMAS
from ..signals import ltx23_sdr_signal, ltx_hdr_working_signal
from ..state import (
    apply_encoded_conditions,
    create_audio_latent_tools,
    create_video_latent_tools,
    init_audio_latent_state,
    init_video_latent_state,
)
from ..text_conditioning import (
    EncodedTextConditioning,
    NativeTextConditioner,
    TextConditioner,
)
from ..types import AudioLatentShape, DistilledRequest, video_latent_shape_from_pixel
from ..upscaler import upsample_video


@dataclass(frozen=True)
class StageLatents:
    """Materialized video and optional audio latents crossing a station."""

    video: mx.array
    audio: mx.array | None = None
    initial_noise_state: NoiseStreamState | None = None


@dataclass(frozen=True)
class PreparedStage:
    """Patchified latent state ready for one denoise schedule."""

    video_state: LatentState
    video_tools: VideoLatentTools
    audio_state: LatentState | None
    audio_tools: AudioLatentTools | None
    video_keyframes_mask: mx.array | None = None
    generated_keyframe_layout: GeneratedKeyframeLayout | None = None


@dataclass(frozen=True)
class DistilledSamplerPlan:
    """Resolved step behavior, effective policy, and independent transition seed."""

    policy: str
    stage_1: DenoiseStepKind
    stage_2: DenoiseStepKind
    ancestral_seed: int | None


@dataclass(frozen=True)
class PreparedDistilledRecipe:
    """Load-free request facts resolved before any heavyweight station."""

    sources: tuple[RawConditionSource, ...]
    frame_count: int | None
    sampler: DistilledSamplerPlan


def _lora_configs(request: DistilledRequest) -> tuple[LoRAConfig, ...]:
    return tuple(
        LoRAConfig(
            path=selection.path,
            strength=selection.strength,
            stage_1_strength=selection.stage_1_strength,
            stage_2_strength=selection.stage_2_strength,
            exclude=selection.exclude,
        )
        for selection in request.resolved_loras()
    )


def resolve_lora_profiles(
    request: DistilledRequest,
) -> tuple[LoRAProfile, LoRAProfile]:
    """Resolve ordered, zero-free, complete profiles for both stages."""
    configured = _lora_configs(request)
    return (
        tuple(lora_configs_for_stage(configured, 1)),
        tuple(lora_configs_for_stage(configured, 2)),
    )


def condition_sources(request: DistilledRequest) -> tuple[RawConditionSource, ...]:
    """Resolve caller-owned condition intent without loading an encoder."""
    sources: list[RawConditionSource] = []
    if request.image is not None:
        hdr_authoring = request.hdr if request.image.path.suffix.lower() == ".exr" else None
        sources.append(
            ImageConditionSource.from_config(
                request.image,
                hdr_authoring=hdr_authoring,
            )
        )
    if request.hdr_reference is not None:
        sources.append(HDRReferenceConditionSource.from_config(request.hdr_reference))
    return tuple(sources)


def _requested_frame_count(request: DistilledRequest) -> int | None:
    return request.frames


def _effective_sampler_policy(
    request: DistilledRequest,
    resources: LTX2Resources,
) -> str:
    """Map public sampler intent to one concrete two-stage policy."""
    declared_policy = resources.capabilities.sampler_policy
    if request.sampler == "auto":
        return declared_policy
    if request.sampler == "deterministic":
        return "deterministic-euler-two-stage"
    if request.sampler == "ancestral":
        return "ancestral-stage1-deterministic-stage2"
    raise ValueError(f"unsupported sampler override {request.sampler!r}")


def resolve_distilled_sampler_plan(
    request: DistilledRequest,
    resources: LTX2Resources,
) -> DistilledSamplerPlan:
    """Resolve checkpoint-default or explicitly overridden stage behavior."""
    policy = _effective_sampler_policy(request, resources)
    if policy == "deterministic-euler-two-stage":
        return DistilledSamplerPlan(
            policy=policy,
            stage_1="deterministic-euler",
            stage_2="deterministic-euler",
            ancestral_seed=None,
        )
    if policy == "ancestral-stage1-deterministic-stage2":
        ancestral_seed = request.seed + 10_000
        if ancestral_seed >= 2**64:
            raise ValueError(
                "ancestral seed derivation requires seed + 10000 to fit an unsigned 64-bit integer"
            )
        return DistilledSamplerPlan(
            policy=policy,
            stage_1="ancestral-rf",
            stage_2="deterministic-euler",
            ancestral_seed=ancestral_seed,
        )
    raise ValueError(f"checkpoint declares unsupported sampler policy {policy!r}")


def validate_distilled_request(
    request: DistilledRequest,
    resources: LTX2Resources,
    sources: Sequence[RawConditionSource],
) -> None:
    """Fail before model loading when resources cannot run this request."""
    capabilities = resources.capabilities
    generation = capabilities.model_generation
    if "distilled" not in capabilities.recipe_families:
        raise ValueError("checkpoint does not support the distilled recipe")
    if request.frames is None and not capabilities.duration_available:
        raise ValueError(f"LTX-{generation} resources do not provide an auto-duration head")
    if request.generated_keyframes and not capabilities.generated_keyframes:
        raise ValueError(f"LTX-{generation} does not provide generated-keyframe positions")
    if request.generated_keyframes and request.frames is not None:
        generated_keyframe_indices(request.frames, request.generated_keyframes)
    if request.generate_audio and not capabilities.generates_audio:
        raise ValueError("checkpoint does not support joint audio generation")
    if request.image is not None and request.image.path.suffix.lower() == ".exr":
        if request.hdr is None:
            raise ValueError("EXR conditioning requires an explicit --hdr signal interpretation")
        if generation != "2.5":
            raise ValueError("EXR conditioning is currently supported only by LTX-2.5 native HDR")
    if request.hdr_reference is not None and request.hdr is None:
        raise ValueError("hdr_reference requires an explicit HDR output mode")
    if request.hdr_reference is not None and generation != "2.3":
        raise ValueError("hdr_reference is supported only by the LTX-2.3 HDR IC-LoRA recipe")
    resolve_hdr_recipe(request, resources)
    supported = frozenset(capabilities.condition_families)
    for source in sources:
        if isinstance(source, HDRReferenceConditionSource):
            continue
        if source.family not in supported:
            raise ValueError(f"checkpoint does not support {source.family} conditioning")


def _apply_execution_policy(resources: LTX2Resources) -> None:
    limit = resources.execution_policy.mlx_cache_limit_bytes
    if limit is not None:
        mx.set_cache_limit(limit)


def prepare_distilled_recipe(
    request: DistilledRequest,
    resources: LTX2Resources,
) -> PreparedDistilledRecipe:
    """Validate one request and apply its immutable execution policy."""
    request.validate_for_generation()
    sources = condition_sources(request)
    validate_distilled_request(request, resources, sources)
    frame_count = _requested_frame_count(request)
    sampler = resolve_distilled_sampler_plan(request, resources)
    _apply_execution_policy(resources)
    return PreparedDistilledRecipe(
        sources=sources,
        frame_count=frame_count,
        sampler=sampler,
    )


def resolve_auto_frame_count(
    request: DistilledRequest,
    resources: LTX2Resources,
    text: EncodedTextConditioning,
    predictor: DurationPredictorPort,
) -> int:
    """Resolve one omitted frame count from connector outputs and validate dependents."""
    frames = predictor.predict_num_frames(
        text.video_encoding,
        text.audio_encoding,
        frame_rate=request.fps,
        temporal_compression_ratio=resources.capabilities.video_compression.time,
    )
    if request.image is not None and request.image.frame_index >= frames:
        raise ValueError(
            f"image frame_index {request.image.frame_index} is outside predicted {frames} frames"
        )
    if request.generated_keyframes:
        generated_keyframe_indices(frames, request.generated_keyframes)
    return frames


def prepare_text_conditioning(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    text_conditioner: TextConditioner | None,
    reporter: Reporter,
    artifacts: ArtifactSink,
    emit_artifact: bool = True,
) -> EncodedTextConditioning:
    selected = text_conditioner if text_conditioner is not None else NativeTextConditioner()
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
    request: DistilledRequest,
    geometry: VideoPixelShape,
    *,
    dtype_policy: LTX2DTypePolicy,
    conditions: Sequence[EncodedCondition] = (),
    initial_latents: StageLatents | None = None,
    generated_keyframe_count: int = 0,
) -> PreparedStage:
    """Construct and materialize one conditioned stage state."""
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
        video_state, generated_layout, video_keyframes_mask = append_generated_keyframe_slots(
            video_state,
            video_tools,
            pixel_frames=geometry.frames,
            count=generated_keyframe_count,
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
    transformer: TransformerPort,
    text: EncodedTextConditioning,
    sigmas: Sequence[float],
    *,
    noiser: GaussianNoiser | None = None,
    seed: int | None = None,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    reporter: Reporter | None = None,
    phase: str = "denoise",
    step_kind: DenoiseStepKind = "deterministic-euler",
    stage_number: int | None = None,
    noise_provider: TransitionNoiseProvider | None = None,
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
        # exact and lets supported text cross-attention calls use STEEL.
        context_mask = None
    else:
        context_mask = text.attention_mask
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
    )
    return StageLatents(
        video=video,
        audio=audio,
        initial_noise_state=stage_noiser.state,
    )


def upscale_between_stages(
    stage: StageLatents,
    upscaler: SpatialUpscalerPort,
    *,
    reporter: Reporter | None = None,
) -> StageLatents:
    """Upscale video latents while carrying the materialized audio product."""
    video = upsample_video(stage.video, upscaler, reporter=reporter)
    mx.eval(video)
    return StageLatents(
        video=video,
        audio=stage.audio,
        initial_noise_state=stage.initial_noise_state,
    )


def prepare_stage_conditions(
    resources: LTX2Resources,
    factory: DistilledComponents,
    sources: Sequence[RawConditionSource],
    geometry: VideoPixelShape,
    *,
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
            compute_dtype=encoder_resources.dtype_policy.video_vae,
            reporter=reporter,
        )


def hdr_reference_receipt(
    request: DistilledRequest,
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
    """Release materialized station products no longer needed downstream."""
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


@contextmanager
def reported_phase(reporter: Reporter, phase: str) -> Iterator[None]:
    """Report one lexical station boundary without letting UX break cleanup."""
    with suppress(Exception):
        reporter.phase_start(phase)
    try:
        yield
    finally:
        with suppress(Exception):
            reporter.phase_end(phase)


def build_generation_output(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    frames: VideoFrameStream,
    waveform: mx.array | None,
    sample_rate: int | None,
    frame_count: int | None = None,
    geometry: VideoPixelShape | None = None,
    generated_keyframe_indices: tuple[int, ...] = (),
    lora_receipts: Sequence[dict[str, object]] = (),
    text_conditioning_receipt: Mapping[str, object] | None = None,
    execution_mode: str | None = None,
    source_latent_stage: str | None = None,
    sampler_plan: DistilledSamplerPlan | None = None,
    initial_noise_state: NoiseStreamState | None = None,
    ancestral_noise_state: NoiseStreamState | None = None,
    hdr_reference_receipts: Sequence[Mapping[str, object]] = (),
) -> GenerationOutput:
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
            "cache_path": None if component.cache_path is None else str(component.cache_path),
        }
        for component in resources.components
    }
    if sampler_plan is None:
        sampler_policy = _effective_sampler_policy(request, resources)
        ancestral_seed = (
            request.seed + 10_000
            if sampler_policy == "ancestral-stage1-deterministic-stage2"
            else None
        )
    else:
        sampler_policy = sampler_plan.policy
        ancestral_seed = sampler_plan.ancestral_seed
    audio_latent_shape = (
        AudioLatentShape.from_video(
            resolved_geometry,
            fps=request.fps,
            reference_aligned=request.reference_aligned_audio,
        )
        if request.generate_audio
        else None
    )
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
        "ancestral_noise_state": (
            None if ancestral_noise_state is None else ancestral_noise_state.to_metadata()
        ),
        "sampler_override": None if request.sampler == "auto" else request.sampler,
        "sampler_policy": sampler_policy,
        "ancestral_noise_seed": ancestral_seed,
        "generated_keyframe_indices": generated_keyframe_indices,
        "lora_receipts": tuple(lora_receipts),
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
    return GenerationOutput(
        frames=frames,
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
        diagnostics_provider=lambda: video_decode_diagnostics(frames),
        metadata=metadata,
    )


def decode_stage_latents(
    request: DistilledRequest,
    resources: LTX2Resources,
    latents: StageLatents,
    *,
    geometry: VideoPixelShape,
    components: DistilledComponents,
    reporter: Reporter,
    generated_keyframe_indices: tuple[int, ...] = (),
    lora_receipts: Sequence[dict[str, object]] = (),
    text_conditioning_receipt: Mapping[str, object] | None = None,
    execution_mode: str | None = None,
    source_latent_stage: str | None = None,
    sampler_plan: DistilledSamplerPlan | None = None,
    ancestral_noise_state: NoiseStreamState | None = None,
    hdr_reference_receipts: Sequence[Mapping[str, object]] = (),
) -> GenerationOutput:
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
    decode_dtype = resolve_video_vae_decode_dtype(
        request.vae_decode_dtype,
        hdr=request.hdr is not None,
        default=resources.dtype_policy.video_vae,
    )
    terminal_resources = resources
    if decode_dtype != resources.dtype_policy.video_vae:
        terminal_resources = replace(
            resources,
            dtype_policy=replace(resources.dtype_policy, video_vae=decode_dtype),
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
            tiling_mode=request.vae_tiling.mode,
            reporter=reporter,
            decoder_seed=request.seed,
            noise_backend=request.noise_backend,
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
            tiling_mode=request.vae_tiling.mode,
            reporter=reporter,
            decoder_seed=request.seed,
            noise_backend=request.noise_backend,
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
        lora_receipts=lora_receipts,
        text_conditioning_receipt=text_conditioning_receipt,
        execution_mode=execution_mode,
        source_latent_stage=source_latent_stage,
        sampler_plan=sampler_plan,
        initial_noise_state=initial_noise_state,
        ancestral_noise_state=ancestral_noise_state,
        hdr_reference_receipts=hdr_reference_receipts,
    )


def generate_distilled(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    components: DistilledComponents | None = None,
    text_conditioner: TextConditioner | None = None,
    reporter: Reporter | None = None,
    artifact_sink: ArtifactSink | None = None,
) -> GenerationOutput:
    """Generate through explicit, bounded component leases."""
    sink = reporter if reporter is not None else NullReporter()
    artifacts = artifact_sink if artifact_sink is not None else NullArtifactSink()
    factory = components if components is not None else NativeLTX2Components(reporter=sink)
    prepared = prepare_distilled_recipe(request, resources)
    hdr_recipe = resolve_hdr_recipe(request, resources)
    sources = prepared.sources
    frame_count = prepared.frame_count
    sampler = prepared.sampler
    text = prepare_text_conditioning(
        request,
        resources,
        text_conditioner=text_conditioner,
        reporter=sink,
        artifacts=artifacts,
    )
    text_conditioning_receipt = text.replay_receipt
    if frame_count is None:
        with factory.duration_predictor(resources) as duration_predictor:
            frame_count = resolve_auto_frame_count(
                request,
                resources,
                text,
                duration_predictor,
            )
    stage_1_profile, stage_2_profile = resolve_lora_profiles(request)
    stage_1_geometry = VideoPixelShape(
        batch=1,
        frames=frame_count,
        height=request.height // 2,
        width=request.width // 2,
    )
    stage_1_conditions = prepare_stage_conditions(
        resources,
        factory,
        sources,
        stage_1_geometry,
        reporter=sink,
    )
    if stage_1_conditions:
        artifacts.save(
            media_conditioning_artifact(
                1,
                sources=sources,
                conditions=stage_1_conditions,
                geometry=stage_1_geometry,
                fps=request.fps,
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

    transformer: ComponentLease[TransformerPort] | None = None
    lora_receipts: list[dict[str, object]] = []
    final_latents: StageLatents
    noiser = GaussianNoiser(request.seed, backend=request.noise_backend)
    transition_noise = (
        None
        if sampler.ancestral_seed is None
        else SeededGaussianNoise(
            sampler.ancestral_seed,
            backend=request.noise_backend,
        )
    )
    try:
        with reported_phase(sink, "load stage 1 transformer"):
            transformer = factory.transformer(
                resources,
                stage_1_profile,
            )
            validate_hdr_adapter_placement(
                hdr_recipe,
                transformer.value.lora_receipts,
                stage=1,
            )
            stages = (1, 2) if stage_1_profile == stage_2_profile else (1,)
            for receipt in transformer.value.lora_receipts:
                metadata = receipt.to_metadata()
                metadata["stages"] = stages
                lora_receipts.append(metadata)
        stage_1_latents = run_stage(
            stage_1,
            transformer.value,
            text,
            DISTILLED_STAGE_1_SIGMAS,
            noiser=noiser,
            reporter=sink,
            phase="distilled stage 1",
            step_kind=sampler.stage_1,
            stage_number=1,
            noise_provider=transition_noise,
        )
        del stage_1
        artifacts.save(
            distilled_stage_latents_artifact(
                1,
                video_latent=stage_1_latents.video,
                audio_latent=stage_1_latents.audio,
                final=False,
                noise_state=stage_1_latents.initial_noise_state,
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
        stage_2_conditions = prepare_stage_conditions(
            resources,
            factory,
            sources,
            stage_2_geometry,
            reporter=sink,
        )
        if stage_2_conditions:
            artifacts.save(
                media_conditioning_artifact(
                    2,
                    sources=sources,
                    conditions=stage_2_conditions,
                    geometry=stage_2_geometry,
                    fps=request.fps,
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
                    transformer.value.lora_receipts,
                    stage=2,
                )
                for receipt in transformer.value.lora_receipts:
                    metadata = receipt.to_metadata()
                    metadata["stages"] = (2,)
                    lora_receipts.append(metadata)
        final_latents = run_stage(
            stage_2,
            transformer.value,
            text,
            DISTILLED_STAGE_2_SIGMAS,
            noiser=noiser,
            reporter=sink,
            phase="distilled stage 2",
            step_kind=sampler.stage_2,
            stage_number=2,
        )
        del stage_2
        artifacts.save(
            distilled_stage_latents_artifact(
                2,
                video_latent=final_latents.video,
                audio_latent=final_latents.audio,
                final=True,
                noise_state=final_latents.initial_noise_state,
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
        lora_receipts=lora_receipts,
        text_conditioning_receipt=text_conditioning_receipt,
        sampler_plan=sampler,
        ancestral_noise_state=(None if transition_noise is None else transition_noise.state),
        hdr_reference_receipts=hdr_reference_receipts,
    )


__all__ = [
    "DistilledSamplerPlan",
    "DistilledRequest",
    "PreparedDistilledRecipe",
    "PreparedStage",
    "StageLatents",
    "build_generation_output",
    "condition_sources",
    "decode_stage_latents",
    "generate_distilled",
    "hdr_reference_receipt",
    "prepare_stage",
    "prepare_stage_conditions",
    "prepare_distilled_recipe",
    "prepare_text_conditioning",
    "release_stage_temporaries",
    "reported_phase",
    "resolve_lora_profiles",
    "resolve_distilled_sampler_plan",
    "run_stage",
    "upscale_between_stages",
    "validate_distilled_request",
]
