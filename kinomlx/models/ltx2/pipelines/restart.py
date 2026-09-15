"""Restart the distilled recipe from a saved stage checkpoint."""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from kinomlx.artifacts import ArtifactSink, NullArtifactSink
from kinomlx.config_rules import validate_record
from kinomlx.io.safetensors import load_weights_with_metadata
from kinomlx.memory_policy import memory_managed
from kinomlx.reporting import NullReporter, Reporter, reported_phase
from kinomlx.samplers.noise import NoiseStreamState
from kinomlx.types import VideoPixelShape

from ..artifacts import distilled_stage_latents_artifact, media_conditioning_artifact
from ..components import DistilledComponents, NativeLTX2Components
from ..generation import GenerationOutput
from ..hdr_profile import resolve_hdr_recipe, validate_hdr_adapter_placement
from ..noising import GaussianNoiser
from ..resources import LTX2Resources
from ..restart import DistilledRestart, LatentStage
from ..sigmas import stage_sigmas
from ..text_conditioning import NativeTextConditioner, TextConditioner
from ..types import AudioLatentShape, DistilledRequest, video_latent_shape_from_pixel
from .distilled import (
    StageLatents,
    condition_sources,
    decode_stage_latents,
    hdr_reference_receipt,
    prepare_stage,
    prepare_stage_conditions,
    prepare_text_conditioning,
    release_stage_temporaries,
    resolve_adapter_profiles,
    resolve_distilled_sampler_plan,
    run_stage,
    upscale_between_stages,
    validate_distilled_request,
)

_LATENT_DTYPES = frozenset({mx.bfloat16, mx.float16, mx.float32})


def load_stage_latents(
    path: Path,
    *,
    stage: LatentStage,
    geometry: VideoPixelShape,
    fps: float,
    generate_audio: bool,
    reference_aligned_audio: bool,
    reporter: Reporter,
    source_model_generation: str | None,
) -> StageLatents:
    """Load and structurally validate one KinoMLX distilled latent sidecar."""

    generation_label = (
        "unknown-generation"
        if source_model_generation is None
        else f"LTX-{source_model_generation}"
    )
    with reported_phase(reporter, f"load {stage} latents"):
        tensors, metadata = load_weights_with_metadata(path)
        try:
            video = tensors["video_latent"]
        except KeyError as exc:
            raise ValueError(
                f"{generation_label} {stage} restart artifact has no video_latent tensor"
            ) from exc
        expected_video = video_latent_shape_from_pixel(geometry).to_tuple()
        if tuple(video.shape) != expected_video:
            raise ValueError(
                f"{generation_label} {stage} video latent shape {tuple(video.shape)} "
                f"does not match {expected_video}"
            )
        _validate_latent_values(
            video,
            name="video_latent",
            source=path,
            generation_label=generation_label,
            stage=stage,
        )

        audio = tensors.get("audio_latent") if generate_audio else None
        if generate_audio:
            if audio is None:
                raise ValueError(
                    f"{generation_label} {stage} restart artifact has no audio_latent tensor"
                )
            expected_audio = AudioLatentShape.from_video(
                geometry,
                fps=fps,
                reference_aligned=reference_aligned_audio,
            ).to_tuple()
            if tuple(audio.shape) != expected_audio:
                raise ValueError(
                    f"{generation_label} {stage} audio latent shape {tuple(audio.shape)} "
                    f"does not match {expected_audio}"
                )
            _validate_latent_values(
                audio,
                name="audio_latent",
                source=path,
                generation_label=generation_label,
                stage=stage,
            )
        mx.eval(video, *(() if audio is None else (audio,)))
    noise_state = NoiseStreamState.from_artifact_metadata(metadata)
    return StageLatents(
        video=video,
        audio=audio,
        initial_noise_state=noise_state,
    )


def _validate_latent_values(
    value: mx.array,
    *,
    name: str,
    source: Path,
    generation_label: str,
    stage: LatentStage,
) -> None:
    if value.dtype not in _LATENT_DTYPES:
        raise ValueError(
            f"{generation_label} {stage} {name} in {source} must use a supported "
            f"floating dtype, got {value.dtype}"
        )
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError(
            f"{generation_label} {stage} {name} in {source} must contain only finite values"
        )


def _with_restart_metadata(
    output: GenerationOutput,
    *,
    source_model_generation: str | None,
) -> GenerationOutput:
    metadata = dict(output.metadata)
    metadata["source_model_generation"] = source_model_generation
    return replace(output, metadata=metadata)


def _decode_saved_stage(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    latent_stage: LatentStage,
    latent_path: Path,
    source_model_generation: str | None,
    factory: DistilledComponents,
    reporter: Reporter,
    artifacts: ArtifactSink,
) -> GenerationOutput:
    if request.frames is None:
        raise ValueError("decode restart requires a resolved source frame count")
    scale = 2 if latent_stage == "stage-1" else 1
    geometry = VideoPixelShape(
        batch=1,
        frames=request.frames,
        height=request.height // scale,
        width=request.width // scale,
    )
    latents = load_stage_latents(
        latent_path,
        stage=latent_stage,
        geometry=geometry,
        fps=request.fps,
        generate_audio=request.generate_audio,
        reference_aligned_audio=request.reference_aligned_audio,
        reporter=reporter,
        source_model_generation=source_model_generation,
    )
    # Re-emit the one latent product this station read, so the bundle it writes
    # can itself be decode-restarted.
    artifacts.save(
        distilled_stage_latents_artifact(
            1 if latent_stage == "stage-1" else 2,
            video_latent=latents.video,
            audio_latent=latents.audio,
            final=latent_stage != "stage-1",
            noise_state=latents.initial_noise_state,
        )
    )
    output = decode_stage_latents(
        request,
        resources,
        latents,
        geometry=geometry,
        components=factory,
        reporter=reporter,
        execution_mode="restart-decode",
        source_latent_stage=latent_stage,
    )
    return _with_restart_metadata(
        output,
        source_model_generation=source_model_generation,
    )


def _restart_stage_2(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    latent_path: Path,
    source_model_generation: str | None,
    factory: DistilledComponents,
    text_conditioner: TextConditioner | None,
    reporter: Reporter,
    artifacts: ArtifactSink,
) -> GenerationOutput:
    if request.frames is None:
        raise ValueError("stage-2 restart requires a resolved source frame count")
    if request.text_conditioning is None:
        raise ValueError("stage-2 restart requires saved text conditioning")
    if request.generate_audio and not resources.capabilities.generates_audio:
        raise ValueError("selected decoder resources do not provide audio generation")

    stage_1_geometry = VideoPixelShape(
        batch=1,
        frames=request.frames,
        height=request.height // 2,
        width=request.width // 2,
    )
    stage_1_latents = load_stage_latents(
        latent_path,
        stage="stage-1",
        geometry=stage_1_geometry,
        fps=request.fps,
        generate_audio=request.generate_audio,
        reference_aligned_audio=request.reference_aligned_audio,
        reporter=reporter,
        source_model_generation=source_model_generation,
    )
    if stage_1_latents.initial_noise_state is None and request.noise_backend != "mlx":
        raise ValueError(
            "legacy stage-1 latents have no Torch-MPS noise position; "
            "use noise_backend='mlx' or regenerate stage 1"
        )
    # Stage-2 re-emits the saved text and the stage-1 latent it consumed, so
    # the bundle it writes can itself be restarted at stage 2.
    text = prepare_text_conditioning(
        request,
        resources,
        text_conditioner=text_conditioner,
        reporter=reporter,
        artifacts=artifacts,
    )
    text_conditioning_receipt = text.replay_receipt
    sources = condition_sources(request)

    artifacts.save(
        distilled_stage_latents_artifact(
            1,
            video_latent=stage_1_latents.video,
            audio_latent=stage_1_latents.audio,
            final=False,
            noise_state=stage_1_latents.initial_noise_state,
        )
    )
    with factory.spatial_upscaler(resources) as upscaler:
        upscaled = upscale_between_stages(
            stage_1_latents,
            upscaler,
            reporter=reporter,
        )
    del stage_1_latents
    release_stage_temporaries()

    stage_2_geometry = VideoPixelShape(
        batch=1,
        frames=request.frames,
        height=request.height,
        width=request.width,
    )
    stage_2_conditions = prepare_stage_conditions(
        resources,
        factory,
        sources,
        stage_2_geometry,
        reporter=reporter,
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
    reference_receipt = hdr_reference_receipt(
        request,
        stage_2_geometry,
        stage_2_conditions,
        stage=2,
    )
    stage_2 = prepare_stage(
        request,
        stage_2_geometry,
        dtype_policy=resources.dtype_policy,
        conditions=stage_2_conditions,
        initial_latents=upscaled,
    )
    initial_noise_state = upscaled.initial_noise_state
    del stage_2_conditions, upscaled

    sampler = resolve_distilled_sampler_plan(request, resources)
    hdr_recipe = resolve_hdr_recipe(request, resources)
    stage_2_profile = resolve_adapter_profiles(request)[1]
    adapter_receipts: list[dict[str, object]] = []
    transformer = None
    try:
        with reported_phase(reporter, "load stage 2 transformer"):
            transformer = factory.transformer(resources, stage_2_profile)
            validate_hdr_adapter_placement(
                hdr_recipe,
                transformer.value.adapter_receipts,
                stage=2,
            )
            for receipt in transformer.value.adapter_receipts:
                metadata = receipt.to_metadata()
                metadata["stages"] = (2,)
                adapter_receipts.append(metadata)
        legacy_noise_position = initial_noise_state is None
        noiser = GaussianNoiser(
            request.seed,
            backend=request.noise_backend,
            position=initial_noise_state,
        )
        if legacy_noise_position:
            # MLX keyed noise advances once per tensor regardless of its shape.
            # The resulting cross-backend position remains unknown, so it is
            # deliberately omitted from the replacement artifact below.
            for _ in range(2 if request.generate_audio else 1):
                noiser.advance((1,))
        final_latents = run_stage(
            stage_2,
            transformer.value,
            text,
            stage_sigmas(request.stage_1_sigmas, request.stage_2_sigmas)[1],
            noiser=noiser,
            reporter=reporter,
            phase="distilled stage 2",
            step_kind=sampler.stage_2,
            stage_number=2,
        )
        if legacy_noise_position:
            final_latents = replace(final_latents, initial_noise_state=None)
        del stage_2
    finally:
        if transformer is not None:
            transformer.close()
    artifacts.save(
        distilled_stage_latents_artifact(
            2,
            video_latent=final_latents.video,
            audio_latent=final_latents.audio,
            final=True,
            noise_state=final_latents.initial_noise_state,
        )
    )
    del text
    release_stage_temporaries()
    output = decode_stage_latents(
        request,
        resources,
        final_latents,
        geometry=stage_2_geometry,
        components=factory,
        reporter=reporter,
        adapter_receipts=adapter_receipts,
        text_conditioning_receipt=text_conditioning_receipt,
        execution_mode="restart-stage-2",
        source_latent_stage="stage-1",
        sampler_plan=sampler,
        hdr_reference_receipts=(() if reference_receipt is None else (reference_receipt,)),
    )
    return _with_restart_metadata(
        output,
        source_model_generation=source_model_generation,
    )


@memory_managed
def restart_distilled(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    restart: DistilledRestart,
    components: DistilledComponents | None = None,
    text_conditioner: TextConditioner | None = None,
    reporter: Reporter | None = None,
    artifact_sink: ArtifactSink | None = None,
) -> GenerationOutput:
    """Restart at stage 2 or decode one selected saved latent directly."""
    validate_record(request, location="[generate]")
    validate_record(restart, location="[restart]")
    if not restart.latents.is_file():
        raise ValueError(f"restart latent artifact does not exist: {restart.latents}")
    if (
        restart.phase == "stage-2"
        and restart.text_conditioning is not None
        and not restart.text_conditioning.is_file()
    ):
        raise ValueError(
            f"restart text-conditioning artifact does not exist: {restart.text_conditioning}"
        )
    sink = reporter if reporter is not None else NullReporter()
    artifacts = artifact_sink if artifact_sink is not None else NullArtifactSink()
    factory = components if components is not None else NativeLTX2Components(reporter=sink)
    validate_distilled_request(request, resources, condition_sources(request))
    if restart.phase == "decode":
        return _decode_saved_stage(
            request,
            resources,
            latent_stage=restart.selected_latent_stage,
            latent_path=restart.latents,
            source_model_generation=restart.source_model_generation,
            factory=factory,
            reporter=sink,
            artifacts=artifacts,
        )
    if restart.phase == "stage-2":
        if restart.text_conditioning is None:
            raise AssertionError("validated stage-2 restart has no text conditioning")
        stage_2_request = replace(request, text_conditioning=restart.text_conditioning)
        restart_text_conditioner = (
            text_conditioner
            if text_conditioner is not None
            else NativeTextConditioner(replay_identity_policy="observe")
        )
        return _restart_stage_2(
            stage_2_request,
            resources,
            latent_path=restart.latents,
            source_model_generation=restart.source_model_generation,
            factory=factory,
            text_conditioner=restart_text_conditioner,
            reporter=sink,
            artifacts=artifacts,
        )
    raise AssertionError(f"validated restart has unknown phase {restart.phase!r}")


__all__ = [
    "load_stage_latents",
    "restart_distilled",
]
