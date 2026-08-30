"""Explicit external composition of the public distilled LTX-2 stations.

This is intentionally more detailed than the ordinary library call. It is the
ownership reference for authors of new recipes: every closeable model appears
inside the lexical region where it is live, and every station consumes a
materialized product from the station above it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kinomlx.artifacts import ArtifactSink, NullArtifactSink
from kinomlx.components import ComponentLease
from kinomlx.media.signals import BT709_SDR_420_DELIVERY, OutputColorPlan
from kinomlx.models.ltx2.artifacts import distilled_stage_latents_artifact
from kinomlx.models.ltx2.components import (
    DistilledComponents,
    NativeLTX2Components,
    TransformerPort,
)
from kinomlx.models.ltx2.decode import (
    decode_audio_mel,
    decode_ltx23_sdr_frames,
    vocode_audio,
)
from kinomlx.models.ltx2.pipelines.distilled import (
    StageLatents,
    build_generation_output,
    prepare_distilled_recipe,
    prepare_stage,
    prepare_stage_conditions,
    prepare_text_conditioning,
    release_stage_temporaries,
    reported_phase,
    resolve_auto_frame_count,
    resolve_lora_profiles,
    run_stage,
    upscale_between_stages,
)
from kinomlx.models.ltx2.resources import LTX2Resources, prepare_resources
from kinomlx.models.ltx2.runner import GenerationOutput
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.sigmas import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
)
from kinomlx.models.ltx2.signals import ltx23_sdr_signal
from kinomlx.models.ltx2.text_conditioning import TextConditioner
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.output import VideoToolboxGenerationSink
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.samplers.noisers import GaussianNoiser, SeededGaussianNoise
from kinomlx.settings import Settings
from kinomlx.types import VideoPixelShape


def compose_distilled(
    request: DistilledRequest,
    resources: LTX2Resources,
    *,
    components: DistilledComponents | None = None,
    text_conditioner: TextConditioner | None = None,
    reporter: Reporter | None = None,
    artifact_sink: ArtifactSink | None = None,
) -> GenerationOutput:
    """Compose every distilled station explicitly using public KinoMLX APIs."""
    progress = reporter if reporter is not None else NullReporter()
    artifacts = artifact_sink if artifact_sink is not None else NullArtifactSink()
    factory = components if components is not None else NativeLTX2Components(reporter=progress)

    prepared = prepare_distilled_recipe(request, resources)
    sources = prepared.sources
    frame_count = prepared.frame_count
    sampler = prepared.sampler
    text = prepare_text_conditioning(
        request,
        resources,
        text_conditioner=text_conditioner,
        reporter=progress,
        artifacts=artifacts,
    )
    if frame_count is None:
        with factory.duration_predictor(resources) as duration_predictor:
            frame_count = resolve_auto_frame_count(
                request,
                resources,
                text,
                duration_predictor,
            )
    stage_1_profile, stage_2_profile = resolve_lora_profiles(request)
    noiser = GaussianNoiser(request.seed, backend=request.noise_backend)
    transition_noise = (
        None
        if sampler.ancestral_seed is None
        else SeededGaussianNoise(
            sampler.ancestral_seed,
            backend=request.noise_backend,
        )
    )

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
        reporter=progress,
    )
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
    try:
        with reported_phase(progress, "load stage 1 transformer"):
            transformer = factory.transformer(
                resources,
                stage_1_profile,
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
            reporter=progress,
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

        # Equal complete profiles intentionally retain one transformer. A
        # different profile closes stage 1 before the upscaler enters memory.
        if stage_1_profile != stage_2_profile:
            transformer.close()
            transformer = None

        with factory.spatial_upscaler(resources) as upscaler:
            upscaled = upscale_between_stages(
                stage_1_latents,
                upscaler,
                reporter=progress,
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
            reporter=progress,
        )
        stage_2 = prepare_stage(
            request,
            stage_2_geometry,
            dtype_policy=resources.dtype_policy,
            conditions=stage_2_conditions,
            initial_latents=upscaled,
        )
        del stage_2_conditions, upscaled

        if transformer is None:
            with reported_phase(progress, "load stage 2 transformer"):
                transformer = factory.transformer(
                    resources,
                    stage_2_profile,
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
            reporter=progress,
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

    waveform = None
    sample_rate = None
    if final_latents.audio is not None:
        with factory.audio_decoder(resources) as audio_decoder:
            mel = decode_audio_mel(
                final_latents.audio,
                audio_decoder,
                reporter=progress,
            )
        with factory.vocoder(resources) as vocoder:
            waveform = vocode_audio(mel, vocoder, reporter=progress)
            sample_rate = int(vocoder.output_sample_rate)
        del mel

    video_latent = final_latents.video
    initial_noise_state = final_latents.initial_noise_state
    del final_latents
    release_stage_temporaries()
    signal = ltx23_sdr_signal(
        width=request.width,
        height=request.height,
        fps=request.fps,
    )
    frames = decode_ltx23_sdr_frames(
        video_latent,
        lambda: factory.video_decoder(resources),
        spec=signal,
        frame_count=frame_count,
        tiling_config=request.vae_tiling.to_runtime_config(),
        auto_tiling=request.vae_tiling.mode == "auto",
        reporter=progress,
        noise_backend=request.noise_backend,
    )
    return build_generation_output(
        request,
        resources,
        frames=frames,
        waveform=waveform,
        sample_rate=sample_rate,
        frame_count=frame_count,
        generated_keyframe_indices=generated_indices,
        lora_receipts=lora_receipts,
        initial_noise_state=initial_noise_state,
        ancestral_noise_state=(None if transition_noise is None else transition_noise.state),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-backend", choices=("mlx", "torch-mps"), default="mlx")
    parser.add_argument("--audio", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare resources, run the explicit recipe, and write one movie."""
    options = _parser().parse_args(argv)
    request = DistilledRequest(
        prompt=options.prompt,
        width=options.width,
        height=options.height,
        frames=options.frames,
        seed=options.seed,
        noise_backend=options.noise_backend,
        generate_audio=options.audio,
    )
    resources = prepare_resources(
        LTX2Settings.from_env(),
        infrastructure=Settings.from_env(),
    )
    with compose_distilled(request, resources) as output:
        plan = OutputColorPlan(
            source=output.signal,
            deliveries=(BT709_SDR_420_DELIVERY,),
        )
        VideoToolboxGenerationSink(
            path=options.output,
            fps=request.fps,
        ).write(output, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
