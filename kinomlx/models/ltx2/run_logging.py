"""Human-readable LTX-2 request and resource logging."""

import logging
from collections.abc import Mapping

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.cli.presentation import (
    log_adapter_selections,
    log_prompt_blocks,
    log_section,
    model_display_name,
)
from kinomlx.models.ltx2.resources import LTX2Resources
from kinomlx.samplers.noise import noise_compatibility_profile
from kinomlx.samplers.sigmas import render_sigmas

from .adapters.selection import LTX2AdapterSelection
from .config import Invocation
from .frame_position import parse_frame_position, resolve_frame_position
from .pipelines.distilled import sampling_facts, stage_sampler_choices
from .settings import LTX2Settings
from .sigmas import stage_sigmas
from .types import LTX2ImageConditioningConfig, LTX2Request
from .vocabularies import LTX2StageSampler


def _stage_strength_details(selection: AdapterSelection) -> tuple[str, ...]:
    """Name what each stage applies, never a common scalar no stage reads.

    A stage override is the whole strength for that stage, so once either stage
    has one the line names both resolved stage values and omits the scalar.
    Without an override the scalar is what both stages apply and stands alone.
    """
    if not isinstance(selection, LTX2AdapterSelection):
        raise TypeError("LTX-2 request returned a non-LTX adapter selection")
    if selection.stage_1_strength is None and selection.stage_2_strength is None:
        return (f"strength={selection.strength:g}",)
    return tuple(
        f"stage {stage} strength={selection.strength_for_stage(stage):g}" for stage in (1, 2)
    )


def _image_strength_detail(image: LTX2ImageConditioningConfig) -> str:
    """Name one strength when both stages hold it, and both when they differ."""
    stage_1, stage_2 = image.strength_for_stage(1), image.strength_for_stage(2)
    if stage_1 == stage_2:
        return f"strength={stage_1:g}"
    return f"strength={stage_1:g} (stage 1) / {stage_2:g} (stage 2)"


def _image_frame_detail(image: LTX2ImageConditioningConfig, frames: int | None) -> str:
    """Spell the position the user wrote, adding the frame a name resolves to."""
    spelling = image.frame_index
    if frames is None or isinstance(parse_frame_position(spelling), int):
        return spelling
    return f"{spelling} ({resolve_frame_position(spelling, frames)})"


def log_request(logger: logging.Logger, invocation: Invocation) -> None:
    """Log the complete effective LTX-2 request."""
    request = invocation.request
    settings = invocation.model_settings
    output = invocation.output
    action = "restart" if invocation.restart is not None else "generate"
    model = (
        f"{model_display_name(invocation.model)} Distilled (generation auto)"
        if settings.model_generation is None
        else f"LTX-{settings.model_generation} Distilled"
    )
    frames = "auto" if request.frames is None else request.frames
    timeline = (
        "auto" if request.frames is None else f"{max(0, request.frames - 1) / request.fps:.3f}s"
    )
    log_section(
        logger,
        "Run",
        (
            ("Model", model),
            ("Action", action),
            ("Resolution", f"{request.width}x{request.height}"),
            ("Frames", frames),
            ("FPS", f"{request.fps:g}"),
            ("Timeline", timeline),
            (
                "Requested duration",
                "auto" if request.duration is None else f"{request.duration:g}s",
            ),
            ("Seed", request.seed),
        ),
    )
    residency = (
        "all blocks"
        if settings.transformer_resident_blocks is None
        else f"{settings.transformer_resident_blocks} blocks"
    )
    tiling: str = request.vae_tiling.mode
    if tiling == "custom":
        tiling += (
            f"; temporal_tile={request.vae_tiling.temporal_tile_frames or 'off'}; "
            f"temporal_overlap={request.vae_tiling.temporal_overlap_frames}; "
            f"spatial_tile={request.vae_tiling.spatial_tile_pixels or 'off'}; "
            f"spatial_overlap={request.vae_tiling.spatial_overlap_pixels}"
        )
    log_section(
        logger,
        "Compute",
        (
            ("Transformer dtype", settings.transformer_dtype),
            ("Dense attention kernel", str(settings.dense_attention_kernel).lower()),
            ("Compile attention", str(settings.compile_attention).lower()),
            ("Fast mode", str(settings.fast_mode).lower()),
            ("Transformer residency", residency),
            ("Retained tensors", ", ".join(settings.transformer_retain) or "none"),
            ("Transformer quantization", settings.transformer_quantize),
            ("Adapter accumulation", settings.resolved_adapter_accumulation()),
            ("Quantization excluded layers", settings.transformer_quantize_exclude_layers),
            ("Video VAE", settings.video_vae),
            ("VAE decode dtype", request.vae_decode_dtype),
            (
                "VAE allocator cache",
                request.vae_cache_limit_gb if request.vae_cache_limit_gb is not None else "auto",
            ),
            ("VAE tiling", tiling),
            (
                "VAE sharpening",
                (
                    f"amount {request.vae_sharpen:g}, radius {request.vae_sharpen_radius:g} px"
                    if request.vae_sharpen
                    else "off"
                ),
            ),
            ("Prompt layout", "padded" if request.pad_prompt_to_max else "compact"),
        ),
    )
    conditioning: list[tuple[str, object]] = [
        ("Audio", "enabled" if request.generate_audio else "disabled"),
        (
            "Audio timing",
            ("reference-aligned" if request.reference_aligned_audio else "coverage")
            if request.generate_audio
            else "inactive (audio disabled)",
        ),
        ("HDR authoring", request.hdr or "off"),
        ("Generated keyframes", request.generated_keyframes),
        ("First frame mode", request.first_frame_mode),
    ]
    for index, image in enumerate(request.images, start=1):
        conditioning.append(
            (
                "Input image" if len(request.images) == 1 else f"Input image {index}",
                f"{image.path}; frame={_image_frame_detail(image, request.frames)}; "
                f"{_image_strength_detail(image)}",
            )
        )
    if request.hdr_reference is not None:
        conditioning.append(
            (
                "HDR reference",
                f"{request.hdr_reference.path}; strength={request.hdr_reference.strength:g}",
            )
        )
    log_section(logger, "Conditioning", conditioning)
    log_section(
        logger,
        "Delivery",
        (
            ("Spatial VSR", output.vsr_spatial_mode),
            ("Target FPS", "source" if output.target_fps is None else f"{output.target_fps:g}"),
            (
                "Temporal VSR",
                output.vsr_temporal_mode
                if output.target_fps is not None and abs(output.target_fps - request.fps) > 1e-6
                else "off",
            ),
            (
                "Cut detection",
                output.cut_detect_mode
                if output.target_fps is not None and abs(output.target_fps - request.fps) > 1e-6
                else "inactive (no frame-rate conversion)",
            ),
            ("Encode quality", f"{output.encode_quality:g}"),
            (
                "Audio codec",
                output.audio_codec if request.generate_audio else "inactive (audio disabled)",
            ),
            (
                "Audio onset trim",
                output.audio_onset_trim if request.generate_audio else "inactive (audio disabled)",
            ),
        ),
    )
    if invocation.restart is not None:
        log_section(
            logger,
            "Restart",
            (
                ("Source run", invocation.restart.source_run),
                ("Stage", invocation.restart.config.stage),
                ("Latent stage", invocation.restart.selected_latent_stage),
                ("Latents", invocation.restart.selected_latents),
            ),
        )
    if request.text_conditioning is None:
        log_prompt_blocks(logger, (request.prompt,))
    else:
        logger.info("Text conditioning: %s", request.text_conditioning)
    log_adapter_selections(
        logger,
        request.resolved_adapters(),
        strength_details=_stage_strength_details,
    )


def _gemma_variant_label(family: str, model_settings: LTX2Settings | None) -> str | None:
    """Name the Gemma 3 variant a run is conditioned with, or None when not applicable."""
    if model_settings is None or not family.startswith("gemma3"):
        return None
    if model_settings.text_encoder_path is not None:
        return f"explicit text_encoder_path ({model_settings.text_encoder_path})"
    return model_settings.gemma_variant


def log_resources(
    logger: logging.Logger,
    resources: LTX2Resources,
    *,
    model_settings: LTX2Settings | None = None,
) -> None:
    """Announce one prepared inventory, for a run or for a resource-only dry run."""
    checkpoint = resources.checkpoint
    family = resources.capabilities.text_encoder_family
    fields: list[tuple[str, object]] = [
        ("Model", f"LTX-{checkpoint.model_generation} Distilled"),
        ("Model generation", checkpoint.model_generation),
        ("Model version", checkpoint.model_version or "undeclared"),
        ("Checkpoint layout", checkpoint.layout.value),
        ("Checkpoint", checkpoint.source_path),
        ("Checkpoint fingerprint", checkpoint.source_fingerprint),
        ("Text encoder", family),
    ]
    variant = _gemma_variant_label(family, model_settings)
    if variant is not None:
        fields.append(("Gemma variant", variant))
    fields.append(("Video VAE", resources.capabilities.video_vae_kind))
    log_section(logger, "Resources", fields)
    log_section(
        logger,
        "Components",
        (
            (
                component.kind.value,
                "source="
                + (
                    "checkpoint"
                    if component.source_path == checkpoint.source_path
                    else str(component.source_path)
                ),
            )
            for component in resources.components
        ),
    )
    logger.debug(
        "Component fingerprints: %s",
        "; ".join(
            f"{component.kind.value}={component.source_fingerprint}"
            for component in resources.components
        ),
    )


def log_runtime_resources(
    logger: logging.Logger,
    runner: object,
    *,
    model_settings: LTX2Settings | None = None,
    request: LTX2Request | None = None,
) -> None:
    """Log prepared LTX-2 resources, when the runner exposes them, and the run's sampling."""

    resources = getattr(runner, "resources", None)
    if isinstance(resources, LTX2Resources):
        log_resources(logger, resources, model_settings=model_settings)
    _log_sampling(logger, request, published_stage_samplers(runner))


def _render_sampler(choice: Mapping[str, object]) -> str:
    """``ancestral (auto)`` when the generation chose it, ``(requested)`` when the run did."""
    origin = "auto" if choice.get("requested") == "auto" else "requested"
    return f"{choice.get('effective')} ({origin})"


def _log_sampling(
    logger: logging.Logger,
    request: LTX2Request | None,
    published: tuple[LTX2StageSampler, LTX2StageSampler] | None,
) -> None:
    """Announce the run's sampling once: request facts always, resolved samplers when known."""
    fields: list[tuple[str, object]] = []
    if request is not None:
        first, second = stage_sigmas(request.stage_1_sigmas, request.stage_2_sigmas)
        fields.extend(
            (
                ("Stage 1 steps", len(first) - 1),
                ("Stage 2 steps", len(second) - 1),
                ("Stage 1 sigmas", render_sigmas(first)),
                ("Stage 2 sigmas", render_sigmas(second)),
            )
        )
        for stage, schedule in enumerate(request.sparse_attention_schedules(), start=1):
            if schedule.active:
                fields.append((f"Stage {stage} sparse attention", schedule.render()))
        if published is None:
            for stage, requested in enumerate(request.stage_samplers(), start=1):
                fields.append((f"Stage {stage} sampler", requested.value))
        else:
            for stage, choice in enumerate(
                stage_sampler_choices(request, published).values(), start=1
            ):
                fields.append((f"Stage {stage} sampler", _render_sampler(choice)))
        fields.append(("Noise backend", request.noise_backend))
        fields.append(("Noise compatibility", noise_compatibility_profile(request.noise_backend)))
    elif published is not None:
        for stage, sampler in enumerate(published, start=1):
            fields.append((f"Stage {stage} sampler", sampler.value))
    log_section(logger, "Sampling", fields)


def published_stage_samplers(
    runner: object,
) -> tuple[LTX2StageSampler, LTX2StageSampler] | None:
    """Return the prepared checkpoint's published samplers when the runner exposes resources."""
    resources = getattr(runner, "resources", None)
    if not isinstance(resources, LTX2Resources):
        return None
    return resources.capabilities.stage_samplers


def log_effective_sampling(
    logger: logging.Logger,
    metadata: Mapping[str, object],
    *,
    request: LTX2Request,
    published: tuple[LTX2StageSampler, LTX2StageSampler] | None,
) -> None:
    """Warn when a recipe's recorded sampler facts differ from the announced ones.

    Without published samplers nothing was resolved and announced, so there is
    nothing to check.
    """
    if published is None:
        return
    announced = sampling_facts(request, published)
    changed = []
    recorded_samplers = metadata.get("samplers")
    for stage, (key, choice) in enumerate(stage_sampler_choices(request, published).items(), 1):
        recorded = recorded_samplers.get(key) if isinstance(recorded_samplers, Mapping) else None
        if not isinstance(recorded, Mapping) or any(
            recorded.get(field) != value for field, value in choice.items()
        ):
            shown = _render_sampler(recorded) if isinstance(recorded, Mapping) else recorded
            changed.append(
                f"Stage {stage} sampler: announced {_render_sampler(choice)!r}, recorded {shown!r}"
            )
    labels = (
        ("noise_backend", "Noise backend"),
        ("noise_compatibility_profile", "Noise compatibility"),
    )
    changed.extend(
        f"{label}: announced {announced[key]!r}, recorded {metadata.get(key)!r}"
        for key, label in labels
        if metadata.get(key) != announced[key]
    )
    if changed:
        logger.warning("Sampling changed at run time: %s", "; ".join(changed))


__all__ = [
    "log_effective_sampling",
    "log_request",
    "log_runtime_resources",
    "published_stage_samplers",
]
