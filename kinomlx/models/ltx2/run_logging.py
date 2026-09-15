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
from .pipelines.distilled import sampling_facts
from .settings import LTX2Settings
from .sigmas import stage_sigmas
from .types import DistilledRequest


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
            ("Attention", "STEEL" if settings.steel_attention else "MLX SDPA"),
            ("Compile attention", str(settings.compile_attention).lower()),
            ("Fast mode", str(settings.fast_mode).lower()),
            ("Transformer residency", residency),
            ("Retained tensors", ", ".join(settings.transformer_retain) or "none"),
            ("Transformer quantization", settings.transformer_quantize),
            ("Adapter accumulation", settings.resolved_adapter_accumulation()),
            ("Quantization excluded layers", settings.transformer_quantize_exclude_layers),
            ("Video VAE", settings.video_vae),
            ("VAE decode dtype", request.vae_decode_dtype),
            ("VAE tiling", tiling),
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
    ]
    if request.image is not None:
        conditioning.append(
            (
                "Input image",
                f"{request.image.path}; frame={request.image.frame_index}; "
                f"strength={request.image.strength:g}",
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
                ("Phase", invocation.restart.config.phase),
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
    if model_settings.gemma_path is not None:
        return f"explicit gemma_path ({model_settings.gemma_path})"
    return model_settings.gemma_variant


def log_runtime_resources(
    logger: logging.Logger,
    runner: object,
    *,
    model_settings: LTX2Settings | None = None,
    request: DistilledRequest | None = None,
) -> None:
    """Log prepared LTX-2 resources, when the runner exposes them, and the run's sampling."""

    resources = getattr(runner, "resources", None)
    if not isinstance(resources, LTX2Resources):
        resources = None
    if resources is not None:
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
    _log_sampling(logger, request, declared_sampler_policy(runner))


def _log_sampling(
    logger: logging.Logger,
    request: DistilledRequest | None,
    declared_policy: str | None,
) -> None:
    """Announce the run's sampling once: request facts always, policy when declared."""
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
        fields.append(("Requested sampler", request.sampler))
        if declared_policy is not None:
            facts = sampling_facts(request, declared_policy)
            fields.append(("Policy", facts["sampler_policy"]))
            fields.append(("Override", facts["sampler_override"] or "none"))
        fields.append(("Noise backend", request.noise_backend))
        fields.append(("Noise compatibility", noise_compatibility_profile(request.noise_backend)))
    elif declared_policy is not None:
        fields.append(("Policy", declared_policy))
    log_section(logger, "Sampling", fields)


def declared_sampler_policy(runner: object) -> str | None:
    """Return the prepared checkpoint's sampler policy when the runner exposes resources."""
    resources = getattr(runner, "resources", None)
    if not isinstance(resources, LTX2Resources):
        return None
    return resources.capabilities.sampler_policy


def log_effective_sampling(
    logger: logging.Logger,
    metadata: Mapping[str, object],
    *,
    request: DistilledRequest,
    declared_policy: str | None,
) -> None:
    """Warn when a recipe's recorded sampler facts differ from the announced ones.

    Without a declared policy nothing was announced, so there is nothing to check.
    """
    if declared_policy is None:
        return
    announced = sampling_facts(request, declared_policy)
    labels = (
        ("sampler_policy", "Policy"),
        ("sampler_override", "Override"),
        ("noise_backend", "Noise backend"),
        ("noise_compatibility_profile", "Noise compatibility"),
    )
    changed = [
        f"{label}: announced {announced[key]!r}, recorded {metadata.get(key)!r}"
        for key, label in labels
        if metadata.get(key) != announced[key]
    ]
    if changed:
        logger.warning("Sampling changed at run time: %s", "; ".join(changed))


__all__ = [
    "declared_sampler_policy",
    "log_effective_sampling",
    "log_request",
    "log_runtime_resources",
]
