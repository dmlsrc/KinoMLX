"""Central, packaging-neutral LTX-2 compatibility inspection.

Generation selects a graph binder. Packaging only supplies candidate files for
logical components. Both LTX-2.3 and LTX-2.5 pass through this same inspection
path and every failure names the effective generation being checked.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio_vae.config import AudioVAEConfig
from .metadata import (
    ConnectorHeaderConfig,
    DurationHeadConfig,
    LatentUpscalerConfig,
    LTX2CheckpointConfig,
    TextEncoderHeaderConfig,
    TextProjectionHeaderConfig,
    checkpoint_config,
    generation_label,
    inspect_audio_vae,
    inspect_connectors,
    inspect_duration_head,
    inspect_latent_upscaler,
    inspect_text_encoder,
    inspect_text_projection,
    inspect_video_vae,
    validate_transformer_header,
)
from .video_vae.config import VideoVAEConfig


@dataclass(frozen=True)
class LTX2ComponentSources:
    """Candidate physical sources for one logical LTX-2 composition."""

    transformer: Path
    text_encoder: Path | None
    video_vae: Path | None
    audio_vae: Path | None = None
    spatial_upscaler: Path | None = None
    temporal_upscaler: Path | None = None
    duration_head: Path | None = None
    text_projection_candidates: tuple[Path, ...] = ()
    connector_candidates: tuple[Path, ...] = ()


@dataclass(frozen=True)
class LTX2CompatibilityReport:
    """Generation-labeled logical component facts and resolved source paths."""

    checkpoint: LTX2CheckpointConfig
    label: str
    declared_generation: str | None
    metadata_notes: tuple[str, ...]
    text_encoder: TextEncoderHeaderConfig | None
    text_encoder_source: Path | None
    text_projection: TextProjectionHeaderConfig | None
    text_projection_source: Path | None
    connectors: ConnectorHeaderConfig | None
    connector_source: Path | None
    video_vae: VideoVAEConfig | None
    video_vae_source: Path | None
    audio_vae: AudioVAEConfig | None
    audio_vae_source: Path | None
    spatial_upscaler: LatentUpscalerConfig | None
    temporal_upscaler: LatentUpscalerConfig | None
    duration_head: DurationHeadConfig | None

    @property
    def model_generation(self) -> str:
        return self.checkpoint.model_generation


def _declared_generation(version: str | None) -> str | None:
    if version is None:
        return None
    match = re.search(r"(?<!\d)2\.(3|5)(?!\d)", version)
    return None if match is None else f"2.{match.group(1)}"


def _unique_files(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    result = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return tuple(result)


def _checked_component[T](label: str, inspect: Callable[[], T]) -> T:
    """Give every centralized component failure an explicit generation label."""
    try:
        return inspect()
    except ValueError as exc:
        if label in str(exc):
            raise
        raise ValueError(f"{label} compatibility: {exc}") from exc


def _resolve_candidate[T](
    paths: tuple[Path, ...],
    *,
    component: str,
    inspect: Callable[[Path], T],
) -> tuple[Path, T]:
    failures: list[str] = []
    for path in _unique_files(paths):
        try:
            return path, inspect(path)
        except ValueError as exc:
            failures.append(str(exc))
    detail = "no candidate artifact was supplied" if not failures else failures[0]
    raise ValueError(
        f"{component} compatibility: no candidate supplies the consumed targets: {detail}"
    )


def inspect_ltx2_compatibility(
    sources: LTX2ComponentSources,
    *,
    parsed_checkpoint: LTX2CheckpointConfig | None = None,
    expected_generation: str | None = None,
) -> LTX2CompatibilityReport:
    """Apply one consumed-target policy to an LTX-2.3 or LTX-2.5 pack."""
    parsed = (
        checkpoint_config(sources.transformer) if parsed_checkpoint is None else parsed_checkpoint
    )
    generation = parsed.model_generation
    label = generation_label(generation)
    if expected_generation is not None:
        expected_label = generation_label(expected_generation)
        if expected_generation != generation:
            raise ValueError(
                f"{expected_label} compatibility was requested, but consumed transformer "
                f"structure selects {label}"
            )
    declared = _declared_generation(parsed.declared_model_version)
    notes = []
    if parsed.transformer.inferred_fields:
        notes.append(
            f"{label} constructor inferred fields: {', '.join(parsed.transformer.inferred_fields)}"
        )
    if declared is not None and declared != generation:
        notes.append(
            f"declared model_version identifies LTX-{declared}, but consumed graph fields "
            f"select {label}"
        )

    _checked_component(
        label,
        lambda: validate_transformer_header(sources.transformer, parsed.transformer),
    )

    text_config = None
    text_encoder_source = sources.text_encoder
    if text_encoder_source is not None:
        text_config = _checked_component(
            label,
            lambda: inspect_text_encoder(
                text_encoder_source,
                model_generation=generation,
            ),
        )
        if text_config.hidden_size != parsed.transformer.caption_channels:
            raise ValueError(
                f"{label} compatibility: text encoder hidden size does not match transformer "
                f"caption channels: {text_config.hidden_size} != "
                f"{parsed.transformer.caption_channels}"
            )

    projection_path = None
    projection_config = None
    if text_config is not None:
        projection_path, projection_config = _resolve_candidate(
            sources.text_projection_candidates,
            component=f"{label} text projection",
            inspect=lambda path: inspect_text_projection(
                path,
                model_generation=generation,
                hidden_size=text_config.hidden_size,
                num_hidden_layers=text_config.num_hidden_layers,
            ),
        )
        if projection_config.video_projection_dim != parsed.transformer.video_context_dim:
            raise ValueError(
                f"{label} compatibility: text video projection does not match transformer "
                f"context: {projection_config.video_projection_dim} != "
                f"{parsed.transformer.video_context_dim}"
            )
        if projection_config.audio_projection_dim != parsed.transformer.audio_context_dim:
            raise ValueError(
                f"{label} compatibility: text audio projection does not match transformer "
                f"context: {projection_config.audio_projection_dim} != "
                f"{parsed.transformer.audio_context_dim}"
            )

    connector_path = None
    connector_config = None
    if sources.connector_candidates:
        connector_path, connector_config = _resolve_candidate(
            sources.connector_candidates,
            component=f"{label} connector",
            inspect=lambda path: inspect_connectors(path, config=parsed.transformer),
        )

    video_config = None
    video_vae_source = sources.video_vae
    if video_vae_source is not None:
        video_config = _checked_component(
            label,
            lambda: inspect_video_vae(
                video_vae_source,
                model_generation=generation,
            ),
        )
        if (
            video_config.latent_channels != parsed.transformer.video_in_channels
            or video_config.latent_channels != parsed.transformer.video_out_channels
        ):
            raise ValueError(
                f"{label} compatibility: video VAE latent channels do not match transformer "
                f"video channels: {video_config.latent_channels} != "
                f"{parsed.transformer.video_in_channels}/"
                f"{parsed.transformer.video_out_channels}"
            )

    audio_config = None
    audio_vae_source = sources.audio_vae
    if audio_vae_source is not None:
        audio_config = _checked_component(
            label,
            lambda: inspect_audio_vae(
                audio_vae_source,
                model_generation=generation,
            ),
        )
        if audio_config.channels != parsed.transformer.audio_out_channels:
            raise ValueError(
                f"{label} compatibility: audio VAE channels do not match transformer audio "
                f"channels: {audio_config.channels} != {parsed.transformer.audio_out_channels}"
            )

    spatial_config = None
    spatial_upscaler_source = sources.spatial_upscaler
    if spatial_upscaler_source is not None:
        spatial_config = _checked_component(
            label,
            lambda: inspect_latent_upscaler(
                spatial_upscaler_source,
                expected_kind="spatial",
                model_generation=generation,
            ),
        )
        if video_config is not None and spatial_config.in_channels != video_config.latent_channels:
            raise ValueError(
                f"{label} compatibility: spatial upscaler channels do not match video latent "
                f"channels: {spatial_config.in_channels} != {video_config.latent_channels}"
            )

    temporal_config = None
    temporal_upscaler_source = sources.temporal_upscaler
    if temporal_upscaler_source is not None:
        temporal_config = _checked_component(
            label,
            lambda: inspect_latent_upscaler(
                temporal_upscaler_source,
                expected_kind="temporal",
                model_generation=generation,
            ),
        )
        if video_config is not None and temporal_config.in_channels != video_config.latent_channels:
            raise ValueError(
                f"{label} compatibility: temporal upscaler channels do not match video latent "
                f"channels: {temporal_config.in_channels} != {video_config.latent_channels}"
            )

    duration_config = None
    duration_head_source = sources.duration_head
    if duration_head_source is not None:
        duration_config = _checked_component(
            label,
            lambda: inspect_duration_head(
                duration_head_source,
                model_generation=generation,
            ),
        )
        if (
            duration_config.video_context_dim != parsed.transformer.video_context_dim
            or duration_config.audio_context_dim != parsed.transformer.audio_context_dim
        ):
            raise ValueError(
                f"{label} compatibility: duration-head context dimensions do not match "
                "the transformer"
            )

    return LTX2CompatibilityReport(
        checkpoint=parsed,
        label=label,
        declared_generation=declared,
        metadata_notes=tuple(notes),
        text_encoder=text_config,
        text_encoder_source=sources.text_encoder,
        text_projection=projection_config,
        text_projection_source=projection_path,
        connectors=connector_config,
        connector_source=connector_path,
        video_vae=video_config,
        video_vae_source=sources.video_vae,
        audio_vae=audio_config,
        audio_vae_source=sources.audio_vae,
        spatial_upscaler=spatial_config,
        temporal_upscaler=temporal_config,
        duration_head=duration_config,
    )


__all__ = [
    "LTX2CompatibilityReport",
    "LTX2ComponentSources",
    "inspect_ltx2_compatibility",
]
