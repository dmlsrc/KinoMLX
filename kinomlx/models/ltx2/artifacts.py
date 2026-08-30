"""LTX-2 artifact vocabulary and output-adjacent path contribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from kinomlx.artifacts import TensorArtifact

if TYPE_CHECKING:
    import mlx.core as mx

    from kinomlx.types import VideoPixelShape

    from .conditioning import EncodedCondition, RawConditionSource


TEXT_CONDITIONING = "text_conditioning"
STAGE_1_CONDITIONING = "stage_1_conditioning"
STAGE_2_CONDITIONING = "stage_2_conditioning"
STAGE_1_LATENTS = "stage_1_latents"
FINAL_LATENTS = "final_latents"
_TEXT_CONDITIONING_V3_FIELDS = frozenset(
    {
        "tokenizer_source_sha256",
        "tokenizer_model_sha256",
        "tokenizer_metadata_sha256",
        "tokenization_policy",
        "text_artifact_identity",
        "projection_source_identity",
        "connector_source_identity",
    }
)


class ArtifactOptions(Protocol):
    """Host output selections consumed by the LTX-2 artifact contribution."""

    save_latents: bool | None
    save_media_conditioning: bool | None
    save_text_conditioning: bool | None


class NoiseArtifactState(Protocol):
    """String metadata surface of a resumable normal-noise stream."""

    def to_artifact_metadata(
        self,
        *,
        prefix: str = "initial_noise_",
    ) -> tuple[tuple[str, str], ...]: ...


@dataclass(frozen=True)
class LTX2ArtifactConfig:
    """Optional LTX-2 artifact selections resolved beside the recipe schema."""

    save_latents: bool | None = None
    save_media_conditioning: bool | None = None
    save_text_conditioning: bool | None = None


def sidecar_paths(video_path: Path) -> dict[str, Path]:
    """Return deterministic LTX-2 tensor sidecars beside ``video_path``."""
    stem = video_path.stem
    parent = video_path.parent
    return {
        STAGE_1_LATENTS: parent / f"{stem}_stage1.safetensors",
        FINAL_LATENTS: parent / f"{stem}.safetensors",
        TEXT_CONDITIONING: parent / f"{stem}_text.safetensors",
        STAGE_1_CONDITIONING: parent / f"{stem}_stage1_conditioning.safetensors",
        STAGE_2_CONDITIONING: parent / f"{stem}_stage2_conditioning.safetensors",
    }


def requested_artifacts(
    options: ArtifactOptions,
    *,
    save_all: bool = False,
    has_media_conditioning: bool = False,
) -> frozenset[str]:
    """Resolve enabled LTX-2 tensor artifacts from host output selections."""
    requested: set[str] = set()
    if options.save_latents is True or (options.save_latents is None and save_all):
        requested.update((STAGE_1_LATENTS, FINAL_LATENTS))
    if options.save_text_conditioning is True or (
        options.save_text_conditioning is None and save_all
    ):
        requested.add(TEXT_CONDITIONING)
    if options.save_media_conditioning is True or (
        has_media_conditioning and options.save_media_conditioning is None and save_all
    ):
        requested.update((STAGE_1_CONDITIONING, STAGE_2_CONDITIONING))
    return frozenset(requested)


def restart_artifacts(
    requested: frozenset[str],
    *,
    phase: str,
) -> frozenset[str]:
    """Return only artifacts newly produced after a selected restart point."""
    if phase == "stage-2":
        return requested & {FINAL_LATENTS, STAGE_2_CONDITIONING}
    if phase == "decode":
        return frozenset()
    raise ValueError(f"unknown restart phase {phase!r}")


def text_conditioning_artifact(
    *,
    prompt: str,
    video_encoding: mx.array,
    audio_encoding: mx.array,
    attention_mask: mx.array,
    provenance: Mapping[str, str],
) -> TensorArtifact:
    """Build the public LTX-2 encoded-prompt artifact."""
    extended = _TEXT_CONDITIONING_V3_FIELDS.intersection(provenance)
    if extended and extended != _TEXT_CONDITIONING_V3_FIELDS:
        missing = sorted(_TEXT_CONDITIONING_V3_FIELDS - extended)
        raise ValueError(
            "text-conditioning schema-3 provenance is incomplete "
            f"(first missing field: {missing[0]})"
        )
    schema_version = "3" if extended else "2"
    return TensorArtifact(
        name=TEXT_CONDITIONING,
        tensors=(
            ("video_encoding", video_encoding),
            ("audio_encoding", audio_encoding),
            ("attention_mask", attention_mask),
        ),
        metadata=(
            ("schema_version", schema_version),
            ("artifact", "ltx2_text_conditioning"),
            ("prompt", prompt),
            *tuple(provenance.items()),
        ),
        reporting_phase="save text conditioning",
    )


def distilled_stage_latents_artifact(
    stage: int,
    *,
    video_latent: mx.array,
    audio_latent: mx.array | None,
    final: bool,
    noise_state: NoiseArtifactState | None = None,
) -> TensorArtifact:
    """Build one materialized two-stage distilled latent artifact."""
    tensors: tuple[tuple[str, mx.array], ...] = (("video_latent", video_latent),)
    if audio_latent is not None:
        tensors += (("audio_latent", audio_latent),)
    return TensorArtifact(
        name=FINAL_LATENTS if final else STAGE_1_LATENTS,
        tensors=tensors,
        metadata=(
            ("schema_version", "2" if noise_state is not None else "1"),
            ("pipeline", "distilled_two_stage"),
            ("stage", str(stage)),
            ("final", str(final).lower()),
            *(
                ()
                if noise_state is None
                else noise_state.to_artifact_metadata(prefix="initial_noise_")
            ),
        ),
        reporting_phase=f"save stage {stage} latents",
    )


def media_conditioning_artifact(
    stage: int,
    *,
    sources: Sequence[RawConditionSource],
    conditions: Sequence[EncodedCondition],
    geometry: VideoPixelShape,
    fps: float,
) -> TensorArtifact:
    """Build one stage's ordered VAE-encoded media-conditioning product."""
    from .conditioning import (
        HDRReferenceConditionSource,
        ImageConditionSource,
        VideoConditionByKeyframeIndex,
        VideoConditionByLatentIndex,
        VideoConditionByReferenceLatent,
    )

    if stage not in {1, 2}:
        raise ValueError("media-conditioning stage must be 1 or 2")
    if not conditions or len(sources) != len(conditions):
        raise ValueError("media-conditioning sources and encoded conditions must be non-empty")

    tensors: list[tuple[str, mx.array]] = []
    metadata: list[tuple[str, str]] = [
        ("schema_version", "1"),
        ("artifact", "ltx2_media_conditioning"),
        ("stage", str(stage)),
        ("condition_count", str(len(conditions))),
        ("frame_count", str(geometry.frames)),
        ("width", str(geometry.width)),
        ("height", str(geometry.height)),
        ("fps", str(fps)),
    ]
    for index, (source, condition) in enumerate(zip(sources, conditions, strict=True)):
        prefix = f"condition_{index}"
        tensor_key = f"{prefix}_latent"
        if isinstance(source, ImageConditionSource) and isinstance(
            condition,
            VideoConditionByLatentIndex | VideoConditionByKeyframeIndex,
        ):
            latent = (
                condition.latent
                if isinstance(condition, VideoConditionByLatentIndex)
                else condition.keyframes
            )
            metadata.extend(
                (
                    (f"{prefix}_family", source.family),
                    (f"{prefix}_source_path", str(source.path)),
                    (f"{prefix}_frame_index", str(source.frame_index)),
                    (f"{prefix}_strength", str(source.strength)),
                    (
                        f"{prefix}_hdr_authoring",
                        "none" if source.hdr_authoring is None else source.hdr_authoring,
                    ),
                    (f"{prefix}_tensor", tensor_key),
                )
            )
        elif isinstance(source, HDRReferenceConditionSource) and isinstance(
            condition,
            VideoConditionByReferenceLatent,
        ):
            latent = condition.latent
            metadata.extend(
                (
                    (f"{prefix}_family", source.family),
                    (f"{prefix}_source_path", str(source.path)),
                    (f"{prefix}_strength", str(source.strength)),
                    (f"{prefix}_tensor", tensor_key),
                )
            )
        else:
            raise TypeError(
                f"media-conditioning source {type(source).__name__} does not match "
                f"encoded condition {type(condition).__name__}"
            )
        tensors.append((tensor_key, latent))

    return TensorArtifact(
        name=STAGE_1_CONDITIONING if stage == 1 else STAGE_2_CONDITIONING,
        tensors=tuple(tensors),
        metadata=tuple(metadata),
        reporting_phase=f"save stage {stage} media conditioning",
    )


__all__ = [
    "FINAL_LATENTS",
    "LTX2ArtifactConfig",
    "STAGE_1_CONDITIONING",
    "STAGE_1_LATENTS",
    "STAGE_2_CONDITIONING",
    "TEXT_CONDITIONING",
    "distilled_stage_latents_artifact",
    "media_conditioning_artifact",
    "requested_artifacts",
    "restart_artifacts",
    "sidecar_paths",
    "text_conditioning_artifact",
]
