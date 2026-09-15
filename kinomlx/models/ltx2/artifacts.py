"""LTX-2 artifact vocabulary and output-adjacent path contribution."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

import mlx.core as mx

from kinomlx.artifacts import TensorArtifact
from kinomlx.config.stations import OutputPolicy
from kinomlx.config_records import config_record
from kinomlx.types import VideoPixelShape

from .conditioning import (
    EncodedCondition,
    HDRReferenceConditionSource,
    ImageConditionSource,
    RawConditionSource,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoConditionByReferenceLatent,
)

TEXT_CONDITIONING = "text_conditioning"
STAGE_1_CONDITIONING = "stage_1_conditioning"
STAGE_2_CONDITIONING = "stage_2_conditioning"
STAGE_1_LATENTS = "stage_1_latents"
FINAL_LATENTS = "final_latents"

# A restart station re-emits what it consumed and still holds, so a bundle it
# writes can itself be restarted. Stage-2 reloads saved text and the stage-1
# latent before it runs; decode reloads exactly one latent product and
# ``requested_artifacts`` narrows the pair below to the stage it consumed.
ARTIFACT_STATIONS = MappingProxyType(
    {
        TEXT_CONDITIONING: OutputPolicy(frozenset({"generate", "stage-2"})),
        STAGE_1_CONDITIONING: OutputPolicy(frozenset({"generate"})),
        STAGE_2_CONDITIONING: OutputPolicy(frozenset({"generate", "stage-2"})),
        STAGE_1_LATENTS: OutputPolicy(frozenset({"generate", "stage-2", "decode"})),
        FINAL_LATENTS: OutputPolicy(frozenset({"generate", "stage-2", "decode"})),
    }
)
ARTIFACT_GROUPS = MappingProxyType(
    {
        "save_latents": (STAGE_1_LATENTS, FINAL_LATENTS),
        "save_text_conditioning": (TEXT_CONDITIONING,),
        "save_media_conditioning": (STAGE_1_CONDITIONING, STAGE_2_CONDITIONING),
    }
)
OUTPUT_POLICIES = MappingProxyType(
    {
        field: OutputPolicy(
            frozenset(
                station for name in artifacts for station in ARTIFACT_STATIONS[name].produced_at
            )
        )
        for field, artifacts in ARTIFACT_GROUPS.items()
    }
)
_TEXT_CONDITIONING_V3_FIELDS = frozenset(
    {
        "tokenizer_source_sha256",
        "tokenization_policy",
        "text_artifact_identity",
        "projection_source_identity",
        "connector_source_identity",
    }
)


class ArtifactOptions(Protocol):
    """Host output selections consumed by the LTX-2 artifact contribution."""

    @property
    def save_latents(self) -> bool | None: ...

    @property
    def save_media_conditioning(self) -> bool | None: ...

    @property
    def save_text_conditioning(self) -> bool | None: ...


class NoiseArtifactState(Protocol):
    """String metadata surface of a resumable normal-noise stream."""

    def to_artifact_metadata(
        self,
        *,
        prefix: str = "initial_noise_",
    ) -> tuple[tuple[str, str], ...]: ...


@config_record
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
    station: Literal["generate", "decode", "stage-2"] = "generate",
    source_latent_stage: Literal["stage-1", "final"] = "final",
) -> frozenset[str]:
    """Resolve enabled LTX-2 tensor artifacts from host output selections.

    ``source_latent_stage`` names the latent product a decode restart consumed.
    Decode re-emits that one product and never the other, so a requested
    artifact is always one the station can actually write.
    """
    if station not in {"generate", "decode", "stage-2"}:
        raise ValueError(f"unknown artifact station {station!r}")
    requested: set[str] = set()
    for field, artifacts in ARTIFACT_GROUPS.items():
        value = getattr(options, field)
        media = field == "save_media_conditioning"
        selected = OUTPUT_POLICIES[field].selected(
            value,
            save_all=save_all and (not media or has_media_conditioning),
            station=station,
            field=field,
        )
        if selected and media and not has_media_conditioning:
            raise ValueError("save_media_conditioning requires --image or --hdr-reference")
        if selected:
            requested.update(
                name for name in artifacts if station in ARTIFACT_STATIONS[name].produced_at
            )
    if station == "decode":
        requested.discard(STAGE_1_LATENTS if source_latent_stage == "final" else FINAL_LATENTS)
    return frozenset(requested)


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
    "STAGE_1_CONDITIONING",
    "STAGE_1_LATENTS",
    "STAGE_2_CONDITIONING",
    "TEXT_CONDITIONING",
    "LTX2ArtifactConfig",
    "distilled_stage_latents_artifact",
    "media_conditioning_artifact",
    "requested_artifacts",
    "sidecar_paths",
    "text_conditioning_artifact",
]
