"""Raw condition sources and bounded encoder preparation."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx

from kinomlx.reporting import Reporter
from kinomlx.types import VideoPixelShape

from ..components import _VideoEncoderCallablePort
from ..encode import encode_image
from ..hdr_reference import encode_reference_video
from ..resources import LTX2Capabilities
from ..types import (
    LTX2HDRAuthoring,
    LTX2HDRReferenceConditioningConfig,
    LTX2ImageConditioningConfig,
)
from ..vocabularies import LTX2FirstFrameMode
from .item import EncodedCondition
from .keyframe import VideoConditionByKeyframeIndex
from .latent import VideoConditionByLatentIndex
from .reference import VideoConditionByReferenceLatent


class LTX2ConditionEncoderPort(_VideoEncoderCallablePort, Protocol):
    """Narrow callable surface used while preparing media conditions."""


@dataclass(frozen=True)
class ImageConditionSource:
    """Caller image intent without encoded tensors or model ownership."""

    path: Path
    frame_index: int = 0
    strength: float = 1.0
    stage1_strength: float | None = None
    stage2_strength: float | None = None
    hdr_authoring: LTX2HDRAuthoring | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.frame_index < 0:
            raise ValueError("condition frame_index must be non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("condition strength must be between 0 and 1")
        for stage, value in ((1, self.stage1_strength), (2, self.stage2_strength)):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"condition stage {stage} strength must be between 0 and 1")

    @classmethod
    def from_config(
        cls,
        config: LTX2ImageConditioningConfig,
        *,
        frame_index: int,
        hdr_authoring: LTX2HDRAuthoring | None = None,
    ) -> ImageConditionSource:
        """Build one source from a record whose named position is already resolved."""
        return cls(
            path=config.path,
            frame_index=frame_index,
            strength=config.strength,
            stage1_strength=config.stage1_strength,
            stage2_strength=config.stage2_strength,
            hdr_authoring=hdr_authoring,
        )

    def strength_for_stage(self, stage: int) -> float:
        """Resolve the strength one stage applies, inheriting the common value."""
        if stage == 1:
            return self.strength if self.stage1_strength is None else self.stage1_strength
        if stage == 2:
            return self.strength if self.stage2_strength is None else self.stage2_strength
        raise ValueError(f"LTX-2 condition stage must be 1 or 2, got {stage}")

    @property
    def family(self) -> str:
        return "image" if self.frame_index == 0 else "keyframe"


@dataclass(frozen=True)
class HDRReferenceConditionSource:
    """Caller-owned SDR reference video intent without encoded tensors."""

    path: Path
    strength: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("HDR reference strength must be between 0 and 1")

    @classmethod
    def from_config(
        cls,
        config: LTX2HDRReferenceConditioningConfig,
    ) -> HDRReferenceConditionSource:
        return cls(path=config.path, strength=config.strength)

    @property
    def family(self) -> str:
        return "hdr-reference"


type RawConditionSource = ImageConditionSource | HDRReferenceConditionSource


def prepare_conditions(
    sources: Iterable[RawConditionSource],
    geometry: VideoPixelShape,
    encoder: LTX2ConditionEncoderPort,
    capabilities: LTX2Capabilities,
    *,
    stage: int,
    compute_dtype: mx.Dtype,
    first_frame_mode: LTX2FirstFrameMode = LTX2FirstFrameMode.ANCHOR,
    reporter: Reporter | None = None,
) -> tuple[EncodedCondition, ...]:
    """Encode ordered raw sources for exactly one stage, at that stage's strengths."""
    if stage not in {1, 2}:
        raise ValueError(f"LTX-2 condition stage must be 1 or 2, got {stage}")
    prepared: list[EncodedCondition] = []
    supported = frozenset(capabilities.condition_families)
    for source in sources:
        if isinstance(source, HDRReferenceConditionSource):
            if capabilities.model_generation != "2.3":
                raise ValueError("HDR reference conditioning requires LTX-2.3")
            latent = encode_reference_video(
                source.path,
                encoder,
                width=geometry.width,
                height=geometry.height,
                frames=geometry.frames,
                compute_dtype=compute_dtype,
                reporter=reporter,
            )
            prepared.append(
                VideoConditionByReferenceLatent(
                    latent=latent,
                    strength=source.strength,
                )
            )
            continue
        if source.family not in supported:
            raise ValueError(f"checkpoint does not support {source.family} conditioning")
        if source.frame_index >= geometry.frames:
            raise ValueError(
                f"condition frame {source.frame_index} is outside {geometry.frames} frames"
            )
        latent = encode_image(
            source.path,
            encoder,
            width=geometry.width,
            height=geometry.height,
            compute_dtype=compute_dtype,
            hdr_authoring=source.hdr_authoring,
            reporter=reporter,
        )
        strength = source.strength_for_stage(stage)
        if source.frame_index == 0 and first_frame_mode == LTX2FirstFrameMode.ANCHOR:
            prepared.append(
                VideoConditionByLatentIndex(
                    latent=latent,
                    strength=strength,
                    latent_idx=0,
                )
            )
        else:
            prepared.append(
                VideoConditionByKeyframeIndex(
                    keyframes=latent,
                    frame_idx=source.frame_index,
                    strength=strength,
                )
            )
    return tuple(prepared)


__all__ = [
    "HDRReferenceConditionSource",
    "ImageConditionSource",
    "LTX2ConditionEncoderPort",
    "RawConditionSource",
    "prepare_conditions",
]
