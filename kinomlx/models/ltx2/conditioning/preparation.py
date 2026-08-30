"""Raw condition sources and bounded encoder preparation."""

from __future__ import annotations

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
    HDRAuthoring,
    HDRReferenceConditioningConfig,
    ImageConditioningConfig,
)
from .item import EncodedCondition
from .keyframe import VideoConditionByKeyframeIndex
from .latent import VideoConditionByLatentIndex
from .reference import VideoConditionByReferenceLatent


class ConditionEncoderPort(_VideoEncoderCallablePort, Protocol):
    """Narrow callable surface used while preparing media conditions."""


@dataclass(frozen=True)
class ImageConditionSource:
    """Caller image intent without encoded tensors or model ownership."""

    path: Path
    frame_index: int = 0
    strength: float = 0.95
    hdr_authoring: HDRAuthoring | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.frame_index < 0:
            raise ValueError("condition frame_index must be non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("condition strength must be between 0 and 1")

    @classmethod
    def from_config(
        cls,
        config: ImageConditioningConfig,
        *,
        hdr_authoring: HDRAuthoring | None = None,
    ) -> ImageConditionSource:
        return cls(
            path=config.path,
            frame_index=config.frame_index,
            strength=config.strength,
            hdr_authoring=hdr_authoring,
        )

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
        config: HDRReferenceConditioningConfig,
    ) -> HDRReferenceConditionSource:
        return cls(path=config.path, strength=config.strength)

    @property
    def family(self) -> str:
        return "hdr-reference"


type RawConditionSource = ImageConditionSource | HDRReferenceConditionSource


def prepare_conditions(
    sources: Iterable[RawConditionSource],
    geometry: VideoPixelShape,
    encoder: ConditionEncoderPort,
    capabilities: LTX2Capabilities,
    *,
    compute_dtype: mx.Dtype,
    reporter: Reporter | None = None,
) -> tuple[EncodedCondition, ...]:
    """Encode ordered raw sources for exactly one stage geometry."""
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
        if source.hdr_authoring is None:
            latent = encode_image(
                source.path,
                encoder,
                width=geometry.width,
                height=geometry.height,
                compute_dtype=compute_dtype,
                reporter=reporter,
            )
        else:
            latent = encode_image(
                source.path,
                encoder,
                width=geometry.width,
                height=geometry.height,
                compute_dtype=compute_dtype,
                hdr_authoring=source.hdr_authoring,
                reporter=reporter,
            )
        if source.frame_index == 0:
            prepared.append(
                VideoConditionByLatentIndex(
                    latent=latent,
                    strength=source.strength,
                    latent_idx=0,
                )
            )
        else:
            prepared.append(
                VideoConditionByKeyframeIndex(
                    keyframes=latent,
                    frame_idx=source.frame_index,
                    strength=source.strength,
                )
            )
    return tuple(prepared)


__all__ = [
    "ConditionEncoderPort",
    "HDRReferenceConditionSource",
    "ImageConditionSource",
    "RawConditionSource",
    "prepare_conditions",
]
