"""Shared still-image restart inputs and field-owned artifact relationships."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kinomlx.config_constraints import ChoiceConstraint, IntegerConstraint
from kinomlx.config_rules import RecordRule

ImageRestartPhase = Literal["decode", "denoise"]
IMAGE_RESTART_PHASES = ("decode", "denoise")


def _selection(values: Mapping[str, object]) -> None:
    if values["phase"] == "denoise" and values["latents"] is not None:
        raise ValueError("restart latents apply only when phase=decode")


def _artifacts(values: Mapping[str, object]) -> None:
    if values["phase"] == "decode":
        if values["latents"] is None:
            raise ValueError("decode restart requires final latents")
        if values["text_conditioning"] is not None:
            raise ValueError("decode restart does not consume text conditioning")
    else:
        if values["text_conditioning"] is None:
            raise ValueError("denoise restart requires text conditioning")
        if values["latents"] is not None:
            raise ValueError("denoise restart does not consume final latents")


IMAGE_RESTART_SELECTION = RecordRule(
    "restart_selection", frozenset({"phase", "latents"}), _selection
)
IMAGE_RESTART_ARTIFACTS = RecordRule(
    "restart_artifacts", frozenset({"phase", "latents", "text_conditioning"}), _artifacts
)


@dataclass(frozen=True)
class ImageRestartSelectionFields:
    """A prior run plus a selected still-image station and optional artifact override."""

    run: Path
    phase: ImageRestartPhase = field(
        default="decode",
        metadata={
            "config_constraints": (ChoiceConstraint(IMAGE_RESTART_PHASES),),
            "config_rules": (IMAGE_RESTART_SELECTION,),
        },
    )
    latents: Path | None = None
    index: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )


@dataclass(frozen=True)
class ImageRestartArtifactFields:
    """Concrete artifacts required by the selected public recipe."""

    latents: Path | None = None
    phase: ImageRestartPhase = field(
        default="decode",
        metadata={
            "config_constraints": (ChoiceConstraint(IMAGE_RESTART_PHASES),),
            "config_rules": (IMAGE_RESTART_ARTIFACTS,),
        },
    )
    text_conditioning: Path | None = None
