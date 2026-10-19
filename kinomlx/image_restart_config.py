"""Shared still-image stage model, restart records, and the rules that read them."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import cast

from kinomlx.config_constraints import ChoiceConstraint, IntegerConstraint
from kinomlx.config_rules import RecordRule


class ImageStage(StrEnum):
    """The stages of a still-image recipe, in execution order.

    A run starts at the first stage, or at a later stage whose inputs a saved
    run's products supply.
    """

    TEXT = "text"
    DENOISE = "denoise"
    DECODE = "decode"


IMAGE_STAGES = tuple(ImageStage)
IMAGE_PRODUCES: Mapping[str, ImageStage] = frozendict(
    {
        "text_conditioning": ImageStage.TEXT,
        "initial_noise": ImageStage.DENOISE,
        "final_latents": ImageStage.DENOISE,
    }
)
# What a restart at each stage reads back. The keys are the stages a saved run
# lets a run start at, in choice order; the first is the default.
IMAGE_CONSUMES: Mapping[str, tuple[str, ...]] = frozendict(
    {
        ImageStage.DECODE: ("final_latents",),
        ImageStage.DENOISE: ("text_conditioning",),
    }
)
IMAGE_RESTARTS = tuple(IMAGE_CONSUMES)
# The record field that carries each artifact a restart can read back.
IMAGE_ARTIFACT_FIELDS: Mapping[str, str] = frozendict(
    {"final_latents": "latents", "text_conditioning": "text_conditioning"}
)


def _consumed(stage: str) -> tuple[str, ...]:
    """What a restart at ``stage`` reads back; an unknown stage is refused, not guessed."""
    if stage not in IMAGE_CONSUMES:
        raise ValueError(f"unknown restart stage {stage!r}")
    return IMAGE_CONSUMES[stage]


def _selection(values: Mapping[str, object]) -> None:
    if values["latents"] is not None and "final_latents" not in _consumed(
        cast(str, values["stage"])
    ):
        stages = "|".join(
            name for name, consumed in IMAGE_CONSUMES.items() if "final_latents" in consumed
        )
        raise ValueError(f"restart latents apply only when stage={stages}")


def _inputs(values: Mapping[str, object]) -> None:
    stage = cast(str, values["stage"])
    consumed = _consumed(stage)
    given = tuple(
        artifact for artifact, name in IMAGE_ARTIFACT_FIELDS.items() if values[name] is not None
    )
    if set(given) != set(consumed):
        raise ValueError(
            f"{stage} restart takes {', '.join(consumed)}, got {', '.join(given) or 'nothing'}"
        )


IMAGE_RESTART_SELECTION = RecordRule(
    "restart_selection", frozenset({"stage", "latents"}), _selection
)
IMAGE_RESTART_INPUTS = RecordRule(
    "restart_inputs", frozenset({"stage", "latents", "text_conditioning"}), _inputs
)


@dataclass(frozen=True)
class ImageRestartSelectionFields:
    """A prior run, the still-image stage to start at, and an optional latent override."""

    run: Path
    stage: ImageStage = field(
        default=ImageStage.DECODE,
        metadata={
            "config_constraints": (ChoiceConstraint(IMAGE_RESTARTS),),
            "config_rules": (IMAGE_RESTART_SELECTION,),
        },
    )
    latents: Path | None = None
    index: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )


@dataclass(frozen=True)
class ImageRestartArtifactFields:
    """The stage a run starts at and the saved artifacts that stage reads."""

    latents: Path | None = None
    stage: ImageStage = field(
        default=ImageStage.DECODE,
        metadata={
            "config_constraints": (ChoiceConstraint(IMAGE_RESTARTS),),
            "config_rules": (IMAGE_RESTART_INPUTS,),
        },
    )
    text_conditioning: Path | None = None
