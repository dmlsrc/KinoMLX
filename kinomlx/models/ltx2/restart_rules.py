"""Pure LTX restart selection and concrete artifact policies."""

from collections.abc import Mapping
from typing import Literal, cast

from kinomlx.config_rules import RecordRule

RestartPhase = Literal["decode", "stage-2"]
LatentStage = Literal["stage-1", "final"]
RESTART_PHASE_CHOICES = ("decode", "stage-2")
LATENT_STAGE_CHOICES = ("final", "stage-1")


def resolved_latent_stage(phase: RestartPhase, selected: LatentStage | None) -> LatentStage:
    return selected if selected is not None else "stage-1" if phase == "stage-2" else "final"


def _selection(values: Mapping[str, object]) -> None:
    if values["phase"] == "stage-2" and values["latent_stage"] is not None:
        raise ValueError("restart latent_stage applies only when phase=decode")


def _artifacts(values: Mapping[str, object]) -> None:
    phase = cast(RestartPhase, values["phase"])
    stage = resolved_latent_stage(phase, cast(LatentStage | None, values["latent_stage"]))
    if phase == "stage-2" and stage != "stage-1":
        raise ValueError("stage-2 restart requires stage-1 latents")
    if phase == "stage-2" and values["text_conditioning"] is None:
        raise ValueError("stage-2 restart requires text conditioning")
    if phase == "decode" and values["text_conditioning"] is not None:
        raise ValueError("decode restart does not consume text conditioning")


RESTART_SELECTION = RecordRule(
    "restart_selection", frozenset({"phase", "latent_stage"}), _selection
)
RESTART_ARTIFACTS = RecordRule(
    "restart_artifacts", frozenset({"phase", "latent_stage", "text_conditioning"}), _artifacts
)
