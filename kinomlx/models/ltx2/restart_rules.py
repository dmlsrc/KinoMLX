"""LTX stage model, restart declarations, and the record rules that read them."""

from collections.abc import Mapping
from enum import StrEnum
from typing import cast

from kinomlx.config_rules import RecordRule


class LTX2Stage(StrEnum):
    """The stages of the distilled recipe, in execution order.

    A run starts at the first stage, or at a later stage whose inputs a saved
    run's products supply. Stage 1 is a stage and not a start, because no
    saved product lets a run resume ahead of it.
    """

    TEXT = "text"
    STAGE_1 = "stage-1"
    STAGE_2 = "stage-2"
    DECODE = "decode"


TEXT_CONDITIONING = "text_conditioning"
STAGE_1_CONDITIONING = "stage_1_conditioning"
STAGE_2_CONDITIONING = "stage_2_conditioning"
STAGE_1_LATENTS = "stage_1_latents"
FINAL_LATENTS = "final_latents"

STAGES = tuple(LTX2Stage)
PRODUCES: Mapping[str, LTX2Stage] = frozendict(
    {
        TEXT_CONDITIONING: LTX2Stage.TEXT,
        STAGE_1_CONDITIONING: LTX2Stage.STAGE_1,
        STAGE_1_LATENTS: LTX2Stage.STAGE_1,
        STAGE_2_CONDITIONING: LTX2Stage.STAGE_2,
        FINAL_LATENTS: LTX2Stage.STAGE_2,
    }
)
# What a restart at each stage reads back. The keys are the stages a saved run
# lets a run start at, in choice order; the first is the default.
CONSUMES: Mapping[str, tuple[str, ...]] = frozendict(
    {
        LTX2Stage.DECODE: (FINAL_LATENTS,),
        LTX2Stage.STAGE_2: (TEXT_CONDITIONING, STAGE_1_LATENTS),
    }
)
RESTARTS = tuple(CONSUMES)
START_STAGES = (STAGES[0], *CONSUMES)
# The latent product each denoising stage writes. A restart reads the latest
# one written before it unless ``latent_stage`` names an earlier stage's.
LATENTS: Mapping[str, str] = frozendict(
    {LTX2Stage.STAGE_1: STAGE_1_LATENTS, LTX2Stage.STAGE_2: FINAL_LATENTS}
)
LATENT_STAGES = tuple(LATENTS)
_ORDER: Mapping[str, int] = {stage: index for index, stage in enumerate(STAGES)}


def _consumed(stage: str) -> tuple[str, ...]:
    """What a restart at ``stage`` reads back; an unknown stage is refused, not guessed."""
    if stage not in CONSUMES:
        raise ValueError(f"unknown restart stage {stage!r}")
    return CONSUMES[stage]


def latent_stages_before(stage: str) -> tuple[LTX2Stage, ...]:
    """The denoising stages whose latents a restart at ``stage`` can read."""
    _consumed(stage)
    return tuple(LTX2Stage(name) for name in LATENTS if _ORDER[name] < _ORDER[stage])


def default_latent_stage(stage: str) -> LTX2Stage:
    """The latent a restart at ``stage`` reads unless told otherwise: the latest before it."""
    return latent_stages_before(stage)[-1]


def consumed_artifacts(stage: str, latent_stage: str | None = None) -> tuple[str, ...]:
    """What a restart at ``stage`` reads, holding the latent of ``latent_stage`` when named."""
    selected = LATENTS[default_latent_stage(stage) if latent_stage is None else latent_stage]
    return tuple(selected if name in LATENTS.values() else name for name in _consumed(stage))


def readable_artifacts(stage: str) -> tuple[str, ...]:
    """Every artifact a restart at ``stage`` can read back, over every latent it may name."""
    readable: dict[str, None] = {}
    for latent_stage in latent_stages_before(stage):
        readable.update(dict.fromkeys(consumed_artifacts(stage, latent_stage)))
    return tuple(readable)


# What each restart stage can read back and so re-emit; availability derives from
# this, while a run reads the one latent ``consumed_artifacts`` selects.
READABLE: Mapping[str, tuple[str, ...]] = frozendict(
    {stage: readable_artifacts(stage) for stage in CONSUMES}
)


def _latent_stage(values: Mapping[str, object]) -> None:
    stage, latent_stage = cast(str, values["stage"]), values["latent_stage"]
    if latent_stage is not None and latent_stage not in latent_stages_before(stage):
        raise ValueError(f"{stage} restart cannot read latents written at {latent_stage}")


def _inputs(values: Mapping[str, object]) -> None:
    _latent_stage(values)
    stage = cast(str, values["stage"])
    latent_stage = cast(str | None, values["latent_stage"])
    consumed = consumed_artifacts(stage, latent_stage)
    given = (
        LATENTS[default_latent_stage(stage) if latent_stage is None else latent_stage],
        *((TEXT_CONDITIONING,) if values["text_conditioning"] is not None else ()),
    )
    if set(given) != set(consumed):
        raise ValueError(f"{stage} restart takes {', '.join(consumed)}, got {', '.join(given)}")


RESTART_SELECTION = RecordRule(
    "restart_selection", frozenset({"stage", "latent_stage"}), _latent_stage
)
RESTART_INPUTS = RecordRule(
    "restart_inputs", frozenset({"stage", "latent_stage", "text_conditioning"}), _inputs
)
