"""Field-owned stage policies and the stage model that derives artifact availability."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class StageMode(StrEnum):
    """What a saved stage admits for one field."""

    MUTABLE = "mutable"
    LOCKED = "locked"
    IRRELEVANT = "irrelevant"


class IrrelevantAction(StrEnum):
    """What a stage does with a field its stage cannot consume."""

    IGNORE = "ignore"
    REJECT = "reject"


@dataclass(frozen=True)
class StagePolicy:
    """One field's admission decision at a saved restart stage."""

    stage: str
    mode: StageMode
    attribute: str | None = None
    reason: str = field(default="", kw_only=True)
    irrelevant_action: IrrelevantAction | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        # Membership stays checked here as well as typed, because a frozen
        # record still accepts whatever a caller constructs it with.
        if not self.stage or self.mode not in StageMode:
            raise ValueError("stage policy requires a stage and valid mode")
        if self.attribute is not None and not self.attribute.isidentifier():
            raise ValueError("stage projection must name a typed record attribute")
        if self.attribute is not None and self.mode != StageMode.LOCKED:
            raise ValueError("only locked stage policies may select a projection")
        if not isinstance(self.reason, str) or (self.reason and not self.reason.strip()):
            raise ValueError("stage reason must be nonempty prose when supplied")
        if (self.mode == StageMode.IRRELEVANT) != (self.irrelevant_action is not None):
            raise ValueError("irrelevant stage policy requires an ignore or reject action")
        if self.irrelevant_action is not None and self.irrelevant_action not in IrrelevantAction:
            raise ValueError("irrelevant stage action must be ignore or reject")


@dataclass(frozen=True)
class OutputPolicy:
    """Stages that can produce one selectable output artifact."""

    produced_at: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.produced_at, frozenset) or any(
            not isinstance(stage, str) or not stage.replace("-", "_").isidentifier()
            for stage in self.produced_at
        ):
            raise ValueError("output policy requires frozen stage names")

    def describe(self) -> dict[str, object]:
        """Expose the same availability used by artifact selection and admission."""
        return {"produced_at": sorted(self.produced_at)}

    def selected(self, value: bool | None, *, save_all: bool, stage: str, field: str) -> bool:
        """Explicit unavailable output requests fail; save-all selects available artifacts."""
        if stage not in self.produced_at:
            if value is True:
                raise ValueError(f"{field} is not produced by a {stage} restart")
            return False
        return save_all if value is None else value


def stage_output_defaults(
    values: Mapping[str, object], policies: Mapping[str, OutputPolicy], *, stage: str
) -> dict[str, object]:
    """A new stage does not inherit requests to reproduce unavailable parent artifacts."""
    return {
        **values,
        **{name: False for name, policy in policies.items() if stage not in policy.produced_at},
    }


def produced_at_stages(
    stages: Sequence[str],
    produces: Mapping[str, str],
    consumes: Mapping[str, Sequence[str]],
) -> dict[str, frozenset[str]]:
    """Derive where each artifact can be produced from one family's stage model.

    ``stages`` is the recipe in execution order, ``produces`` names the stage
    that writes each artifact, and ``consumes`` names the saved artifacts each
    restart phase reads back. A run is entered at the first stage or at a
    restart phase. An artifact is producible from an entry point when its
    producing stage runs from there, or when that entry point consumed it and
    so can re-emit it.
    """
    order = {stage: index for index, stage in enumerate(stages)}
    if not stages or len(order) != len(stages):
        raise ValueError("stage model requires distinct stages in execution order")
    for artifact, stage in produces.items():
        if stage not in order:
            raise ValueError(f"{artifact} is produced at an undeclared stage {stage!r}")
    for phase, artifacts in consumes.items():
        if phase not in order or phase == stages[0]:
            raise ValueError(f"restart phase {phase!r} must be a stage after the first")
        for artifact in artifacts:
            if artifact not in produces or order[produces[artifact]] >= order[phase]:
                raise ValueError(f"{phase} cannot consume {artifact!r}: not produced earlier")
    entry_points = (stages[0], *consumes)
    return {
        artifact: frozenset(
            entry
            for entry in entry_points
            if order[entry] <= order[stage] or artifact in consumes.get(entry, ())
        )
        for artifact, stage in produces.items()
    }
