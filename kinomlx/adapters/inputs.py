"""Presence-preserving adapter input fields, separate from resolved runtime selections."""

from dataclasses import dataclass, field
from pathlib import Path

from kinomlx.config_constraints import (
    ItemChoicesConstraint,
    LengthConstraint,
    NumberConstraint,
    NumberSequenceConstraint,
)
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_record

from .input_rules import ADAPTER_SCHEDULE
from .schedule import AdapterSchedule, parse_sigma_window
from .selection import AdapterSelection


@dataclass(frozen=True)
class SigmaWindowConstraint(LengthConstraint):
    """Admit the existing band grammar wherever a window member is supplied."""

    minimum: int = 1
    when_supplied: bool = True

    def validate(self, value: object) -> None:
        super().validate(value)
        if not isinstance(value, str):
            raise ValueError("must be sigma-window text")
        parse_sigma_window(value)

    def describe(self) -> dict[str, object]:
        return {"kind": "sigma_window", "when_supplied": self.when_supplied}


def exclusion_field(categories: frozenset[str]) -> tuple[str, ...]:
    """Own the common empty default while each model owns its accepted vocabulary."""
    return field(default=(), metadata={"config_constraints": (ItemChoicesConstraint(categories),)})


@config_record
@dataclass(frozen=True)
class AdapterInput:
    """Common input controls; concrete model records declare their exclusion vocabulary."""

    path: Path
    strength: float = field(default=1.0, metadata={"config_constraints": (NumberConstraint(),)})
    exclude: tuple[str, ...] = exclusion_field(frozenset())


@config_record
@dataclass(frozen=True)
class ScheduledAdapterInput(AdapterInput):
    """Controls for models that resolve per-step strengths or noise-level bands."""

    step_strengths: tuple[float, ...] = field(
        default=(),
        metadata={
            "config_constraints": (NumberSequenceConstraint(1, when_supplied=True),),
            "config_rules": (ADAPTER_SCHEDULE,),
        },
    )
    sigma_window: str = field(
        default="", metadata={"config_constraints": (SigmaWindowConstraint(),)}
    )

    def selection(self, *, location: str = "adapter") -> AdapterSelection:
        """Resolve surviving intent after precedence, preserving explicit-strength conflicts."""
        validate_record(self, location=location)
        return AdapterSelection(
            path=self.path,
            strength=self.strength,
            exclude=self.exclude,
            schedule=AdapterSchedule(self.step_strengths) if self.step_strengths else None,
            sigma_window=parse_sigma_window(self.sigma_window) if self.sigma_window else None,
        )
