"""Field-owned input admission for explicit denoise schedules."""

from dataclasses import dataclass
from typing import cast

from kinomlx.config_constraints import NumberSequenceConstraint

from .sigmas import SigmaSchedule


@dataclass(frozen=True)
class SigmaScheduleConstraint(NumberSequenceConstraint):
    minimum_length: int = 0

    def validate(self, value: object) -> None:
        super().validate(value)
        if value:
            SigmaSchedule(cast(tuple[float, ...], value))

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "kind": "sigma_schedule"}
