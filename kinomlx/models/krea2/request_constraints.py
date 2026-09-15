"""Local Krea request grammars shared by configuration and API admission."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kinomlx.adapters.inputs import SigmaWindowConstraint
from kinomlx.config_constraints import LengthConstraint, NumberConstraint

from .adapters.inputs import profile_selection
from .reference_regions import parse_reference_region


@dataclass(frozen=True)
class NonzeroStrengthConstraint(NumberConstraint):
    def validate(self, value: object) -> None:
        super().validate(value)
        if value == 0:
            raise ValueError("must be nonzero")

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "nonzero": True}


@dataclass(frozen=True)
class ProfileExclusionsConstraint(LengthConstraint):
    def validate(self, value: object) -> None:
        super().validate(value)
        profile_selection(Path(), strength=1.0, exclusions=cast(str, value), location="profile")

    def describe(self) -> dict[str, object]:
        return {"kind": "krea_profile_exclusions"}


@dataclass(frozen=True)
class ProfileWindowConstraint(SigmaWindowConstraint):
    minimum: int = 0
    when_supplied: bool = False

    def validate(self, value: object) -> None:
        if value != "":
            super().validate(value)


@dataclass(frozen=True)
class ReferenceRegionsConstraint(LengthConstraint):
    def validate(self, value: object) -> None:
        super().validate(value)
        for region in cast(tuple[str, ...], value):
            parse_reference_region(region)

    def describe(self) -> dict[str, object]:
        return {"kind": "krea_reference_regions"}
