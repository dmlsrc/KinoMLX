"""LTX-2 adapter inputs with independent optional stage strengths."""

from dataclasses import dataclass, field

from kinomlx.adapters.inputs import AdapterInput, exclusion_field
from kinomlx.config_constraints import NumberConstraint
from kinomlx.config_records import config_record

from .categories import ADAPTER_EXCLUDE_CATEGORIES
from .selection import LTX2AdapterSelection


@config_record
@dataclass(frozen=True)
class LTX2AdapterConfig(AdapterInput):
    """One ordered user adapter; omitted stage strengths inherit common strength."""

    exclude: tuple[str, ...] = exclusion_field(ADAPTER_EXCLUDE_CATEGORIES)
    stage1_strength: float | None = field(
        default=None, metadata={"config_constraints": (NumberConstraint(when_supplied=True),)}
    )
    stage2_strength: float | None = field(
        default=None, metadata={"config_constraints": (NumberConstraint(when_supplied=True),)}
    )

    def selection(self, *, location: str = "adapter") -> LTX2AdapterSelection:  # noqa: ARG002
        """Resolve surviving stage intent into the existing runtime representation."""
        return LTX2AdapterSelection(
            path=self.path,
            strength=self.strength,
            exclude=self.exclude,
            stage_1_strength=self.stage1_strength,
            stage_2_strength=self.stage2_strength,
        )
