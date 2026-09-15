"""LTX-2 stage-specific adapter selection and profile resolution."""

import math
from dataclasses import dataclass

from kinomlx.adapters.lora import LoRAConfig, LoRAProfile
from kinomlx.adapters.selection import AdapterSelection


@dataclass(frozen=True)
class LTX2AdapterSelection(AdapterSelection):
    """One LTX-2 adapter with optional per-stage strength overrides."""

    stage_1_strength: float | None = None
    stage_2_strength: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("stage 1 strength", self.stage_1_strength),
            ("stage 2 strength", self.stage_2_strength),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"adapter {label} must be finite, got {value}")

    def strength_for_stage(self, stage: int) -> float:
        """Resolve the effective adapter strength for one LTX-2 stage."""
        if stage == 1:
            return self.strength if self.stage_1_strength is None else self.stage_1_strength
        if stage == 2:
            return self.strength if self.stage_2_strength is None else self.stage_2_strength
        raise ValueError(f"LTX-2 adapter stage must be 1 or 2, got {stage}")


def lora_profile_for_stage(
    selections: tuple[LTX2AdapterSelection, ...],
    stage: int,
) -> LoRAProfile:
    """Resolve one zero-free LTX-2 stage profile for the LoRA loader."""
    if stage not in {1, 2}:
        raise ValueError(f"LTX-2 adapter stage must be 1 or 2, got {stage}")
    return tuple(
        LoRAConfig(
            path=selection.path,
            strength=strength,
            exclude=selection.exclude,
        )
        for selection in selections
        if (strength := selection.strength_for_stage(stage)) != 0.0
    )


__all__ = [
    "LTX2AdapterSelection",
    "lora_profile_for_stage",
]
