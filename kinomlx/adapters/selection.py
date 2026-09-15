"""Resolved runtime adapter selections shared by model pipelines."""

from dataclasses import dataclass
from pathlib import Path

from .schedule import (
    AdapterSchedule,
    AdapterSigmaWindow,
    require_schedule_length,
)


@dataclass(frozen=True)
class AdapterSelection:
    """One fully expanded per-run community adapter selection."""

    path: Path
    strength: float
    exclude: tuple[str, ...] = ()
    schedule: AdapterSchedule | None = None
    sigma_window: AdapterSigmaWindow | None = None

    @property
    def effective_strength(self) -> float:
        """The first magnitude this selection applies, not an unread fallback.

        A window's bands may declare their own magnitude, in which case the
        scalar is a fallback the window never reads, so the first band's own
        strength is what a reader should see. Bands are highest-priority first,
        which makes that the window's analogue of a schedule's first step.
        """
        if self.schedule is not None:
            return self.schedule.per_step[0]
        if self.sigma_window is not None:
            leading = self.sigma_window.bands[0].strength
            return self.strength if leading is None else leading
        return self.strength

    @property
    def effective(self) -> bool:
        """Whether this selection can change any weight it is placed on."""
        if self.schedule is not None:
            return self.schedule.effective
        if self.sigma_window is not None:
            # Each band applies its own magnitude when it declares one and the
            # scalar fallback otherwise, so the window is effective exactly when
            # some band would apply a nonzero strength.
            return any(
                (self.strength if band.strength is None else band.strength) != 0.0
                for band in self.sigma_window.bands
            )
        return self.strength != 0.0

    @property
    def strength_plan(self) -> AdapterSchedule | AdapterSigmaWindow | None:
        """The per-step array or noise-level window, or ``None`` for a constant."""
        return self.schedule if self.schedule is not None else self.sigma_window

    @property
    def fuses(self) -> bool:
        """Whether one constant strength expresses everything this selection asks.

        A step array naming the same strength at every step is a constant, and
        fusion applies a constant once instead of paying the activation path for
        the whole run. A window is never constant here: it resolves against the
        sampler's own schedule when the run binds it, which is not known yet.
        """
        return self.sigma_window is None and (self.schedule is None or self.schedule.constant)

    @property
    def varies(self) -> bool:
        """Whether this selection's strength changes over the run."""
        return self.strength_plan is not None


def validate_adapter_gate(
    *,
    has_window: bool,
    has_steps: bool,
    window_field: str,
    steps_field: str,
) -> None:
    """Reject competing gate controls using the caller's actual field names."""
    if has_window and has_steps:
        raise ValueError(f"use {window_field} or {steps_field}, not both")


def require_adapter_schedule_steps(
    selections: tuple[AdapterSelection, ...], steps: int, *, item_name: str = "adapter"
) -> None:
    """Check every scheduled selection against the step count the recipe resolved.

    Called before any weights load, because a model profile can rewrite the step
    count after request construction and a wrong array length should not wait for
    the first denoise step to surface.
    """
    for selection in selections:
        if selection.schedule is not None:
            require_schedule_length(
                selection.schedule, steps, label=f"{item_name} {selection.path.name}"
            )


__all__ = [
    "AdapterSelection",
    "require_adapter_schedule_steps",
    "validate_adapter_gate",
]
