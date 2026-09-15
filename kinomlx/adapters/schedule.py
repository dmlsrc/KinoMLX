"""Per-step adapter strength arrays resolved against one sampler recipe.

A schedule holds one strength for every denoise step, so the sampler reads a
plain float per step rather than rebuilding weights. Any spelling a caller
offers - an explicit array, a gate, a ramp - resolves to this one representation
before it reaches the runtime, which therefore cannot tell them apart.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterSchedule:
    """One resolved strength for every denoise step of a single adapter."""

    per_step: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.per_step)
        if not values:
            raise ValueError("an adapter step schedule needs at least one strength")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("adapter step strengths must be finite")
        object.__setattr__(self, "per_step", values)

    def __len__(self) -> int:
        return len(self.per_step)

    def at(self, step: int) -> float:
        """Return the strength one zero-based denoise step applies."""
        if not 0 <= step < len(self.per_step):
            raise IndexError(
                f"denoise step {step} is outside a {len(self.per_step)}-step adapter schedule"
            )
        return self.per_step[step]

    @property
    def effective(self) -> bool:
        """Whether any step applies a nonzero strength."""
        return any(value != 0.0 for value in self.per_step)

    @property
    def constant(self) -> bool:
        """Whether every step shares one strength, which fusion would apply instead."""
        return len(set(self.per_step)) == 1

    def to_metadata(self) -> dict[str, object]:
        return {
            "per_step": list(self.per_step),
            "active_steps": [index for index, value in enumerate(self.per_step) if value != 0.0],
        }


@dataclass(frozen=True)
class AdapterSigmaBand:
    """One closed noise-level interval and the strength that applies inside it."""

    low: float
    high: float = 1.0
    strength: float | None = None

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("high", self.high)):
            if not math.isfinite(value):
                raise ValueError(f"adapter sigma band {name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"adapter sigma band {name} must lie between 0 and 1")
        if self.low > self.high:
            raise ValueError(f"adapter sigma band low {self.low:g} is above high {self.high:g}")
        if self.strength is not None and not math.isfinite(self.strength):
            raise ValueError("adapter sigma band strength must be finite")

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def render(self) -> str:
        span = f"{self.low:g} to {self.high:g}"
        return span if self.strength is None else f"{span} at {self.strength:g}"

    def to_metadata(self) -> dict[str, object]:
        return {"low": self.low, "high": self.high, "strength": self.strength}


@dataclass(frozen=True)
class AdapterSigmaWindow:
    """Ordered noise-level bands; the first band holding a level sets its strength.

    A step-indexed array says which steps an adapter runs on; bands say which noise
    levels it runs at, which survives a change to the step count. That is the right
    spelling when an adapter was trained at a known level: the Krea community SDA
    adapter records its own training band, and selecting that band reproduces the
    gate its model card prescribes on the eight-step schedule without naming a step.

    Several bands give a different strength at different levels, so a taper is one
    spec rather than a step array recomputed for every step count. First match wins,
    which makes overlap deterministic and lets a coarse band sit last as a fallback.
    Any level no band holds contributes nothing.

    The two spellings genuinely disagree on some schedules, so neither is a default
    for the other. Resolution happens against the schedule the sampler actually
    walks, never against an estimate of it.
    """

    bands: tuple[AdapterSigmaBand, ...]

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("an adapter sigma window needs at least one band")

    def resolve(self, sigmas: Sequence[float], *, strength: float, label: str) -> AdapterSchedule:
        """Select each step's strength from the first band holding its noise level.

        A schedule carries one more level than the steps it drives, and step ``i`` is
        evaluated at ``sigmas[i]``, so the final level is never a step's own.
        """
        if len(sigmas) < 2:
            raise ValueError(f"{label} needs a schedule of at least two levels to select steps")
        per_step = []
        held = False
        for value in sigmas[:-1]:
            level = float(value)
            band = next((item for item in self.bands if item.contains(level)), None)
            if band is None:
                per_step.append(0.0)
                continue
            held = True
            per_step.append(strength if band.strength is None else band.strength)
        # No band holding any level is the wrong-schedule mistake: the bands were
        # written for a different step count or a different model. A band that holds
        # levels and declares zero is instead a deliberate carve-out, so it stands.
        if not held:
            raise ValueError(
                f"{label} selects no step: no level of the resolved schedule falls inside "
                f"{self.render()}"
            )
        return AdapterSchedule(tuple(per_step))

    def render(self) -> str:
        return "; ".join(band.render() for band in self.bands)

    def to_metadata(self) -> dict[str, object]:
        return {"bands": [band.to_metadata() for band in self.bands]}


def _parse_band(text: str) -> AdapterSigmaBand:
    body, _, magnitude = text.partition("@")
    bounds = [part.strip() for part in body.split(":")]
    if len(bounds) > 2 or any(not part for part in bounds) or (("@" in text) and not magnitude):
        raise ValueError(
            'an adapter sigma band must be "low", "low:high", or either with "@strength", '
            'such as "0.93", "0.93:0.97", or "0.6:0.93@0.5"'
        )
    try:
        values = [float(part) for part in bounds]
        level = None if not magnitude else float(magnitude)
    except ValueError:
        raise ValueError(f"adapter sigma band numbers are unreadable in {text!r}") from None
    low, high = (values[0], values[1]) if len(values) == 2 else (values[0], 1.0)
    return AdapterSigmaBand(low, high, level)


def parse_sigma_window(text: str) -> AdapterSigmaWindow:
    """Parse comma-separated noise-level bands, highest-priority first."""
    if not isinstance(text, str):
        raise ValueError("an adapter sigma window must be given as text")
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError('an adapter sigma window must be one or more bands, such as "0.93"')
    return AdapterSigmaWindow(tuple(_parse_band(part) for part in parts))


def parse_step_strengths(text: str) -> tuple[float, ...]:
    """Parse one comma-separated per-step strength array from CLI or override text."""
    if not isinstance(text, str):
        raise ValueError("adapter step strengths must be given as text")
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            "adapter step strengths must be a comma-separated list of numbers, "
            'such as "1,1,0,0,0,0,0,0"'
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError(f"adapter step strengths must all be numbers, got {text!r}") from None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("adapter step strengths must be finite")
    return values


def render_strength_plan(
    schedule: AdapterSchedule | None, window: AdapterSigmaWindow | None
) -> tuple[str, str] | None:
    """Render whichever varying-strength spelling a selection used, for one log line."""
    if schedule is not None:
        return ("step strengths", ", ".join(f"{value:g}" for value in schedule.per_step))
    if window is not None:
        return ("sigma window", window.render())
    return None


def require_schedule_length(schedule: AdapterSchedule, steps: int, *, label: str) -> None:
    """Reject a schedule whose length disagrees with the resolved step count.

    The step count is only final after recipe resolution, because a model profile
    can rewrite it, so this check belongs to planning rather than to request
    construction.
    """
    if len(schedule) != steps:
        raise ValueError(
            f"{label} has {len(schedule)} step strengths, but the resolved recipe "
            f"runs {steps} steps"
        )


__all__ = [
    "AdapterSchedule",
    "AdapterSigmaBand",
    "AdapterSigmaWindow",
    "parse_sigma_window",
    "parse_step_strengths",
    "render_strength_plan",
    "require_schedule_length",
]
