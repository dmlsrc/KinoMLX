"""Model-neutral sigma schedules: presentation and validated user schedules.

A schedule holds ``N + 1`` noise levels driving ``N`` denoise steps, so the step
count follows from the array rather than being declared beside it. Rendering is
deliberately separate from validation: reporting a resolved schedule must never
fail a run, while a schedule the user supplied is checked before it reaches a
sampler.
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import mlx.core as mx

RENDER_LIMIT = 12


def _values(schedule: Sequence[float] | mx.array) -> tuple[float, ...]:
    if isinstance(schedule, mx.array):
        return tuple(cast(list[float], schedule.reshape(-1).tolist()))
    return tuple(float(value) for value in schedule)


def render_sigmas(schedule: Sequence[float] | mx.array, *, limit: int = RENDER_LIMIT) -> str:
    """Render one resolved schedule for the console, abbreviating a long one.

    Never validates: a resolved schedule is reported as it is, and the head is
    kept in full because the early high-noise steps are the ones a step-gated
    adapter is aimed at.
    """
    values = _values(schedule)
    if not values:
        return "none"
    rendered = [f"{value:g}" for value in values]
    if len(rendered) <= limit:
        return ", ".join(rendered)
    head = ", ".join(rendered[: limit - 3])
    tail = ", ".join(rendered[-2:])
    return f"{head}, ... , {tail} ({len(rendered)} values)"


def sigma_metadata(schedule: Sequence[float] | mx.array | None) -> list[float] | None:
    """Return one resolved schedule as plain JSON numbers."""
    return None if schedule is None else list(_values(schedule))


@dataclass(frozen=True)
class SigmaSchedule:
    """One validated user-supplied denoise schedule.

    The contract is model-neutral: at least one step, finite levels inside the
    unit interval, strictly decreasing. A first value below one is legitimate -
    LTX-2 stage 2 continues from stage 1's tail - and so is a final value above
    zero, which simply stops the denoise early.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if len(values) < 2:
            raise ValueError(
                "a sigma schedule needs at least two values, one more than its step count"
            )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("sigma values must be finite")
        outside = [f"{value:g}" for value in values if not 0.0 <= value <= 1.0]
        if outside:
            raise ValueError(f"sigma values must lie between 0 and 1, got {', '.join(outside)}")
        descents = itertools.pairwise(values)
        if any(current <= following for current, following in descents):
            raise ValueError(f"sigma values must decrease strictly, got {render_sigmas(values)}")
        object.__setattr__(self, "values", values)

    @property
    def steps(self) -> int:
        """The number of denoise steps this schedule drives."""
        return len(self.values) - 1

    def to_array(self, dtype: mx.Dtype = mx.float32) -> mx.array:
        return mx.array(self.values, dtype=dtype)

    def to_metadata(self) -> list[float]:
        return list(self.values)

    def render(self, *, limit: int = RENDER_LIMIT) -> str:
        return render_sigmas(self.values, limit=limit)


def parse_sigmas(text: str) -> tuple[float, ...]:
    """Parse one comma-separated sigma schedule from CLI or override text."""
    if not isinstance(text, str):
        raise ValueError("a sigma schedule must be given as text")
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError('sigmas must be a comma-separated list of numbers, such as "1,0.75,0.5,0"')
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError(f"sigmas must all be numbers, got {text!r}") from None
    return SigmaSchedule(values).values


def resolve_sigma_schedule(
    values: Sequence[float], *, steps: int | None, label: str
) -> SigmaSchedule:
    """Build a user schedule and reject a step count that contradicts it."""
    schedule = SigmaSchedule(tuple(values))
    if steps is not None and steps != schedule.steps:
        raise ValueError(
            f"{label} declares {len(schedule.values)} sigma values, which drive "
            f"{schedule.steps} steps, but steps is set to {steps}"
        )
    return schedule


def resolved_sigmas(
    explicit: Sequence[float],
    generate: Callable[[], Sequence[float] | mx.array],
    *,
    label: str,
) -> SigmaSchedule:
    """Prefer a supplied schedule over a model's own, validating whichever is used.

    Every model derives or pins a schedule and accepts an override of it, so this
    holds the single answer to which one wins. Validating the generated branch as
    well means a spacing that emits a non-decreasing or out-of-range level is
    refused here, naming the schedule, rather than reaching a denoise loop.
    """
    try:
        return SigmaSchedule(_values(explicit or generate()))
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc


@dataclass(frozen=True)
class ScheduleRequest:
    """What a scheduler needs before it can produce a schedule.

    A scheduler is configured long before either value is known: the step count
    can come from a profile or from a supplied array, and a resolution-aware
    shift needs the token grid. Both are settled together at the one point each
    model resolves its recipe against its geometry.
    """

    steps: int
    image_tokens: int = 0
    """The target token grid, for a scheduler that derives its own shift from it.

    Zero when the recipe already resolved every resolution-dependent term, as
    Krea 2 does by carrying an effective shift on the recipe itself.
    """


class SigmaScheduler(Protocol):
    """Produces one denoise schedule once steps and geometry are known.

    Implementations carry their own configuration and stay free of model state,
    so a new spacing - Karras, beta, a tail-weighted variant - is a class here
    plus the one line that selects it. Step dynamics such as added noise or
    momentum are deliberately not part of this seam: they belong to the Euler
    step, not to the schedule it walks.
    """

    @property
    def name(self) -> str:
        """Short identifier recorded in run metadata."""
        ...

    def prepare(self, request: ScheduleRequest) -> tuple[float, ...]:
        """Return ``steps + 1`` noise levels for this request."""
        ...


@dataclass(frozen=True)
class ExplicitScheduler:
    """Returns one fixed schedule, ignoring the request that asks for it.

    Serves both a user-supplied array and a model's literal pinned schedule.
    """

    values: tuple[float, ...]
    name: str = "explicit"

    def prepare(self, request: ScheduleRequest) -> tuple[float, ...]:  # noqa: ARG002
        return self.values


def resolve_schedule(scheduler: SigmaScheduler, request: ScheduleRequest) -> SigmaSchedule:
    """Run one scheduler and validate what it produced.

    Validation applies to built-in schedulers too: a spacing that emits a
    non-decreasing or out-of-range level is a defect, and finding it here names
    the scheduler instead of failing later inside a denoise loop.
    """
    values = scheduler.prepare(request)
    schedule = SigmaSchedule(tuple(values))
    if schedule.steps != request.steps:
        raise ValueError(
            f"scheduler {scheduler.name!r} returned {len(schedule.values)} values, "
            f"which drive {schedule.steps} steps, but {request.steps} were requested"
        )
    return schedule


__all__ = [
    "RENDER_LIMIT",
    "ExplicitScheduler",
    "ScheduleRequest",
    "SigmaSchedule",
    "SigmaScheduler",
    "parse_sigmas",
    "render_sigmas",
    "resolve_schedule",
    "resolve_sigma_schedule",
    "resolved_sigmas",
    "sigma_metadata",
]
