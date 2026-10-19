"""Per-stage sparse attention schedules for LTX video self-attention.

Each denoise stage takes a schedule written as text, giving each step a Sol-Attn tau or
``dense``:

- ``dense`` (the default) keeps every step of the stage dense.
- ``auto`` takes Sol-Engine's published stage-2 schedule, tau 1, 1.25 and 1.5 on the distilled
  stage's three steps (NVlabs Sana ``sol-engine``, LTX-2.5 on an RTX 5090, layer 0 dense there
  too). Stage 1 has no published schedule and refuses it.
- ``1,1.25,1.5`` gives one entry per step, in order; a single entry applies to every step.
- ``5-8=1.5`` sets steps by number (from 1) or inclusive range; steps it leaves out are dense.

Larger tau routes fewer key blocks to exact attention: faster, and further from dense. Dense
layers are written as layer numbers and inclusive ranges, ``0,46-47``, or ``none``; they keep
dense attention on every sparse step. Only video self-attention is ever sparse.
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from kinomlx.config_constraints import LengthConstraint

from .kernels.sol_attention import morton_order

DENSE = "dense"
AUTO = "auto"
# Sol-Engine's published stage-2 taus, one per step of the distilled three-step stage 2.
AUTO_STAGE_2_TAUS: tuple[float, ...] = (1.0, 1.25, 1.5)
NO_LAYERS = "none"
# Both LTX-2 generations stack 48 transformer blocks.
LAYER_COUNT = 48

type StepEntry = float | None
"""A step's Sol-Attn tau, or None for a dense step."""


def _entry(text: str) -> StepEntry:
    value = text.strip()
    if value == DENSE:
        return None
    try:
        tau = float(value)
    except ValueError:
        raise ValueError(f'a step takes a tau number or "{DENSE}", got {value!r}') from None
    if not math.isfinite(tau):
        raise ValueError(f"a step's tau must be finite, got {value!r}")
    return tau


def _step_range(text: str) -> tuple[int, int]:
    first, separator, last = text.strip().partition("-")
    try:
        start = int(first)
        end = int(last) if separator else start
    except ValueError:
        raise ValueError(
            f"steps are numbered from 1, like 3 or 5-8, got {text.strip()!r}"
        ) from None
    if start < 1 or end < start:
        raise ValueError(f"steps are numbered from 1, like 3 or 5-8, got {text.strip()!r}")
    return start, end


@dataclass(frozen=True)
class StepSchedule:
    """A parsed stage schedule, before it meets the stage's step count.

    Exactly one form is set: ``entries`` lists one entry per step (or one for every step),
    ``ranges`` sets inclusive step ranges and leaves the others dense, and ``auto`` takes
    Sol-Engine's stage-2 schedule.
    """

    entries: tuple[StepEntry, ...] = ()
    ranges: tuple[tuple[int, int, StepEntry], ...] = ()
    auto: bool = False

    def resolve(self, steps: int, *, label: str, stage: int) -> tuple[StepEntry, ...]:
        """Give every one of stage ``stage``'s ``steps`` its entry."""
        if self.auto:
            if stage != 2:
                raise ValueError(
                    f'{label}: "{AUTO}" is Sol-Engine\'s stage-2 schedule; stage 1 takes taus '
                    f'or "{DENSE}"'
                )
            if steps != len(AUTO_STAGE_2_TAUS):
                raise ValueError(
                    f'{label}: "{AUTO}" sets the taus of the distilled three-step stage 2 '
                    f"(1, 1.25 and 1.5), but the stage runs {steps} steps; write the taus out"
                )
            return AUTO_STAGE_2_TAUS
        if self.ranges:
            resolved: list[StepEntry] = [None] * steps
            for start, end, entry in self.ranges:
                if end > steps:
                    raise ValueError(f"{label} names step {end}, but the stage runs {steps} steps")
                resolved[start - 1 : end] = [entry] * (end - start + 1)
            return tuple(resolved)
        if len(self.entries) == 1:
            return self.entries * steps
        if len(self.entries) != steps:
            raise ValueError(
                f"{label} lists {len(self.entries)} steps, but the stage runs {steps}; give "
                "one entry per step or a single entry for every step"
            )
        return self.entries


def parse_step_schedule(text: str) -> StepSchedule:
    """Parse a stage schedule: ``dense``, ``auto``, ``1,1.25,1.5`` or ``5-8=1.5``."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f'a sparse attention schedule must not be empty; use "{DENSE}"')
    if text.strip() == AUTO:
        return StepSchedule(auto=True)
    items = [item.strip() for item in text.split(",")]
    if any(not item for item in items):
        raise ValueError(f"a sparse attention schedule has an empty entry: {text!r}")
    keyed = ["=" in item for item in items]
    if not any(keyed):
        return StepSchedule(entries=tuple(_entry(item) for item in items))
    if not all(keyed):
        raise ValueError(
            "a sparse attention schedule lists entries per step or sets STEPS=ENTRY items, "
            f"not both: {text!r}"
        )
    ranges: list[tuple[int, int, StepEntry]] = []
    for item in items:
        steps, _, entry = item.partition("=")
        start, end = _step_range(steps)
        for other_start, other_end, _ in ranges:
            if start <= other_end and other_start <= end:
                raise ValueError(f"a sparse attention schedule sets step {start} twice: {text!r}")
        ranges.append((start, end, _entry(entry)))
    return StepSchedule(ranges=tuple(sorted(ranges)))


def _render_entry(entry: StepEntry) -> str:
    if entry is None:
        return DENSE
    text = repr(entry)
    return text.removesuffix(".0")


def canonical_step_schedule(text: str) -> str:
    """Write a schedule in its canonical spelling."""
    schedule = parse_step_schedule(text)
    if schedule.auto:
        return AUTO
    if schedule.ranges:
        return ",".join(
            (f"{start}" if start == end else f"{start}-{end}") + f"={_render_entry(entry)}"
            for start, end, entry in schedule.ranges
        )
    return ",".join(_render_entry(entry) for entry in schedule.entries)


def is_dense_schedule(text: str) -> bool:
    """Whether a schedule leaves every step dense, whatever its step count."""
    schedule = parse_step_schedule(text)
    if schedule.auto:
        return False
    if schedule.ranges:
        return all(entry is None for _, _, entry in schedule.ranges)
    return all(entry is None for entry in schedule.entries)


def parse_layer_ranges(text: str) -> tuple[int, ...]:
    """Parse ``0,46-47`` or ``none`` into sorted, distinct transformer layer numbers."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f'dense layers must not be empty; use "{NO_LAYERS}" for no layers')
    if text.strip() == NO_LAYERS:
        return ()
    layers: set[int] = set()
    for item in text.split(","):
        first, separator, last = item.strip().partition("-")
        try:
            start = int(first)
            end = int(last) if separator else start
        except ValueError:
            raise ValueError(
                f"dense layers are numbers and ranges like 0,46-47, got {item.strip()!r}"
            ) from None
        if not 0 <= start <= end < LAYER_COUNT:
            raise ValueError(
                f"dense layers run from 0 to {LAYER_COUNT - 1}, in increasing ranges, got "
                f"{item.strip()!r}"
            )
        span = set(range(start, end + 1))
        repeated = layers & span
        if repeated:
            raise ValueError(f"dense layers repeat layer {min(repeated)}")
        layers |= span
    return tuple(sorted(layers))


def canonical_layer_ranges(text: str) -> str:
    """Write dense layers as ascending numbers and merged ranges."""
    return render_layer_ranges(parse_layer_ranges(text))


def render_layer_ranges(layers: tuple[int, ...]) -> str:
    """Write sorted, distinct layer numbers as merged ranges, or ``none``."""
    if not layers:
        return NO_LAYERS
    runs: list[list[int]] = []
    for layer in layers:
        if runs and layer == runs[-1][1] + 1:
            runs[-1][1] = layer
        else:
            runs.append([layer, layer])
    return ",".join(f"{start}" if start == end else f"{start}-{end}" for start, end in runs)


def _canonical(text: object, writer: object) -> object:
    if not isinstance(text, str) or not callable(writer):
        return text
    try:
        return writer(text)
    except ValueError:
        # Leave an invalid spelling for validation to reject with its own message.
        return text


@dataclass(frozen=True)
class StepScheduleConstraint(LengthConstraint):
    """Own the per-stage sparse attention schedule spelling."""

    minimum: int = 1

    def canonicalize(self, value: object) -> object:
        return _canonical(value, canonical_step_schedule)

    def validate(self, value: object) -> None:
        if not isinstance(value, str):
            raise ValueError("must be text")
        super().validate(value)
        parse_step_schedule(value)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "kind": "sparse_attention_steps"}


@dataclass(frozen=True)
class LayerRangesConstraint(LengthConstraint):
    """Own the dense-layer spelling: numbers and inclusive ranges from 0 to 47, or none."""

    minimum: int = 1

    def canonicalize(self, value: object) -> object:
        return _canonical(value, canonical_layer_ranges)

    def validate(self, value: object) -> None:
        if not isinstance(value, str):
            raise ValueError("must be text")
        super().validate(value)
        parse_layer_ranges(value)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "kind": "layer_ranges",
            "first_layer": 0,
            "last_layer": LAYER_COUNT - 1,
        }


@dataclass(frozen=True)
class SparseAttentionSchedule:
    """One stage's resolved sparse attention: each step's tau or None, and the dense layers."""

    taus: tuple[StepEntry, ...]
    dense_layers: tuple[int, ...]

    @property
    def active(self) -> bool:
        return any(tau is not None for tau in self.taus)

    def render(self) -> str:
        """Each step's entry and the dense layers, for logs."""
        steps = ", ".join(_render_entry(tau) for tau in self.taus)
        return f"{steps}; dense layers {render_layer_ranges(self.dense_layers)}"

    def to_metadata(self) -> dict[str, object] | None:
        """The run receipt: None for a dense stage."""
        if not self.active:
            return None
        return {
            "method": "sol-attn",
            "token_order": "morton",
            "taus": list(self.taus),
            "dense_layers": list(self.dense_layers),
        }


def resolve_schedule(
    schedule: str, dense_layers: str, *, steps: int, label: str, stage: int
) -> SparseAttentionSchedule:
    """Resolve stage ``stage``'s schedule against its step count."""
    taus = parse_step_schedule(schedule).resolve(steps, label=label, stage=stage)
    return SparseAttentionSchedule(taus, parse_layer_ranges(dense_layers))


@dataclass(frozen=True, eq=False)
class SparseAttentionStep:
    """Sparse video self-attention for one transformer forward.

    ``order`` lists the video tokens in routing order: the generated latent grid along a 3-D
    Morton curve, then any appended conditioning tokens in their own order.
    """

    tau: float
    dense_layers: frozenset[int]
    order: mx.array

    def applies_to(self, layer: int) -> bool:
        """Whether a transformer block computes its video self-attention sparsely."""
        return layer not in self.dense_layers


def stage_steps(
    schedule: SparseAttentionSchedule,
    *,
    grid: tuple[int, int, int],
    tokens: int,
) -> tuple[SparseAttentionStep | None, ...]:
    """Build each step's forward record over a ``(frames, height, width)`` token grid."""
    if not schedule.active:
        return (None,) * len(schedule.taus)
    order = morton_order(*grid, tokens=tokens)
    dense_layers = frozenset(schedule.dense_layers)
    return tuple(
        None if tau is None else SparseAttentionStep(tau, dense_layers, order)
        for tau in schedule.taus
    )


__all__ = [
    "AUTO",
    "AUTO_STAGE_2_TAUS",
    "DENSE",
    "LAYER_COUNT",
    "NO_LAYERS",
    "LayerRangesConstraint",
    "SparseAttentionSchedule",
    "SparseAttentionStep",
    "StepEntry",
    "StepSchedule",
    "StepScheduleConstraint",
    "canonical_layer_ranges",
    "canonical_step_schedule",
    "is_dense_schedule",
    "parse_layer_ranges",
    "parse_step_schedule",
    "render_layer_ranges",
    "resolve_schedule",
    "stage_steps",
]
