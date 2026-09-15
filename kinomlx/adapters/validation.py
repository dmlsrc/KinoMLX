"""One first-use finite scan owned by a freshly adapted transformer lease."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx

from kinomlx.modeling.loading import WeightTarget
from kinomlx.reporting import Reporter


class AdapterWeightError(ValueError):
    """An evaluated adapted tensor is nonfinite."""


def _finite(value: mx.array) -> mx.array:
    return mx.all(mx.isfinite(value))


# Compiled so the predicate fuses into one pass over each tensor. Uncompiled,
# isfinite materializes several full-size Boolean intermediates per tensor:
# 6.5 s against 0.6 s over 34.5 GiB of LTX-2.5 weights.
_compiled_finite = mx.compile(_finite)


@dataclass
class AdapterWeightCheck:
    """Retain live target accessors, never base weights, factors, or reductions."""

    enabled: bool = True
    targets: dict[str, Callable[[], mx.array]] = field(default_factory=dict)

    def add(self, name: str, target: WeightTarget) -> None:
        self.targets[name] = lambda: cast(mx.array, getattr(target.owner, target.attribute))

    def run(
        self,
        *,
        context: str,
        reporter: Reporter,
        active: Callable[[str], bool] | None = None,
        aggregate_total: int | None = None,
    ) -> None:
        if not self.enabled:
            self.targets.clear()
        if not self.targets:
            return
        selected = [name for name in self.targets if active is None or active(name)]
        if not selected:
            return
        phase = "check adapter weights"
        reporter.phase_start(
            phase, total=len(selected), unit="tensor", aggregate_total=aggregate_total
        )
        try:
            values = [(name, self.targets[name]()) for name in selected]
            finite = cast(
                list[bool], mx.stack([_compiled_finite(value) for _, value in values]).tolist()
            )
            failures = [
                f"{name} ({value.dtype})"
                for (name, value), valid in zip(values, finite, strict=True)
                if not valid
            ]
            if failures:
                raise AdapterWeightError(
                    f"{context}: nonfinite adapter weights: {', '.join(failures)}"
                )
            reporter.phase_advance(phase, len(values))
            for name in selected:
                del self.targets[name]
        finally:
            reporter.phase_end(phase)


def pending_weight_check(model: object) -> AdapterWeightCheck | None:
    """Custom transformer ports without adapters have no pending validation."""
    value = getattr(model, "adapter_weight_check", None)
    return value if isinstance(value, AdapterWeightCheck) else None


def check_adapter_weights(
    model: object, *, context: str, reporter: Reporter, active: Callable[[str], bool] | None = None
) -> None:
    pending = pending_weight_check(model)
    if pending is not None:
        pending.run(context=context, reporter=reporter, active=active)
