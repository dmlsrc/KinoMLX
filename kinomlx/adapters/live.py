"""Scheduled adapters applied to activations, leaving base weights unchanged.

Fusion overwrites a target weight, so a strength that changes between denoise
steps would mean rewriting every adapted weight at each change point - and for
any format whose delta is a function of the base weight, the original base is
gone and the change is not even invertible. A scheduled adapter therefore stays
unfused: its factors are retained in the compute dtype and applied to
activations, which is cheap exactly when the delta has a form that never builds
the dense ``[out, in]`` product.

Three structures qualify, and ``live_applier`` declines everything else so the
caller can refuse the schedule by name rather than silently compute a wrong
result:

- low-rank products, applied as two skinny products (LoRA, AdaLoRA, and LoHa
  after its Hadamard product is expanded into one wider pair, while that pair is
  still smaller than the dense delta);
- Kronecker products, applied by reshaping the input axis (LoKr).

A scheduled term always lands after anything fused into the same weight, so a
scheduled adapter never participates in another adapter's base-dependent
reconstruction.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.modeling.loading import WeightTarget

from .algebra import intrinsic_scale, loha_pairs, lokr_factors, plain_lora_entry
from .formats import AdapterOperation
from .lora import validated_lora_scale
from .schedule import AdapterSchedule, AdapterSigmaWindow

_log = logging.getLogger(__name__)

type LiveApply = Callable[[mx.array, mx.array, float], mx.array]


type AdapterStrengthPlan = AdapterSchedule | AdapterSigmaWindow


@dataclass(frozen=True)
class ScheduledAdapterOperation:
    """One operation whose strength varies over the run instead of being constant.

    The plan is either a resolved per-step array or a noise-level window, which
    only becomes an array once the sampler's own schedule is known.
    """

    operation: AdapterOperation
    schedule: AdapterStrengthPlan
    order_key: str
    strength: float = 1.0


@dataclass
class LiveAdapterTerm:
    """One scheduled operation bound to one Linear, plus its current strength.

    ``strength`` is an ordinary Python float rewritten once per step. Nothing in
    the forward graph closes over it, so a step change costs no device work.
    """

    order_key: str
    kind: str
    target: str
    plan: AdapterStrengthPlan
    apply: LiveApply
    retained_bytes: int
    window_strength: float = 1.0
    schedule: AdapterSchedule | None = None
    strength: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.plan, AdapterSchedule):
            self.schedule = self.plan

    @property
    def varies(self) -> bool:
        """Whether this term's strength changes between steps.

        An unresolved window counts as varying: selecting steps by noise level is
        how a gate is expressed, so a caller must not treat it as constant.
        """
        return True if self.schedule is None else not self.schedule.constant

    def bind(self, sigmas: Sequence[float]) -> None:
        """Resolve a noise-level window against the schedule the sampler walks."""
        if self.schedule is None and isinstance(self.plan, AdapterSigmaWindow):
            self.schedule = self.plan.resolve(
                sigmas, strength=self.window_strength, label=f"adapter {self.order_key}"
            )


@dataclass(frozen=True)
class _LowRank:
    """``delta = scale * (b @ a)``, applied as two skinny products."""

    a_t: mx.array
    b_t: mx.array
    scale: float

    def __call__(self, value: mx.array, result: mx.array, strength: float) -> mx.array:
        # The addmm epilogue folds the scale and the accumulation into the second
        # product, so no full-width delta is written and reread.
        return mx.addmm(result, value @ self.a_t, self.b_t, alpha=self.scale * strength)


@dataclass(frozen=True)
class _Kronecker:
    """``delta = scale * (w1 (x) w2)``, applied by reshaping the input axis.

    ``delta[(o1, o2), (i1, i2)] = w1[o1, i1] * w2[o2, i2]``, so folding the input
    axis into ``(in1, in2)`` turns the product into ``w1 @ X @ w2.T``. The dense
    Kronecker expansion is never built.
    """

    w1: mx.array
    w2_t: mx.array
    scale: float
    grid: tuple[int, int]
    pairs: tuple[int, int]

    def __call__(self, value: mx.array, result: mx.array, strength: float) -> mx.array:
        shape = result.shape
        folded = value.reshape(*value.shape[:-1], *self.grid)
        paired = result.reshape(*shape[:-1], *self.pairs)
        applied = mx.addmm(paired, self.w1 @ folded, self.w2_t, alpha=self.scale * strength)
        return applied.reshape(shape)


def _nbytes(*values: mx.array) -> int:
    return sum(value.nbytes for value in values)


def _low_rank_applier(
    operation: AdapterOperation, shape: tuple[int, ...], dtype: mx.Dtype
) -> tuple[LiveApply, int] | None:
    entry = plain_lora_entry(operation)
    if entry is None:
        return None
    if (entry.b.shape[0], entry.a.shape[1]) != shape:
        raise ValueError(f"adapter {operation.prefix!r} does not match its logical target")
    scale = validated_lora_scale(entry, operation.scalar)
    a_t, b_t = entry.a.T.astype(dtype), entry.b.T.astype(dtype)
    return _LowRank(a_t, b_t, scale), _nbytes(a_t, b_t)


def _loha_expansion_exceeds_dense(operation: AdapterOperation, shape: tuple[int, ...]) -> bool:
    """Whether expanding this LoHa would retain more than the dense delta.

    A Hadamard product of two rank-r products is a single product of rank r*r:
    pairing every column of one factor with every column of the other. Retained
    size and applied cost therefore both grow with the square of the rank, and
    past a crossover the expanded pair is larger than the dense delta it exists
    to avoid building. On a 3072-wide target that crossover is rank 40, which
    ordinary LyCORIS LoHa ranks reach.
    """
    extracted = loha_pairs(operation, shape)
    if extracted is None:
        return False
    rank = extracted[2]
    return rank * rank * (shape[0] + shape[1]) >= shape[0] * shape[1]


def _loha_applier(
    operation: AdapterOperation, shape: tuple[int, ...], dtype: mx.Dtype
) -> tuple[LiveApply, int] | None:
    extracted = loha_pairs(operation, shape)
    if extracted is None:
        return None
    (up_one, down_one), (up_two, down_two), rank = extracted
    # Decline rather than expand past the crossover: the caller then refuses the
    # schedule by name and the adapter fuses at one constant strength.
    if _loha_expansion_exceeds_dense(operation, shape):
        return None
    up = (up_one[:, :, None] * up_two[:, None, :]).reshape(shape[0], rank * rank)
    down = (down_one[:, None, :] * down_two[None, :, :]).reshape(rank * rank, shape[1])
    scale = intrinsic_scale(operation, rank)
    a_t, b_t = down.T.astype(dtype), up.T.astype(dtype)
    return _LowRank(a_t, b_t, scale), _nbytes(a_t, b_t)


def _kronecker_applier(
    operation: AdapterOperation, shape: tuple[int, ...], dtype: mx.Dtype
) -> tuple[LiveApply, int] | None:
    if operation.kind != "lokr" or operation.part != "weight" or len(shape) != 2:
        return None
    if "magnitude" in operation.factors:
        return None
    first, second, rank = lokr_factors(operation, shape)
    if first.ndim != 2 or second.ndim != 2:
        return None
    scale = intrinsic_scale(operation, rank)
    w1 = first.astype(dtype)
    w2_t = second.T.astype(dtype)
    return (
        _Kronecker(
            w1,
            w2_t,
            scale,
            grid=(first.shape[1], second.shape[1]),
            pairs=(first.shape[0], second.shape[0]),
        ),
        _nbytes(w1, w2_t),
    )


_APPLIERS = (_low_rank_applier, _loha_applier, _kronecker_applier)


def live_applier(
    operation: AdapterOperation, shape: tuple[int, ...], *, dtype: mx.Dtype
) -> tuple[LiveApply, int] | None:
    """Build one activation applier, or ``None`` when the format needs a dense delta.

    ``None`` is a capability answer, not a failure: the caller reports that this
    adapter format cannot carry a per-step schedule and must be fused at one
    constant strength.
    """
    if len(shape) != 2:
        return None
    for build in _APPLIERS:
        applier = build(operation, shape, dtype)
        if applier is not None:
            return applier
    return None


def live_decline_reason(operation: AdapterOperation, shape: tuple[int, ...]) -> str:
    """Say why ``live_applier`` declined, so the refusal states the real cause."""
    if _loha_expansion_exceeds_dense(operation, shape):
        return (
            "expands to a factor pair larger than the dense delta it would avoid "
            "building and cannot carry a per-step schedule"
        )
    return "has no activation form and cannot carry a per-step schedule"


_LIVE_CLASSES: dict[type, type] = {}


class _LiveLinear:
    """Mixed into one Linear class so scheduled terms add to its output."""

    # Underscore-prefixed, so MLX parameter traversal never sees the factors the
    # terms hold. The installer sets it when it reassigns the owner's class.
    _live_terms: tuple[LiveAdapterTerm, ...]

    def _with_live_terms(self, value: mx.array, result: mx.array) -> mx.array:
        for term in self._live_terms:
            # A zero strength is skipped outright, so a gated-off step computes
            # exactly what an unadapted weight computes.
            if term.strength != 0.0:
                result = term.apply(value, result, term.strength)
        return result


class LiveAdapterLinear(_LiveLinear, nn.Linear):
    """Linear whose scheduled adapters add to its output."""

    def __call__(self, value: mx.array) -> mx.array:
        return self._with_live_terms(value, nn.Linear.__call__(self, value))


class LiveAdapterPretransposedLinear(_LiveLinear, PretransposedLinear):
    """Pretransposed Linear whose scheduled adapters add to its output."""

    def __call__(self, value: mx.array) -> mx.array:
        return self._with_live_terms(value, PretransposedLinear.__call__(self, value))


_LIVE_CLASSES[nn.Linear] = LiveAdapterLinear
_LIVE_CLASSES[PretransposedLinear] = LiveAdapterPretransposedLinear


@dataclass(frozen=True)
class LiveAdapters:
    """Per-step strength fan-out for the scheduled adapters of one lease."""

    terms: tuple[LiveAdapterTerm, ...]

    @property
    def needs_sigmas(self) -> bool:
        """Whether any term still waits for the sampler's resolved schedule."""
        return any(term.schedule is None for term in self.terms)

    def bind_sigmas(self, sigmas: Sequence[float]) -> None:
        """Resolve every noise-level window once, before the first step."""
        for term in self.terms:
            term.bind(sigmas)

    def set_step(self, index: int) -> None:
        """Select the strength every scheduled term applies at one denoise step."""
        for term in self.terms:
            if term.schedule is None:
                raise RuntimeError(
                    f"adapter {term.order_key} selects steps by noise level, so the "
                    "sampler must bind its resolved schedule before stepping"
                )
            term.strength = term.schedule.at(index)

    @property
    def retained_bytes(self) -> int:
        return sum(term.retained_bytes for term in self.terms)

    def to_metadata(self) -> dict[str, object]:
        schedules: dict[str, object] = {}
        for term in self.terms:
            resolved = None if term.schedule is None else term.schedule.to_metadata()
            entry: dict[str, object] = {"selection": "steps", "resolved": resolved}
            if isinstance(term.plan, AdapterSigmaWindow):
                entry = {
                    "selection": "sigma-window",
                    "window": term.plan.to_metadata(),
                    "strength": term.window_strength,
                    "resolved": resolved,
                }
            schedules.setdefault(term.order_key, entry)
        return {
            "targets": len({term.target for term in self.terms}),
            "terms": len(self.terms),
            "kinds": sorted({term.kind for term in self.terms}),
            "retained_factor_bytes": self.retained_bytes,
            "schedules": schedules,
        }


def live_adapters(model: object) -> LiveAdapters | None:
    """Custom transformer ports without scheduled adapters have no controller."""
    value = getattr(model, "live_adapters", None)
    return value if isinstance(value, LiveAdapters) else None


def set_live_adapter_step(model: object, index: int) -> None:
    """Advance any scheduled adapters to one zero-based denoise step."""
    controller = live_adapters(model)
    if controller is not None:
        controller.set_step(index)


def bind_live_adapter_sigmas(model: object, sigmas: Sequence[float] | mx.array) -> None:
    """Resolve any noise-level adapter window against the sampler's own schedule.

    Called once before the denoise loop. Binding here rather than at install keeps
    a window matched to the schedule the run actually walks, including one the
    user supplied and one a resolution-dependent shift produced.
    """
    controller = live_adapters(model)
    if controller is not None and controller.needs_sigmas:
        values = sigmas.tolist() if isinstance(sigmas, mx.array) else list(sigmas)
        controller.bind_sigmas(cast(list[float], values))


def _live_class(owner: object, target: str) -> type:
    current = type(owner)
    if current in _LIVE_CLASSES.values():
        return current
    live = _LIVE_CLASSES.get(current)
    if live is None:
        raise ValueError(
            f"adapter target {target!r} is a {current.__name__}, which cannot carry a "
            "per-step schedule; remove its step strengths to fuse it at one strength"
        )
    return live


def install_live_adapters(
    targets: Mapping[str, WeightTarget],
    operations: Mapping[str, Sequence[ScheduledAdapterOperation]],
    *,
    model: str,
) -> LiveAdapters:
    """Bind scheduled operations to their Linear targets without touching weights.

    Every factor is cast once to the target's own compute dtype, which is what
    the matmul it joins already uses. One batched finite check covers the cast
    results, because a narrowing cast is where an out-of-range factor becomes
    nonfinite.
    """
    terms: list[LiveAdapterTerm] = []
    checks: list[mx.array] = []
    checked: list[str] = []
    for name in sorted(operations):
        if name not in targets:
            raise ValueError(f"adapter target {name!r} is absent")
        target = targets[name]
        if target.attribute != "weight":
            raise ValueError(
                f"adapter target {name!r} is not a weight; only Linear weights carry "
                "a per-step schedule"
            )
        live = _live_class(target.owner, name)
        weight = cast(mx.array, getattr(target.owner, target.attribute))
        dtype = weight.dtype
        bound: list[LiveAdapterTerm] = []
        for scheduled in sorted(operations[name], key=lambda item: item.order_key):
            built = live_applier(scheduled.operation, target.shape, dtype=dtype)
            if built is None:
                reason = live_decline_reason(scheduled.operation, target.shape)
                raise ValueError(
                    f"{model} adapter {scheduled.order_key} uses "
                    f"{scheduled.operation.kind!r} at {name!r}, which {reason}; "
                    "remove its step strengths to fuse it at one constant strength"
                )
            applier, retained = built
            bound.append(
                LiveAdapterTerm(
                    order_key=scheduled.order_key,
                    kind=scheduled.operation.kind,
                    target=name,
                    plan=scheduled.schedule,
                    window_strength=scheduled.strength,
                    apply=applier,
                    retained_bytes=retained,
                )
            )
            for factor in _applier_factors(applier):
                checks.append(mx.all(mx.isfinite(factor)))
                checked.append(f"{name} ({factor.dtype})")
        target.owner.__class__ = live
        # Underscore-prefixed attributes stay outside MLX parameter traversal, so
        # retained factors never reach parameters(), set_dtype, or weight counts.
        object.__setattr__(target.owner, "_live_terms", tuple(bound))
        terms.extend(bound)
    if checks:
        finite = cast(list[bool], mx.stack(checks).tolist())
        failures = [label for label, valid in zip(checked, finite, strict=True) if not valid]
        if failures:
            raise ValueError(
                f"{model} scheduled adapter factors are nonfinite in the compute dtype: "
                f"{', '.join(sorted(set(failures)))}"
            )
    controller = LiveAdapters(tuple(terms))
    _log.info(
        "%s scheduled adapters: %d live terms over %d targets, %.1f MiB of retained "
        "factors in compute dtype",
        model,
        len(terms),
        len({term.target for term in terms}),
        controller.retained_bytes / (1 << 20),
    )
    return controller


def _applier_factors(applier: LiveApply) -> tuple[mx.array, ...]:
    if isinstance(applier, _LowRank):
        return (applier.a_t, applier.b_t)
    if isinstance(applier, _Kronecker):
        return (applier.w1, applier.w2_t)
    return ()


__all__ = [
    "AdapterStrengthPlan",
    "LiveAdapterLinear",
    "LiveAdapterPretransposedLinear",
    "LiveAdapterTerm",
    "LiveAdapters",
    "ScheduledAdapterOperation",
    "bind_live_adapter_sigmas",
    "install_live_adapters",
    "live_adapters",
    "live_applier",
    "live_decline_reason",
    "set_live_adapter_step",
]
