"""LoRA fusion math: ``W_fused = W + strength * scaling * (B @ A)``.

Given a matched :class:`LoRAEntry` and a user strength, computes
the delta and/or applies it across a base-weights dict.  Per-model
key mapping is the caller's responsibility - see
``kinomlx/models/<name>/cache/keys/`` for cache-aware prefix conventions.

Fusion preflights every adapter before changing weights. Placeable targets are
always fused, while coverage below the fixed 50 percent observability threshold
warns and a zero-percent adapter becomes a warned no-op. Malformed pairs,
unsafe scaling, and narrow-dtype overflow remain fatal before mutation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import cast

import mlx.core as mx

from kinomlx.lora.loading import LoRAEntry, iter_lora_entries

_log = logging.getLogger(__name__)

# Finite ceilings for narrow target dtypes. bf16 and fp32 share fp32's exponent
# range, so a finite fp32 fused weight can't overflow them - they are absent
# here and skip the magnitude check entirely (no per-weight readback on the
# default bf16 path).
_DTYPE_MAX_FINITE: dict[mx.Dtype, float] = {
    mx.float16: 65504.0,
}
_RANGE_CHECK_BATCH = 24
LOW_COVERAGE_WARNING_THRESHOLD = 0.5
# A mathematical bound computed in fp32 can round down by a few ulps. Treat the
# final 1% of FP16's range as inconclusive and verify the exact fused peak.
_RANGE_BOUND_EXACT_FRACTION = 0.99
_FusionRecord = tuple[LoRAEntry, float, bool]
_FusionTable = dict[str, list[_FusionRecord]]
_RangeValidationInputs = tuple[
    dict[str, mx.array],
    Iterable[tuple[dict[str, mx.array], float]],
    dict[str, float] | None,
]
_RangeCheck = tuple[mx.array, mx.array, list[_FusionRecord], mx.Dtype, float, str]


def _delta_abs_bound(entry: LoRAEntry, strength: float) -> mx.array:
    """Return a cheap upper bound on ``max(abs(scale * (B @ A)))``.

    For each rank row, ``max_j(abs(A[r, j]))`` bounds every input column.
    Multiplying that vector by ``abs(B)`` and reducing its output rows is a
    guaranteed bound without constructing the full out-by-in delta matrix.
    """
    effective_scale = abs(_validated_scale(entry, strength))
    a_row_peak = mx.max(mx.abs(entry.a.astype(mx.float32)), axis=1)
    output_bounds = mx.matmul(mx.abs(entry.b.astype(mx.float32)), a_row_peak)
    return mx.max(output_bounds) * effective_scale


def _fused_abs_bound(
    base: mx.array,
    records: list[_FusionRecord],
    *,
    base_peak: float | None = None,
) -> mx.array:
    """Bound a fused weight's absolute peak without forming its full delta."""
    bound = (
        mx.array(base_peak, dtype=mx.float32)
        if base_peak is not None
        else mx.max(mx.abs(base.astype(mx.float32)))
    )
    for entry, strength, _transpose in records:
        bound = bound + _delta_abs_bound(entry, strength)
    return bound


def _fused_range_check(
    base: mx.array,
    records: list[_FusionRecord],
    target_dtype: mx.Dtype,
    key: str,
    *,
    base_peak: float | None = None,
) -> _RangeCheck | None:
    """Return a deferred bounded narrow-dtype range check when required.

    A no-op for bf16 / fp32 targets. The cheap bound proves almost all normal
    FP16 adapters safe. A bound above 65504 is inconclusive rather than an
    error; evaluation falls back to the exact fused peak for that target.
    """
    limit = _DTYPE_MAX_FINITE.get(target_dtype)
    if limit is None:
        return None
    return (
        _fused_abs_bound(base, records, base_peak=base_peak),
        base,
        records,
        target_dtype,
        limit,
        key,
    )


def _raise_for_fused_peak(
    peak: float,
    target_dtype: mx.Dtype,
    limit: float,
    key: str,
) -> None:
    """Raise when one host-visible fused-weight peak is out of range."""
    if not math.isfinite(peak):
        raise ValueError(
            f"LoRA fusion produced a non-finite weight at '{key}': the base "
            f"weight or the delta already overflowed float32. Lower the LoRA "
            f"strength or check the adapter."
        )
    if peak > limit:
        raise ValueError(
            f"LoRA fusion overflows {target_dtype} at '{key}': fused "
            f"|max|={peak:.4g} exceeds the {limit:.4g} ceiling. Use a wider "
            f"transformer dtype (bf16) or lower the LoRA strength."
        )


def _evaluate_range_checks(
    checks: list[_RangeCheck],
) -> None:
    """Evaluate one bounded range-only batch with one normal-path transfer.

    The normal path reads base and factor arrays but never constructs the full
    fused weights. Only inconclusive bounds build exact fused peaks, one target
    at a time to cap exceptional-path memory. Stacking normal-path scalars
    avoids one ``item()`` synchronization per target.
    """
    if not checks:
        return
    peaks = mx.stack([check[0] for check in checks])
    mx.eval(peaks)
    host_peaks = cast(list[int | float], peaks.tolist())
    exact_checks = [
        check
        for peak, check in zip(host_peaks, checks, strict=True)
        if not math.isfinite(float(peak)) or float(peak) > check[4] * _RANGE_BOUND_EXACT_FRACTION
    ]
    if exact_checks:
        # Inconclusive bounds are an exceptional safety path. Evaluate them one
        # at a time so a pathological adapter cannot materialize a batch of
        # full transformer matrices merely to report an overflow.
        for _bound, base, records, target_dtype, limit, key in exact_checks:
            exact_peak = mx.max(mx.abs(_fused_weight_f32(base, records)))
            mx.eval(exact_peak)
            _raise_for_fused_peak(
                float(cast(int | float, exact_peak.item())),
                target_dtype,
                limit,
                key,
            )
    checks.clear()


def _validated_scale(entry: LoRAEntry, strength: float) -> float:
    """Validate a matrix pair and return its finite effective scale."""
    if entry.a.ndim != 2 or entry.b.ndim != 2:
        raise ValueError(
            "LoRA factors must both be 2D matrices; got "
            f"A={tuple(entry.a.shape)}, B={tuple(entry.b.shape)}"
        )
    if entry.a.shape[0] <= 0 or entry.b.shape[1] != entry.a.shape[0]:
        raise ValueError(
            "LoRA factors have incompatible ranks: "
            f"A={tuple(entry.a.shape)}, B={tuple(entry.b.shape)}"
        )
    try:
        effective_scale = float(strength) * float(entry.scaling)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid LoRA strength or scaling: {strength!r}") from exc
    if not math.isfinite(effective_scale):
        raise ValueError(f"LoRA effective scale must be finite, got {effective_scale!r}")
    return effective_scale


def _delta_shape(entry: LoRAEntry, strength: float = 1.0) -> tuple[int, int]:
    """Return ``B @ A`` shape after validating the low-rank factors."""
    _validated_scale(entry, strength)
    return int(entry.b.shape[0]), int(entry.a.shape[1])


def compute_delta(entry: LoRAEntry, strength: float = 1.0) -> mx.array:
    """Compute the additive LoRA delta in fp32.

    ``delta = strength * entry.scaling * (B @ A)`` where ``scaling``
    is ``alpha/rank`` if the entry has alpha, else ``1.0``.

    Returns an fp32 array regardless of input dtype; the caller
    is responsible for casting back to the base weight's dtype
    after summing into the base.
    """
    effective_scale = _validated_scale(entry, strength)
    a_f32 = entry.a.astype(mx.float32)
    b_f32 = entry.b.astype(mx.float32)
    return mx.matmul(b_f32, a_f32) * effective_scale


def _fused_weight_f32(
    base: mx.array,
    records: list[_FusionRecord],
) -> mx.array:
    """Build one fused fp32 weight graph from its preflighted records."""
    delta_sum: mx.array | None = None
    for entry, strength, transpose in records:
        delta = compute_delta(entry, strength)
        if transpose:
            delta = delta.T
        delta_sum = delta if delta_sum is None else delta_sum + delta
    if delta_sum is None:  # pragma: no cover - records guarantees a term
        raise AssertionError("LoRA fusion table contained no delta")
    return base.astype(mx.float32) + delta_sum


def _build_fusion_table(
    base_weights: dict[str, mx.array],
    adapter_specs: list[tuple[dict[str, mx.array], float]],
    *,
    emit_diagnostics: bool,
    consume_adapters: bool = False,
) -> _FusionTable:
    """Preflight adapter placement and return target-keyed fusion records."""
    base_by_prefix: dict[str, str] = {}
    for key in base_weights:
        prefix = key.removesuffix(".weight")
        previous = base_by_prefix.get(prefix)
        if previous is not None and previous != key:
            raise ValueError(
                f"ambiguous base weights {previous!r} and {key!r} share LoRA prefix {prefix!r}"
            )
        base_by_prefix[prefix] = key

    table: _FusionTable = {}
    for adapter_index, (lora_weights, strength) in enumerate(adapter_specs, start=1):
        try:
            strength_value = float(strength)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LoRA adapter {adapter_index} has invalid strength") from exc
        if not math.isfinite(strength_value):
            raise ValueError(
                f"LoRA adapter {adapter_index} strength must be finite, got {strength!r}"
            )

        entries = list(iter_lora_entries(lora_weights))
        if consume_adapters:
            lora_weights.clear()
        if not entries:
            raise RuntimeError(
                f"LoRA adapter {adapter_index} contains no complete supported A/B weight pairs"
            )

        placed = 0
        missing = 0
        shape_mismatched = 0
        unplaced: list[str] = []
        for prefix, entry in entries:
            delta_shape = _delta_shape(entry, strength_value)
            base_key = base_by_prefix.get(prefix)
            if base_key is None:
                missing += 1
                unplaced.append(prefix)
                continue

            base_shape = tuple(int(dim) for dim in base_weights[base_key].shape)
            declared_transpose = base_key.endswith(".weight_t")
            declared_normal = base_key.endswith(".weight")
            if declared_transpose:
                transpose = True
                shape_matches = base_shape == tuple(reversed(delta_shape))
            elif declared_normal:
                transpose = False
                shape_matches = base_shape == delta_shape
            elif base_shape == delta_shape:
                transpose = False
                shape_matches = True
            else:
                transpose = True
                shape_matches = len(base_shape) == 2 and base_shape == tuple(reversed(delta_shape))
            if not shape_matches:
                if emit_diagnostics:
                    _log.warning(
                        "LoRA shape mismatch for %s: base=%s delta=%s; skipping target",
                        base_key,
                        base_shape,
                        delta_shape,
                    )
                shape_mismatched += 1
                unplaced.append(prefix)
                continue

            table.setdefault(base_key, []).append((entry, strength_value, transpose))
            placed += 1

        coverage = placed / len(entries)
        if emit_diagnostics:
            _log.info(
                "LoRA adapter %d: strength=%.4g, placed %d/%d targets "
                "(coverage %.0f%%, %d missing, %d shape mismatches)",
                adapter_index,
                strength_value,
                placed,
                len(entries),
                coverage * 100.0,
                missing,
                shape_mismatched,
            )
        if coverage < LOW_COVERAGE_WARNING_THRESHOLD and emit_diagnostics:
            outcome = "no targets will be fused" if placed == 0 else "fusing placeable targets"
            _log.warning(
                "LoRA adapter %d coverage %.0f%% is below the %.0f%% warning threshold; "
                "%s (%d/%d targets, unplaced examples: %s)",
                adapter_index,
                coverage * 100.0,
                LOW_COVERAGE_WARNING_THRESHOLD * 100.0,
                outcome,
                placed,
                len(entries),
                unplaced[:3],
            )
        elif unplaced and emit_diagnostics:
            _log.warning(
                "LoRA adapter %d placed %d/%d targets (coverage %.0f%%); unplaced examples: %s",
                adapter_index,
                placed,
                len(entries),
                coverage * 100.0,
                unplaced[:3],
            )
    return table


def _assert_equivalent_range_table(
    production: _FusionTable,
    validation: _FusionTable,
) -> None:
    """Reject substitute validation arrays whose fusion plan differs."""
    if production.keys() != validation.keys():
        raise ValueError("LoRA range-validation targets do not match production targets")
    for key, production_records in production.items():
        validation_records = validation[key]
        production_signature = [
            (entry.a.shape, entry.b.shape, strength, transpose)
            for entry, strength, transpose in production_records
        ]
        validation_signature = [
            (entry.a.shape, entry.b.shape, strength, transpose)
            for entry, strength, transpose in validation_records
        ]
        if production_signature != validation_signature:
            raise ValueError(f"LoRA range-validation plan differs from production at {key!r}")


def _validate_fused_ranges(
    base_weights: dict[str, mx.array],
    table: _FusionTable,
    target_dtype: mx.Dtype | None,
    *,
    consume: bool,
    base_peaks: dict[str, float] | None = None,
) -> None:
    """Validate every narrow fused target before caller-visible mutation."""
    pending_checks: list[_RangeCheck] = []
    for key in list(table):
        records = table.pop(key) if consume else table[key]
        base = base_weights.pop(key) if consume else base_weights[key]
        out_dtype = target_dtype if target_dtype is not None else base.dtype
        limit = _DTYPE_MAX_FINITE.get(out_dtype)
        base_peak = None
        if limit is not None and base_peaks is not None:
            if key not in base_peaks:
                raise ValueError(
                    f"FP16 range sidecar is missing LoRA target {key!r}; "
                    "rebuild the transformer cache"
                )
            base_peak = base_peaks[key]
            if not math.isfinite(base_peak) or base_peak < 0.0 or base_peak > limit:
                raise ValueError(f"invalid cached FP16 base peak for {key!r}: {base_peak!r}")
        range_check = _fused_range_check(
            base,
            records,
            out_dtype,
            key,
            base_peak=base_peak,
        )
        if range_check is None:
            continue
        pending_checks.append(range_check)
        if len(pending_checks) >= _RANGE_CHECK_BATCH:
            _evaluate_range_checks(pending_checks)
    _evaluate_range_checks(pending_checks)


def fuse_many(
    base_weights: dict[str, mx.array],
    adapters: Iterable[tuple[dict[str, mx.array], float]],
    *,
    target_dtype: mx.Dtype | None = None,
    in_place: bool = False,
    before_mutation: Callable[[], None] | None = None,
    range_validation_inputs: _RangeValidationInputs | None = None,
) -> dict[str, mx.array]:
    """Apply one or more LoRAs in a single fp32 accumulation pass.

    ``adapters`` contains ``(weights, strength)`` pairs. Every adapter is
    preflighted independently against the base dictionary. Coverage is the
    number of placeable adapter targets divided by the number of complete
    LoRA pairs in that adapter. Values below 50 percent warn but still fuse
    every placeable target; zero percent is a warned no-op.

    Exact and transposed 2D target layouts are supported. Replacement weights
    remain lazy in both modes; there is no eager per-target materialization
    path. FP16 targets first run a bounded range preflight before any dictionary
    mutation. It proves normal adapters safe from base/factor reductions and
    computes an exact fused peak only for an inconclusive bound.

    A cache bridge may provide independently reloaded, equivalent base and
    adapter dictionaries plus cache-build FP16 base peaks through
    ``range_validation_inputs``. They are ownership-transfer scratch inputs:
    this function verifies their placement plan, consumes them in bounded
    batches, and clears them before building the returned graph from the
    untouched production arrays. This is why the live cache path does not use
    the stock production arrays for FP16 validation: evaluating a reduction
    through an MLX array realizes that array and would recreate the eager
    model-memory peak. ``before_mutation`` runs once immediately before the
    first successful in-place assignment. The default remains non-mutating.
    """
    adapter_specs = list(adapters)
    table = _build_fusion_table(
        base_weights,
        adapter_specs,
        emit_diagnostics=True,
    )

    if range_validation_inputs is None:
        _validate_fused_ranges(
            base_weights,
            table,
            target_dtype,
            consume=False,
        )
    else:
        validation_base, validation_adapters, validation_base_peaks = range_validation_inputs
        validation_specs = list(validation_adapters)
        try:
            validation_table = _build_fusion_table(
                validation_base,
                validation_specs,
                emit_diagnostics=False,
                consume_adapters=True,
            )
            _assert_equivalent_range_table(table, validation_table)
            for key in table:
                production_dtype = (
                    target_dtype if target_dtype is not None else base_weights[key].dtype
                )
                validation_dtype = (
                    target_dtype if target_dtype is not None else validation_base[key].dtype
                )
                if production_dtype != validation_dtype:
                    raise ValueError(
                        f"LoRA range-validation dtype differs at {key!r}: "
                        f"{validation_dtype} != {production_dtype}"
                    )
            _validate_fused_ranges(
                validation_base,
                validation_table,
                target_dtype,
                consume=True,
                base_peaks=validation_base_peaks,
            )
        finally:
            validation_base.clear()
            if validation_base_peaks is not None:
                validation_base_peaks.clear()
            for weights, _strength in validation_specs:
                weights.clear()

    fused = base_weights if in_place else {}
    mutation_started = False
    for key, base in base_weights.items():
        out_dtype = target_dtype if target_dtype is not None else base.dtype
        records = table.get(key)
        if not records:
            value = base if base.dtype == out_dtype else base.astype(out_dtype)
        else:
            value = _fused_weight_f32(base, records).astype(out_dtype)

        if in_place and value is not base:
            if not mutation_started and before_mutation is not None:
                before_mutation()
            mutation_started = True
        fused[key] = value
    return fused


def fuse(
    base_weights: dict[str, mx.array],
    lora_weights: dict[str, mx.array],
    strength: float = 1.0,
    *,
    target_dtype: mx.Dtype | None = None,
    in_place: bool = False,
    before_mutation: Callable[[], None] | None = None,
) -> dict[str, mx.array]:
    """Apply one LoRA to a base-weights dict.

    This is the one-adapter convenience wrapper around :func:`fuse_many`.
    It accepts exact or transposed 2D target layouts. Low coverage warns while
    preserving every placeable target. ``in_place=True`` changes only
    dictionary ownership; fused replacement weights remain lazy in either mode.
    """
    return fuse_many(
        base_weights,
        [(lora_weights, strength)],
        target_dtype=target_dtype,
        in_place=in_place,
        before_mutation=before_mutation,
    )
