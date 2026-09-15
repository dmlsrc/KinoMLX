"""Bounded preflight and fusion for one freshly loaded transformer lease."""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import mlx.core as mx

from kinomlx.modeling.loading import WeightTarget

from .algebra import plain_lora_entry, reconstruct_delta
from .formats import AdapterOperation
from .lora import validated_lora_scale
from .validation import AdapterWeightCheck

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightedAdapterOperation:
    operation: AdapterOperation
    strength: float
    order_key: str


def logical_weight(target: WeightTarget, *, transposed: bool) -> mx.array:
    weight = cast(mx.array, getattr(target.owner, target.attribute))
    logical = weight.T if transposed else weight
    if logical.shape != target.shape:
        raise ValueError(
            f"adapter target has physical shape {weight.shape}, expected logical {target.shape}"
        )
    return logical


def _fused_weight(
    target: WeightTarget,
    operations: tuple[WeightedAdapterOperation, ...],
    *,
    transposed: bool,
    base_weight: mx.array | None = None,
    accumulation: str = "float32",
) -> mx.array:
    original = logical_weight(target, transposed=transposed)
    ordered = sorted(operations, key=lambda item: (item.order_key, item.strength))
    entries = [(plain_lora_entry(selected.operation), selected) for selected in ordered]
    if (
        accumulation == "float16"
        and original.dtype == mx.float16
        and all(
            entry is not None and selected.operation.kind == "lora" for entry, selected in entries
        )
    ):
        # Half targets accumulate plain low-rank terms in half: the GEMM sums in
        # FP32 internally, but each term rounds to float16 before the next is added,
        # so stacked terms and cancellations round differently from the FP32 chain.
        # Callers admit this path with ``half_accumulation_fallbacks``: the factors
        # are cast to float16 first, so factor magnitudes beyond its range overflow
        # even when the FP32 product would fit.
        fused = original
        for entry, selected in entries:
            assert entry is not None
            scale = validated_lora_scale(entry, selected.strength * selected.operation.scalar)
            if (entry.b.shape[0], entry.a.shape[1]) != original.shape:
                raise ValueError("LoRA shape mismatch for its logical target")
            if scale != 0:
                fused = mx.addmm(
                    fused, entry.b.astype(mx.float16), entry.a.astype(mx.float16), alpha=scale
                )
        return fused
    base = original.astype(mx.float32)
    fused = base
    for selected in ordered:
        entry = plain_lora_entry(selected.operation)
        if entry is None:
            delta = reconstruct_delta(selected.operation, base, base_weight=base_weight)
            fused = fused + delta * selected.strength
        else:
            scale = validated_lora_scale(entry, selected.strength * selected.operation.scalar)
            if (entry.b.shape[0], entry.a.shape[1]) != original.shape:
                raise ValueError("LoRA shape mismatch for its logical target")
            # Keep FP32 multiply/accumulation and cast only the final weight.
            # The MLX addmm epilogue avoids writing then rereading a full delta.
            if scale != 0:
                fused = mx.addmm(
                    fused, entry.b.astype(mx.float32), entry.a.astype(mx.float32), alpha=scale
                )
    return fused.astype(original.dtype)


FLOAT16_MAX = 65504.0


def half_accumulation_fallbacks(
    operations: Mapping[str, Sequence[WeightedAdapterOperation]],
) -> frozenset[str]:
    """Find eligible half targets needing FP32 factor range, once before fusion.

    Callers select only dense FP16 destinations. One batched evaluation covers
    all wider factors; existing FP16 factors cannot overflow from a no-op cast.
    First-use checks still own nonfinite input/output rejection.
    """
    names = []
    checks = []
    for name, group in operations.items():
        entries = [(plain_lora_entry(item.operation), item) for item in group if _effective(item)]
        if not entries or any(
            entry is None or item.operation.kind != "lora" for entry, item in entries
        ):
            continue
        peaks: list[mx.array] = []
        for entry, _item in entries:
            assert entry is not None
            peaks.extend(
                mx.max(mx.abs(factor))
                for factor in (entry.a, entry.b)
                if factor.dtype != mx.float16
            )
        if peaks:
            names.append(name)
            checks.append(mx.max(mx.stack(peaks).astype(mx.float32)) <= FLOAT16_MAX)
    if not checks:
        return frozenset()
    admissible = cast(list[bool], mx.stack(checks).tolist())
    rejected = frozenset(name for name, fits in zip(names, admissible, strict=True) if not fits)
    if rejected:
        _log.warning(
            "Adapter factors exceed the float16 range; accumulating %d affected targets in float32",
            len(rejected),
        )
    return rejected


def fuse_weight(
    value: mx.array,
    operations: tuple[WeightedAdapterOperation, ...],
    *,
    accumulation: str = "float32",
) -> mx.array:
    """Return one logical ``[out, in]`` weight with its effective adapters applied.

    This is the runtime fuser's arithmetic outside a live model: FP32
    multiply-accumulate and a single cast back to the input dtype, built
    without evaluation. Callers that pack the result afterward use it to fuse
    adapters into source weights before quantization.
    """
    if any(not math.isfinite(item.strength) for item in operations):
        raise ValueError("adapter strength must be finite")
    selected = tuple(item for item in operations if _effective(item))
    if not selected:
        return value
    target = WeightTarget(SimpleNamespace(weight=value), "weight", tuple(value.shape))
    return _fused_weight(target, selected, transposed=False, accumulation=accumulation)


def _companion_weight(
    name: str,
    operations: tuple[WeightedAdapterOperation, ...],
    targets: Mapping[str, WeightTarget],
    transposed: frozenset[str],
) -> mx.array | None:
    if not any(
        item.operation.kind == "glora_peft" and item.operation.part == "bias" for item in operations
    ):
        return None
    weight_name = name.removesuffix("bias") + "weight"
    if weight_name not in targets:
        raise ValueError(f"adapter bias {name!r} needs its unchanged companion weight")
    target = targets[weight_name]
    return logical_weight(target, transposed=weight_name in transposed)


def _effective(item: WeightedAdapterOperation) -> bool:
    if item.strength == 0:
        return False
    operation = item.operation
    # A magnitude or saved base bias can change the target even when the
    # low-rank delta is scaled to zero.
    if "magnitude" in operation.factors or (
        operation.part == "bias" and "base_bias" in operation.factors
    ):
        return True
    if operation.scalar == 0:
        return False
    return not (operation.kind == "adalora" and operation.factors["down"].shape[0] == 0)


def fuse_adapter_operations(
    targets: Mapping[str, WeightTarget],
    operations: Mapping[str, tuple[WeightedAdapterOperation, ...]],
    *,
    pending: AdapterWeightCheck,
    transposed: frozenset[str] = frozenset(),
    accumulation: str = "float32",
) -> None:
    """Validate structure, bind lazy expressions, and register first-use checks.

    The caller owns the first-use destination scan. DoRA's defined normalization
    direction is mandatory independently of that optional scan; exact checks
    evaluate one target at a time to bound dense intermediates.
    Installation failures discard the freshly loaded transformer lease.
    """
    selected = {}
    for name in sorted(operations):
        if name not in targets:
            raise ValueError(f"adapter target {name!r} is absent")
        if any(not math.isfinite(item.strength) for item in operations[name]):
            raise ValueError("adapter strength must be finite")
        effective = tuple(item for item in operations[name] if _effective(item))
        if not effective:
            continue
        selected[name] = effective
        replacements = [item for item in effective if item.operation.kind == "set"]
        if len(replacements) > 1:
            sources = ", ".join(sorted(item.order_key for item in replacements))
            raise ValueError(
                f"adapter target {name!r} has multiple replacement operations: {sources}"
            )
        _fused_weight(
            targets[name],
            effective,
            transposed=name in transposed,
            base_weight=_companion_weight(name, effective, targets, transposed),
            accumulation=accumulation,
        )
        for item in effective:
            if "magnitude" in item.operation.factors:
                delta = reconstruct_delta(
                    item.operation, logical_weight(targets[name], transposed=name in transposed)
                )
                if not mx.all(mx.isfinite(delta)).item():
                    raise ValueError(f"invalid DoRA normalization or expansion for {name}")
    fallbacks = (
        half_accumulation_fallbacks(
            {
                name: group
                for name, group in selected.items()
                if logical_weight(targets[name], transposed=name in transposed).dtype == mx.float16
            }
        )
        if accumulation == "float16"
        else frozenset()
    )
    # Base-dependent bias updates must precede replacement of their companion matrix.
    for name in sorted(selected, key=lambda key: (not key.endswith("bias"), key)):
        target = targets[name]
        candidate = _fused_weight(
            target,
            selected[name],
            transposed=name in transposed,
            base_weight=_companion_weight(name, selected[name], targets, transposed),
            accumulation="float32" if name in fallbacks else accumulation,
        )
        physical = mx.contiguous(candidate.T if name in transposed else candidate)
        setattr(target.owner, target.attribute, physical)
        pending.add(name, target)
