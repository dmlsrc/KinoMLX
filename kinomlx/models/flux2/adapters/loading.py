"""Structure-first Klein adapter placement and fresh-lease shared fusion."""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.adapters.algebra import ia3_input_scale, reconstruct_delta
from kinomlx.adapters.formats import AdapterInventory, AdapterKind, AdapterOperation
from kinomlx.adapters.fusion import (
    WeightedAdapterOperation,
    fuse_adapter_operations,
    logical_weight,
)
from kinomlx.adapters.glora import glora_has_bias_paths
from kinomlx.adapters.live import ScheduledAdapterOperation, install_live_adapters
from kinomlx.adapters.peft import parse_configured_adapter, parse_lazy_adapter
from kinomlx.adapters.receipt import AdapterPlacement, AdapterReceipt, log_adapter_receipt
from kinomlx.adapters.scaling import (
    UNKNOWN_SCALE_WARNING,
    ScaleDetermination,
    determine_scale,
    log_scale_determination,
)
from kinomlx.adapters.schedule import render_strength_plan
from kinomlx.adapters.selection import AdapterSelection
from kinomlx.adapters.source import (
    MODEL_METADATA_KEYS,
    AdapterSource,
    inspect_adapter_source,
)
from kinomlx.adapters.validation import AdapterWeightCheck
from kinomlx.io.safetensors import load_weights
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.modeling.loading import WeightTarget

from ..shapes import transformer_shapes
from ..source_loading import _target
from ..transformer import Flux2Transformer
from .categories import ADAPTER_EXCLUDE_CATEGORIES, target_categories
from .keys import Flux2AdapterKeys, place_operation

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAdapter:
    selection: AdapterSelection
    source_identity: str
    placements: tuple[AdapterPlacement, ...]
    adapter_types: tuple[str, ...]


def adapter_shapes() -> dict[str, tuple[int, ...]]:
    shapes = transformer_shapes()
    return {
        **shapes,
        **{
            name.removesuffix("weight") + "bias": (shape[0],)
            for name, shape in shapes.items()
            if len(shape) == 2
        },
    }


def _source(selection: AdapterSelection) -> AdapterSource:
    source = inspect_adapter_source(selection, model="Klein", categories=ADAPTER_EXCLUDE_CATEGORIES)
    for key in MODEL_METADATA_KEYS:
        declared = source.metadata.get(key, "").lower().replace("_", "-")
        if any(
            token in declared
            for token in ("klein-4b", "klein-4-b", "flux.1", "sdxl", "krea", "ltx")
        ):
            raise ValueError(f"adapter declares an incompatible Klein 9B topology: {declared}")
    return source


def _place(
    inventory: AdapterInventory,
    source: AdapterSource,
    selection: AdapterSelection,
    base_for: Callable[[str], mx.array],
) -> tuple[tuple[AdapterPlacement, ...], tuple[tuple[str, AdapterOperation], ...]]:
    keys, shapes = Flux2AdapterKeys(), adapter_shapes()
    placements, applied = [], []
    occupied: set[str] = set()
    for operation in inventory.operations:
        binding = keys.resolve(operation.prefix)
        if binding is None:
            if any(
                marker in operation.prefix
                for marker in ("double_blocks", "single_blocks", "transformer_blocks")
            ):
                raise ValueError(
                    f"adapter target contradicts the Klein 9B graph: {operation.prefix}"
                )
            placements.append(
                AdapterPlacement(
                    operation.source_keys,
                    None,
                    operation.kind,
                    "unmapped",
                    operation.dependency_keys,
                )
            )
            continue
        operations = [operation]
        needs_bias = operation.part == "weight" and (
            operation.kind in (AdapterKind.GLORA_PEFT, AdapterKind.ROAD)
            or (operation.kind == AdapterKind.IA3_PEFT and not ia3_input_scale(operation))
        )
        if needs_bias and (
            "base_bias" in operation.factors
            or (operation.kind == AdapterKind.GLORA_PEFT and glora_has_bias_paths(operation))
        ):
            operations.append(replace(operation, part="bias", source_keys=()))
        first = True
        for original in operations:
            for name in binding.targets:
                if f"{name}.{original.part}" not in shapes:
                    raise ValueError(
                        f"Klein target has no applicable {original.part} operation: {name}"
                    )
            for target, placed in place_operation(original, binding, base_for):
                if target not in shapes:
                    raise ValueError(
                        f"Klein target has no applicable {original.part} operation: {target}"
                    )
                if target in occupied:
                    raise ValueError(f"adapter maps multiple operations onto {target}")
                occupied.add(target)
                delta = reconstruct_delta(
                    placed,
                    base_for(target),
                    base_weight=(
                        base_for(target.removesuffix("bias") + "weight")
                        if placed.kind == AdapterKind.GLORA_PEFT and placed.part == "bias"
                        else None
                    ),
                )
                if delta.shape != shapes[target]:
                    raise ValueError(
                        f"adapter {original.prefix} has shape {delta.shape}, expected {shapes[target]}"
                    )
                status = (
                    "excluded" if set(selection.exclude) & target_categories(target) else "applied"
                )
                placements.append(
                    AdapterPlacement(
                        operation.source_keys if first else (),
                        target,
                        operation.kind,
                        status,
                        operation.dependency_keys
                        if first
                        else operation.source_keys + operation.dependency_keys,
                    )
                )
                first = False
                if status == "applied":
                    applied.append((target, placed))
    for names, kind, status in (
        (inventory.shared_keys, "shared_basis", "shared"),
        (inventory.training_keys, "training_only", "unused_inference"),
    ):
        if names:
            placements.append(AdapterPlacement(names, None, kind, status))
    for name in inventory.unknown_keys:
        if any(
            keys.resolve(name[:position]) is not None
            for position, char in enumerate(name)
            if char == "."
        ):
            raise ValueError(f"unsupported adapter semantics on known Klein target: {name}")
        placements.append(AdapterPlacement((name,), None, "unknown", "unmapped"))
    consumed = [*placement.source_keys for placement in placements]
    if len(consumed) != len(set(consumed)) or set(consumed) != set(source.tensor_names):
        raise RuntimeError(
            "Klein adapter receipt must account for every source tensor exactly once"
        )
    return tuple(placements), tuple(applied)


def prepare_flux2_adapters(selections: tuple[AdapterSelection, ...]) -> tuple[PreparedAdapter, ...]:
    """Reject malformed structure before text encoding, retaining no factor arrays."""
    shapes = adapter_shapes()
    prepared = []
    for selection in selections:
        source = _source(selection)
        inventory = parse_configured_adapter(
            load_weights(source.path), source.config, validate_values=selection.strength == 0
        )
        placements, _ = _place(
            inventory, source, selection, lambda name: mx.zeros(shapes[name], dtype=mx.float32)
        )
        prepared.append(
            PreparedAdapter(selection, source.source_identity, placements, inventory.adapter_types)
        )
    return tuple(prepared)


def apply_flux2_adapters(
    model: Flux2Transformer,
    prepared: tuple[PreparedAdapter, ...],
    *,
    variant: str,
    check_weights: bool = True,
) -> tuple[AdapterReceipt, ...]:
    """Install lazy adapter expressions and track effective targets on a fresh model."""
    if not prepared:
        return ()
    pending = AdapterWeightCheck(enabled=check_weights)
    object.__setattr__(model, "adapter_weight_check", pending)
    shapes = adapter_shapes()
    targets = {name: _target(model, name, shape) for name, shape in transformer_shapes().items()}
    transposed = frozenset(
        name for name, target in targets.items() if isinstance(target.owner, PretransposedLinear)
    )
    synthetic_biases: dict[str, nn.Linear] = {}
    for name, shape in shapes.items():
        if name.endswith(".bias"):
            weight = targets[name.removesuffix("bias") + "weight"]
            owner = SimpleNamespace(
                bias=mx.zeros(shape, dtype=getattr(weight.owner, weight.attribute).dtype)
            )
            targets[name] = WeightTarget(owner, "bias", shape)
            synthetic_biases[name] = cast(nn.Linear, weight.owner)

    def live_base(name: str) -> mx.array:
        return logical_weight(targets[name], transposed=name in transposed)

    planned: dict[str, list[WeightedAdapterOperation]] = defaultdict(list)
    scheduled: dict[str, list[ScheduledAdapterOperation]] = defaultdict(list)
    receipts = []
    for index, item in enumerate(prepared, 1):
        selection = item.selection
        source = _source(selection)
        if source.source_identity != item.source_identity:
            raise ValueError("Klein adapter changed after structural preflight")
        placements = item.placements
        # A step array naming one strength for every step is a constant, and
        # fusion applies a constant once rather than paying two operations
        # per adapted Linear per forward for the whole run.
        live_plan = None if selection.fuses else selection.strength_plan
        scale = ScaleDetermination()
        if selection.effective:
            inventory = parse_lazy_adapter(load_weights(source.path), source.config)
            scale = determine_scale(source.config_source, source.config, inventory.operations)
            placements, ops = _place(inventory, source, selection, live_base)
            for target, operation in ops:
                if live_plan is None:
                    planned[target].append(
                        WeightedAdapterOperation(
                            operation, selection.effective_strength, str(source.path)
                        )
                    )
                else:
                    scheduled[target].append(
                        ScheduledAdapterOperation(
                            operation,
                            live_plan,
                            str(source.path),
                            selection.strength,
                        )
                    )
        declared = next(
            (source.metadata[key] for key in MODEL_METADATA_KEYS if source.metadata.get(key)), None
        )
        lower = (declared or "").lower()
        declared_variant = (
            "kv"
            if "kv" in lower
            else "base"
            if "base" in lower
            else "distilled"
            if "flux.2-klein-9b" in lower or "distilled" in lower
            else None
        )
        unmapped = tuple(
            f"Unmapped Klein adapter input: {p.source_keys}"
            for p in placements
            if p.status == "unmapped"
        )
        scale_unknown = source.scale_unknown and selection.effective
        warnings = unmapped + ((UNKNOWN_SCALE_WARNING,) if scale_unknown else ())
        receipt = AdapterReceipt(
            source.path,
            source.source_identity,
            selection.strength,
            selection.exclude,
            variant,
            declared,
            declared_variant,
            placements,
            item.adapter_types,
            variant_pairing="undeclared"
            if declared_variant is None
            else "same-variant"
            if declared_variant == variant
            else "cross-variant",
            effective=selection.effective and any(p.status == "applied" for p in placements),
            warnings=warnings,
            step_strengths=() if selection.schedule is None else selection.schedule.per_step,
            sigma_window=(
                None if selection.sigma_window is None else selection.sigma_window.to_metadata()
            ),
            application="fused" if live_plan is None else "live",
            scale_source=scale.source,
            intrinsic_scales=scale.scales,
        )
        receipts.append(receipt)
        _plan_detail = render_strength_plan(selection.schedule, selection.sigma_window)
        log_adapter_receipt(
            _log,
            model=f"Klein {variant}",
            path=selection.path,
            position=(index, len(prepared)),
            adapter_types=receipt.adapter_types,
            strength=selection.effective_strength,
            placed_targets=receipt.placed_targets,
            complete_targets=receipt.complete_targets,
            structural_coverage=receipt.coverage,
            exclusions=selection.exclude,
            details=(
                ("pairing", receipt.variant_pairing),
                ("effective", str(receipt.effective).lower()),
                *(() if _plan_detail is None else (_plan_detail,)),
            ),
        )
        log_scale_determination(
            _log,
            model="Klein",
            name=selection.path.name,
            determination=scale,
            strength=selection.effective_strength,
        )
        for warning in unmapped:
            _log.warning("%s", warning)
        if scale_unknown:
            _log.warning("Klein adapter %s: %s", selection.path.name, UNKNOWN_SCALE_WARNING)
        if not receipt.effective:
            _log.warning("Klein adapter has no effective merge: %s", source.path.name)
        elif receipt.coverage < 0.5:
            _log.warning("Klein adapter has low structural coverage: %s", source.path.name)
    if scheduled:
        object.__setattr__(
            model,
            "live_adapters",
            install_live_adapters(
                targets,
                {name: tuple(items) for name, items in scheduled.items()},
                model=f"Klein {variant}",
            ),
        )
    if planned:
        fuse_adapter_operations(
            targets,
            {name: tuple(ops) for name, ops in planned.items()},
            transposed=transposed,
            pending=pending,
        )
        for name, live_owner in synthetic_biases.items():
            if name in pending.targets:
                live_owner.bias = cast(SimpleNamespace, targets[name].owner).bias
                pending.add(name, WeightTarget(live_owner, "bias", shapes[name]))
    return tuple(receipts)
