"""Plan and apply local static adapters to a fresh Krea 2 transformer."""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx

from kinomlx.adapters.algebra import ia3_input_scale, reconstruct_delta
from kinomlx.adapters.formats import (
    AdapterInventory,
    AdapterOperation,
)
from kinomlx.adapters.fusion import (
    WeightedAdapterOperation,
    fuse_adapter_operations,
    logical_weight,
)
from kinomlx.adapters.glora import glora_has_bias_paths
from kinomlx.adapters.live import ScheduledAdapterOperation, install_live_adapters
from kinomlx.adapters.peft import parse_configured_adapter, parse_lazy_adapter
from kinomlx.adapters.receipt import (
    LOW_COVERAGE_WARNING_THRESHOLD,
    AdapterPlacement,
    AdapterReceipt,
    log_adapter_receipt,
)
from kinomlx.adapters.schedule import render_strength_plan
from kinomlx.adapters.selection import AdapterSelection
from kinomlx.adapters.source import AdapterSource, inspect_adapter_source
from kinomlx.adapters.validation import AdapterWeightCheck
from kinomlx.io.safetensors import load_weights

from ..metadata import Krea2TransformerConfig
from ..transformer.loading import transformer_weight_targets
from ..transformer.model import Krea2Transformer
from ..weight_keys import transformer_weight_transform
from .categories import ADAPTER_EXCLUDE_CATEGORIES, target_categories
from .keys import Krea2AdapterKeys, canonicalize_adapter_operation

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Krea2PreparedAdapter:
    """One validated source identity without retained adapter tensors."""

    selection: AdapterSelection
    source_identity: str
    placements: tuple[AdapterPlacement, ...] = ()
    zero_alpha_count: int = 0
    adapter_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Krea2AdapterPlan:
    """Structure-only adapter plan safe to retain across text conditioning."""

    transformer_config_digest: str
    adapters: tuple[Krea2PreparedAdapter, ...] = ()


@dataclass(frozen=True)
class _PlacedInventory:
    placements: tuple[AdapterPlacement, ...]
    operations: tuple[tuple[str, AdapterOperation], ...]


def _declared_variant(metadata: Mapping[str, str]) -> str | None:
    for key in (
        "base_model",
        "base_model_name_or_path",
        "ss_sd_model_name",
        "ss_base_model_version",
        "model_version",
    ):
        value = metadata.get(key, "").lower()
        if "krea" in value:
            if "turbo" in value:
                return "turbo"
            if "raw" in value:
                return "raw"
    return None


def _declared_model(metadata: Mapping[str, str]) -> str | None:
    for key in (
        "base_model_name_or_path",
        "base_model",
        "ss_sd_model_name",
        "ss_base_model_version",
        "model_version",
    ):
        value = metadata.get(key)
        if value:
            return value
    return None


def _inspect_selection(selected: AdapterSelection) -> AdapterSource:
    return inspect_adapter_source(selected, model="Krea 2", categories=ADAPTER_EXCLUDE_CATEGORIES)


def _needs_bias_operation(operation: AdapterOperation) -> bool:
    return operation.kind in ("glora_peft", "road") or (
        operation.kind == "ia3_peft" and not ia3_input_scale(operation)
    )


def _place_inventory(
    inventory: AdapterInventory,
    *,
    keys: Krea2AdapterKeys,
    target_shapes: Mapping[str, tuple[int, ...]],
    selected: AdapterSelection,
    path: Path,
    base_for: Callable[[str], mx.array],
) -> _PlacedInventory:
    placements = []
    applied = []
    occupied: set[str] = set()
    exclusions = set(selected.exclude)
    for operation in inventory.operations:
        binding = keys.resolve(operation.prefix, operation.part)
        target = None
        status = "unmapped"
        if binding is not None:
            target = binding.target_key
            operation = canonicalize_adapter_operation(operation, binding)
            if target not in target_shapes:
                raise RuntimeError(f"Krea adapter key map produced absent target {target!r}")
            if target in occupied:
                raise ValueError(f"adapter {path.name!r} maps multiple operations onto {target!r}")
            occupied.add(target)
            base = base_for(target)
            delta = reconstruct_delta(operation, base)
            if delta.shape != target_shapes[target]:
                raise ValueError(
                    f"adapter {operation.prefix!r} reconstructed {delta.shape}, "
                    f"expected {target_shapes[target]}"
                )
            status = "excluded" if exclusions & target_categories(target) else "applied"
            if status == "applied":
                applied.append((target, operation))

            if _needs_bias_operation(operation):
                bias_target = target.removesuffix(".weight") + ".bias"
                if bias_target not in target_shapes and (
                    "base_bias" in operation.factors
                    or (operation.kind == "glora_peft" and glora_has_bias_paths(operation))
                ):
                    raise ValueError(
                        f"PEFT {operation.kind} bias paths require a biased Linear target"
                    )
                if bias_target in target_shapes:
                    if bias_target in occupied:
                        raise ValueError(
                            f"adapter has conflicting bias operations at {bias_target!r}"
                        )
                    occupied.add(bias_target)
                    bias_operation = replace(operation, part="bias", source_keys=())
                    bias_delta = reconstruct_delta(
                        bias_operation,
                        base_for(bias_target),
                        base_weight=base,
                    )
                    if bias_delta.shape != target_shapes[bias_target]:
                        raise ValueError(
                            f"adapter {operation.prefix!r} reconstructed bias {bias_delta.shape}, "
                            f"expected {target_shapes[bias_target]}"
                        )
                    bias_status = (
                        "excluded" if exclusions & target_categories(bias_target) else "applied"
                    )
                    if bias_status == "applied":
                        applied.append((bias_target, bias_operation))
                    placements.append(
                        AdapterPlacement(
                            (),
                            bias_target,
                            operation.kind,
                            bias_status,
                            operation.source_keys,
                        )
                    )
        placements.append(
            AdapterPlacement(
                operation.source_keys,
                target,
                operation.kind,
                status,
                operation.dependency_keys,
            )
        )

    if inventory.shared_keys:
        placements.append(AdapterPlacement(inventory.shared_keys, None, "shared_basis", "shared"))
    if inventory.training_keys:
        placements.append(
            AdapterPlacement(inventory.training_keys, None, "training_only", "unused_inference")
        )
    for unknown in inventory.unknown_keys:
        if keys.belongs_to_known_module(unknown):
            raise ValueError(
                f"adapter tensor {unknown!r} has unsupported semantics on a known Krea module"
            )
        placements.append(AdapterPlacement((unknown,), None, "unknown", "unmapped"))
    return _PlacedInventory(tuple(placements), tuple(applied))


def _verify_accounting(placed: _PlacedInventory, source: AdapterSource) -> None:
    consumed = [key for item in placed.placements for key in item.source_keys]
    if len(consumed) != len(set(consumed)) or set(consumed) != set(source.tensor_names):
        raise RuntimeError(
            "adapter placement receipt does not account for every source tensor exactly once"
        )


def prepare_krea2_adapters(
    config: Krea2TransformerConfig,
    selections: tuple[AdapterSelection, ...],
) -> Krea2AdapterPlan:
    """Validate adapter structure and target shapes without retaining arrays."""
    if not selections:
        return Krea2AdapterPlan(config.config_digest)
    shell = Krea2Transformer(config)
    target_shapes = {
        name: target.shape for name, target in transformer_weight_targets(shell).items()
    }
    keys = Krea2AdapterKeys(config)
    prepared = []
    for selected in selections:
        source = _inspect_selection(selected)
        weights = load_weights(source.path)
        inventory = parse_configured_adapter(
            weights,
            source.config,
            validate_values=selected.strength == 0.0,
        )
        placed = _place_inventory(
            inventory,
            keys=keys,
            target_shapes=target_shapes,
            selected=selected,
            path=source.path,
            base_for=lambda name: mx.zeros(target_shapes[name], dtype=mx.float32),
        )
        _verify_accounting(placed, source)
        zero_alpha_count = sum(
            operation.alpha == 0.0 and operation.kind in ("lora", "loha", "lokr")
            for operation in inventory.operations
        )
        prepared.append(
            Krea2PreparedAdapter(
                selected,
                source.source_identity,
                placed.placements,
                zero_alpha_count,
                inventory.adapter_types,
            )
        )
    return Krea2AdapterPlan(config.config_digest, tuple(prepared))


def apply_krea2_adapters(
    model: Krea2Transformer,
    plan: Krea2AdapterPlan,
    *,
    variant: str,
    check_weights: bool = True,
) -> tuple[AdapterReceipt, ...]:
    """Validate values and fuse a previously structure-checked adapter plan."""
    if plan.transformer_config_digest != model.config.config_digest:
        raise ValueError("Krea adapter plan does not match the transformer configuration")
    if not plan.adapters:
        return ()
    pending = AdapterWeightCheck(enabled=check_weights)
    object.__setattr__(model, "adapter_weight_check", pending)
    keys = Krea2AdapterKeys(model.config)
    targets = transformer_weight_targets(model)
    target_shapes = {name: target.shape for name, target in targets.items()}
    transposed = frozenset(
        name
        for name in targets
        if not model.reference_denoise and transformer_weight_transform(name) is not None
    )

    def live_base(name: str) -> mx.array:
        return logical_weight(targets[name], transposed=name in transposed)

    planned: dict[str, list[WeightedAdapterOperation]] = defaultdict(list)
    scheduled: dict[str, list[ScheduledAdapterOperation]] = defaultdict(list)
    receipts = []
    for index, prepared in enumerate(plan.adapters, 1):
        selected = prepared.selection
        source = _inspect_selection(selected)
        if source.source_identity != prepared.source_identity:
            raise ValueError(
                f"Krea adapter {source.path.name!r} changed after structural preflight"
            )
        if not selected.effective:
            placed = _PlacedInventory(prepared.placements, ())
            zero_alpha = prepared.zero_alpha_count
        else:
            inventory = parse_lazy_adapter(load_weights(source.path), source.config)
            placed = _place_inventory(
                inventory,
                keys=keys,
                target_shapes=target_shapes,
                selected=selected,
                path=source.path,
                base_for=live_base,
            )
            zero_alpha = sum(
                operation.alpha == 0.0 and operation.kind in ("lora", "loha", "lokr")
                for operation in inventory.operations
            )
        _verify_accounting(placed, source)
        # A step array naming one strength for every step is a constant, and
        # fusion applies a constant once rather than paying two operations per
        # adapted Linear per forward for the whole run.
        live_plan = None if selected.fuses else selected.strength_plan
        for target, operation in placed.operations:
            if live_plan is None:
                planned[target].append(
                    WeightedAdapterOperation(
                        operation, selected.effective_strength, str(source.path)
                    )
                )
            else:
                scheduled[target].append(
                    ScheduledAdapterOperation(
                        operation,
                        live_plan,
                        str(source.path),
                        selected.strength,
                    )
                )
        declared_variant = _declared_variant(source.metadata)
        pairing = (
            "undeclared"
            if declared_variant is None
            else "recommended-raw-to-turbo"
            if (declared_variant, variant) == ("raw", "turbo")
            else "same-variant"
            if declared_variant == variant
            else "cross-variant"
        )
        parsed_unmapped = sum(
            item.status == "unmapped"
            and item.kind not in ("unknown", "shared_basis", "training_only")
            for item in placed.placements
        )
        unknown_tensors = sum(item.kind == "unknown" for item in placed.placements)
        warnings = []
        if parsed_unmapped:
            warnings.append(
                f"{parsed_unmapped} parsed adapter operations did not map to Krea 2 targets"
            )
        if unknown_tensors:
            warnings.append(f"{unknown_tensors} unknown adapter tensors were not placed")
        receipt = AdapterReceipt(
            path=source.path,
            source_identity=source.source_identity,
            strength=selected.strength,
            application="fused" if live_plan is None else "live",
            exclusions=selected.exclude,
            base_variant=variant,
            declared_model=_declared_model(source.metadata),
            declared_variant=declared_variant,
            placements=placed.placements,
            adapter_types=prepared.adapter_types,
            variant_pairing=pairing,
            effective=selected.effective and bool(placed.operations),
            warnings=tuple(warnings),
            step_strengths=() if selected.schedule is None else selected.schedule.per_step,
            sigma_window=(
                None if selected.sigma_window is None else selected.sigma_window.to_metadata()
            ),
        )
        receipts.append(receipt)
        _plan_detail = render_strength_plan(selected.schedule, selected.sigma_window)
        log_adapter_receipt(
            _log,
            model=f"Krea 2 {variant.title()}",
            path=selected.path,
            position=(index, len(plan.adapters)),
            adapter_types=receipt.adapter_types,
            strength=selected.effective_strength,
            placed_targets=receipt.placed_targets,
            complete_targets=receipt.complete_targets,
            structural_coverage=receipt.coverage,
            exclusions=selected.exclude,
            details=(
                ("pairing", pairing),
                ("effective", str(receipt.effective).lower()),
                *(() if _plan_detail is None else (_plan_detail,)),
            ),
        )
        if zero_alpha:
            _log.info(
                "Krea 2 adapter %s: %d saved zero-alpha operations use unit intrinsic scale",
                source.path.name,
                zero_alpha,
            )
        if receipt.coverage < LOW_COVERAGE_WARNING_THRESHOLD or not receipt.effective:
            _log.warning(
                "Krea 2 adapter %s has %s; coverage is diagnostic, not a rejection",
                source.path.name,
                (
                    "no effective merge (0% effective placement)"
                    if not receipt.effective
                    else "low target coverage"
                ),
            )
        if parsed_unmapped:
            _log.warning(
                "Krea 2 adapter %s: %d parsed operations did not map to transformer targets",
                source.path.name,
                parsed_unmapped,
            )
        if unknown_tensors:
            _log.warning(
                "Krea 2 adapter %s: %d unknown tensors are recorded in the run metadata",
                source.path.name,
                unknown_tensors,
            )
        if pairing == "recommended-raw-to-turbo":
            _log.info("Krea 2 adapter: Raw-trained on Turbo, the officially recommended pairing")
        elif receipt.declared_variant is not None and receipt.declared_variant != variant:
            _log.info(
                "Krea 2 adapter declares %s; applying to %s (cross-variant use is allowed)",
                receipt.declared_variant,
                variant,
            )

    if scheduled:
        object.__setattr__(
            model,
            "live_adapters",
            install_live_adapters(
                targets,
                {name: tuple(items) for name, items in scheduled.items()},
                model="Krea 2",
            ),
        )
    if planned:
        fuse_adapter_operations(
            targets,
            {name: tuple(items) for name, items in planned.items()},
            transposed=transposed,
            pending=pending,
        )
        _log.info(
            "Krea 2 adapter weights are installed lazily; fusion arithmetic is deferred "
            "to the first denoising step"
        )
    return tuple(receipts)
