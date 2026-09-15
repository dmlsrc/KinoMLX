"""Apply community LoRAs to normal or pretransposed weights."""

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.adapters.formats import AdapterOperation
from kinomlx.adapters.fusion import (
    WeightedAdapterOperation,
    fuse_adapter_operations,
)
from kinomlx.adapters.lora import LoRAConfig, iter_lora_entries
from kinomlx.adapters.peft import companion_config, parse_configured_adapter
from kinomlx.adapters.receipt import LOW_COVERAGE_WARNING_THRESHOLD, log_adapter_receipt
from kinomlx.adapters.validation import AdapterWeightCheck
from kinomlx.io.identity import file_signature, payload_digest
from kinomlx.io.safetensors import load_weights, read_metadata
from kinomlx.modeling.loading import WeightTarget
from kinomlx.models.ltx2.weights.layout import get_layout_weight, install_layout_weight
from kinomlx.models.ltx2.weights.quantization import is_quantized_linear
from kinomlx.models.ltx2.weights.schema import LAYOUT_KEY_PREFIX, QUANT_KEY_PREFIX
from kinomlx.reporting import Reporter

from .keys import (
    convert_lora_base_to_mlx,
    lora_key_categories,
    validate_adapter_exclusions,
)
from .receipt import LTX2AdapterReceipt

_log = logging.getLogger(__name__)


class _LiveWeightOwner(Protocol):
    weight: mx.array


class _LiveIndexable(Protocol):
    def __getitem__(self, index: int, /) -> object: ...


@dataclass(frozen=True)
class _LoRAMappingStats:
    resolved: int
    present: int
    placed: int
    excluded: int
    unmapped: int
    missing: int
    shape_mismatched: int
    target_categories: tuple[tuple[str, int], ...]
    skipped_reasons: tuple[tuple[str, int], ...]

    @property
    def considered(self) -> int:
        return self.placed + self.unmapped + self.missing + self.shape_mismatched


@dataclass(frozen=True)
class _LTXAdapterClassification:
    """LTX-supported operations plus harmless tensors the legacy mapper skips."""

    adapter_types: tuple[str, ...]
    extra_unmapped: int = 0


def _adapter_fingerprint(path: Path) -> str:
    # LoRAConfig canonicalizes paths with Path.resolve(), so a selected HF
    # snapshot symlink normally arrives here as its content-addressed blob.
    # Preserve that cheap exact identity whether the caller passed the link or
    # the already-resolved blob path.
    resolved = path.resolve()
    if resolved.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", resolved.name):
        return f"sha256:{resolved.name}"
    return f"source:{payload_digest({'source': file_signature(path)})}"


def _declared_generation(path: Path) -> str | None:
    version = read_metadata(path).get("model_version")
    if version is None:
        return None
    match = re.search(r"(?<!\d)(2\.[35])(?:\D|$)", version)
    return None if match is None else match.group(1)


def _classify_adapter(path: Path, weights: dict[str, mx.array]) -> _LTXAdapterClassification:
    """Classify one source and admit only ordinary two-dimensional LoRA weights."""
    inventory = parse_configured_adapter(
        weights,
        companion_config(path),
        validate_values=False,
    )
    signatures = tuple(sorted({f"{item.kind}:{item.part}" for item in inventory.operations}))
    unsupported = []
    ignored_operations = 0
    for operation in inventory.operations:
        factors = operation.factors
        plain_weight = (
            operation.kind == "lora"
            and operation.part == "weight"
            and "magnitude" not in factors
            and "mid" not in factors
            and factors["down"].ndim == 2
            and factors["up"].ndim == 2
        )
        if plain_weight:
            continue
        if operation.kind == "lora" and operation.part == "bias":
            # The historical LTX mapper applies the paired weight and ignores
            # the optional LoRA B bias. Keep it visible in coverage instead of
            # rejecting an otherwise usable ordinary-LoRA file.
            ignored_operations += 1
            continue
        unsupported.append(operation)
    if unsupported:
        parsed = ", ".join(signatures) or "none"
        first = unsupported[0]
        raise ValueError(
            "LTX-2 currently supports ordinary LoRA weight adapters only; "
            f"parsed adapter operations: {parsed}; first unsupported operation: "
            f"{first.prefix!r} ({first.kind}:{first.part})"
        )
    return _LTXAdapterClassification(
        inventory.adapter_types,
        ignored_operations + len(inventory.unknown_keys) + len(inventory.shared_keys),
    )


def _shape_matches_target(
    target_key: str,
    target: mx.array,
    entry_a: mx.array,
    entry_b: mx.array,
) -> bool:
    if (
        entry_a.ndim != 2
        or entry_b.ndim != 2
        or entry_a.shape[0] <= 0
        or entry_b.shape[1] != entry_a.shape[0]
    ):
        return False
    delta = (int(entry_b.shape[0]), int(entry_a.shape[1]))
    target_shape = tuple(int(value) for value in target.shape)
    return target_shape == (tuple(reversed(delta)) if target_key.endswith(".weight_t") else delta)


def _normalize_lora_for_weights(
    base_weights: dict[str, mx.array],
    lora_weights: dict[str, mx.array],
    *,
    include_audio: bool,
    excluded_categories: frozenset[str],
    extra_unmapped: int = 0,
) -> tuple[dict[str, mx.array], _LoRAMappingStats]:
    normalized: dict[str, mx.array] = {}
    resolved = 0
    present = 0
    placed = 0
    excluded = 0
    unmapped = extra_unmapped
    missing = 0
    shape_mismatched = 0
    category_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {"unmapped": extra_unmapped} if extra_unmapped else {}
    if extra_unmapped:
        category_counts["unmapped"] = extra_unmapped

    entries = list(iter_lora_entries(lora_weights))
    if not entries:
        raise RuntimeError("LoRA contains no complete supported A/B weight pairs")

    for index, (raw_base, entry) in enumerate(entries):
        mlx_base = convert_lora_base_to_mlx(
            raw_base,
            include_audio=include_audio,
        )
        if mlx_base is None:
            unmapped += 1
            reason_counts["unmapped"] = reason_counts.get("unmapped", 0) + 1
            category_counts["unmapped"] = category_counts.get("unmapped", 0) + 1
            target = f"__unmapped__.{index}.{raw_base}"
            normalized[f"{target}.lora_A.weight"] = entry.a
            normalized[f"{target}.lora_B.weight"] = entry.b
            continue
        categories = lora_key_categories(mlx_base)
        if excluded_categories & categories:
            excluded += 1
            reason_counts["excluded"] = reason_counts.get("excluded", 0) + 1
            continue
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        resolved += 1
        normal_key = f"{mlx_base}.weight"
        layout_key = f"{LAYOUT_KEY_PREFIX}{mlx_base}.weight_t"
        quant_key = f"{QUANT_KEY_PREFIX}{mlx_base}.weight"
        if quant_key in base_weights:
            raise RuntimeError(
                f"LoRA target {mlx_base!r} is already quantized; "
                "apply community LoRAs before quantizing the weights"
            )
        if normal_key in base_weights:
            target = mlx_base
            present += 1
            matched_key = normal_key
        elif layout_key in base_weights:
            target = layout_key
            present += 1
            matched_key = layout_key
        else:
            # Keep resolved-but-absent targets in the generic fuser's input.
            # They must count against coverage so wrong-model and partially
            # compatible adapters cannot silently appear fully placed.
            target = mlx_base
            matched_key = None

        if matched_key is None:
            missing += 1
            reason_counts["missing_target"] = reason_counts.get("missing_target", 0) + 1
        elif _shape_matches_target(matched_key, base_weights[matched_key], entry.a, entry.b):
            placed += 1
        else:
            shape_mismatched += 1
            reason_counts["shape_mismatch"] = reason_counts.get("shape_mismatch", 0) + 1

        # Community checkpoints use direct-strength products for this model.
        # Re-emit a canonical pair and intentionally omit alpha metadata.
        normalized[f"{target}.lora_A.weight"] = entry.a
        normalized[f"{target}.lora_B.weight"] = entry.b

    return normalized, _LoRAMappingStats(
        resolved=resolved,
        present=present,
        placed=placed,
        excluded=excluded,
        unmapped=unmapped,
        missing=missing,
        shape_mismatched=shape_mismatched,
        target_categories=tuple(sorted(category_counts.items())),
        skipped_reasons=tuple(sorted(reason_counts.items())),
    )


def _adapter_receipt(
    config: LoRAConfig,
    stats: _LoRAMappingStats,
    *,
    adapter_types: tuple[str, ...],
    model_generation: str | None,
) -> LTX2AdapterReceipt:
    declared = _declared_generation(config.path)
    mismatch = (
        None if declared is None or model_generation is None else declared != model_generation
    )
    complete_targets = stats.considered + stats.excluded
    coverage = stats.placed / complete_targets if complete_targets else 0.0
    effective = config.strength != 0.0 and stats.placed > 0
    warnings = []
    if stats.unmapped:
        warnings.append(
            f"{stats.unmapped} adapter operations or tensors did not map to LTX-2 targets"
        )
    if stats.missing:
        warnings.append(f"{stats.missing} adapter targets are absent from this LTX-2 model")
    if stats.shape_mismatched:
        warnings.append(f"{stats.shape_mismatched} adapter targets have incompatible shapes")
    if not effective:
        warnings.append("no effective merge (0% effective placement)")
    elif coverage < LOW_COVERAGE_WARNING_THRESHOLD:
        warnings.append(f"low target coverage ({coverage:.1%})")
    return LTX2AdapterReceipt(
        path=config.path,
        fingerprint=_adapter_fingerprint(config.path),
        base_model_generation=model_generation,
        declared_model_generation=declared,
        generation_mismatch=mismatch,
        strength=float(config.strength),
        knockouts=tuple(sorted(config.exclude)),
        complete_targets=complete_targets,
        placed_targets=stats.placed,
        structural_coverage=coverage,
        target_categories=stats.target_categories,
        skipped_reasons=stats.skipped_reasons,
        effective=effective,
        warnings=tuple(warnings),
        adapter_types=adapter_types,
    )


def _log_receipt(
    receipt: LTX2AdapterReceipt,
    stats: _LoRAMappingStats,
    *,
    model_generation: str | None,
    position: tuple[int, int],
) -> None:
    log_adapter_receipt(
        _log,
        model=f"LTX-{model_generation or 'unknown'}",
        path=receipt.path,
        position=position,
        adapter_types=receipt.adapter_types,
        strength=receipt.strength,
        placed_targets=receipt.placed_targets,
        complete_targets=receipt.complete_targets,
        structural_coverage=receipt.structural_coverage,
        exclusions=receipt.knockouts,
        details=(
            ("resolved", stats.resolved),
            ("weight_matches", stats.present),
            ("excluded_targets", stats.excluded),
            ("unmapped", stats.unmapped),
            ("missing", stats.missing),
            ("shape_mismatched", stats.shape_mismatched),
            ("effective", str(receipt.effective).lower()),
        ),
    )
    for warning in receipt.warnings:
        _log.warning("LTX-2 adapter %s: %s", receipt.path.name, warning)


def _resolve_live_module(model: object, path: str) -> object | None:
    current = model
    for part in path.split("."):
        if part.isdecimal():
            try:
                current = cast(_LiveIndexable, current)[int(part)]
            except IndexError, KeyError, TypeError:
                return None
        else:
            current = getattr(current, part, None)
            if current is None:
                return None
    return current


def _live_weight_owner(value: object | None) -> _LiveWeightOwner | None:
    """Narrow a dynamically traversed leaf to a mutable MLX weight owner."""
    if value is None or not isinstance(getattr(value, "weight", None), mx.array):
        return None
    return cast(_LiveWeightOwner, value)


def _add_live_lora_target(
    table: dict[str, mx.array],
    model: object,
    mlx_base: str,
) -> None:
    """Add one placeable normal, layout, or packed target to a live table.

    This defensive compatibility branch treats a recognized community LoRA
    key whose live model exposes no compatible weight-owning leaf as absent.
    Normalization records it as ``missing_target`` while every other placeable
    target still fuses.
    """
    normal_key = f"{mlx_base}.weight"
    module = _resolve_live_module(model, mlx_base)
    owner = _live_weight_owner(module)
    if module is not None and is_quantized_linear(module):
        # Compatible packed targets were fused at source preparation. A remaining
        # packed target was not placeable; record it as missing instead of treating
        # packed storage as a dense weight or asking the user to fuse again.
        return
    if owner is not None:
        table[normal_key] = owner.weight
        return
    layout_key = f"{mlx_base}.weight_t"
    layout = get_layout_weight(model, layout_key)
    if layout is not None:
        table[f"{LAYOUT_KEY_PREFIX}{layout_key}"] = layout


def _live_lora_weight_table(
    model: object,
    adapters: tuple[
        tuple[LoRAConfig, frozenset[str], dict[str, mx.array], _LTXAdapterClassification],
        ...,
    ],
    *,
    include_audio: bool,
    prepared_targets: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, mx.array]:
    """Collect only live weights targeted by the already-loaded adapters.

    ``prepared_targets`` names logical bases whose adapters were fused into the
    source weights before packing; they enter the table as shape placeholders
    so receipts count them, and the live fuser leaves their packed weights alone.
    """
    table: dict[str, mx.array] = {}
    for _config, excluded_categories, raw, _classification in adapters:
        for raw_base, _entry in iter_lora_entries(raw):
            mlx_base = convert_lora_base_to_mlx(
                raw_base,
                include_audio=include_audio,
            )
            if mlx_base is None:
                continue
            if excluded_categories & lora_key_categories(mlx_base):
                continue
            if prepared_targets is not None and mlx_base in prepared_targets:
                table[f"{mlx_base}.weight"] = mx.zeros(
                    prepared_targets[mlx_base], dtype=mx.bfloat16
                )
                continue
            _add_live_lora_target(table, model, mlx_base)
    return table


def streamed_adapter_check(
    targets: frozenset[str], *, enabled: bool, reporter: Reporter
) -> Callable[[tuple[tuple[int, nn.Module], ...], Callable[[str], bool]], None]:
    """Check each consumed logical target after its first resident-group evaluation."""
    by_block: dict[int, set[str]] = {}
    for base in targets:
        by_block.setdefault(int(base.split(".", 2)[1]), set()).add(base)
    aggregate_total: int | None = None

    def after_evaluation(
        blocks: tuple[tuple[int, nn.Module], ...], active: Callable[[str], bool]
    ) -> None:
        nonlocal aggregate_total
        if not enabled or not by_block:
            return
        if aggregate_total is None:
            aggregate_total = sum(active(base + ".weight") for base in targets)
        pending = AdapterWeightCheck()
        selected = set()
        for index, block in blocks:
            view = SimpleNamespace(transformer_blocks={index: block})
            for base in by_block.get(index, ()):
                if not active(base + ".weight"):
                    continue
                if is_quantized_linear(_resolve_live_module(view, base)):
                    # Packed targets were fused before quantization. Their integer
                    # storage is not a dense destination for the finite check.
                    selected.add(base)
                    continue
                table: dict[str, mx.array] = {}
                _add_live_lora_target(table, view, base)
                if not table:
                    raise RuntimeError(f"Streamed adapter target is not bound: {base}")
                for value in table.values():
                    pending.add(
                        base, WeightTarget(SimpleNamespace(weight=value), "weight", value.shape)
                    )
                selected.add(base)
        pending.run(
            context="LTX-2 streamed block group",
            reporter=reporter,
            aggregate_total=aggregate_total,
        )
        for index, _block in blocks:
            remaining = by_block.get(index)
            if remaining is not None:
                remaining.difference_update(selected)
                if not remaining:
                    del by_block[index]

    return after_evaluation


def community_adapter_operations(
    adapters: Iterable[LoRAConfig],
    *,
    target_shapes: Mapping[str, tuple[int, ...]],
    include_audio: bool = True,
) -> dict[str, tuple[WeightedAdapterOperation, ...]]:
    """Map admitted adapters onto logical targets that are fused before packing.

    ``target_shapes`` maps logical bases to ``[out, in]`` shapes. The result is
    keyed by ``<base>.weight`` and holds the same weighted operations the live
    fuser would apply, so a preparation hook can fuse them with ``fuse_weight``
    before the weight is quantized. Receipts and logs come from the later live
    fusion pass, which treats these bases as already prepared.
    """
    configs = tuple(adapters)
    if not configs or not target_shapes:
        return {}
    table = {
        f"{base}.weight": mx.zeros(shape, dtype=mx.bfloat16)
        for base, shape in target_shapes.items()
    }
    operations: dict[str, list[WeightedAdapterOperation]] = {}
    for index, config in enumerate(configs):
        excluded = validate_adapter_exclusions(config.exclude)
        raw = load_weights(Path(config.path))
        classification = _classify_adapter(config.path, raw)
        normalized, stats = _normalize_lora_for_weights(
            table,
            raw,
            include_audio=include_audio,
            excluded_categories=excluded,
            extra_unmapped=classification.extra_unmapped,
        )
        if not normalized or not stats.placed or config.strength == 0.0:
            continue
        for prefix, entry in iter_lora_entries(normalized):
            key = prefix if prefix in table else prefix + ".weight"
            if key not in table or not _shape_matches_target(key, table[key], entry.a, entry.b):
                continue
            operation = AdapterOperation(
                prefix,
                "lora",
                "weight",
                {"down": entry.a, "up": entry.b},
                (),
                alpha=entry.alpha,
            )
            operations.setdefault(key, []).append(
                WeightedAdapterOperation(operation, config.strength, f"{index:012d}")
            )
    return {key: tuple(items) for key, items in operations.items()}


def fuse_community_loras_into_model(
    model: object,
    adapters: Iterable[LoRAConfig],
    *,
    include_audio: bool = True,
    model_generation: str | None = None,
    pending: AdapterWeightCheck | None = None,
    prepared_targets: Mapping[str, tuple[int, ...]] | None = None,
    accumulation: str = "float32",
) -> tuple[LTX2AdapterReceipt, ...]:
    """Map admitted ordinary LoRAs into the shared lazy adapter fuser.

    Keep model-specific format admission, exclusions, layouts, and receipts here.
    The shared fuser owns accumulation and first-use target tracking. Each fresh
    transformer receives its own pending check.
    ``prepared_targets`` names bases already fused before
    packing: they count in the receipts but are neither fused again nor added
    to the first-use weight check, whose packed values the latent check covers.
    """
    target = getattr(model, "velocity_model", model)
    configs = tuple(adapters)
    exclusions = tuple(validate_adapter_exclusions(config.exclude) for config in configs)
    loaded = []
    for config, excluded in zip(configs, exclusions, strict=True):
        raw = load_weights(Path(config.path))
        loaded.append((config, excluded, raw, _classify_adapter(config.path, raw)))
    classified = tuple(loaded)
    table = _live_lora_weight_table(
        target,
        classified,
        include_audio=include_audio,
        prepared_targets=prepared_targets,
    )
    prepared_keys = frozenset(f"{base}.weight" for base in (prepared_targets or ()))
    mapped = []
    receipts = []
    for index, (config, excluded, raw, classification) in enumerate(classified, 1):
        normalized, stats = _normalize_lora_for_weights(
            table,
            raw,
            include_audio=include_audio,
            excluded_categories=excluded,
            extra_unmapped=classification.extra_unmapped,
        )
        receipt = _adapter_receipt(
            config,
            stats,
            adapter_types=classification.adapter_types,
            model_generation=model_generation,
        )
        receipts.append(receipt)
        _log_receipt(
            receipt, stats, model_generation=model_generation, position=(index, len(configs))
        )
        if normalized and stats.placed and config.strength != 0.0:
            mapped.append((normalized, config.strength))

    if not mapped:
        return tuple(receipts)

    check = pending if pending is not None else AdapterWeightCheck()
    if pending is None:
        object.__setattr__(model, "adapter_weight_check", check)
    targets = {}
    operations: dict[str, list[WeightedAdapterOperation]] = {}
    transposed = frozenset(key for key in table if key.endswith(".weight_t"))
    for key, value in table.items():
        if key in prepared_keys:
            continue
        shape = tuple(reversed(value.shape)) if key in transposed else value.shape
        targets[key] = WeightTarget(SimpleNamespace(weight=value), "weight", shape)
    for index, (weights, strength) in enumerate(mapped):
        for prefix, entry in iter_lora_entries(weights):
            key = prefix if prefix in table else prefix + ".weight"
            if key in prepared_keys:
                continue
            if key not in table or not _shape_matches_target(key, table[key], entry.a, entry.b):
                continue
            operation = AdapterOperation(
                prefix,
                "lora",
                "weight",
                {"down": entry.a, "up": entry.b},
                (),
                alpha=entry.alpha,
            )
            operations.setdefault(key, []).append(
                WeightedAdapterOperation(operation, strength, f"{index:012d}")
            )
    fuse_adapter_operations(
        targets,
        {key: tuple(items) for key, items in operations.items()},
        transposed=transposed,
        pending=check,
        accumulation=accumulation,
    )
    if prepared_keys:
        _log.info(
            "Adapters fused into %d targets during source preparation",
            len(prepared_keys),
        )
    table = {key: getattr(slot.owner, slot.attribute) for key, slot in targets.items()}
    for key, value in table.items():
        if key.startswith(LAYOUT_KEY_PREFIX):
            install_layout_weight(
                target,
                key[len(LAYOUT_KEY_PREFIX) :],
                value,
            )
        elif not key.startswith(QUANT_KEY_PREFIX):
            owner = _live_weight_owner(_resolve_live_module(target, key.removesuffix(".weight")))
            if owner is None:
                raise RuntimeError(f"LoRA target disappeared during fusion: {key}")
            owner.weight = value
    for key in tuple(check.targets):
        if key.startswith(LAYOUT_KEY_PREFIX):

            def layout_value(key: str = key) -> mx.array:
                value = get_layout_weight(target, key[len(LAYOUT_KEY_PREFIX) :])
                if value is None:
                    raise RuntimeError(f"adapter layout target disappeared: {key}")
                return value

            check.targets[key] = layout_value
        else:
            owner = _live_weight_owner(_resolve_live_module(target, key.removesuffix(".weight")))
            if owner is None:
                raise RuntimeError(f"adapter target disappeared: {key}")
            check.add(key, WeightTarget(owner, "weight", owner.weight.shape))
    return tuple(receipts)


__all__ = [
    "community_adapter_operations",
    "fuse_community_loras_into_model",
]
