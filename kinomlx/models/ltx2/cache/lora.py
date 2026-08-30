"""Apply community LoRAs to normal or pretransposed cache weights."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from kinomlx.io.safetensors import load_weights, read_metadata
from kinomlx.lora.fusion import LOW_COVERAGE_WARNING_THRESHOLD, fuse_many
from kinomlx.lora.loading import LoRAConfig, iter_lora_entries

from .keys.lora import (
    convert_lora_base_to_mlx,
    lora_key_categories,
    validate_lora_exclusions,
)
from .layout import get_layout_weight, install_layout_weight
from .quantization import is_cache_quantized_linear
from .schema import LAYOUT_KEY_PREFIX, QUANT_KEY_PREFIX, file_signature, payload_digest
from .storage import load_cache_weights, load_transformer_fp16_ranges

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
class LoRAAdapterReceipt:
    """Factual placement receipt for one adapter against one transformer graph."""

    path: Path
    fingerprint: str
    base_model_generation: str | None
    declared_model_generation: str | None
    generation_mismatch: bool | None
    strength: float
    knockouts: tuple[str, ...]
    complete_targets: int
    placed_targets: int
    structural_coverage: float
    target_categories: tuple[tuple[str, int], ...]
    skipped_reasons: tuple[tuple[str, int], ...]
    warning: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "base_model_generation": self.base_model_generation,
            "declared_model_generation": self.declared_model_generation,
            "generation_mismatch": self.generation_mismatch,
            "strength": self.strength,
            "knockouts": self.knockouts,
            "complete_targets": self.complete_targets,
            "placed_targets": self.placed_targets,
            "structural_coverage": self.structural_coverage,
            "target_categories": dict(self.target_categories),
            "skipped_reasons": dict(self.skipped_reasons),
            "warning": self.warning,
        }


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


def _normalize_lora_for_cache(
    base_weights: dict[str, mx.array],
    lora_weights: dict[str, mx.array],
    *,
    include_audio: bool,
    excluded_categories: frozenset[str],
) -> tuple[dict[str, mx.array], _LoRAMappingStats]:
    normalized: dict[str, mx.array] = {}
    resolved = 0
    present = 0
    placed = 0
    excluded = 0
    unmapped = 0
    missing = 0
    shape_mismatched = 0
    category_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

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
                f"LoRA target {mlx_base!r} is packed in a quantized cache; "
                "rebuild an unquantized cache before applying community LoRAs"
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
    model_generation: str | None,
) -> LoRAAdapterReceipt:
    declared = _declared_generation(config.path)
    mismatch = (
        None if declared is None or model_generation is None else declared != model_generation
    )
    coverage = stats.placed / stats.considered if stats.considered else 0.0
    return LoRAAdapterReceipt(
        path=config.path,
        fingerprint=_adapter_fingerprint(config.path),
        base_model_generation=model_generation,
        declared_model_generation=declared,
        generation_mismatch=mismatch,
        strength=float(config.strength),
        knockouts=tuple(sorted(config.exclude)),
        complete_targets=stats.considered + stats.excluded,
        placed_targets=stats.placed,
        structural_coverage=coverage,
        target_categories=stats.target_categories,
        skipped_reasons=stats.skipped_reasons,
        warning=stats.considered > 0 and coverage < LOW_COVERAGE_WARNING_THRESHOLD,
    )


def normalize_lora_for_cache(
    base_weights: dict[str, mx.array],
    lora_weights: dict[str, mx.array],
    *,
    include_audio: bool = True,
    exclude: tuple[str, ...] = (),
) -> dict[str, mx.array]:
    """Map one community adapter onto actual normal/layout cache slots.

    Pretransposed slots use ``weight_t`` and are fused with the transposed
    delta by the generic fuser. Quantized packed slots cannot accept a dense
    low-rank update without dequantize/requantize, so they fail explicitly.
    """
    normalized, _stats = _normalize_lora_for_cache(
        base_weights,
        lora_weights,
        include_audio=include_audio,
        excluded_categories=validate_lora_exclusions(exclude),
    )
    return normalized


def fuse_community_loras(
    base_weights: dict[str, mx.array],
    adapters: Iterable[LoRAConfig],
    *,
    include_audio: bool = True,
    in_place: bool = True,
    model_generation: str | None = None,
    receipt_collector: list[LoRAAdapterReceipt] | None = None,
) -> dict[str, mx.array]:
    """Load, map, and fuse community adapters in one fp32 accumulation pass."""
    configs = tuple(adapters)
    exclusions = tuple(validate_lora_exclusions(config.exclude) for config in configs)
    mapped = []
    for config, excluded_categories in zip(configs, exclusions, strict=True):
        raw = load_weights(Path(config.path))
        normalized, stats = _normalize_lora_for_cache(
            base_weights,
            raw,
            include_audio=include_audio,
            excluded_categories=excluded_categories,
        )
        _log.info(
            "LoRA %s against LTX-%s: strength=%.4g, %d resolved, %d cache matches, "
            "%d placed, %d excluded, %d unmapped",
            config.path.name,
            model_generation or "unknown",
            config.strength,
            stats.resolved,
            stats.present,
            stats.placed,
            stats.excluded,
            stats.unmapped,
        )
        receipt = _adapter_receipt(config, stats, model_generation=model_generation)
        if receipt_collector is not None:
            receipt_collector.append(receipt)
        if normalized:
            mapped.append((normalized, config.strength))
    if not mapped:
        return base_weights if in_place else dict(base_weights)
    return fuse_many(
        base_weights,
        mapped,
        in_place=in_place,
    )


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
    if module is not None and is_cache_quantized_linear(module):
        if owner is None:
            raise RuntimeError(f"packed LoRA target {mlx_base!r} has no MLX weight")
        table[f"{QUANT_KEY_PREFIX}{normal_key}"] = owner.weight
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
        tuple[LoRAConfig, frozenset[str], dict[str, mx.array]],
        ...,
    ],
    *,
    include_audio: bool,
) -> dict[str, mx.array]:
    """Collect only live weights targeted by the already-loaded adapters."""
    table: dict[str, mx.array] = {}
    for _config, excluded_categories, raw in adapters:
        for raw_base, _entry in iter_lora_entries(raw):
            mlx_base = convert_lora_base_to_mlx(
                raw_base,
                include_audio=include_audio,
            )
            if mlx_base is None:
                continue
            if excluded_categories & lora_key_categories(mlx_base):
                continue
            _add_live_lora_target(table, model, mlx_base)
    return table


def fuse_community_loras_into_model(
    model: object,
    adapters: Iterable[LoRAConfig],
    *,
    include_audio: bool = True,
    model_generation: str | None = None,
    transformer_cache_path: Path | str | None = None,
) -> tuple[LoRAAdapterReceipt, ...]:
    """Fuse adapters while retaining only their live target weights.

    Every replacement stays lazy so first transformer use can schedule its
    low-rank product without a synchronization per target. FP16 validates
    independently reloaded cache/adapter arrays and consumes them in bounded
    batches. This is intentionally not done through the live arrays: evaluating
    even a reduction would realize the model and adapter inputs, recreating the
    eager memory peak this bridge exists to avoid.
    """
    target = getattr(model, "velocity_model", model)
    configs = tuple(adapters)
    exclusions = tuple(validate_lora_exclusions(config.exclude) for config in configs)
    loaded = tuple(
        (config, excluded, load_weights(Path(config.path)))
        for config, excluded in zip(configs, exclusions, strict=True)
    )
    table = _live_lora_weight_table(
        target,
        loaded,
        include_audio=include_audio,
    )
    mapped = []
    receipts = []
    active_configs: list[tuple[LoRAConfig, frozenset[str]]] = []
    for config, excluded, raw in loaded:
        normalized, stats = _normalize_lora_for_cache(
            table,
            raw,
            include_audio=include_audio,
            excluded_categories=excluded,
        )
        _log.info(
            "LoRA %s: strength=%.4g, %d resolved, %d cache matches, %d excluded, %d unmapped",
            config.path.name,
            config.strength,
            stats.resolved,
            stats.present,
            stats.excluded,
            stats.unmapped,
        )
        receipts.append(_adapter_receipt(config, stats, model_generation=model_generation))
        if normalized:
            mapped.append((normalized, config.strength))
            active_configs.append((config, excluded))

    if not mapped:
        return tuple(receipts)

    range_validation_inputs = None
    if any(value.dtype == mx.float16 for value in table.values()):
        if transformer_cache_path is None:
            raise ValueError("transformer_cache_path is required for FP16 LoRA range validation")
        cache_path = Path(transformer_cache_path)
        loaded_validation_base = load_cache_weights(cache_path)
        validation_base = {
            key: loaded_validation_base[key] for key in table if key in loaded_validation_base
        }
        loaded_validation_base.clear()
        validation_base_peaks = load_transformer_fp16_ranges(cache_path)
        validation_mapped = []
        for config, excluded in active_configs:
            validation_raw = load_weights(Path(config.path))
            normalized, _stats = _normalize_lora_for_cache(
                validation_base,
                validation_raw,
                include_audio=include_audio,
                excluded_categories=excluded,
            )
            validation_raw.clear()
            validation_mapped.append((normalized, config.strength))
        range_validation_inputs = (
            validation_base,
            validation_mapped,
            validation_base_peaks,
        )

    fuse_many(
        table,
        mapped,
        in_place=True,
        range_validation_inputs=range_validation_inputs,
    )
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
    return tuple(receipts)


__all__ = [
    "LoRAAdapterReceipt",
    "fuse_community_loras",
    "fuse_community_loras_into_model",
    "normalize_lora_for_cache",
]
