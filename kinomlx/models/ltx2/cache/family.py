"""Build and validate auxiliary split weight-family caches."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx._typing import JsonObject
from kinomlx.reporting import NullReporter, Reporter

from .keys import weight_family_for_key
from .layout import bake_conv_layout_for_family
from .schema import (
    WEIGHT_FAMILIES,
    WEIGHT_FAMILY_LABELS,
    weight_family_cache_paths,
)
from .storage import (
    cache_artifacts_exist,
    metadata_matches,
    save_weights_atomic,
    write_metadata,
)
from .weights import iter_checkpoint_weights

_log = logging.getLogger(__name__)

_SOURCE_COMPONENT_FAMILIES = {
    "video_vae": frozenset({"video_vae"}),
    "audio_vae_vocoder": frozenset({"audio_vae", "vocoder"}),
}


@dataclass(frozen=True)
class WeightFamilyCacheResult:
    """Result metadata for auxiliary family cache builds."""

    cache_paths: dict[str, Path]
    rebuilt: bool
    loaded_count: int


def _validated_families(families: tuple[str, ...]) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(families))
    for family in unique:
        if family not in WEIGHT_FAMILIES:
            raise ValueError(f"Unsupported weight family: {family}")
    return unique


def _validated_source_component(
    source_component: str | None,
    families: tuple[str, ...],
) -> str | None:
    if source_component is None:
        return None
    allowed = _SOURCE_COMPONENT_FAMILIES.get(source_component)
    if allowed is None:
        choices = ", ".join(sorted(_SOURCE_COMPONENT_FAMILIES))
        raise ValueError(f"Unsupported source component {source_component!r}; expected {choices}")
    incompatible = sorted(set(families) - allowed)
    if incompatible:
        raise ValueError(
            f"Source component {source_component!r} cannot provide families: "
            + ", ".join(incompatible)
        )
    return source_component


def _save_weight_family_cache(
    family: str,
    cache_file: Path,
    metadata_file: Path,
    payload: JsonObject,
    weights: dict[str, mx.array],
) -> int:
    if not weights:
        raise ValueError(f"Checkpoint contains no tensors for requested {family} family")
    converted = bake_conv_layout_for_family(family, weights)
    save_weights_atomic(cache_file, converted)
    write_metadata(metadata_file, payload)
    _log.info(
        "Built %s cache: %d tensors",
        WEIGHT_FAMILY_LABELS[family],
        len(converted),
    )
    return len(converted)


def build_weight_family_caches(
    weights_path: Path | str,
    cache_root: Path | str | None,
    families: tuple[str, ...],
    *,
    source_component: str | None = None,
    reporter: Reporter | None = None,
) -> int:
    """Build requested source-level family caches in one checkpoint pass."""
    requested = _validated_families(families)
    source_component = _validated_source_component(source_component, requested)
    if not requested:
        return 0
    sink = reporter if reporter is not None else NullReporter()
    phase = "build weight family caches"
    sink.phase_start(phase, total=None, unit="tensor")
    buckets: dict[str, dict[str, mx.array]] = {family: {} for family in requested}
    try:
        for key, value in iter_checkpoint_weights(weights_path):
            family = weight_family_for_key(
                key,
                source_component=source_component,
            )
            if family in buckets:
                buckets[family][key] = value
                sink.phase_advance(phase)

        absent = [family for family, weights in buckets.items() if not weights]
        if absent:
            raise ValueError(
                "Checkpoint contains no tensors for requested families: " + ", ".join(absent)
            )

        loaded_count = 0
        for family, family_weights in buckets.items():
            cache_file, metadata_file, payload = weight_family_cache_paths(
                weights_path,
                cache_root,
                family,
                source_component=source_component,
            )
            loaded_count += _save_weight_family_cache(
                family,
                cache_file,
                metadata_file,
                payload,
                family_weights,
            )
        return loaded_count
    finally:
        buckets.clear()
        gc.collect()
        mx.clear_cache()
        sink.phase_end(phase)


def ensure_weight_family_caches(
    weights_path: Path | str,
    *,
    families: tuple[str, ...],
    cache_mode: str,
    cache_root: Path | str | None,
    source_component: str | None = None,
    reporter: Reporter | None = None,
) -> WeightFamilyCacheResult:
    """Ensure named auxiliary family caches exist and return their paths."""
    if cache_mode not in {"auto", "rebuild"}:
        raise ValueError(f"Unsupported weight family cache mode: {cache_mode}")
    requested = _validated_families(families)
    source_component = _validated_source_component(source_component, requested)
    cache_paths: dict[str, Path] = {}
    missing: list[str] = []
    for family in requested:
        cache_file, metadata_file, payload = weight_family_cache_paths(
            weights_path,
            cache_root,
            family,
            source_component=source_component,
        )
        cache_paths[family] = cache_file
        valid = cache_artifacts_exist(cache_file) and metadata_matches(
            metadata_file,
            payload,
        )
        if cache_mode == "rebuild" or not valid:
            missing.append(family)
    if missing:
        for family in missing:
            _log.info(
                "%s cache: building %s",
                WEIGHT_FAMILY_LABELS[family],
                cache_paths[family],
            )
        loaded_count = build_weight_family_caches(
            weights_path,
            cache_root,
            tuple(missing),
            source_component=source_component,
            reporter=reporter,
        )
        rebuilt = True
    else:
        loaded_count = 0
        rebuilt = False
        for family in requested:
            _log.info(
                "%s cache: using %s",
                WEIGHT_FAMILY_LABELS[family],
                cache_paths[family],
            )
    return WeightFamilyCacheResult(
        cache_paths=cache_paths,
        rebuilt=rebuilt,
        loaded_count=loaded_count,
    )


__all__ = [
    "WeightFamilyCacheResult",
    "build_weight_family_caches",
    "ensure_weight_family_caches",
]
