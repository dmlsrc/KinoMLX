"""Central consumed-target binding for LTX text stations."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx._typing import JsonValue
from kinomlx.io.safetensors import load_weights, read_header
from kinomlx.reporting import NullReporter, Reporter


@dataclass(frozen=True)
class WeightTarget:
    """One model attribute and the exact logical shape it consumes."""

    owner: object
    attribute: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class WeightBinding:
    """One preflighted source tensor bound to one logical target."""

    source_path: Path
    source_key: str
    logical_key: str
    target: WeightTarget


def _source_matches(header: Mapping[str, JsonValue], suffix: str) -> tuple[str, ...]:
    exact = (suffix,) if suffix in header else ()
    wrapped = tuple(
        sorted(
            key
            for key in header
            if key != "__metadata__" and key != suffix and key.endswith("." + suffix)
        )
    )
    return exact + wrapped


def resolve_weight_bindings(
    paths: Sequence[Path],
    targets: Mapping[str, WeightTarget],
    source_suffixes: Callable[[str], tuple[str, ...]],
    *,
    component: str,
) -> tuple[WeightBinding, ...]:
    """Resolve every consumed target exactly once without policing source baggage."""
    headers = tuple((path, read_header(path)) for path in paths)
    resolved: list[WeightBinding] = []
    missing: list[str] = []
    for logical_key, target in targets.items():
        matches: list[tuple[Path, str, Mapping[str, JsonValue]]] = []
        for path, header in headers:
            for suffix in source_suffixes(logical_key):
                for source_key in _source_matches(header, suffix):
                    entry = header[source_key]
                    if isinstance(entry, Mapping):
                        matches.append((path, source_key, entry))
        if not matches:
            missing.append(logical_key)
            continue
        if len(matches) != 1:
            names = ", ".join(f"{path.name}:{key}" for path, key, _entry in matches[:3])
            raise ValueError(
                f"{component} consumed target {logical_key!r} resolves more than once: {names}"
            )
        source_path, source_key, entry = matches[0]
        raw_shape = entry.get("shape")
        shape = (
            tuple(raw_shape)
            if isinstance(raw_shape, list)
            and all(not isinstance(value, bool) and isinstance(value, int) for value in raw_shape)
            else ()
        )
        if shape != target.shape:
            raise ValueError(
                f"{component} consumed tensor {source_key!r} has shape {shape}, "
                f"expected {target.shape}"
            )
        resolved.append(
            WeightBinding(
                source_path=source_path,
                source_key=source_key,
                logical_key=logical_key,
                target=target,
            )
        )
    if missing:
        raise ValueError(
            f"{component} checkpoint is incomplete: missing {len(missing)} consumed tensors "
            f"(first: {missing[0]})"
        )
    return tuple(resolved)


def bind_weight_targets(
    paths: Sequence[Path],
    targets: Mapping[str, WeightTarget],
    source_suffixes: Callable[[str], tuple[str, ...]],
    *,
    component: str,
    phase: str,
    reporter: Reporter | None = None,
) -> int:
    """Preflight and bind one complete consumed target set, ignoring unrelated tensors."""
    bindings = resolve_weight_bindings(
        paths,
        targets,
        source_suffixes,
        component=component,
    )
    return bind_resolved_weights(
        paths,
        bindings,
        phase=phase,
        reporter=reporter,
    )


def bind_resolved_weights(
    paths: Sequence[Path],
    bindings: Sequence[WeightBinding],
    *,
    phase: str,
    reporter: Reporter | None = None,
) -> int:
    """Apply already-preflighted bindings in source-file order."""
    by_path = {
        path: tuple(binding for binding in bindings if binding.source_path == path)
        for path in paths
    }
    sink = reporter if reporter is not None else NullReporter()
    sink.phase_start(phase, total=len(paths), unit="file")
    try:
        for path in paths:
            weights = load_weights(path)
            try:
                for binding in by_path[path]:
                    setattr(
                        binding.target.owner,
                        binding.target.attribute,
                        weights[binding.source_key],
                    )
            finally:
                weights.clear()
                del weights
                gc.collect()
                mx.clear_cache()
            sink.phase_advance(phase)
    finally:
        sink.phase_end(phase)
    return len(bindings)


__all__ = [
    "WeightTarget",
    "bind_resolved_weights",
    "bind_weight_targets",
    "resolve_weight_bindings",
]
