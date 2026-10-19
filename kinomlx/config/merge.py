"""Recursive config merge: tables merge, scalars and arrays replace."""

from collections.abc import Mapping
from typing import cast


def _merge_two(base: object, overlay: object) -> object:
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        base_mapping = cast(Mapping[str, object], base)
        overlay_mapping = cast(Mapping[str, object], overlay)
        merged = dict(base_mapping)
        for key, value in overlay_mapping.items():
            merged[key] = _merge_two(merged[key], value) if key in merged else value
        return merged
    return overlay


def merge_configs(*configs: Mapping[str, object]) -> dict[str, object]:
    """Merge mappings from left to right without mutating an input mapping."""
    merged: dict[str, object] = {}
    for config in configs:
        merged = cast(dict[str, object], _merge_two(merged, config))
    return merged


def field_paths(
    config: Mapping[str, object], prefix: tuple[str, ...] = ()
) -> frozenset[tuple[str, ...]]:
    """Return present leaf identities; an empty group supplies no member assignment."""
    return frozenset(
        *(
            field_paths(cast(Mapping[str, object], value), (*prefix, name))
            if isinstance(value, Mapping)
            else ((*prefix, name),)
        )
        for name, value in config.items()
    )


__all__ = ["field_paths", "merge_configs"]
