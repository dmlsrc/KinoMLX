"""Recursive config merge: tables merge, scalars and arrays replace."""

from __future__ import annotations

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


def normalize_output_selection(config: Mapping[str, object]) -> dict[str, object]:
    """Make one precedence layer choose exact-path or generated-path mode.

    A directory or prefix in a higher-precedence layer must be able to replace
    an exact path selected below it. Within one layer, a non-``None`` exact
    path still wins over directory/prefix fields supplied beside it.
    """
    result = dict(config)
    output = result.get("output")
    if not isinstance(output, Mapping):
        return result
    normalized = dict(cast(Mapping[str, object], output))
    if normalized.get("path") is None and ({"directory", "prefix"} & normalized.keys()):
        normalized["path"] = None
    result["output"] = normalized
    return result


__all__ = ["merge_configs", "normalize_output_selection"]
