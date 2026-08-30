"""Transformer checkpoint-to-MLX key conversion."""

from __future__ import annotations

from typing import cast

import mlx.core as mx

from kinomlx._typing import TensorTree

from ...transformer.graph import (
    DIFFUSION_PREFIX,
)
from ...transformer.graph import (
    convert_checkpoint_key as _convert_checkpoint_key,
)


def convert_pytorch_key_to_mlx(
    pytorch_key: str,
    include_audio: bool = False,
) -> str | None:
    """Convert a prefix-free LTX transformer key to its MLX parameter path."""
    converted = _convert_checkpoint_key(pytorch_key, include_audio=include_audio)
    return None if converted is None else converted.target_key


def convert_checkpoint_key(
    checkpoint_key: str,
    *,
    include_audio: bool = False,
) -> str | None:
    """Convert a fully qualified checkpoint key or return ``None`` if unrelated."""
    converted = _convert_checkpoint_key(checkpoint_key, include_audio=include_audio)
    return None if converted is None else converted.target_key


def flatten_to_nested(flat: dict[str, mx.array]) -> dict[str, TensorTree]:
    """Convert dotted MLX parameter paths to ``Module.update`` structures."""
    nested: dict[str, TensorTree] = {}
    for key, value in flat.items():
        current = nested
        parts = key.split(".")
        for part in parts[:-1]:
            current = cast(dict[str, TensorTree], current.setdefault(part, {}))
        current[parts[-1]] = value
    result = _numeric_dicts_to_lists(nested)
    if not isinstance(result, dict):
        raise ValueError("top-level MLX parameter keys must not be numeric")
    return result


def _numeric_dicts_to_lists(value: TensorTree) -> TensorTree:
    if not isinstance(value, dict):
        return value
    processed = {key: _numeric_dicts_to_lists(item) for key, item in value.items()}
    if processed and all(key.isdigit() for key in processed):
        result: list[TensorTree] = [None] * (max(map(int, processed)) + 1)
        for key, item in processed.items():
            result[int(key)] = item
        return result
    return processed


__all__ = [
    "DIFFUSION_PREFIX",
    "convert_checkpoint_key",
    "convert_pytorch_key_to_mlx",
    "flatten_to_nested",
]
