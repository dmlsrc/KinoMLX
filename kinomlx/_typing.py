"""Internal structural types for dynamic serialization and tensor trees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mlx.core as mx

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | Sequence[JsonValue] | Mapping[str, JsonValue]
type JsonObject = dict[str, JsonValue]

# MLX Module.update trees may contain dictionaries and numerically indexed
# lists. None is permitted while a list is assembled from dotted keys.
type TensorTree = None | mx.array | list[TensorTree] | dict[str, TensorTree]

__all__ = ["JsonObject", "JsonScalar", "JsonValue", "TensorTree"]
