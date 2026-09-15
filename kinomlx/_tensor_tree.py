"""Recursive MLX tensor-tree type used by module weight updates."""

import mlx.core as mx

# MLX Module.update trees may contain dictionaries and numerically indexed
# lists. None is permitted while a list is assembled from dotted keys.
type TensorTree = mx.array | list[TensorTree] | dict[str, TensorTree] | None

__all__ = ["TensorTree"]
