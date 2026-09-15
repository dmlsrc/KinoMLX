"""Klein weight dtype and execution-layout preparation."""

import mlx.core as mx

from .layout import pretransposed_weight


def prepare_weight(kind: str, key: str, value: mx.array, dtype: mx.Dtype) -> mx.array:
    """Cast one consumed source tensor and bake the measured transformer layout.

    The result stays lazy. Selected transformer down-projections are stored
    transposed and contiguous for the pretransposed linear layers; every other
    tensor keeps its source layout in the requested dtype.
    """
    prepared = value.astype(dtype)
    if kind == "transformer" and pretransposed_weight(key):
        prepared = mx.contiguous(prepared.T)
    return prepared


__all__ = ["prepare_weight"]
