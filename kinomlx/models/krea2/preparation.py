"""Lazy Krea component transformations shared by all loading routes."""

import mlx.core as mx

from .resource_types import Krea2TensorBinding
from .weight_keys import (
    KREA2_TRANSFORMER_PRETRANSPOSE,
    KREA2_VAE_CONV2D_OHWI,
    KREA2_VAE_CURRENT_FRAME_OHWI,
    KREA2_VAE_FLATTEN,
)


def prepare_weight(
    value: mx.array,
    binding: Krea2TensorBinding,
    *,
    dtype: mx.Dtype | None,
) -> mx.array:
    converted = value
    if binding.reshape_rows is not None:
        rows = binding.reshape_rows
        if converted.ndim != 1 or converted.size % rows:
            raise ValueError(
                f"Krea 2 tensor {binding.source_key!r} cannot reshape into {rows} rows"
            )
        converted = converted.reshape(rows, converted.size // rows)
    if binding.transform == KREA2_TRANSFORMER_PRETRANSPOSE:
        if converted.ndim != 2:
            raise ValueError(f"Krea 2 tensor {binding.source_key!r} must be a 2D Linear weight")
        converted = converted.T
    elif binding.transform == KREA2_VAE_CURRENT_FRAME_OHWI:
        if converted.ndim != 5:
            raise ValueError(
                f"Krea 2 tensor {binding.source_key!r} must be a 5D causal Conv3d weight"
            )
        converted = converted[:, :, -1, :, :].transpose(0, 2, 3, 1)
    elif binding.transform == KREA2_VAE_CONV2D_OHWI:
        if converted.ndim != 4:
            raise ValueError(f"Krea 2 tensor {binding.source_key!r} must be a 4D Conv2d weight")
        converted = converted.transpose(0, 2, 3, 1)
    elif binding.transform == KREA2_VAE_FLATTEN:
        converted = converted.reshape(-1)
    elif binding.transform is not None:
        raise ValueError(f"unsupported Krea 2 tensor transform: {binding.transform}")
    if dtype is not None and converted.dtype != dtype:
        converted = converted.astype(dtype)
    return mx.contiguous(converted)
