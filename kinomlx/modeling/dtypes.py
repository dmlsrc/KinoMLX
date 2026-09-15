"""Small shared dtype resolution for floating MLX model policies."""

import mlx.core as mx

_FLOAT_DTYPES = frozenset((mx.float16, mx.bfloat16, mx.float32))
_FLOAT_DTYPE_BY_NAME = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


def resolve_float_dtype(value: str | mx.Dtype, *, owner: str = "model") -> mx.Dtype:
    """Resolve one supported floating compute/weight dtype."""
    if isinstance(value, str):
        try:
            return _FLOAT_DTYPE_BY_NAME[value]
        except KeyError as exc:
            raise ValueError(f"unsupported {owner} dtype: {value}") from exc
    if value not in _FLOAT_DTYPES:
        raise ValueError(f"unsupported {owner} dtype: {value}")
    return value


def safetensors_float_dtype(value: str | mx.Dtype) -> str:
    """Return the safetensors header spelling for a floating model policy."""
    return {mx.float16: "F16", mx.bfloat16: "BF16", mx.float32: "F32"}[resolve_float_dtype(value)]


__all__ = ["resolve_float_dtype", "safetensors_float_dtype"]
