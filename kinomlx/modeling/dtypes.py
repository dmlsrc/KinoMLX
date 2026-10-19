"""Small shared dtype resolution for floating MLX model policies."""

from enum import StrEnum

import mlx.core as mx


class FloatDType(StrEnum):
    """The floating dtypes a model component may be bound at."""

    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class DTypeDirective(StrEnum):
    """A component dtype policy that instructs rather than naming a dtype."""

    JSON = "json"
    """The component carries no tensors; its source is JSON."""

    SOURCE = "source"
    """Bind whatever dtype the source declares."""


type DTypePolicy = FloatDType | DTypeDirective
"""What a component's dtype policy may be: a dtype, or an instruction about one.

A StrEnum cannot be extended once it has members, so the two vocabularies stay
separate and the union names their sum. Their members compare and hash as their
values, so a policy can be tested against either side without conversion.
"""

FLOAT_DTYPE_CHOICES = tuple(item.value for item in FloatDType)

_FLOAT_DTYPES = frozenset((mx.float16, mx.bfloat16, mx.float32))
_FLOAT_DTYPE_BY_NAME = {
    FloatDType.BFLOAT16: mx.bfloat16,
    FloatDType.FLOAT16: mx.float16,
    FloatDType.FLOAT32: mx.float32,
}


_FLOAT_DTYPE_NAMES = {dtype: name for name, dtype in _FLOAT_DTYPE_BY_NAME.items()}


def resolve_float_dtype(value: FloatDType | mx.Dtype, *, owner: str = "model") -> mx.Dtype:
    """Resolve one supported floating compute/weight dtype."""
    if isinstance(value, str):
        try:
            return _FLOAT_DTYPE_BY_NAME[FloatDType(value)]
        except ValueError as exc:
            raise ValueError(f"unsupported {owner} dtype: {value}") from exc
    if value not in _FLOAT_DTYPES:
        raise ValueError(f"unsupported {owner} dtype: {value}")
    return value


def resolve_bound_dtype(policy: DTypePolicy, *, owner: str = "model") -> mx.Dtype:
    """Resolve the dtype a component binds at, refusing a policy that names none.

    A directive is not a dtype: ``json`` belongs to a component with no
    tensors and ``source`` defers to whatever the file declares, so neither
    can answer what a graph executes in. Typing this boundary is how a policy
    that cannot be bound is refused by name instead of reaching the resolver
    as an unrecognized string.
    """
    if isinstance(policy, DTypeDirective):
        raise ValueError(f"{owner} dtype policy {policy.value!r} names no dtype to bind")
    return resolve_float_dtype(policy, owner=owner)


def float_dtype_name(value: FloatDType | mx.Dtype, *, owner: str = "model") -> FloatDType:
    """Name one supported floating dtype, whichever side of the pair it arrives from.

    ``str(mx.bfloat16)`` is ``"mlx.core.bfloat16"``, so reading a dtype's name
    by trimming that prefix produces a bare string that no longer belongs to
    this vocabulary. This returns the member instead.
    """
    if isinstance(value, FloatDType):
        return value
    try:
        return _FLOAT_DTYPE_NAMES[value]
    except KeyError as exc:
        raise ValueError(f"unsupported {owner} dtype: {value}") from exc


def safetensors_float_dtype(value: FloatDType | mx.Dtype) -> str:
    """Return the safetensors header spelling for a floating model policy.

    The header vocabulary is the storage format's, not this project's, so it
    stays a string rather than earning a name here.
    """
    return {mx.float16: "F16", mx.bfloat16: "BF16", mx.float32: "F32"}[resolve_float_dtype(value)]


__all__ = [
    "FLOAT_DTYPE_CHOICES",
    "DTypeDirective",
    "DTypePolicy",
    "FloatDType",
    "float_dtype_name",
    "resolve_bound_dtype",
    "resolve_float_dtype",
    "safetensors_float_dtype",
]
