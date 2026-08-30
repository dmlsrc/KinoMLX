"""Install cache-backed quantized transformer linear modules."""

from __future__ import annotations

from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn

from .schema import (
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)
from .weights import (
    CACHE_QUANTIZED_BLOCK_LINEAR_BASES,
    cache_quant_pretransposed,
    quant_defaults,
    quant_mode_for_target,
)


class _PretransposedQuantizedLinear(nn.Module):
    """Quantized linear whose packed cache weight was built from ``weight.T``."""

    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    bias: mx.array
    group_size: int
    bits: int
    mode: str

    def __call__(self, value: mx.array) -> mx.array:
        weight = cast(mx.array, self["weight"])
        scales = cast(mx.array, self["scales"])
        biases = cast(mx.array | None, self.get("biases"))
        result = mx.quantized_matmul(
            value,
            weight,
            scales=scales,
            biases=biases,
            transpose=False,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            result = result + cast(mx.array, self["bias"])
        return result


_QUANTIZED_LINEAR_TYPES = (nn.QuantizedLinear, _PretransposedQuantizedLinear)
type _CacheQuantizedLinear = nn.QuantizedLinear | _PretransposedQuantizedLinear


def is_cache_quantized_linear(value: object) -> bool:
    """Return whether a live linear owns packed cache-quantized weights."""
    return isinstance(value, _QUANTIZED_LINEAR_TYPES)


def _empty_linear(*, has_bias: bool = True) -> nn.Linear:
    linear = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(linear)
    linear.weight = mx.zeros((0, 0), dtype=mx.float32)
    if has_bias:
        linear.bias = mx.zeros((0,), dtype=mx.float32)
    return linear


def _resolve_linear_parent(
    block: nn.Module,
    base: str,
) -> tuple[nn.Module, str] | None:
    parts = base.split(".")
    parent = block
    for part in parts[:-1]:
        candidate = getattr(parent, part, None)
        if not isinstance(candidate, nn.Module):
            return None
        parent = candidate
    return parent, parts[-1]


def restore_block_quantized_linears(
    block: nn.Module,
    keep_bases: set[str],
) -> None:
    """Replace stale cache-backed quantized linears with empty base linears."""
    for base in CACHE_QUANTIZED_BLOCK_LINEAR_BASES:
        if base in keep_bases:
            continue
        resolved = _resolve_linear_parent(block, base)
        if resolved is None:
            continue
        parent, attribute = resolved
        current = getattr(parent, attribute, None)
        if isinstance(current, _QUANTIZED_LINEAR_TYPES):
            setattr(
                parent,
                attribute,
                _empty_linear(has_bias="bias" in current),
            )


def quant_bases_for_block_keys(
    quant_keys: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> set[str]:
    """Return logical linear bases represented by cached quant tensors."""
    bases: set[str] = set()
    for _full_key, block_key in quant_keys:
        for suffix in (".weight", ".scales", ".biases"):
            if block_key.endswith(suffix):
                bases.add(block_key[: -len(suffix)])
                break
    return bases


def _quant_params_for_base(
    quant_keys: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    base: str,
) -> set[str]:
    prefix = f"{base}."
    return {
        block_key[len(prefix) :]
        for _full_key, block_key in quant_keys
        if block_key.startswith(prefix)
    }


def _empty_quantized_linear(
    *,
    mode: str,
    group_size: int,
    bits: int,
    has_bias: bool,
    has_quant_biases: bool,
    pretransposed: bool,
) -> _CacheQuantizedLinear:
    if pretransposed:
        linear: _CacheQuantizedLinear = _PretransposedQuantizedLinear.__new__(
            _PretransposedQuantizedLinear
        )
    else:
        linear = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(linear)
    linear.group_size = group_size
    linear.bits = bits
    linear.mode = mode
    linear.weight = mx.zeros((0, 0), dtype=mx.uint32)
    linear.scales = mx.zeros((0, 0), dtype=mx.uint8)
    if has_quant_biases:
        linear.biases = mx.zeros((0, 0), dtype=mx.float32)
    if has_bias:
        linear.bias = mx.zeros((0,), dtype=mx.float32)
    linear.freeze()
    return linear


def _ensure_quantized_linear(
    current: object,
    *,
    mode: str,
    group_size: int,
    bits: int,
    has_bias: bool,
    has_quant_biases: bool,
    pretransposed: bool,
) -> _CacheQuantizedLinear:
    type_matches = (
        isinstance(current, _PretransposedQuantizedLinear)
        if pretransposed
        else isinstance(current, nn.QuantizedLinear)
    )
    if not type_matches:
        return _empty_quantized_linear(
            mode=mode,
            group_size=group_size,
            bits=bits,
            has_bias=has_bias,
            has_quant_biases=has_quant_biases,
            pretransposed=pretransposed,
        )
    linear = cast(_CacheQuantizedLinear, current)
    if linear.mode != mode or linear.group_size != group_size or linear.bits != bits:
        return _empty_quantized_linear(
            mode=mode,
            group_size=group_size,
            bits=bits,
            has_bias=has_bias,
            has_quant_biases=has_quant_biases,
            pretransposed=pretransposed,
        )
    if has_quant_biases and "biases" not in linear:
        linear.biases = mx.zeros((0, 0), dtype=mx.float32)
    elif not has_quant_biases and "biases" in linear:
        delattr(linear, "biases")
    if has_bias and "bias" not in linear:
        linear.bias = mx.zeros((0,), dtype=mx.float32)
    elif not has_bias and "bias" in linear:
        delattr(linear, "bias")
    return linear


def _quant_mode_for_base(
    base: str,
    *,
    transformer_cache_quantize: str,
    quantization_specs: tuple[tuple[str, str], ...],
) -> str | None:
    if (
        transformer_cache_quantize
        in {
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
        }
        and base in CACHE_QUANTIZED_BLOCK_LINEAR_BASES
    ):
        return "mxfp8"
    if base == "ff.project_in.proj":
        return quant_mode_for_target(quantization_specs, "project_in")
    if base == "ff.project_out":
        return quant_mode_for_target(quantization_specs, "project_out")
    return None


def prepare_block_quantized_linears(
    block: nn.Module,
    quant_bases: set[str],
    quant_keys: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    normal_keys: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    *,
    transformer_cache_quantize: str,
    quantization_specs: tuple[tuple[str, str], ...],
    group_size: int | None,
    bits: int | None,
) -> None:
    """Install correctly configured quantized linears before weight binding."""
    normal_block_keys = {block_key for _full_key, block_key in normal_keys}
    pretransposed = cache_quant_pretransposed(transformer_cache_quantize)
    for base in quant_bases:
        mode = _quant_mode_for_base(
            base,
            transformer_cache_quantize=transformer_cache_quantize,
            quantization_specs=quantization_specs,
        )
        if mode is None:
            raise ValueError(f"Missing cached quantization mode for transformer target: {base}")
        normalized_group_size, normalized_bits = quant_defaults(
            mode,
            group_size,
            bits,
        )
        parameters = _quant_params_for_base(quant_keys, base)
        missing_parameters = {"weight", "scales"} - parameters
        if missing_parameters:
            missing = ", ".join(sorted(missing_parameters))
            raise ValueError(f"Incomplete quantized cache target {base}: missing {missing}")
        has_bias = f"{base}.bias" in normal_block_keys
        has_quant_biases = "biases" in parameters
        resolved = _resolve_linear_parent(block, base)
        if resolved is None:
            raise ValueError(f"Missing transformer module for cached quantization target: {base}")
        parent, attribute = resolved
        current = getattr(parent, attribute, None)
        if current is None:
            raise ValueError(f"Missing transformer linear for cached quantization target: {base}")
        setattr(
            parent,
            attribute,
            _ensure_quantized_linear(
                current,
                mode=mode,
                group_size=normalized_group_size,
                bits=normalized_bits,
                has_bias=has_bias,
                has_quant_biases=has_quant_biases,
                pretransposed=pretransposed,
            ),
        )


__all__ = [
    "is_cache_quantized_linear",
    "prepare_block_quantized_linears",
    "quant_bases_for_block_keys",
    "restore_block_quantized_linears",
]
