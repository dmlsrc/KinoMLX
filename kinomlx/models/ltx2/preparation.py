"""Lazy LTX transformer conversion and weight binding."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import inspect_bindings, inspect_safetensors, load_weights, read_dtypes
from kinomlx.modeling.conversion import Conversion, SourceTensor, plan_sources
from kinomlx.modeling.dtypes import FloatDType
from kinomlx.models.ltx2.weights.dtypes import normalize_transformer_dtypes
from kinomlx.models.ltx2.weights.layout import bake_conv_layout, layout_weight_key
from kinomlx.models.ltx2.weights.schema import (
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
    TransformerQuantize,
    dtype_payload_name,
)
from kinomlx.models.ltx2.weights.tensors import (
    dequantize_fp8_weight,
    ff_dtype_for_key,
    fp8_scale_companions,
    quant_defaults,
    quant_mode_for_key,
    quant_pretransposed,
    reject_unsupported_fp8,
)

from .weights.bindings import (
    FP8_SCALE_TARGET_PREFIX,
    family_source_bindings,
    transformer_source_bindings,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerPreparation:
    sources: tuple[SourceTensor, ...]
    convert: Callable[[SourceTensor, mx.array], Conversion]


def prepare_transformer_source(
    weights_path: Path | str,
    *,
    transformer_dtype: FloatDType | mx.Dtype | None = None,
    include_audio: bool,
    video_ff_layout_specs: tuple[tuple[str, str], ...],
    video_ff_layout_layers: tuple[int, ...],
    video_attn_layout_specs: tuple[tuple[str, str], ...],
    video_attn_layout_layers: tuple[int, ...],
    audio_ff_layout_specs: tuple[tuple[str, str], ...] = (),
    audio_ff_layout_layers: tuple[int, ...] = (),
    audio_attn_layout_specs: tuple[tuple[str, str], ...] = (),
    audio_attn_layout_layers: tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
    transformer_quantize: TransformerQuantize = TransformerQuantize.OFF,
    transformer_quantize_exclude_layers: tuple[int, ...] = (),
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    source_bindings: Mapping[str, str] | None = None,
    adjust_quantized: Callable[[str, mx.array], mx.array] | None = None,
    adjust_dense: Callable[[str, mx.array], mx.array] | None = None,
) -> TransformerPreparation:
    """Plan the lazy source conversion for model binding.

    ``adjust_quantized`` receives each quantized target's logical weight after
    FP8 decoding and before packing; adapters fuse there so a packed weight already carries them.
    """
    transformer_dtype, video_ff_dtype, audio_ff_dtype = normalize_transformer_dtypes(
        transformer_dtype,
        video_ff_dtype,
        audio_ff_dtype,
    )

    source = Path(weights_path).expanduser().absolute()
    header_dtypes = read_dtypes(source)
    fp8_keys, weight_scales, input_scales, comfy_tags = fp8_scale_companions(header_dtypes)
    fp8_value_dtype = transformer_dtype or mx.bfloat16
    if fp8_keys:
        reject_unsupported_fp8(header_dtypes)
        _log.info(
            "FP8 checkpoint: dequantizing %d tensors (%d weight scales, "
            "%d input scales and %d ComfyUI tags removed) to %s",
            len(fp8_keys),
            len(weight_scales),
            len(input_scales),
            len(comfy_tags),
            dtype_payload_name(fp8_value_dtype),
        )
    bindings, _source_has_fp8 = transformer_source_bindings(
        source,
        include_audio=include_audio,
        source_bindings=source_bindings,
    )
    if source_bindings is not None:
        missing_targets = sorted(
            set(source_bindings.values()) - {binding.target_key for binding in bindings}
        )
        if missing_targets:
            raise ValueError(
                f"Checkpoint {weights_path} changed while preparing transformer weights; "
                f"missing consumed target {missing_targets[0]}"
            )
    sources = plan_sources(
        inspect_bindings(bindings),
        convert=lambda entry: not entry.target_key.startswith(FP8_SCALE_TARGET_PREFIX),
    )
    scales = (
        {key: value for key, value in load_weights(source).items() if key in weight_scales}
        if fp8_keys
        else {}
    )

    def convert(item: SourceTensor, raw: mx.array) -> Conversion:
        mlx_key = item.target_key
        value = raw
        if item.source_key in fp8_keys:
            value = dequantize_fp8_weight(
                raw,
                item.source_key,
                scales.get(f"{item.source_key}_scale"),
                fp8_value_dtype,
            )
        quant_mode = quant_mode_for_key(
            mlx_key,
            transformer_quantize=transformer_quantize,
            transformer_quantize_exclude_layers=transformer_quantize_exclude_layers,
            video_ff_quantize_specs=video_ff_quantize_specs,
            video_ff_quantize_layers=video_ff_quantize_layers,
        )
        if quant_mode is not None:
            if adjust_quantized is not None:
                value = adjust_quantized(mlx_key, value)
            quant_value = (
                mx.contiguous(value.T) if quant_pretransposed(transformer_quantize) else value
            )
            if transformer_dtype is not None and quant_value.dtype == mx.bfloat16:
                quant_value = quant_value.astype(transformer_dtype)
            group_size, bits = quant_defaults(
                quant_mode,
                video_ff_quantize_group_size,
                video_ff_quantize_bits,
            )
            quantized = mx.quantize(quant_value, group_size, bits, mode=quant_mode)
            base_key = f"{QUANT_KEY_PREFIX}{mlx_key}"[: -len(".weight")]
            tensors = [(f"{base_key}.weight", quantized[0]), (f"{base_key}.scales", quantized[1])]
            if len(quantized) > 2:
                tensors.append((f"{base_key}.biases", quantized[2]))
            return Conversion(tuple(tensors))

        logical_layout_key = layout_weight_key(
            mlx_key,
            video_ff_layout_specs=video_ff_layout_specs,
            video_ff_layout_layers=video_ff_layout_layers,
            video_attn_layout_specs=video_attn_layout_specs,
            video_attn_layout_layers=video_attn_layout_layers,
            audio_ff_layout_specs=audio_ff_layout_specs,
            audio_ff_layout_layers=audio_ff_layout_layers,
            audio_attn_layout_specs=audio_attn_layout_specs,
            audio_attn_layout_layers=audio_attn_layout_layers,
            adaln_pretranspose=adaln_pretranspose,
        )
        target_dtype = ff_dtype_for_key(
            mlx_key,
            video_ff_dtype=video_ff_dtype,
            audio_ff_dtype=audio_ff_dtype,
        )
        if target_dtype is None and transformer_dtype is not None and value.dtype == mx.bfloat16:
            target_dtype = transformer_dtype
        if target_dtype is not None:
            value = value.astype(target_dtype)
        if adjust_dense is not None:
            value = adjust_dense(mlx_key, value)
        if logical_layout_key is not None:
            target_key = f"{LAYOUT_KEY_PREFIX}{logical_layout_key}"
            stored = mx.contiguous(value.T)
        else:
            target_key = mlx_key
            stored = value
        if target_dtype is not None and stored.dtype != target_dtype:
            stored = stored.astype(target_dtype)
        return Conversion(((target_key, stored),))

    return TransformerPreparation(tuple(sources), convert)


@dataclass(frozen=True)
class FamilyPreparation:
    """Lazy conversion of one auxiliary weight family for model binding."""

    family: str
    source: Path
    sources: tuple[SourceTensor, ...]
    convert: Callable[[SourceTensor, mx.array], Conversion]


def family_source_preparation(
    source: Path | str,
    *,
    family: str,
    source_component: str | None,
    select: Callable[[str], bool] | None = None,
) -> FamilyPreparation:
    """Inspect once and attach lazy FP8 decoding and native layouts to a family."""
    path = Path(source).expanduser().resolve()
    inventory = inspect_safetensors(path)
    bindings = family_source_bindings(inventory, family=family, source_component=source_component)
    if select is not None:
        bindings = tuple(binding for binding in bindings if select(binding.target_key))
    sources = plan_sources(inspect_bindings(bindings, inventories={path: inventory}))
    header_dtypes = {tensor.name: tensor.dtype for tensor in inventory.tensors}
    fp8_keys, weight_scales, input_scales, comfy_tags = fp8_scale_companions(header_dtypes)
    if fp8_keys:
        reject_unsupported_fp8(header_dtypes)
    companions = weight_scales | input_scales | comfy_tags
    scales = (
        {key: value for key, value in load_weights(path).items() if key in weight_scales}
        if fp8_keys
        else {}
    )

    def convert(item: SourceTensor, raw: mx.array) -> Conversion:
        if item.source_key in companions:
            return Conversion()
        value = raw
        if item.source_key in fp8_keys:
            value = dequantize_fp8_weight(
                raw,
                item.source_key,
                scales.get(f"{item.source_key}_scale"),
                mx.bfloat16,
            )
        stored = bake_conv_layout(family, item.target_key, value)
        return Conversion(((item.target_key, stored),))

    return FamilyPreparation(family=family, source=path, sources=sources, convert=convert)
