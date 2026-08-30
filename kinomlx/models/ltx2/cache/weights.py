"""Checkpoint dtype handling and transformer cache quantization routing."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx.io.safetensors import load_weights, read_dtypes

from .schema import (
    TRANSFORMER_CACHE_QUANTIZE_MODES,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)

FP8_SAFETENSORS_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})
FP16_MAX = 65504.0
SUPPORTED_TRANSFORMER_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}

_VIDEO_FF_KEY_PATTERNS = ("ff.project_in.proj", "ff.project_out")
_AUDIO_FF_KEY_PATTERNS = (
    "audio_ff.project_in.proj",
    "audio_ff.project_out",
)
_ATTENTION_QUANT_PROJECTIONS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "to_gate_logits",
)
_BLOCK_ATTENTION_MODULES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)
CACHE_QUANTIZED_BLOCK_LINEAR_BASES = tuple(
    f"{attention}.{projection}"
    for attention in _BLOCK_ATTENTION_MODULES
    for projection in _ATTENTION_QUANT_PROJECTIONS
) + (
    "ff.project_in.proj",
    "ff.project_out",
    "audio_ff.project_in.proj",
    "audio_ff.project_out",
)


def resolve_transformer_dtype(
    dtype: str | mx.Dtype | None,
) -> mx.Dtype | None:
    """Resolve the Settings spelling for a transformer cache/compute dtype."""
    if dtype is None or not isinstance(dtype, str):
        return dtype
    try:
        return SUPPORTED_TRANSFORMER_DTYPES[dtype.lower()]
    except KeyError as exc:
        valid = ", ".join(SUPPORTED_TRANSFORMER_DTYPES)
        raise ValueError(f"Unsupported transformer dtype {dtype!r}; valid values: {valid}") from exc


def normalize_transformer_cache_dtypes(
    transformer_dtype: str | mx.Dtype | None,
    video_ff_dtype: str | mx.Dtype | None,
    audio_ff_dtype: str | mx.Dtype | None,
) -> tuple[mx.Dtype | None, mx.Dtype | None, mx.Dtype | None]:
    """Canonicalize global and targeted transformer cache dtype requests."""
    resolved_transformer = resolve_transformer_dtype(transformer_dtype)
    resolved_video = resolve_transformer_dtype(video_ff_dtype)
    resolved_audio = resolve_transformer_dtype(audio_ff_dtype)

    if resolved_transformer is None or resolved_transformer == mx.bfloat16:
        return (
            None,
            (
                None
                if resolved_video is not None and resolved_video == mx.bfloat16
                else resolved_video
            ),
            (
                None
                if resolved_audio is not None and resolved_audio == mx.bfloat16
                else resolved_audio
            ),
        )

    for label, override in (
        ("video FF", resolved_video),
        ("audio FF", resolved_audio),
    ):
        if override is not None and override != resolved_transformer:
            raise ValueError(
                f"{label} cache dtype {override} conflicts with transformer "
                f"dtype {resolved_transformer}"
            )
    return resolved_transformer, None, None


def ff_cache_dtype_for_key(
    mlx_key: str,
    *,
    video_ff_dtype: mx.Dtype | None,
    audio_ff_dtype: mx.Dtype | None = None,
) -> mx.Dtype | None:
    """Return the requested cache dtype for a video or audio FF parameter."""
    if video_ff_dtype is None and audio_ff_dtype is None:
        return None
    base = mlx_key.rsplit(".", 1)[0]
    if not base.startswith("transformer_blocks."):
        return None
    parts = base.split(".", 2)
    if len(parts) < 3:
        return None
    suffix = parts[2]
    if video_ff_dtype is not None and suffix in _VIDEO_FF_KEY_PATTERNS:
        return video_ff_dtype
    if audio_ff_dtype is not None and suffix in _AUDIO_FF_KEY_PATTERNS:
        return audio_ff_dtype
    return None


def cast_for_cache(
    value: mx.array,
    target_dtype: mx.Dtype,
    key: str,
    *,
    fp16_peak_sink: Callable[[float], None] | None = None,
) -> mx.array:
    """Cast a tensor while preventing silent FP16 overflow.

    Cache builders may provide ``fp16_peak_sink`` to retain the exact peak of
    the stored FP16 value. The live LoRA bridge later uses those one-time cache
    statistics instead of reading every base weight during each run.
    """
    if target_dtype == mx.float16 and value.dtype != mx.float16:
        max_abs = float(cast(int | float, mx.max(mx.abs(value.astype(mx.float32))).item()))
        if not math.isfinite(max_abs):
            raise ValueError(f"{key}: source contains non-finite values; cannot bake to float16")
        if max_abs > FP16_MAX:
            raise ValueError(
                f"{key}: max|w|={max_abs:.4g} exceeds the float16 range "
                f"({FP16_MAX:.0f}); use bfloat16 for this checkpoint"
            )
    stored = value.astype(target_dtype)
    if target_dtype == mx.float16 and fp16_peak_sink is not None:
        stored_peak = float(cast(int | float, mx.max(mx.abs(stored.astype(mx.float32))).item()))
        fp16_peak_sink(stored_peak)
    return stored


def checkpoint_has_fp8_tensors(weights_path: Path | str) -> bool:
    """Return whether a safetensors header declares any FP8 tensor."""
    try:
        dtypes = set(read_dtypes(weights_path).values())
    except OSError, ValueError:
        return False
    return bool(FP8_SAFETENSORS_DTYPES & dtypes)


def fp8_scale_companions(
    header_dtypes: Mapping[str, str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Classify FP8 weights and vLLM/ComfyUI companion tensors.

    The returned sets are ``(fp8, weight_scales, input_scales, comfy_tags)``.
    Weight scales are consumed during dequantization. Input scales and ComfyUI
    format tags describe activation/static-quantization state and are dropped
    after the cache has converted the weight to a runnable MLX dtype.
    """
    fp8_keys = {key for key, dtype in header_dtypes.items() if dtype in FP8_SAFETENSORS_DTYPES}
    scale_suffix = "_scale"
    for key in header_dtypes:
        if not key.endswith(scale_suffix):
            continue
        parent = key[: -len(scale_suffix)]
        if parent in header_dtypes and parent not in fp8_keys:
            raise ValueError(f"{key}: quantization scale has non-FP8 parent {parent}")
    weight_scale_keys = {
        key
        for key in header_dtypes
        if key.endswith(scale_suffix) and key[: -len(scale_suffix)] in fp8_keys
    }
    weight_suffix = ".weight"
    input_scale_keys = {
        key[: -len(weight_suffix)] + ".input_scale"
        for key in fp8_keys
        if key.endswith(weight_suffix)
        and key[: -len(weight_suffix)] + ".input_scale" in header_dtypes
    }
    comfy_suffix = ".comfy_quant"
    comfy_quant_keys = {
        key
        for key in header_dtypes
        if key.endswith(comfy_suffix) and key[: -len(comfy_suffix)] + ".weight" in fp8_keys
    }
    return (
        fp8_keys,
        weight_scale_keys,
        input_scale_keys,
        comfy_quant_keys,
    )


def iter_fp8_checkpoint_weights(
    weights_path: Path | str,
    value_dtype: mx.Dtype,
) -> Iterator[tuple[str, mx.array]]:
    """Yield an FP8 checkpoint dequantized with pure MLX operations."""
    header_dtypes = read_dtypes(weights_path)
    if any(dtype == "F8_E5M2" for dtype in header_dtypes.values()):
        raise ValueError("F8_E5M2 checkpoints are not supported; mx.from_fp8 decodes E4M3 only")
    fp8_keys, weight_scales, input_scales, comfy_tags = fp8_scale_companions(header_dtypes)
    raw_weights = load_weights(weights_path)
    for key, value in raw_weights.items():
        if key in weight_scales or key in input_scales or key in comfy_tags:
            continue
        stored = value
        if key in fp8_keys:
            scale_key = f"{key}_scale"
            if scale_key in weight_scales:
                scale = raw_weights[scale_key]
                if scale.ndim != 0:
                    raise ValueError(
                        f"{scale_key}: FP8 weight scale must be scalar, "
                        f"got shape {tuple(scale.shape)}"
                    )
                stored = cast_for_cache(
                    mx.from_fp8(value, dtype=mx.float32) * scale.astype(mx.float32),
                    value_dtype,
                    key,
                )
            else:
                stored = mx.from_fp8(value, dtype=value_dtype)
        yield key, stored


def iter_checkpoint_weights(
    weights_path: Path | str,
    *,
    fp8_value_dtype: mx.Dtype = mx.bfloat16,
) -> Iterator[tuple[str, mx.array]]:
    """Yield checkpoint tensors, transparently dequantizing FP8 sources."""
    if checkpoint_has_fp8_tensors(weights_path):
        yield from iter_fp8_checkpoint_weights(weights_path, fp8_value_dtype)
        return
    yield from load_weights(weights_path).items()


def selected_layers(
    layers: tuple[int, ...],
    *,
    num_layers: int = 48,
) -> set[int]:
    return set(layers) if layers else set(range(num_layers))


def quant_mode_for_target(
    specs: tuple[tuple[str, str], ...],
    target: str,
) -> str | None:
    for spec_target, mode in specs:
        if spec_target == target:
            return mode
    return None


def quant_defaults(
    mode: str,
    group_size: int | None,
    bits: int | None,
) -> tuple[int, int]:
    defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "mxfp8": (32, 8),
        "nvfp4": (16, 4),
    }
    if mode not in defaults:
        raise ValueError(f"Unsupported FF quantization mode: {mode}")
    default_group_size, default_bits = defaults[mode]
    normalized_group_size = default_group_size if group_size is None else group_size
    normalized_bits = default_bits if bits is None else bits
    if normalized_group_size <= 0 or normalized_bits <= 0:
        raise ValueError("quantization group size and bits must be positive")
    return normalized_group_size, normalized_bits


def block_linear_base_for_key(
    mlx_key: str,
) -> tuple[int, str, str] | None:
    parts = mlx_key.split(".")
    if len(parts) < 5 or parts[0] != "transformer_blocks" or parts[-1] != "weight":
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    base = ".".join(parts[2:-1])
    if base not in CACHE_QUANTIZED_BLOCK_LINEAR_BASES:
        return None
    return layer, base, parts[-1]


def ff_quant_mode_for_key(
    mlx_key: str,
    *,
    video_ff_quantize_specs: tuple[tuple[str, str], ...],
    video_ff_quantize_layers: tuple[int, ...],
) -> str | None:
    if not video_ff_quantize_specs:
        return None
    block_linear = block_linear_base_for_key(mlx_key)
    if block_linear is None:
        return None
    layer, base, _parameter = block_linear
    if layer not in selected_layers(video_ff_quantize_layers):
        return None
    targets = {
        "project_in": "ff.project_in.proj",
        "project_out": "ff.project_out",
    }
    for target, expected_base in targets.items():
        mode = quant_mode_for_target(video_ff_quantize_specs, target)
        if mode is not None and base == expected_base:
            return mode
    return None


def cache_quant_mode_for_key(
    mlx_key: str,
    *,
    transformer_cache_quantize: str,
    video_ff_quantize_specs: tuple[tuple[str, str], ...],
    video_ff_quantize_layers: tuple[int, ...],
) -> str | None:
    """Return the cache quantization mode for one converted transformer key."""
    if transformer_cache_quantize not in TRANSFORMER_CACHE_QUANTIZE_MODES:
        raise ValueError(
            f"Unsupported transformer cache quantization mode: {transformer_cache_quantize}"
        )
    if (
        transformer_cache_quantize
        in {
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
            TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
        }
        and block_linear_base_for_key(mlx_key) is not None
    ):
        return "mxfp8"
    return ff_quant_mode_for_key(
        mlx_key,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
    )


def cache_quant_pretransposed(transformer_cache_quantize: str) -> bool:
    return transformer_cache_quantize == TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE


__all__ = [
    "CACHE_QUANTIZED_BLOCK_LINEAR_BASES",
    "FP16_MAX",
    "FP8_SAFETENSORS_DTYPES",
    "SUPPORTED_TRANSFORMER_DTYPES",
    "block_linear_base_for_key",
    "cache_quant_mode_for_key",
    "cache_quant_pretransposed",
    "cast_for_cache",
    "checkpoint_has_fp8_tensors",
    "ff_cache_dtype_for_key",
    "ff_quant_mode_for_key",
    "fp8_scale_companions",
    "iter_checkpoint_weights",
    "iter_fp8_checkpoint_weights",
    "normalize_transformer_cache_dtypes",
    "quant_defaults",
    "quant_mode_for_target",
    "resolve_transformer_dtype",
    "selected_layers",
]
