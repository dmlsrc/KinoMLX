"""Checkpoint dtype handling and transformer weight quantization routing."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import load_weights, read_dtypes
from kinomlx.models.ltx2.weights.schema import (
    TRANSFORMER_QUANTIZE_MODES,
    TransformerQuantize,
)

FP8_SAFETENSORS_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})

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
QUANTIZED_BLOCK_LINEAR_BASES = (
    *tuple(
        f"{attention}.{projection}"
        for attention in _BLOCK_ATTENTION_MODULES
        for projection in _ATTENTION_QUANT_PROJECTIONS
    ),
    "ff.project_in.proj",
    "ff.project_out",
    "audio_ff.project_in.proj",
    "audio_ff.project_out",
)


def ff_dtype_for_key(
    mlx_key: str,
    *,
    video_ff_dtype: mx.Dtype | None,
    audio_ff_dtype: mx.Dtype | None = None,
) -> mx.Dtype | None:
    """Return the requested weight dtype for a video or audio FF parameter."""
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
    after preparation converts the weight to a runnable MLX dtype.
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


def reject_unsupported_fp8(header_dtypes: Mapping[str, str]) -> None:
    """Refuse the FP8 flavor ``mx.from_fp8`` cannot decode."""
    if any(dtype == "F8_E5M2" for dtype in header_dtypes.values()):
        raise ValueError("F8_E5M2 checkpoints are not supported; mx.from_fp8 decodes E4M3 only")


def dequantize_fp8_weight(
    value: mx.array,
    key: str,
    scale: mx.array | None,
    value_dtype: mx.Dtype,
) -> mx.array:
    """Decode one FP8 tensor, applying its scalar weight scale when it has one."""
    if scale is None:
        return mx.from_fp8(value, dtype=value_dtype)
    if scale.ndim != 0:
        raise ValueError(
            f"{key}_scale: FP8 weight scale must be scalar, got shape {tuple(scale.shape)}"
        )
    return (mx.from_fp8(value, dtype=mx.float32) * scale.astype(mx.float32)).astype(value_dtype)


def iter_fp8_checkpoint_weights(
    weights_path: Path | str,
    value_dtype: mx.Dtype,
) -> Iterator[tuple[str, mx.array]]:
    """Yield an FP8 checkpoint dequantized with pure MLX operations."""
    header_dtypes = read_dtypes(weights_path)
    reject_unsupported_fp8(header_dtypes)
    fp8_keys, weight_scales, input_scales, comfy_tags = fp8_scale_companions(header_dtypes)
    raw_weights = load_weights(weights_path)
    # Pop consumed entries so evaluated buffers do not accumulate in the dict;
    # scale companions are tiny and stay for lookup.
    for key in sorted(raw_weights):
        if key in weight_scales or key in input_scales or key in comfy_tags:
            continue
        value = raw_weights.pop(key)
        if key in fp8_keys:
            value = dequantize_fp8_weight(value, key, raw_weights.get(f"{key}_scale"), value_dtype)
        yield key, value


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
    if base not in QUANTIZED_BLOCK_LINEAR_BASES:
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


def quant_mode_for_key(
    mlx_key: str,
    *,
    transformer_quantize: str,
    transformer_quantize_exclude_layers: tuple[int, ...] = (),
    video_ff_quantize_specs: tuple[tuple[str, str], ...],
    video_ff_quantize_layers: tuple[int, ...],
) -> str | None:
    """Return the quantization mode for one converted transformer key."""
    if transformer_quantize not in TRANSFORMER_QUANTIZE_MODES:
        raise ValueError(
            f"Unsupported transformer weight quantization mode: {transformer_quantize}"
        )
    block = block_linear_base_for_key(mlx_key)
    if block is not None and block[0] in transformer_quantize_exclude_layers:
        return None
    if (
        transformer_quantize
        in {
            TransformerQuantize.MXFP8_BLOCKS,
            TransformerQuantize.MXFP8_BLOCKS_PRETRANSPOSE,
        }
        and block_linear_base_for_key(mlx_key) is not None
    ):
        return "mxfp8"
    return ff_quant_mode_for_key(
        mlx_key,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
    )


def quant_pretransposed(transformer_quantize: str) -> bool:
    return transformer_quantize == TransformerQuantize.MXFP8_BLOCKS_PRETRANSPOSE


__all__ = [
    "FP8_SAFETENSORS_DTYPES",
    "QUANTIZED_BLOCK_LINEAR_BASES",
    "block_linear_base_for_key",
    "checkpoint_has_fp8_tensors",
    "dequantize_fp8_weight",
    "ff_dtype_for_key",
    "ff_quant_mode_for_key",
    "fp8_scale_companions",
    "iter_fp8_checkpoint_weights",
    "quant_defaults",
    "quant_mode_for_key",
    "quant_mode_for_target",
    "quant_pretransposed",
    "reject_unsupported_fp8",
    "selected_layers",
]
