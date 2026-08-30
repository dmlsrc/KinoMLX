"""Cache-baked Conv and transformer linear layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import mlx.core as mx

import kinomlx._mlx_nn as nn

from .policy import ensure_ff_pretranspose_for_dtype
from .weights import selected_layers

_ADALN_TOP_LEVEL_LINEARS = (
    "adaln_single.linear",
    "audio_adaln_single.linear",
    "prompt_adaln_single.linear",
    "audio_prompt_adaln_single.linear",
    "av_ca_video_scale_shift_adaln_single.linear",
    "av_ca_audio_scale_shift_adaln_single.linear",
    "av_ca_a2v_gate_adaln_single.linear",
    "av_ca_v2a_gate_adaln_single.linear",
)
_VIDEO_ATTN_MODULES = ("attn1", "attn2", "audio_to_video_attn")
_AUDIO_ATTN_MODULES = ("audio_attn1", "audio_attn2", "video_to_audio_attn")
_PROJECTION_TARGETS = ("to_out", "to_q", "to_k", "to_v", "to_gate_logits")
_ATTENTION_MODULES = _VIDEO_ATTN_MODULES + _AUDIO_ATTN_MODULES
_ATTENTION_CACHE_ATTRIBUTES = {
    "to_out": "_to_out_weight_t",
    "to_q": "_to_q_weight_t",
    "to_k": "_to_k_weight_t",
    "to_v": "_to_v_weight_t",
    "to_gate_logits": "_to_gate_logits_weight_t",
}


class _WeightSlot(Protocol):
    weight: mx.array


def _has_pretranspose(
    specs: tuple[tuple[str, str], ...],
    target: str,
) -> bool:
    return (target, "pretranspose") in specs


def layout_cache_key(
    mlx_key: str,
    *,
    video_ff_layout_specs: tuple[tuple[str, str], ...],
    video_ff_layout_layers: tuple[int, ...],
    video_attn_layout_specs: tuple[tuple[str, str], ...],
    video_attn_layout_layers: tuple[int, ...],
    audio_ff_layout_specs: tuple[tuple[str, str], ...] = (),
    audio_ff_layout_layers: tuple[int, ...] = (),
    audio_attn_layout_specs: tuple[tuple[str, str], ...] = (),
    audio_attn_layout_layers: tuple[int, ...] = (),
    adaln_pretranspose: bool = False,
) -> str | None:
    """Return the logical ``weight_t`` cache key requested for ``mlx_key``."""
    parts = mlx_key.split(".")
    if not parts or parts[-1] != "weight":
        return None

    if adaln_pretranspose and parts[0] != "transformer_blocks":
        base = ".".join(parts[:-1])
        if base in _ADALN_TOP_LEVEL_LINEARS:
            return f"{base}.weight_t"

    if len(parts) < 5 or parts[0] != "transformer_blocks":
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    suffix = ".".join(parts[2:])

    if layer in selected_layers(video_ff_layout_layers):
        if (
            _has_pretranspose(video_ff_layout_specs, "project_in")
            and suffix == "ff.project_in.proj.weight"
        ):
            return f"transformer_blocks.{layer}.ff.project_in.proj.weight_t"
        if (
            _has_pretranspose(video_ff_layout_specs, "project_out")
            and suffix == "ff.project_out.weight"
        ):
            return f"transformer_blocks.{layer}.ff.project_out.weight_t"

    if layer in selected_layers(video_attn_layout_layers):
        for projection in _PROJECTION_TARGETS:
            if not _has_pretranspose(video_attn_layout_specs, projection):
                continue
            for module in _VIDEO_ATTN_MODULES:
                if suffix == f"{module}.{projection}.weight":
                    return f"transformer_blocks.{layer}.{module}.{projection}.weight_t"

    if layer in selected_layers(audio_ff_layout_layers):
        if (
            _has_pretranspose(audio_ff_layout_specs, "project_in")
            and suffix == "audio_ff.project_in.proj.weight"
        ):
            return f"transformer_blocks.{layer}.audio_ff.project_in.proj.weight_t"
        if (
            _has_pretranspose(audio_ff_layout_specs, "project_out")
            and suffix == "audio_ff.project_out.weight"
        ):
            return f"transformer_blocks.{layer}.audio_ff.project_out.weight_t"

    if layer in selected_layers(audio_attn_layout_layers):
        for projection in _PROJECTION_TARGETS:
            if not _has_pretranspose(audio_attn_layout_specs, projection):
                continue
            for module in _AUDIO_ATTN_MODULES:
                if suffix == f"{module}.{projection}.weight":
                    return f"transformer_blocks.{layer}.{module}.{projection}.weight_t"
    return None


def bake_conv_layout_for_family(
    family: str,
    weights: Mapping[str, mx.array],
) -> dict[str, mx.array]:
    """Materialize PyTorch Conv weights in the MLX-native family layout.

    Video Conv3d tensors become OTHWI and audio Conv2d tensors become OHWI.
    Vocoder Conv1d and ConvTranspose1d need structure-specific permutations,
    so they remain unchanged for their loader to distinguish.
    """
    permutation: tuple[int, ...]
    if family == "video_vae":
        target_ndim = 5
        permutation = (0, 2, 3, 4, 1)
    elif family == "audio_vae":
        target_ndim = 4
        permutation = (0, 2, 3, 1)
    else:
        return dict(weights)

    converted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if value.ndim == target_ndim and key.endswith(".weight"):
            converted[key] = mx.contiguous(value.transpose(*permutation))
        else:
            converted[key] = value
    return converted


def clear_block_layout_weights(block: nn.Module) -> None:
    """Release previously installed layout tensors before rebinding a block."""
    for feed_forward_name in ("ff", "audio_ff"):
        feed_forward = getattr(block, feed_forward_name, None)
        if feed_forward is None:
            continue
        for attribute, linear_path in (
            ("_project_in_weight_t", ("project_in", "proj")),
            ("_project_out_weight_t", ("project_out",)),
        ):
            if hasattr(feed_forward, attribute):
                cached = getattr(feed_forward, attribute)
                linear = feed_forward
                for part in linear_path:
                    linear = getattr(linear, part, None)
                    if linear is None:
                        break
                _restore_empty_weight(linear, cached)
                setattr(feed_forward, attribute, None)

    for attention_name in _ATTENTION_MODULES:
        attention = getattr(block, attention_name, None)
        if attention is None:
            continue
        for projection, attribute in _ATTENTION_CACHE_ATTRIBUTES.items():
            if hasattr(attention, attribute):
                cached = getattr(attention, attribute)
                _restore_empty_weight(getattr(attention, projection, None), cached)
                setattr(attention, attribute, None)


def clear_layout_weights(model: object) -> None:
    """Release every private pretransposed tensor before a cache rebind."""
    blocks = cast(Sequence[nn.Module], getattr(model, "transformer_blocks", ()))
    for block in blocks:
        clear_block_layout_weights(block)

    for module_name in _ADALN_TOP_LEVEL_LINEARS:
        module = getattr(model, module_name.rsplit(".", 1)[0], None)
        if module is not None and hasattr(module, "_linear_weight_t"):
            cached = module._linear_weight_t
            _restore_empty_weight(
                getattr(module, "linear", None),
                cached if isinstance(cached, mx.array) else None,
            )
            module._linear_weight_t = None


def _restore_empty_weight(module: object, cached: mx.array | None) -> None:
    """Restore a zero-sized update slot after a layout loader deleted weight."""
    if isinstance(module, nn.Module) and cached is not None and "weight" not in module:
        cast(_WeightSlot, module).weight = mx.zeros((0, 0), dtype=cached.dtype)


def _drop_weight(module: object) -> None:
    if isinstance(module, nn.Module) and "weight" in module:
        delattr(module, "weight")


def install_block_layout_weight(
    block: nn.Module,
    layout_key: str,
    value: mx.array,
) -> None:
    """Install one pretransposed tensor on a resident transformer block."""
    feed_forward_targets = {
        "ff.project_in.proj.weight_t": (
            "ff",
            "_project_in_weight_t",
            ("project_in", "proj"),
        ),
        "ff.project_out.weight_t": (
            "ff",
            "_project_out_weight_t",
            ("project_out",),
        ),
        "audio_ff.project_in.proj.weight_t": (
            "audio_ff",
            "_project_in_weight_t",
            ("project_in", "proj"),
        ),
        "audio_ff.project_out.weight_t": (
            "audio_ff",
            "_project_out_weight_t",
            ("project_out",),
        ),
    }
    target = feed_forward_targets.get(layout_key)
    if target is not None:
        owner_name, cache_attribute, linear_path = target
        owner = getattr(block, owner_name, None)
        if owner is None:
            raise ValueError(f"Missing module for cache key: {layout_key}")
        setattr(owner, cache_attribute, value)
        linear = owner
        for part in linear_path:
            linear = getattr(linear, part, None)
            if linear is None:
                break
        _drop_weight(linear)
        return

    for attention_name in _ATTENTION_MODULES:
        for projection, cache_attribute in _ATTENTION_CACHE_ATTRIBUTES.items():
            if layout_key != f"{attention_name}.{projection}.weight_t":
                continue
            attention = getattr(block, attention_name, None)
            if attention is None:
                raise ValueError(f"Missing attention module for cache key: {layout_key}")
            setattr(attention, cache_attribute, value)
            _drop_weight(getattr(attention, projection, None))
            return
    raise ValueError(f"Unsupported transformer cache layout key: {layout_key}")


def install_top_level_layout_weight(
    model: object,
    layout_key: str,
    value: mx.array,
) -> None:
    """Install a top-level AdaLayerNormSingle pretransposed linear weight."""
    suffix = ".linear.weight_t"
    if not layout_key.endswith(suffix):
        raise ValueError(f"Unsupported top-level layout key: {layout_key}")
    base = layout_key[: -len(suffix)]
    module = getattr(model, base, None)
    if module is None:
        raise ValueError(f"Missing module for layout key: {layout_key}")
    if not hasattr(module, "_linear_weight_t"):
        raise ValueError(f"Module {base} does not support the cached AdaLN layout")
    module._linear_weight_t = value
    _drop_weight(getattr(module, "linear", None))


def install_layout_weight(
    model: object,
    layout_key: str,
    value: mx.array,
) -> None:
    """Install a top-level or per-block pretransposed cache tensor."""
    parts = layout_key.split(".")
    if not parts:
        raise ValueError(f"Invalid transformer cache layout key: {layout_key}")
    if parts[0] != "transformer_blocks":
        install_top_level_layout_weight(model, layout_key, value)
        return
    if len(parts) < 5:
        raise ValueError(f"Invalid transformer cache layout key: {layout_key}")
    try:
        layer = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid transformer cache layout key: {layout_key}") from exc
    blocks = cast(Sequence[nn.Module], getattr(model, "transformer_blocks", ()))
    if layer >= len(blocks):
        raise ValueError(f"Layout cache has block {layer}, but model only has {len(blocks)} blocks")
    install_block_layout_weight(
        blocks[layer],
        ".".join(parts[2:]),
        value,
    )


def get_layout_weight(
    model: object,
    layout_key: str,
) -> mx.array | None:
    """Return one installed private layout tensor without flattening the model."""
    top_level_suffix = ".linear.weight_t"
    if layout_key.endswith(top_level_suffix) and not layout_key.startswith("transformer_blocks."):
        module_name = layout_key[: -len(top_level_suffix)]
        module = getattr(model, module_name, None)
        return None if module is None else getattr(module, "_linear_weight_t", None)

    prefix = "transformer_blocks."
    if not layout_key.startswith(prefix):
        return None
    remainder = layout_key[len(prefix) :]
    index_text, separator, block_key = remainder.partition(".")
    if not separator:
        return None
    try:
        blocks = cast(Sequence[nn.Module], getattr(model, "transformer_blocks", ()))
        block = blocks[int(index_text)]
    except IndexError, TypeError, ValueError:
        return None

    feed_forward_targets = {
        "ff.project_in.proj.weight_t": ("ff", "_project_in_weight_t"),
        "ff.project_out.weight_t": ("ff", "_project_out_weight_t"),
        "audio_ff.project_in.proj.weight_t": (
            "audio_ff",
            "_project_in_weight_t",
        ),
        "audio_ff.project_out.weight_t": (
            "audio_ff",
            "_project_out_weight_t",
        ),
    }
    feed_forward = feed_forward_targets.get(block_key)
    if feed_forward is not None:
        owner = getattr(block, feed_forward[0], None)
        return None if owner is None else getattr(owner, feed_forward[1], None)

    for attention_name in _ATTENTION_MODULES:
        for projection, attribute in _ATTENTION_CACHE_ATTRIBUTES.items():
            if block_key != f"{attention_name}.{projection}.weight_t":
                continue
            attention = getattr(block, attention_name, None)
            return None if attention is None else getattr(attention, attribute, None)
    return None


__all__ = [
    "bake_conv_layout_for_family",
    "clear_block_layout_weights",
    "clear_layout_weights",
    "ensure_ff_pretranspose_for_dtype",
    "get_layout_weight",
    "install_block_layout_weight",
    "install_layout_weight",
    "install_top_level_layout_weight",
    "layout_cache_key",
]
