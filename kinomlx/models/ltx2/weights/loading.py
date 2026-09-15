"""Bind and stream prepared transformer weights."""

import gc
import logging
from collections.abc import Callable
from typing import Protocol, cast

import mlx.core as mx
from mlx.utils import tree_flatten

import kinomlx._mlx_nn as nn
from kinomlx.models.ltx2.weights.dtypes import (
    resolve_transformer_dtype,
)
from kinomlx.models.ltx2.weights.keys import flatten_to_nested
from kinomlx.models.ltx2.weights.layout import clear_layout_weights, install_layout_weight
from kinomlx.models.ltx2.weights.quantization import (
    prepare_block_quantized_linears,
    quant_bases_for_block_keys,
    restore_block_quantized_linears,
)
from kinomlx.models.ltx2.weights.schema import (
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
    TRANSFORMER_QUANTIZE_OFF,
)
from kinomlx.models.ltx2.weights.validation import validate_model_weight_graph

from .streaming import TransformerBlockStreamer

_log = logging.getLogger(__name__)


class _TransformerBinding(Protocol):
    transformer_blocks: list[nn.Module]
    transformer_block_streamer: TransformerBlockStreamer | None


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, list) or not all(isinstance(block, nn.Module) for block in blocks):
        raise TypeError("transformer weight binding requires a list of MLX transformer blocks")
    return cast(list[nn.Module], blocks)


def _parameter_schema(block: nn.Module) -> frozenset[str]:
    flattened = cast(list[tuple[str, object]], tree_flatten(block.parameters()))
    return frozenset(key for key, _value in flattened)


def _validate_model_compute_dtype(
    model: nn.Module,
    transformer_dtype: str | mx.Dtype | None,
) -> None:
    actual = resolve_transformer_dtype(getattr(model, "compute_dtype", None))
    if actual is None:
        return
    requested = resolve_transformer_dtype(transformer_dtype) or mx.bfloat16
    if actual != requested:
        raise ValueError(
            f"model compute dtype {actual} does not match transformer weight dtype {requested}"
        )


def bind_transformer_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    *,
    include_audio: bool | None = None,
    transformer_quantize: str = TRANSFORMER_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
) -> tuple[int, int, int]:
    """Bind prepared tensors from a lazy source graph."""
    validate_model_weight_graph(
        model,
        weights,
        include_audio=include_audio,
    )
    normal_weights: dict[str, mx.array] = {}
    layout_weights: dict[str, mx.array] = {}
    quant_weights: dict[str, mx.array] = {}
    normal_keys_by_block: dict[int, list[tuple[str, str]]] = {}
    quant_keys_by_block: dict[int, list[tuple[str, str]]] = {}
    layout_block_indices: set[int] = set()

    for key, value in weights.items():
        if key.startswith(LAYOUT_KEY_PREFIX):
            logical_key = key[len(LAYOUT_KEY_PREFIX) :]
            layout_weights[logical_key] = value
            parts = logical_key.split(".")
            if len(parts) >= 3 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                layout_block_indices.add(int(parts[1]))
            continue
        if key.startswith(QUANT_KEY_PREFIX):
            logical_key = key[len(QUANT_KEY_PREFIX) :]
            quant_weights[logical_key] = value
            target_table = quant_keys_by_block
        else:
            logical_key = key
            normal_weights[key] = value
            target_table = normal_keys_by_block
        parts = logical_key.split(".")
        if len(parts) < 3 or parts[0] != "transformer_blocks":
            continue
        try:
            block_index = int(parts[1])
        except ValueError:
            continue
        target_table.setdefault(block_index, []).append((key, ".".join(parts[2:])))

    blocks = _transformer_blocks(model)
    block_indices = set(normal_keys_by_block) | set(quant_keys_by_block) | layout_block_indices
    invalid_block_indices = sorted(index for index in block_indices if index >= len(blocks))
    if invalid_block_indices:
        raise ValueError(
            f"Transformer weight has block {invalid_block_indices[0]}, but model "
            f"only has {len(blocks)} blocks"
        )
    missing_block_indices = sorted(set(range(len(blocks))) - block_indices)
    if missing_block_indices:
        raise ValueError(
            f"Transformer weight is missing block weights for layer {missing_block_indices[0]}"
        )

    # Clear every private layout slot before rebinding so stale tensors cannot
    # win over newly loaded normal, quantized, or differently laid-out weights.
    clear_layout_weights(model)
    for block_index in block_indices:
        block = blocks[block_index]
        quant_keys = quant_keys_by_block.get(block_index, ())
        quant_bases = quant_bases_for_block_keys(quant_keys)
        restore_block_quantized_linears(block, keep_bases=quant_bases)
        if quant_bases:
            prepare_block_quantized_linears(
                block,
                quant_bases,
                quant_keys,
                normal_keys_by_block.get(block_index, ()),
                transformer_quantize=transformer_quantize,
                quantization_specs=video_ff_quantize_specs,
                group_size=video_ff_quantize_group_size,
                bits=video_ff_quantize_bits,
            )

    # Bind a single block tree: separate normal/packed updates leave holes
    # in the block list when quantization excludes the first or middle blocks.
    if normal_weights or quant_weights:
        model.update(flatten_to_nested(normal_weights | quant_weights))
    for key, value in layout_weights.items():
        install_layout_weight(model, key, value)

    loaded_count = len(normal_weights) + len(layout_weights) + len(quant_weights)
    layout_count = len(layout_weights)
    quant_count = len(quant_weights)
    del weights, normal_weights, layout_weights, quant_weights
    gc.collect()
    return loaded_count, layout_count, quant_count


def bind_transformer(
    model: nn.Module,
    *,
    weight_loader: Callable[[int | None], dict[str, mx.array]],
    transformer_dtype: str | mx.Dtype | None = None,
    include_audio: bool | None = None,
    resident_blocks: int | None = None,
    transformer_quantize: str = TRANSFORMER_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
) -> tuple[int, int, int]:
    """Validate and bind a reloadable lazy source."""
    _validate_model_compute_dtype(model, transformer_dtype)
    if resident_blocks is None:
        return bind_transformer_weights(
            model,
            weight_loader(None),
            include_audio=include_audio,
            transformer_quantize=transformer_quantize,
            video_ff_quantize_specs=video_ff_quantize_specs,
            video_ff_quantize_group_size=video_ff_quantize_group_size,
            video_ff_quantize_bits=video_ff_quantize_bits,
        )

    if resident_blocks <= 0:
        raise ValueError("resident_blocks must be positive")
    blocks = _transformer_blocks(model)
    model_block_count = len(blocks)
    if resident_blocks > model_block_count:
        raise ValueError(
            f"resident_blocks={resident_blocks} exceeds model block count {model_block_count}"
        )
    expected_block_schemas = tuple(_parameter_schema(block) for block in blocks)
    streamer = TransformerBlockStreamer(
        weight_loader=weight_loader,
        expected_model=model,
        include_audio=include_audio,
        expected_block_schemas=expected_block_schemas,
        transformer_quantize=transformer_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
    )
    try:
        weights = streamer.take_non_block_weights()
        non_block_weights: dict[str, mx.array] = {}
        layout_weights: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith(LAYOUT_KEY_PREFIX):
                layout_weights[key[len(LAYOUT_KEY_PREFIX) :]] = value
            elif key.startswith(QUANT_KEY_PREFIX):
                raise ValueError(f"Unexpected non-block quantized weight key: {key}")
            else:
                non_block_weights[key] = value

        clear_layout_weights(model)
        if non_block_weights:
            model.update(flatten_to_nested(non_block_weights))
        for key, value in layout_weights.items():
            install_layout_weight(model, key, value)
        binding = cast(_TransformerBinding, model)
        binding.transformer_blocks = blocks[:resident_blocks]
        binding.transformer_block_streamer = streamer
        del weights, non_block_weights, layout_weights
        gc.collect()
    except BaseException:
        streamer.close()
        raise
    _log.info(
        "Loaded transformer weight: %d streamed block tensors (%d layout, "
        "%d quantized), %d/%d blocks resident",
        streamer.loaded_count,
        streamer.layout_count,
        streamer.quant_count,
        resident_blocks,
        streamer.block_count,
    )
    return streamer.loaded_count, streamer.layout_count, streamer.quant_count
