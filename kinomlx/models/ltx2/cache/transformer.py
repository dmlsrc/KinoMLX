"""Ensure, bind, and stream transformer cache artifacts."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
from mlx.utils import tree_flatten

import kinomlx._mlx_nn as nn
from kinomlx.reporting import Reporter

from ..metadata import TransformerConstructorConfig, resolve_transformer_bindings
from .building import build_transformer_cache
from .keys import flatten_to_nested
from .layout import (
    clear_layout_weights,
    install_layout_weight,
)
from .policy import (
    DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    LayoutSpecs,
    normalize_transformer_layout_policy,
)
from .quantization import (
    prepare_block_quantized_linears,
    quant_bases_for_block_keys,
    restore_block_quantized_linears,
)
from .schema import (
    DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
    TRANSFORMER_CACHE_QUANTIZE_OFF,
    TransformerCacheOptions,
    transformer_cache_paths,
    transformer_fp16_ranges_path,
)
from .storage import (
    cache_artifacts_exist,
    existing_cache_shards,
    load_cache_weights,
    load_transformer_fp16_ranges,
    metadata_matches,
)
from .streaming import TransformerBlockStreamer
from .validation import validate_model_cache_graph
from .weights import normalize_transformer_cache_dtypes, resolve_transformer_dtype

_log = logging.getLogger(__name__)


class _TransformerCacheBinding(Protocol):
    transformer_blocks: list[nn.Module]
    transformer_block_streamer: TransformerBlockStreamer | None


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, list) or not all(isinstance(block, nn.Module) for block in blocks):
        raise TypeError("transformer cache binding requires a list of MLX transformer blocks")
    return cast(list[nn.Module], blocks)


def _parameter_schema(block: nn.Module) -> frozenset[str]:
    flattened = cast(list[tuple[str, object]], tree_flatten(block.parameters()))
    return frozenset(key for key, _value in flattened)


@dataclass(frozen=True)
class TransformerCacheResult:
    """Result metadata for a transformer cache build or load."""

    cache_path: Path
    rebuilt: bool
    loaded_count: int
    layout_count: int
    quant_count: int = 0


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
            f"model compute dtype {actual} does not match transformer cache dtype {requested}"
        )


def load_transformer_cache(
    model: nn.Module,
    cache_file: Path,
    *,
    include_audio: bool | None = None,
    require_graph: bool = True,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
) -> tuple[int, int, int]:
    """Load a converted transformer cache into an existing model instance.

    ``require_graph=False`` is reserved for intentionally partial unit fixtures.
    """
    cached_weights = load_cache_weights(Path(cache_file))
    validate_model_cache_graph(
        model,
        cached_weights,
        include_audio=include_audio,
        require_graph=require_graph,
    )
    normal_weights: dict[str, mx.array] = {}
    layout_weights: dict[str, mx.array] = {}
    quant_weights: dict[str, mx.array] = {}
    normal_keys_by_block: dict[int, list[tuple[str, str]]] = {}
    quant_keys_by_block: dict[int, list[tuple[str, str]]] = {}
    layout_block_indices: set[int] = set()

    for key, value in cached_weights.items():
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
            f"Transformer cache has block {invalid_block_indices[0]}, but model "
            f"only has {len(blocks)} blocks"
        )
    missing_block_indices = sorted(set(range(len(blocks))) - block_indices)
    if missing_block_indices:
        raise ValueError(
            f"Transformer cache is missing block weights for layer {missing_block_indices[0]}"
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
                transformer_cache_quantize=transformer_cache_quantize,
                quantization_specs=video_ff_quantize_specs,
                group_size=video_ff_quantize_group_size,
                bits=video_ff_quantize_bits,
            )

    if normal_weights:
        model.update(flatten_to_nested(normal_weights))
    if quant_weights:
        model.update(flatten_to_nested(quant_weights))
    for key, value in layout_weights.items():
        install_layout_weight(model, key, value)

    loaded_count = len(normal_weights) + len(layout_weights) + len(quant_weights)
    layout_count = len(layout_weights)
    quant_count = len(quant_weights)
    del cached_weights, normal_weights, layout_weights, quant_weights
    gc.collect()
    return loaded_count, layout_count, quant_count


def ensure_transformer_cache(
    weights_path: Path | str,
    *,
    transformer_dtype: str | mx.Dtype | None = None,
    cache_mode: str,
    cache_root: Path | str | None,
    include_audio: bool,
    video_ff_layout_specs: LayoutSpecs = DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: LayoutSpecs = DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    video_attn_layout_layers: tuple[int, ...] = (),
    audio_ff_layout_specs: LayoutSpecs | None = None,
    audio_ff_layout_layers: tuple[int, ...] | None = None,
    audio_attn_layout_specs: LayoutSpecs | None = None,
    audio_attn_layout_layers: tuple[int, ...] | None = None,
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    reporter: Reporter | None = None,
    shard_limit_bytes: int = DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
    constructor_config: TransformerConstructorConfig | None = None,
) -> TransformerCacheResult:
    """Build a matching transformer cache when no valid artifact is present."""
    if cache_mode not in {"auto", "rebuild"}:
        raise ValueError(f"Unsupported transformer cache mode: {cache_mode}")
    transformer_dtype, video_ff_dtype, audio_ff_dtype = normalize_transformer_cache_dtypes(
        transformer_dtype,
        video_ff_dtype,
        audio_ff_dtype,
    )
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        if video_ff_quantize_specs:
            raise ValueError(
                "whole-transformer cache quantization and targeted FF "
                "quantization cannot be combined"
            )
        if video_ff_dtype is not None or audio_ff_dtype is not None:
            raise ValueError(
                "whole-transformer cache quantization and targeted FF dtype "
                "overrides cannot be combined"
            )
    layout_policy = normalize_transformer_layout_policy(
        include_audio=include_audio,
        transformer_dtype=transformer_dtype,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        layouts_enabled=(transformer_cache_quantize == TRANSFORMER_CACHE_QUANTIZE_OFF),
    )
    options: TransformerCacheOptions = {
        "transformer_dtype": transformer_dtype,
        "include_audio": include_audio,
        "video_ff_layout_specs": layout_policy.video_ff_specs,
        "video_ff_layout_layers": layout_policy.video_ff_layers,
        "video_attn_layout_specs": layout_policy.video_attn_specs,
        "video_attn_layout_layers": layout_policy.video_attn_layers,
        "audio_ff_layout_specs": layout_policy.audio_ff_specs,
        "audio_ff_layout_layers": layout_policy.audio_ff_layers,
        "audio_attn_layout_specs": layout_policy.audio_attn_specs,
        "audio_attn_layout_layers": layout_policy.audio_attn_layers,
        "adaln_pretranspose": adaln_pretranspose,
        "transformer_cache_quantize": transformer_cache_quantize,
        "video_ff_quantize_specs": video_ff_quantize_specs,
        "video_ff_quantize_layers": layout_policy.video_ff_quantize_layers,
        "video_ff_quantize_group_size": video_ff_quantize_group_size,
        "video_ff_quantize_bits": video_ff_quantize_bits,
        "video_ff_dtype": video_ff_dtype,
        "audio_ff_dtype": audio_ff_dtype,
        "constructor_identity": (
            None if constructor_config is None else constructor_config.cache_identity()
        ),
    }
    cache_file, metadata_file, payload = transformer_cache_paths(
        weights_path,
        cache_root,
        **options,
    )
    artifacts_valid = cache_artifacts_exist(
        cache_file,
        metadata_file=metadata_file,
    )
    metadata_valid = metadata_matches(metadata_file, payload)
    fp16_ranges_file = transformer_fp16_ranges_path(cache_file)
    fp16_ranges_valid = False
    if artifacts_valid and fp16_ranges_file.is_file():
        try:
            load_transformer_fp16_ranges(cache_file)
        except ValueError:
            pass
        else:
            fp16_ranges_valid = True
    valid = artifacts_valid and metadata_valid and fp16_ranges_valid
    rebuilt = cache_mode == "rebuild" or not valid
    if rebuilt:
        _log.info("Transformer cache: building %s", cache_file)
        source_bindings = None
        if constructor_config is not None:
            source_bindings = {
                binding.source_key: binding.target_key
                for binding in resolve_transformer_bindings(
                    Path(weights_path),
                    constructor_config,
                    include_audio=include_audio,
                )
            }
        build_transformer_cache(
            weights_path,
            cache_file,
            metadata_file,
            payload,
            reporter=reporter,
            shard_limit_bytes=shard_limit_bytes,
            source_bindings=source_bindings,
            transformer_dtype=transformer_dtype,
            include_audio=include_audio,
            video_ff_layout_specs=layout_policy.video_ff_specs,
            video_ff_layout_layers=layout_policy.video_ff_layers,
            video_attn_layout_specs=layout_policy.video_attn_specs,
            video_attn_layout_layers=layout_policy.video_attn_layers,
            audio_ff_layout_specs=layout_policy.audio_ff_specs,
            audio_ff_layout_layers=layout_policy.audio_ff_layers,
            audio_attn_layout_specs=layout_policy.audio_attn_specs,
            audio_attn_layout_layers=layout_policy.audio_attn_layers,
            adaln_pretranspose=adaln_pretranspose,
            transformer_cache_quantize=transformer_cache_quantize,
            video_ff_quantize_specs=video_ff_quantize_specs,
            video_ff_quantize_layers=layout_policy.video_ff_quantize_layers,
            video_ff_quantize_group_size=video_ff_quantize_group_size,
            video_ff_quantize_bits=video_ff_quantize_bits,
            video_ff_dtype=video_ff_dtype,
            audio_ff_dtype=audio_ff_dtype,
        )
    else:
        shard_count = len(existing_cache_shards(cache_file))
        _log.info(
            "Transformer cache: using %s (%d shards)",
            cache_file.parent,
            shard_count,
        )
    return TransformerCacheResult(
        cache_path=cache_file,
        rebuilt=rebuilt,
        loaded_count=0,
        layout_count=0,
    )


def load_transformer_weights_cached(
    model: nn.Module,
    weights_path: Path | str,
    *,
    transformer_dtype: str | mx.Dtype | None = None,
    cache_mode: str,
    cache_root: Path | str | None,
    include_audio: bool,
    video_ff_layout_specs: LayoutSpecs = DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: LayoutSpecs = DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    video_attn_layout_layers: tuple[int, ...] = (),
    audio_ff_layout_specs: LayoutSpecs | None = None,
    audio_ff_layout_layers: tuple[int, ...] | None = None,
    audio_attn_layout_specs: LayoutSpecs | None = None,
    audio_attn_layout_layers: tuple[int, ...] | None = None,
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    reporter: Reporter | None = None,
    shard_limit_bytes: int = DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
    constructor_config: TransformerConstructorConfig | None = None,
) -> TransformerCacheResult:
    """Ensure and load a full-residency transformer cache."""
    _validate_model_compute_dtype(model, transformer_dtype)
    result = ensure_transformer_cache(
        weights_path,
        transformer_dtype=transformer_dtype,
        cache_mode=cache_mode,
        cache_root=cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        adaln_pretranspose=adaln_pretranspose,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
        reporter=reporter,
        shard_limit_bytes=shard_limit_bytes,
        constructor_config=constructor_config,
    )
    loaded_count, layout_count, quant_count = load_transformer_cache(
        model,
        result.cache_path,
        include_audio=include_audio,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
    )
    _log.info(
        "Loaded transformer cache: %d tensors (%d layout, %d quantized)",
        loaded_count,
        layout_count,
        quant_count,
    )
    return TransformerCacheResult(
        cache_path=result.cache_path,
        rebuilt=result.rebuilt,
        loaded_count=loaded_count,
        layout_count=layout_count,
        quant_count=quant_count,
    )


def bind_transformer_cache(
    model: nn.Module,
    cache_file: Path | str,
    *,
    transformer_dtype: str | mx.Dtype | None = None,
    include_audio: bool | None = None,
    resident_blocks: int | None = None,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
) -> TransformerCacheResult:
    """Validate and bind an existing prepared cache without rebuilding it."""
    _validate_model_compute_dtype(model, transformer_dtype)
    cache_path = Path(cache_file)
    if resident_blocks is None:
        loaded_count, layout_count, quant_count = load_transformer_cache(
            model,
            cache_path,
            include_audio=include_audio,
            transformer_cache_quantize=transformer_cache_quantize,
            video_ff_quantize_specs=video_ff_quantize_specs,
            video_ff_quantize_group_size=video_ff_quantize_group_size,
            video_ff_quantize_bits=video_ff_quantize_bits,
        )
        return TransformerCacheResult(
            cache_path=cache_path,
            rebuilt=False,
            loaded_count=loaded_count,
            layout_count=layout_count,
            quant_count=quant_count,
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
        cache_path,
        expected_model=model,
        include_audio=include_audio,
        expected_block_schemas=expected_block_schemas,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
    )
    try:
        cached_weights = streamer.take_non_block_weights()
        non_block_weights: dict[str, mx.array] = {}
        layout_weights: dict[str, mx.array] = {}
        for key, value in cached_weights.items():
            if key.startswith(LAYOUT_KEY_PREFIX):
                layout_weights[key[len(LAYOUT_KEY_PREFIX) :]] = value
            elif key.startswith(QUANT_KEY_PREFIX):
                raise ValueError(f"Unexpected non-block quant cache key: {key}")
            else:
                non_block_weights[key] = value

        clear_layout_weights(model)
        if non_block_weights:
            model.update(flatten_to_nested(non_block_weights))
        for key, value in layout_weights.items():
            install_layout_weight(model, key, value)
        binding = cast(_TransformerCacheBinding, model)
        binding.transformer_blocks = blocks[:resident_blocks]
        binding.transformer_block_streamer = streamer
        del cached_weights, non_block_weights, layout_weights
        gc.collect()
    except BaseException:
        streamer.close()
        raise
    _log.info(
        "Loaded transformer cache: %d streamed block tensors (%d layout, "
        "%d quantized), %d/%d blocks resident",
        streamer.loaded_count,
        streamer.layout_count,
        streamer.quant_count,
        resident_blocks,
        streamer.block_count,
    )
    return TransformerCacheResult(
        cache_path=cache_path,
        rebuilt=False,
        loaded_count=streamer.loaded_count,
        layout_count=streamer.layout_count,
        quant_count=streamer.quant_count,
    )


def load_transformer_weights_cached_streaming(
    model: nn.Module,
    weights_path: Path | str,
    *,
    transformer_dtype: str | mx.Dtype | None = None,
    cache_mode: str,
    cache_root: Path | str | None,
    include_audio: bool,
    video_ff_layout_specs: LayoutSpecs = DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: LayoutSpecs = DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    video_attn_layout_layers: tuple[int, ...] = (),
    resident_blocks: int,
    audio_ff_layout_specs: LayoutSpecs | None = None,
    audio_ff_layout_layers: tuple[int, ...] | None = None,
    audio_attn_layout_specs: LayoutSpecs | None = None,
    audio_attn_layout_layers: tuple[int, ...] | None = None,
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    reporter: Reporter | None = None,
    shard_limit_bytes: int = DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
    constructor_config: TransformerConstructorConfig | None = None,
) -> TransformerCacheResult:
    """Load non-block tensors and attach a cache-backed block streamer."""
    if resident_blocks <= 0:
        raise ValueError("resident_blocks must be positive")
    _validate_model_compute_dtype(model, transformer_dtype)
    model_block_count = len(_transformer_blocks(model))
    if resident_blocks > model_block_count:
        raise ValueError(
            f"resident_blocks={resident_blocks} exceeds model block count {model_block_count}"
        )
    result = ensure_transformer_cache(
        weights_path,
        transformer_dtype=transformer_dtype,
        cache_mode=cache_mode,
        cache_root=cache_root,
        include_audio=include_audio,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        adaln_pretranspose=adaln_pretranspose,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
        reporter=reporter,
        shard_limit_bytes=shard_limit_bytes,
        constructor_config=constructor_config,
    )
    bound = bind_transformer_cache(
        model,
        result.cache_path,
        transformer_dtype=transformer_dtype,
        include_audio=include_audio,
        resident_blocks=resident_blocks,
        transformer_cache_quantize=transformer_cache_quantize,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_group_size=video_ff_quantize_group_size,
        video_ff_quantize_bits=video_ff_quantize_bits,
    )
    return TransformerCacheResult(
        cache_path=result.cache_path,
        rebuilt=result.rebuilt,
        loaded_count=bound.loaded_count,
        layout_count=bound.layout_count,
        quant_count=bound.quant_count,
    )


__all__ = [
    "TransformerBlockStreamer",
    "TransformerCacheResult",
    "bind_transformer_cache",
    "build_transformer_cache",
    "ensure_transformer_cache",
    "load_transformer_cache",
    "load_transformer_weights_cached",
    "load_transformer_weights_cached_streaming",
]
