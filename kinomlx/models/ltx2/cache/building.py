"""Build bounded-memory transformer cache shards from checkpoints."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import mlx.core as mx

from kinomlx._typing import JsonObject
from kinomlx.reporting import NullReporter, Reporter

from .keys import DIFFUSION_PREFIX, convert_checkpoint_key
from .layout import layout_cache_key
from .schema import (
    DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
    TRANSFORMER_CACHE_QUANTIZE_OFF,
    dtype_payload_name,
    transformer_fp16_ranges_path,
)
from .storage import (
    ShardedCacheWriter,
    clear_cache_artifacts,
    prepare_cache_build,
    transformer_artifact_manifest,
    write_metadata,
)
from .weights import (
    cache_quant_mode_for_key,
    cache_quant_pretransposed,
    cast_for_cache,
    checkpoint_has_fp8_tensors,
    ff_cache_dtype_for_key,
    fp8_scale_companions,
    iter_checkpoint_weights,
    normalize_transformer_cache_dtypes,
    quant_defaults,
)

_log = logging.getLogger(__name__)


def _peak_recorder(peaks: dict[str, float], key: str) -> Callable[[float], None]:
    def record(peak: float) -> None:
        peaks[key] = peak

    return record


def build_transformer_cache(
    weights_path: Path | str,
    cache_file: Path,
    metadata_file: Path,
    payload: JsonObject,
    *,
    transformer_dtype: str | mx.Dtype | None = None,
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
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    source_bindings: Mapping[str, str] | None = None,
    reporter: Reporter | None = None,
    shard_limit_bytes: int = DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
) -> tuple[int, int, int]:
    """Build a converted, layout-aware, optionally quantized transformer cache."""
    transformer_dtype, video_ff_dtype, audio_ff_dtype = normalize_transformer_cache_dtypes(
        transformer_dtype,
        video_ff_dtype,
        audio_ff_dtype,
    )
    cache_file = Path(cache_file)
    metadata_file = Path(metadata_file)
    fp16_ranges_file = transformer_fp16_ranges_path(cache_file)
    fp16_ranges_file.unlink(missing_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prepare_cache_build(cache_file, metadata_file)
    fp16_peaks: dict[str, float] = {}

    fp8_build = checkpoint_has_fp8_tensors(weights_path)
    fp8_value_dtype = transformer_dtype or mx.bfloat16
    if fp8_build:
        from kinomlx.io.safetensors import read_dtypes

        fp8_keys, weight_scales, input_scales, comfy_tags = fp8_scale_companions(
            read_dtypes(weights_path)
        )
        _log.info(
            "FP8 checkpoint: dequantizing %d tensors (%d weight scales, "
            "%d input scales and %d ComfyUI tags removed) to %s",
            len(fp8_keys),
            len(weight_scales),
            len(input_scales),
            len(comfy_tags),
            dtype_payload_name(fp8_value_dtype),
        )

    sink = reporter if reporter is not None else NullReporter()
    phase = "build transformer cache"
    sink.phase_start(phase, total=None, unit="tensor")
    writer = ShardedCacheWriter(
        cache_file,
        shard_limit_bytes=shard_limit_bytes,
    )
    loaded_count = 0
    layout_count = 0
    quant_count = 0
    skipped_count = 0
    seen_targets: set[str] = set()
    source_items = iter_checkpoint_weights(
        weights_path,
        fp8_value_dtype=fp8_value_dtype,
    )
    try:
        for checkpoint_key, value in source_items:
            if source_bindings is None:
                if not checkpoint_key.startswith(DIFFUSION_PREFIX):
                    continue
                mlx_key = convert_checkpoint_key(
                    checkpoint_key,
                    include_audio=include_audio,
                )
            else:
                mlx_key = source_bindings.get(checkpoint_key)
            if mlx_key is None:
                skipped_count += 1
                continue
            seen_targets.add(mlx_key)
            sink.phase_advance(phase)

            quant_mode = cache_quant_mode_for_key(
                mlx_key,
                transformer_cache_quantize=transformer_cache_quantize,
                video_ff_quantize_specs=video_ff_quantize_specs,
                video_ff_quantize_layers=video_ff_quantize_layers,
            )
            if quant_mode is not None:
                quant_value = (
                    mx.contiguous(value.T)
                    if cache_quant_pretransposed(transformer_cache_quantize)
                    else value
                )
                if transformer_dtype is not None and quant_value.dtype == mx.bfloat16:
                    quant_value = cast_for_cache(
                        quant_value,
                        transformer_dtype,
                        mlx_key,
                    )
                group_size, bits = quant_defaults(
                    quant_mode,
                    video_ff_quantize_group_size,
                    video_ff_quantize_bits,
                )
                quantized = mx.quantize(
                    quant_value,
                    group_size,
                    bits,
                    mode=quant_mode,
                )
                quant_key = f"{QUANT_KEY_PREFIX}{mlx_key}"
                base_key = quant_key[: -len(".weight")]
                writer.add(f"{base_key}.weight", quantized[0])
                writer.add(f"{base_key}.scales", quantized[1])
                quant_count += 2
                if len(quantized) > 2:
                    writer.add(f"{base_key}.biases", quantized[2])
                    quant_count += 1
                loaded_count += 1
                continue

            logical_layout_key = layout_cache_key(
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
            target_dtype = ff_cache_dtype_for_key(
                mlx_key,
                video_ff_dtype=video_ff_dtype,
                audio_ff_dtype=audio_ff_dtype,
            )
            if (
                target_dtype is None
                and transformer_dtype is not None
                and value.dtype == mx.bfloat16
            ):
                target_dtype = transformer_dtype

            if logical_layout_key is not None:
                cache_key = f"{LAYOUT_KEY_PREFIX}{logical_layout_key}"
                stored = mx.contiguous(value.T)
                if target_dtype is not None and stored.dtype != target_dtype:
                    stored = cast_for_cache(stored, target_dtype, mlx_key)
                if stored.dtype == mx.float16:
                    stored = cast_for_cache(
                        stored,
                        mx.float16,
                        mlx_key,
                        fp16_peak_sink=_peak_recorder(fp16_peaks, cache_key),
                    )
                writer.add(cache_key, stored)
                layout_count += 1
            else:
                cache_key = mlx_key
                stored = value
                if target_dtype is not None and value.dtype != target_dtype:
                    stored = cast_for_cache(value, target_dtype, mlx_key)
                if stored.dtype == mx.float16:
                    stored = cast_for_cache(
                        stored,
                        mx.float16,
                        mlx_key,
                        fp16_peak_sink=_peak_recorder(fp16_peaks, cache_key),
                    )
                writer.add(cache_key, stored)
            loaded_count += 1

        if loaded_count == 0:
            raise ValueError(
                f"Checkpoint {weights_path} contains no compatible transformer tensors"
            )
        if source_bindings is not None:
            missing_targets = sorted(set(source_bindings.values()) - seen_targets)
            if missing_targets:
                raise ValueError(
                    f"Checkpoint {weights_path} changed while building the transformer cache; "
                    f"missing consumed target {missing_targets[0]}"
                )
        shard_count = writer.close()
        # Publish this for every new cache, including an empty table for caches
        # with no realized FP16 tensors. That makes native-FP16 checkpoints and
        # explicit FP16 conversions share one cache contract.
        write_metadata(
            fp16_ranges_file,
            {
                "schema_version": 1,
                "max_abs_by_key": fp16_peaks,
            },
        )
        write_metadata(
            metadata_file,
            {
                **payload,
                "artifacts": transformer_artifact_manifest(cache_file),
            },
        )
        _log.info(
            "Built transformer cache: %d tensors (%d pretransposed, "
            "%d quantized, %d skipped, %d shards)",
            loaded_count,
            layout_count,
            quant_count,
            skipped_count,
            shard_count,
        )
        return loaded_count, layout_count, quant_count
    except BaseException:
        # Metadata was invalidated before the build. Remove partial shards too,
        # so an interrupted conversion cannot strand tens of gigabytes.
        clear_cache_artifacts(cache_file)
        fp16_ranges_file.unlink(missing_ok=True)
        raise
    finally:
        del source_items
        gc.collect()
        mx.clear_cache()
        sink.phase_end(phase)


__all__ = ["build_transformer_cache"]
