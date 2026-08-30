"""FP8 conversion, quant routing, and sharded storage tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import read_dtypes, read_header
from kinomlx.models.ltx2.cache.schema import (
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)
from kinomlx.models.ltx2.cache.storage import (
    ShardedCacheWriter,
    cache_artifacts_exist,
    cache_shard_path,
    load_cache_weights,
    metadata_matches,
    prepare_cache_build,
    transformer_artifact_manifest,
    write_metadata,
)
from kinomlx.models.ltx2.cache.weights import (
    cache_quant_mode_for_key,
    cache_quant_pretransposed,
    cast_for_cache,
    checkpoint_has_fp8_tensors,
    fp8_scale_companions,
    iter_checkpoint_weights,
    iter_fp8_checkpoint_weights,
    normalize_transformer_cache_dtypes,
    quant_defaults,
)


def _write_raw_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for key, dtype, shape, data in tensors:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(blobs))


def test_checkpoint_loader_handles_resolved_huggingface_blob_without_extension(
    tmp_path: Path,
) -> None:
    blob = tmp_path / "b33b7fe4bbfe084f"
    staged = blob.with_suffix(".safetensors")
    mx.save_safetensors(str(staged), {"tensor": mx.array([1.0, 2.0])})
    staged.rename(blob)

    loaded = dict(iter_checkpoint_weights(blob))

    assert mx.array_equal(loaded["tensor"], mx.array([1.0, 2.0])).item()


def test_header_reader_reports_raw_fp8_dtype_without_tensor_load(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fp8.safetensors"
    _write_raw_safetensors(path, [("weight", "F8_E4M3", [1], bytes([0x38]))])
    assert read_dtypes(path) == {"weight": "F8_E4M3"}
    assert read_header(path)["weight"]["shape"] == [1]
    assert checkpoint_has_fp8_tensors(path)


def test_header_reader_rejects_truncated_prefix_and_header(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix.safetensors"
    prefix.write_bytes(b"short")
    with pytest.raises(ValueError, match="length prefix"):
        read_header(prefix)
    header = tmp_path / "header.safetensors"
    header.write_bytes(struct.pack("<Q", 10) + b"{}")
    with pytest.raises(ValueError, match="truncated safetensors header"):
        read_header(header)


def test_fp8_companion_classification_is_scoped_to_real_fp8_weights() -> None:
    dtypes = {
        "layer.weight": "F8_E4M3",
        "layer.weight_scale": "F32",
        "layer.input_scale": "F32",
        "layer.comfy_quant": "U8",
        "unrelated.comfy_quant": "U8",
        "scale_shift_table": "F32",
    }
    fp8, weight_scales, input_scales, tags = fp8_scale_companions(dtypes)
    assert fp8 == {"layer.weight"}
    assert weight_scales == {"layer.weight_scale"}
    assert input_scales == {"layer.input_scale"}
    assert tags == {"layer.comfy_quant"}


def test_fp8_iterator_applies_scale_and_drops_activation_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fp8.safetensors"
    _write_raw_safetensors(
        path,
        [
            ("layer.weight", "F8_E4M3", [1], bytes([0x38])),
            ("layer.weight_scale", "F32", [], struct.pack("<f", 2.0)),
            ("layer.input_scale", "F32", [1], struct.pack("<f", 7.0)),
            (
                "layer.comfy_quant",
                "U8",
                [26],
                b'{"format":"float8_e4m3fn"}',
            ),
            ("ordinary", "F32", [1], struct.pack("<f", 3.0)),
        ],
    )
    converted = dict(iter_fp8_checkpoint_weights(path, mx.float16))
    assert set(converted) == {"layer.weight", "ordinary"}
    assert converted["layer.weight"].dtype == mx.float16
    assert float(converted["layer.weight"].item()) == 2.0
    assert float(converted["ordinary"].item()) == 3.0


def test_fp8_iterator_rejects_e5m2_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "e5m2.safetensors"
    _write_raw_safetensors(path, [("weight", "F8_E5M2", [1], bytes([0]))])
    with pytest.raises(ValueError, match="F8_E5M2"):
        dict(iter_fp8_checkpoint_weights(path, mx.bfloat16))


def test_fp8_iterator_rejects_non_scalar_weight_scale(tmp_path: Path) -> None:
    path = tmp_path / "vector-scale.safetensors"
    _write_raw_safetensors(
        path,
        [
            ("layer.weight", "F8_E4M3", [2, 2], bytes([0x38] * 4)),
            ("layer.weight_scale", "F32", [2], struct.pack("<ff", 2.0, 3.0)),
        ],
    )

    with pytest.raises(ValueError, match="layer.weight_scale.*scalar"):
        dict(iter_fp8_checkpoint_weights(path, mx.bfloat16))


def test_fp8_companion_classifier_rejects_scale_for_non_fp8_parent() -> None:
    with pytest.raises(ValueError, match="ordinary.weight_scale.*non-FP8"):
        fp8_scale_companions(
            {
                "quantized.weight": "F8_E4M3",
                "ordinary.weight": "BF16",
                "ordinary.weight_scale": "F32",
            }
        )


def test_fp16_cache_cast_rejects_range_overflow_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="exceeds the float16 range"):
        cast_for_cache(mx.array([65536.0]), mx.float16, "too_large")
    with pytest.raises(ValueError, match="non-finite"):
        cast_for_cache(mx.array([float("nan")]), mx.float16, "nan_weight")
    peaks: list[float] = []
    safe = cast_for_cache(
        mx.array([1.5]),
        mx.float16,
        "safe",
        fp16_peak_sink=peaks.append,
    )
    assert safe.dtype == mx.float16
    assert peaks == [1.5]


def test_transformer_cache_dtype_normalization_is_canonical() -> None:
    assert normalize_transformer_cache_dtypes(
        "bfloat16",
        mx.bfloat16,
        None,
    ) == (None, None, None)
    assert normalize_transformer_cache_dtypes(
        "float16",
        mx.float16,
        mx.float16,
    ) == (mx.float16, None, None)
    assert normalize_transformer_cache_dtypes(
        "bfloat16",
        mx.float16,
        None,
    ) == (None, mx.float16, None)
    with pytest.raises(ValueError, match="conflicts with transformer dtype"):
        normalize_transformer_cache_dtypes(
            "float16",
            mx.bfloat16,
            None,
        )


def test_quant_routing_covers_whole_block_and_targeted_ff_modes() -> None:
    attention = "transformer_blocks.0.attn1.to_q.weight"
    ff = "transformer_blocks.3.ff.project_out.weight"
    assert (
        cache_quant_mode_for_key(
            attention,
            transformer_cache_quantize=TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
            video_ff_quantize_specs=(),
            video_ff_quantize_layers=(),
        )
        == "mxfp8"
    )
    assert (
        cache_quant_mode_for_key(
            ff,
            transformer_cache_quantize="off",
            video_ff_quantize_specs=(("project_out", "nvfp4"),),
            video_ff_quantize_layers=(3,),
        )
        == "nvfp4"
    )
    assert (
        cache_quant_mode_for_key(
            ff,
            transformer_cache_quantize="off",
            video_ff_quantize_specs=(("project_out", "nvfp4"),),
            video_ff_quantize_layers=(2,),
        )
        is None
    )
    assert cache_quant_pretransposed(TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE)
    assert quant_defaults("mxfp8", None, None) == (32, 8)
    assert quant_defaults("affine", 128, 6) == (128, 6)
    with pytest.raises(ValueError, match="must be positive"):
        quant_defaults("affine", 0, 4)
    with pytest.raises(ValueError, match="must be positive"):
        quant_defaults("affine", 64, 0)


def test_sharded_writer_round_trips_and_enforces_contiguous_sequence(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    writer = ShardedCacheWriter(cache_file, shard_limit_bytes=16)
    writer.add("a", mx.arange(4, dtype=mx.float32))
    writer.add("b", mx.arange(4, dtype=mx.float32) + 10)
    assert writer.close() == 2
    assert not cache_file.exists()
    assert cache_artifacts_exist(cache_file)
    loaded = load_cache_weights(cache_file)
    assert set(loaded) == {"a", "b"}
    assert mx.array_equal(loaded["b"], mx.arange(4) + 10).item()

    cache_shard_path(cache_file, 0).rename(cache_shard_path(cache_file, 2))
    assert not cache_artifacts_exist(cache_file)
    with pytest.raises(ValueError, match="missing or misnumbered shard"):
        load_cache_weights(cache_file)


def test_transformer_manifest_detects_tail_truncation_before_load(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    metadata_file = tmp_path / "metadata.json"
    writer = ShardedCacheWriter(cache_file, shard_limit_bytes=16)
    writer.add("a", mx.arange(4, dtype=mx.float32))
    writer.add("b", mx.arange(4, dtype=mx.float32) + 10)
    assert writer.close() == 2
    identity = {"schema_version": 2, "source": {"size": 1}}
    write_metadata(
        metadata_file,
        {
            **identity,
            "artifacts": transformer_artifact_manifest(cache_file),
        },
    )
    assert metadata_matches(metadata_file, identity)
    assert cache_artifacts_exist(cache_file, metadata_file=metadata_file)

    cache_shard_path(cache_file, 1).unlink()

    assert not cache_artifacts_exist(cache_file, metadata_file=metadata_file)
    with pytest.raises(ValueError, match="expected 2 shards, found 1"):
        load_cache_weights(cache_file)


def test_shard_loader_rejects_duplicate_tensor_names(tmp_path: Path) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    for index in range(2):
        mx.save_safetensors(
            str(cache_shard_path(cache_file, index)),
            {"same": mx.array([index])},
        )
    with pytest.raises(ValueError, match="Duplicate tensor"):
        load_cache_weights(cache_file)


def test_sharded_writer_rejects_duplicates_across_flushed_shards(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    writer = ShardedCacheWriter(cache_file, shard_limit_bytes=4)
    writer.add("same", mx.array([1.0]))
    writer.add("other", mx.array([2.0]))
    with pytest.raises(ValueError, match="duplicate cache tensor"):
        writer.add("same", mx.array([3.0]))


def test_loader_rejects_ambiguous_single_file_and_shards(tmp_path: Path) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    mx.save_safetensors(str(cache_file), {"base": mx.array([1.0])})
    mx.save_safetensors(
        str(cache_shard_path(cache_file, 0)),
        {"shard": mx.array([2.0])},
    )
    assert not cache_artifacts_exist(cache_file)
    with pytest.raises(ValueError, match="both a single-file artifact"):
        load_cache_weights(cache_file)


def test_metadata_is_exact_and_invalidated_before_rebuild(tmp_path: Path) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    metadata_file = tmp_path / "metadata.json"
    mx.save_safetensors(str(cache_file), {"x": mx.array([1.0])})
    payload = {"schema_version": 1, "source": {"size": 1}}
    write_metadata(metadata_file, payload)
    assert metadata_matches(metadata_file, payload)
    assert not metadata_matches(metadata_file, {**payload, "schema_version": 2})

    prepare_cache_build(cache_file, metadata_file)
    assert not cache_file.exists()
    assert not metadata_file.exists()
