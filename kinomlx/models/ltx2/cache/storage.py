"""Atomic single-file and sharded safetensors cache storage."""

from __future__ import annotations

import gc
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinomlx._typing import JsonObject, JsonValue
from kinomlx.io.safetensors import load_weights

from .schema import DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES, transformer_fp16_ranges_path


def cache_shard_path(cache_file: Path, index: int) -> Path:
    """Return the stable numbered shard path."""
    if index < 0:
        raise ValueError("cache shard index must be non-negative")
    return cache_file.with_name(f"{cache_file.stem}-{index:05d}{cache_file.suffix}")


def existing_cache_shards(cache_file: Path) -> list[Path]:
    """Return strictly named cache shards in index order."""
    return sorted(
        cache_file.parent.glob(f"{cache_file.stem}-[0-9][0-9][0-9][0-9][0-9]{cache_file.suffix}")
    )


def _validate_shard_sequence(cache_file: Path, shards: list[Path]) -> None:
    for index, shard in enumerate(shards):
        expected = cache_shard_path(cache_file, index)
        if shard != expected:
            raise ValueError(
                f"Transformer cache has a missing or misnumbered shard: "
                f"expected {expected.name}, found {shard.name}"
            )


def _read_transformer_artifact_manifest(
    metadata_file: Path,
    cache_file: Path,
) -> JsonObject | None:
    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    if artifacts.get("cache_file") != cache_file.name:
        return None
    return cast(JsonObject, artifacts)


def _validate_transformer_artifact_manifest(
    cache_file: Path,
    artifacts: Mapping[str, JsonValue],
) -> None:
    expected_count = artifacts.get("shard_count")
    expected_shards = artifacts.get("shards")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or not isinstance(expected_shards, list)
        or len(expected_shards) != expected_count
    ):
        raise ValueError(f"Transformer cache has invalid artifact metadata: {cache_file}")
    shards = existing_cache_shards(cache_file)
    if len(shards) != expected_count:
        raise ValueError(
            f"Transformer cache expected {expected_count} shards, found {len(shards)}: {cache_file}"
        )
    _validate_shard_sequence(cache_file, shards)
    for shard, expected in zip(shards, expected_shards, strict=True):
        if not isinstance(expected, dict):
            raise ValueError(f"Transformer cache has invalid artifact metadata: {cache_file}")
        expected_name = expected.get("name")
        expected_size = expected.get("size")
        if expected_name != shard.name or expected_size != shard.stat().st_size:
            raise ValueError(
                f"Transformer cache shard identity mismatch for {shard.name}: {cache_file}"
            )


def transformer_artifact_manifest(cache_file: Path) -> JsonObject:
    """Describe the complete numbered transformer shard set."""
    shards = existing_cache_shards(cache_file)
    if not shards:
        raise ValueError(f"Transformer cache has no numbered shards: {cache_file}")
    _validate_shard_sequence(cache_file, shards)
    return {
        "cache_file": cache_file.name,
        "shard_count": len(shards),
        "shards": [{"name": shard.name, "size": shard.stat().st_size} for shard in shards],
    }


def cache_artifacts_exist(
    cache_file: Path,
    *,
    metadata_file: Path | None = None,
) -> bool:
    """Return whether a single file or complete-looking shard sequence exists."""
    base_exists = cache_file.is_file()
    shards = existing_cache_shards(cache_file)
    if base_exists and shards:
        return False
    if base_exists:
        return True
    if not shards:
        return False
    try:
        _validate_shard_sequence(cache_file, shards)
    except ValueError:
        return False
    if metadata_file is not None:
        artifacts = _read_transformer_artifact_manifest(metadata_file, cache_file)
        if artifacts is None:
            return False
        try:
            _validate_transformer_artifact_manifest(cache_file, artifacts)
        except OSError, ValueError:
            return False
    return True


def clear_cache_artifacts(cache_file: Path) -> None:
    """Remove only the exact single-file/sharded artifact set for a rebuild."""
    cache_file.unlink(missing_ok=True)
    for shard in existing_cache_shards(cache_file):
        shard.unlink(missing_ok=True)


def prepare_cache_build(cache_file: Path, metadata_file: Path) -> None:
    """Invalidate metadata first, then clear stale artifacts before rebuilding."""
    metadata_file.unlink(missing_ok=True)
    clear_cache_artifacts(cache_file)


def load_cache_weights(cache_file: Path) -> dict[str, mx.array]:
    """Lazily load one cache file or merge a contiguous shard sequence."""
    base_exists = cache_file.is_file()
    shards = existing_cache_shards(cache_file)
    if base_exists and shards:
        raise ValueError(f"Cache has both a single-file artifact and numbered shards: {cache_file}")
    if base_exists:
        return load_weights(cache_file)
    if not shards:
        raise FileNotFoundError(f"No cache at {cache_file} and no shards beside it")
    metadata_file = cache_file.with_name("metadata.json")
    artifacts = _read_transformer_artifact_manifest(metadata_file, cache_file)
    if (
        cache_file.name == "transformer.safetensors"
        and metadata_file.is_file()
        and artifacts is None
    ):
        raise ValueError(f"Transformer cache has invalid artifact metadata: {cache_file}")
    if artifacts is not None:
        _validate_transformer_artifact_manifest(cache_file, artifacts)
    _validate_shard_sequence(cache_file, shards)
    merged: dict[str, mx.array] = {}
    for shard in shards:
        loaded = load_weights(shard)
        duplicate = merged.keys() & loaded.keys()
        if duplicate:
            sample = sorted(duplicate)[0]
            raise ValueError(f"Duplicate tensor {sample!r} across cache shards at {cache_file}")
        merged.update(loaded)
    return merged


def load_transformer_fp16_ranges(cache_file: Path | str) -> dict[str, float]:
    """Load and validate cached FP16 tensor absolute peaks."""
    path = transformer_fp16_ranges_path(cache_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FP16 range sidecar: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid FP16 range sidecar schema: {path}")
    raw_peaks = payload.get("max_abs_by_key")
    if not isinstance(raw_peaks, dict):
        raise ValueError(f"FP16 range sidecar has no max_abs_by_key table: {path}")
    peaks: dict[str, float] = {}
    for key, value in raw_peaks.items():
        if not isinstance(key, str) or isinstance(value, bool):
            raise ValueError(f"FP16 range sidecar contains an invalid entry: {path}")
        try:
            peak = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FP16 range sidecar contains an invalid entry: {path}") from exc
        if not math.isfinite(peak) or peak < 0.0 or peak > 65504.0:
            raise ValueError(f"FP16 range sidecar contains an invalid peak for {key!r}: {peak}")
        peaks[key] = peak
    return peaks


def metadata_matches(
    metadata_path: Path,
    expected_payload: Mapping[str, JsonValue],
) -> bool:
    """Return whether a JSON sidecar exactly matches the expected identity."""
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return False
    if not isinstance(actual, dict):
        return False
    identity = dict(actual)
    identity.pop("artifacts", None)
    return identity == expected_payload


def _unique_temp_path(directory: Path, stem: str, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{stem}.",
        suffix=suffix,
        dir=directory,
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def write_metadata(metadata_file: Path, payload: Mapping[str, JsonValue]) -> None:
    """Atomically publish a cache metadata sidecar."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _unique_temp_path(
        metadata_file.parent,
        metadata_file.stem,
        ".tmp.json",
    )
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(metadata_file)
    finally:
        temp_path.unlink(missing_ok=True)


def save_weights_atomic(
    cache_file: Path,
    weights: Mapping[str, mx.array],
) -> None:
    """Evaluate and atomically publish a single safetensors cache artifact."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(weights.values())
    if values:
        mx.eval(*values)
    temp_path = _unique_temp_path(
        cache_file.parent,
        cache_file.stem,
        ".tmp.safetensors",
    )
    try:
        mx.save_safetensors(str(temp_path), dict(weights))
        temp_path.replace(cache_file)
    finally:
        temp_path.unlink(missing_ok=True)


class ShardedCacheWriter:
    """Bound transformer cache build memory and publish numbered shards."""

    def __init__(
        self,
        cache_file: Path,
        *,
        shard_limit_bytes: int = DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES,
        eval_batch_size: int = 24,
    ) -> None:
        if shard_limit_bytes <= 0:
            raise ValueError("shard_limit_bytes must be positive")
        if eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        self.cache_file = cache_file
        self.shard_limit_bytes = shard_limit_bytes
        self.eval_batch_size = eval_batch_size
        self._weights: dict[str, mx.array] = {}
        self._seen_keys: set[str] = set()
        self._pending: list[mx.array] = []
        self._shard_bytes = 0
        self.shard_count = 0

    def add(self, key: str, value: mx.array) -> None:
        """Add one unique tensor, flushing first when the shard is full."""
        if key in self._seen_keys:
            raise ValueError(f"duplicate cache tensor key: {key}")
        if self._weights and self._shard_bytes + value.nbytes > self.shard_limit_bytes:
            self.flush()
        if len(self._pending) >= self.eval_batch_size:
            self._flush_pending()
        self._weights[key] = value
        self._seen_keys.add(key)
        self._pending.append(value)
        self._shard_bytes += value.nbytes

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        mx.eval(*self._pending)
        self._pending.clear()
        mx.clear_cache()

    def flush(self) -> None:
        """Publish the current shard and release its evaluated buffers."""
        if not self._weights:
            return
        self._flush_pending()
        mx.eval(*self._weights.values())
        shard = cache_shard_path(self.cache_file, self.shard_count)
        shard.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _unique_temp_path(
            shard.parent,
            shard.stem,
            ".tmp.safetensors",
        )
        try:
            mx.save_safetensors(str(temp_path), self._weights)
            temp_path.replace(shard)
        finally:
            temp_path.unlink(missing_ok=True)
        self._weights.clear()
        self._shard_bytes = 0
        self.shard_count += 1
        gc.collect()
        mx.clear_cache()

    def close(self) -> int:
        """Flush the final shard and return the published shard count."""
        self.flush()
        return self.shard_count


__all__ = [
    "ShardedCacheWriter",
    "cache_artifacts_exist",
    "cache_shard_path",
    "clear_cache_artifacts",
    "existing_cache_shards",
    "load_cache_weights",
    "load_transformer_fp16_ranges",
    "metadata_matches",
    "prepare_cache_build",
    "save_weights_atomic",
    "transformer_artifact_manifest",
    "write_metadata",
]
