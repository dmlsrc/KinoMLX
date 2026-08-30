"""Behavioral tests for ``kinomlx.io.safetensors``."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import pytest

import kinomlx.io.safetensors as safetensors_module
from kinomlx.io.safetensors import (
    load_weights,
    load_weights_with_metadata,
    read_header,
    read_metadata,
    read_u8_tensor,
    save_weights,
)


def _write_raw_safetensors(path: Path, header: object, payload: bytes = b"") -> None:
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def _sample_weights() -> dict[str, mx.array]:
    return {
        "transformer.block_0.attn.weight": mx.arange(12, dtype=mx.float32).reshape(3, 4),
        "transformer.block_0.attn.bias": mx.zeros((4,), dtype=mx.float32),
        "transformer.block_1.norm.weight": mx.ones((8,), dtype=mx.bfloat16),
    }


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    """True iff dtype, shape, and every element match."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    # Cast to a common type that allclose accepts for both fp and bf16.
    return bool(mx.array_equal(a, b))


# ---------------------------------------------------------------------------
# Round-trip - weights only
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_weights(tmp_path: Path) -> None:
    out = tmp_path / "weights.safetensors"
    original = _sample_weights()
    save_weights(out, original)

    loaded = load_weights(out)
    assert set(loaded) == set(original)
    for key, ref in original.items():
        assert _arrays_equal(loaded[key], ref), f"mismatch on {key}"


def test_loaders_accept_resolved_huggingface_blob_without_extension(tmp_path: Path) -> None:
    staged = tmp_path / "model.safetensors"
    blob = tmp_path / "b33b7fe4bbfe084f"
    save_weights(staged, _sample_weights(), metadata={"model": "test"})
    staged.rename(blob)

    assert set(load_weights(blob)) == set(_sample_weights())
    _, metadata = load_weights_with_metadata(blob)
    assert metadata == {"model": "test"}


def test_save_failure_preserves_previous_safetensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import kinomlx.io.safetensors as safetensors_module

    out = tmp_path / "weights.safetensors"
    out.write_bytes(b"previous")

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(safetensors_module.mx, "save_safetensors", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected save failure"):
        save_weights(out, _sample_weights())

    assert out.read_bytes() == b"previous"
    assert tuple(tmp_path.glob(".weights.*.tmp.safetensors")) == ()


# ---------------------------------------------------------------------------
# Round-trip - with metadata
# ---------------------------------------------------------------------------


def test_save_with_metadata_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "weights.safetensors"
    weights = _sample_weights()
    metadata = {"schema_version": "1", "layout": "ndhwc"}

    save_weights(out, weights, metadata=metadata)

    loaded_weights, loaded_metadata = load_weights_with_metadata(out)
    assert loaded_metadata == metadata
    assert set(loaded_weights) == set(weights)


def test_save_with_no_metadata_loads_empty_metadata(tmp_path: Path) -> None:
    out = tmp_path / "weights.safetensors"
    save_weights(out, _sample_weights())

    _, metadata = load_weights_with_metadata(out)
    assert metadata == {}


# ---------------------------------------------------------------------------
# Metadata-only read - doesn't materialize tensors
# ---------------------------------------------------------------------------


def test_read_metadata_returns_strings(tmp_path: Path) -> None:
    out = tmp_path / "weights.safetensors"
    metadata = {"schema_version": "7", "fast_mode": "1"}
    save_weights(out, _sample_weights(), metadata=metadata)

    assert read_metadata(out) == metadata


def test_read_metadata_on_file_without_metadata(tmp_path: Path) -> None:
    out = tmp_path / "no_meta.safetensors"
    save_weights(out, _sample_weights())

    # safetensors writes an empty metadata dict in this case; our helper
    # normalizes that to ``{}`` either way.
    assert read_metadata(out) == {}


def test_read_u8_tensor_seek_reads_one_embedded_asset(tmp_path: Path) -> None:
    path = tmp_path / "assets.safetensors"
    expected = b'{"model":{"type":"BPE"}}'
    mx.save_safetensors(
        str(path),
        {
            "large_neighbor": mx.zeros((1024,), dtype=mx.float32),
            "tokenizer_json": mx.array(list(expected), dtype=mx.uint8),
        },
    )

    assert read_u8_tensor(path, "tokenizer_json") == expected


def test_read_u8_tensor_rejects_non_u8_and_missing_assets(tmp_path: Path) -> None:
    path = tmp_path / "assets.safetensors"
    mx.save_safetensors(str(path), {"not_bytes": mx.array([1.0])})

    with pytest.raises(ValueError, match="is not U8"):
        read_u8_tensor(path, "not_bytes")
    with pytest.raises(KeyError, match="missing"):
        read_u8_tensor(path, "missing")


def test_read_u8_tensor_rejects_offsets_beyond_file_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated-asset.safetensors"
    _write_raw_safetensors(
        path,
        {"asset": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}},
        payload=b"x",
    )

    with pytest.raises(ValueError, match="extends beyond the file payload"):
        read_u8_tensor(path, "asset")


@pytest.mark.parametrize("error_type", [RecursionError, ValueError])
def test_read_header_normalizes_json_resource_limit_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    path = tmp_path / "header.safetensors"
    _write_raw_safetensors(path, {})

    def fail_json_decode(_payload: str) -> object:
        raise error_type("injected JSON parser limit")

    monkeypatch.setattr(safetensors_module.json, "loads", fail_json_decode)
    with pytest.raises(ValueError, match="invalid safetensors header") as exc_info:
        read_header(path)

    assert isinstance(exc_info.value.__cause__, error_type)


# ---------------------------------------------------------------------------
# Path-vs-str - both accepted
# ---------------------------------------------------------------------------


def test_helpers_accept_str_paths(tmp_path: Path) -> None:
    out = tmp_path / "weights.safetensors"
    save_weights(str(out), _sample_weights())
    loaded = load_weights(str(out))
    assert "transformer.block_0.attn.weight" in loaded
