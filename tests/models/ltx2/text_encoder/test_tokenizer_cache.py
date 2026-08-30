"""Content-addressed tokenizer derivation and recovery contracts."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import kinomlx.models.ltx2.text_encoder.tokenizer_cache as cache_module
from kinomlx.models.ltx2.text_encoder.tokenizer import GemmaTokenizer
from kinomlx.models.ltx2.text_encoder.tokenizer_cache import (
    TOKENIZER_CACHE_SCHEMA_VERSION,
    derive_tokenizer_model,
    ensure_tokenizer_cache,
    resolve_tokenizer_source,
    tokenizer_cache_payload,
)


def _tokenizer_document(*, prepends_bos: bool = True) -> dict[str, object]:
    pieces = ["<pad>", "<eos>", "<bos>", "<unk>"]
    pieces.extend(f"<0x{value:02X}>" for value in range(256))
    pieces.extend(["a", "b", "ab", "\u2581", "<unused0>", "<|video|>"])
    added_tokens = [
        {"id": index, "content": pieces[index], "special": True}
        for index in (0, 1, 2, 3, len(pieces) - 2, len(pieces) - 1)
    ]
    post_processor: dict[str, object] | None = None
    if prepends_bos:
        post_processor = {
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ]
        }
    return {
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "byte_fallback": True,
            "vocab": {piece: index for index, piece in enumerate(pieces)},
            "merges": [["a", "b"]],
        },
        "added_tokens": added_tokens,
        "post_processor": post_processor,
    }


def _tokenizer_json(*, prepends_bos: bool = True) -> bytes:
    return json.dumps(
        _tokenizer_document(prepends_bos=prepends_bos),
        sort_keys=True,
    ).encode("utf-8")


def _write_embedded_asset(path: Path, payload: bytes) -> None:
    neighbor_bytes = 4096
    header = {
        "neighbor": {
            "dtype": "F32",
            "shape": [1024],
            "data_offsets": [0, neighbor_bytes],
        },
        "tokenizer_json": {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [neighbor_bytes, neighbor_bytes + len(payload)],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(neighbor_bytes) + payload)


def test_derivation_emits_loadable_model_and_wrapper_metadata() -> None:
    model, metadata = derive_tokenizer_model(_tokenizer_json())

    from sentencepiece import SentencePieceProcessor

    processor = SentencePieceProcessor(model_proto=model)
    assert processor.get_piece_size() == 266
    assert metadata["derivation_version"] == TOKENIZER_CACHE_SCHEMA_VERSION
    assert metadata["special_ids"] == {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    assert metadata["special_token_ids"] == [0, 1, 2, 3, 264, 265]
    assert metadata["post_processor_prepends_bos"] is True
    assert metadata["wrapper_literal_tokens"] == {"<unk>": 3}
    assert processor.decode([2, 3]) == "<bos><unk>"


def test_derivation_preserves_dense_added_tokens_beyond_model_vocab() -> None:
    document = _tokenizer_document()
    model = document["model"]
    assert isinstance(model, dict)
    vocab = model["vocab"]
    assert isinstance(vocab, dict)
    added = document["added_tokens"]
    assert isinstance(added, list)
    added.append({"id": len(vocab), "content": "<image_soft_token>"})

    blob, metadata = derive_tokenizer_model(json.dumps(document).encode("utf-8"))

    from sentencepiece import SentencePieceProcessor

    processor = SentencePieceProcessor(model_proto=blob)
    assert processor.piece_to_id("<image_soft_token>") == len(vocab)
    assert metadata["added_tokens"]["<image_soft_token>"] == len(vocab)


def test_tokenizer_cache_cold_build_warm_reuse_and_corruption_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "gemma"
    source_dir.mkdir()
    (source_dir / "tokenizer.json").write_bytes(_tokenizer_json())
    calls = 0
    original = cache_module.derive_tokenizer_model

    def counted(source: bytes):
        nonlocal calls
        calls += 1
        return original(source)

    monkeypatch.setattr(cache_module, "derive_tokenizer_model", counted)

    cold = ensure_tokenizer_cache(source_dir, cache_root=tmp_path / "cache")
    warm = ensure_tokenizer_cache(source_dir, cache_root=tmp_path / "cache")

    assert cold == warm
    assert calls == 1
    assert cold.model_path.is_file()
    assert cold.metadata_path.is_file()
    sidecar = json.loads(cold.metadata_path.read_text(encoding="utf-8"))
    assert sidecar["cache_payload"]["schema_version"] == TOKENIZER_CACHE_SCHEMA_VERSION

    cold.model_path.write_bytes(b"corrupt")
    repaired = ensure_tokenizer_cache(source_dir, cache_root=tmp_path / "cache")
    assert repaired == cold
    assert calls == 2
    assert repaired.model_path.read_bytes() != b"corrupt"

    cold.metadata_path.write_text("not json", encoding="utf-8")
    ensure_tokenizer_cache(source_dir, cache_root=tmp_path / "cache")
    assert calls == 3

    sidecar = json.loads(cold.metadata_path.read_text(encoding="utf-8"))
    sidecar["derivation"]["derivation_version"] = -1
    cold.metadata_path.write_text(json.dumps(sidecar), encoding="utf-8")
    ensure_tokenizer_cache(source_dir, cache_root=tmp_path / "cache")
    assert calls == 4


def test_source_json_change_selects_a_new_content_address(tmp_path: Path) -> None:
    source = tmp_path / "tokenizer.json"
    source.write_bytes(_tokenizer_json(prepends_bos=True))
    first = ensure_tokenizer_cache(source, cache_root=tmp_path / "cache")

    source.write_bytes(_tokenizer_json(prepends_bos=False))
    second = ensure_tokenizer_cache(source, cache_root=tmp_path / "cache")

    assert first.source_json_sha256 != second.source_json_sha256
    assert first.model_path.parent != second.model_path.parent
    assert first.model_path.is_file()
    assert second.model_path.is_file()


def test_embedded_and_external_identical_json_share_content_address(tmp_path: Path) -> None:
    source_json = _tokenizer_json()
    external = tmp_path / "tokenizer.json"
    external.write_bytes(source_json)
    packaged = tmp_path / "renamed-text-artifact.bin"
    _write_embedded_asset(packaged, source_json)

    external_cache = ensure_tokenizer_cache(external, cache_root=tmp_path / "cache")
    embedded_cache = ensure_tokenizer_cache(packaged, cache_root=tmp_path / "cache")

    assert resolve_tokenizer_source(packaged).embedded_key == "tokenizer_json"
    assert external_cache == embedded_cache


def test_cache_payload_is_content_and_policy_only() -> None:
    payload = tokenizer_cache_payload(_tokenizer_json())

    assert set(payload) == {
        "schema_version",
        "source_json_sha256",
        "serialization_policy",
    }
    assert "path" not in payload
    assert "model_version" not in payload


def test_tokenizer_uses_cached_bos_and_literal_token_policy(tmp_path: Path) -> None:
    source = tmp_path / "tokenizer.json"
    source.write_bytes(_tokenizer_json(prepends_bos=True))
    cache = ensure_tokenizer_cache(source, cache_root=tmp_path / "cache")

    tokenizer = GemmaTokenizer(cache)
    ids = tokenizer.encode_ids(" <unk>a ", max_length=16)

    assert ids[0] == 2
    assert ids[1] == 3
    assert 1 not in ids
    assert tokenizer.decode_ids(ids, skip_special_tokens=False) == "<bos><unk>a"
    assert tokenizer.decode_ids(ids, skip_special_tokens=True) == "a"


def test_directory_source_never_falls_back_to_stock_tokenizer_model(tmp_path: Path) -> None:
    source = tmp_path / "gemma"
    source.mkdir()
    (source / "tokenizer.model").write_bytes(b"stock model is not source truth")

    with pytest.raises(FileNotFoundError, match="expected tokenizer.json"):
        ensure_tokenizer_cache(source, cache_root=tmp_path / "cache")


def test_warm_tokenizer_station_does_not_import_build_or_huggingface_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tokenizer.json"
    source.write_bytes(_tokenizer_json())
    cache_root = tmp_path / "cache"
    ensure_tokenizer_cache(source, cache_root=cache_root)
    probe = f"""
import sys

BLOCKED = ("google.protobuf", "tokenizers", "huggingface_hub")

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED):
            raise ImportError(f"{{name}} blocked during warm tokenizer load")
        return None

sys.meta_path.insert(0, Blocker())

from kinomlx.models.ltx2.text_encoder import GemmaTokenizer

tokenizer = GemmaTokenizer({str(source)!r}, cache_root={str(cache_root)!r})
assert tokenizer.encode_ids("plain prompt")[0] == 2
offenders = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED)
)
assert not offenders, offenders
"""
    process = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert process.returncode == 0, process.stderr
