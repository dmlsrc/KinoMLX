"""Content-addressed tokenizer derivation and recovery contracts."""

import json
import struct
from pathlib import Path

import pytest

from kinomlx.models.ltx2.text_encoder.tokenizer import GemmaTokenizer
from kinomlx.models.ltx2.text_encoder.tokenizer_derivation import (
    TOKENIZER_DERIVATION_VERSION,
    derive_tokenizer,
    derive_tokenizer_model,
    resolve_tokenizer_source,
)

from ._tokenizer_fixtures import tokenizer_document, tokenizer_json


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
    model, metadata = derive_tokenizer_model(tokenizer_json())

    from sentencepiece import SentencePieceProcessor

    processor = SentencePieceProcessor(model_proto=model)
    assert processor.get_piece_size() == 266
    assert metadata["derivation_version"] == TOKENIZER_DERIVATION_VERSION
    assert metadata["special_ids"] == {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    assert metadata["special_token_ids"] == [0, 1, 2, 3, 264, 265]
    assert metadata["post_processor_prepends_bos"] is True
    assert metadata["wrapper_literal_tokens"] == {"<unk>": 3}
    assert processor.decode([2, 3]) == "<bos><unk>"


def test_derivation_preserves_dense_added_tokens_beyond_model_vocab() -> None:
    document = tokenizer_document()
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


def test_embedded_and_external_identical_json_share_content_address(tmp_path: Path) -> None:
    source_json = tokenizer_json()
    external = tmp_path / "tokenizer.json"
    external.write_bytes(source_json)
    packaged = tmp_path / "renamed-text-artifact.bin"
    _write_embedded_asset(packaged, source_json)

    external_cache = derive_tokenizer(
        external,
    )
    embedded_cache = derive_tokenizer(
        packaged,
    )

    assert resolve_tokenizer_source(packaged).embedded_key == "tokenizer_json"
    assert external_cache == embedded_cache


def test_tokenizer_uses_weightsd_bos_and_literal_token_policy(tmp_path: Path) -> None:
    source = tmp_path / "tokenizer.json"
    source.write_bytes(tokenizer_json(prepends_bos=True))
    cache = derive_tokenizer(
        source,
    )

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
        derive_tokenizer(
            source,
        )
