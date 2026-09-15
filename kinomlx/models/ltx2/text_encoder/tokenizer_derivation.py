"""In-memory SentencePiece derivation from checkpoint tokenizer JSON."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sentencepiece import sentencepiece_model_pb2 as sentencepiece_model

from kinomlx._json import JsonObject
from kinomlx.io.safetensors import read_u8_tensor

TOKENIZER_DERIVATION_VERSION = 3
TOKENIZER_SERIALIZATION_POLICY = "sentencepiece-bpe-modelproto-v2"
EMBEDDED_TOKENIZER_KEY = "tokenizer_json"

_BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")
_WHITESPACE_RUN = re.compile(r"^(?:\n+|\t+|\r+| +|\u2581{2,})$")
_RESERVED_SHAPE = re.compile(
    r"^(?:<unused\d+>|<\|[^<>]*>|<[^<>|]*\|>|<mask>|\[multimodal\]"
    r"|\n+|\t+|\r+| +|\u2581{2,})$"
)

# SentencePiece ModelProto piece-type values.
_NORMAL = 1
_UNKNOWN = 2
_USER_DEFINED = 4
_BYTE = 6


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TokenizerSource:
    """A tokenizer JSON source outside or inside a safetensors artifact."""

    path: Path
    embedded_key: str | None = None

    def read_json_bytes(self) -> bytes:
        if self.embedded_key is None:
            return self.path.read_bytes()

        return read_u8_tensor(self.path, self.embedded_key)


@dataclass(frozen=True)
class TokenizerDerivation:
    """Tokenizer bytes and the wrapper policy derived from one source."""

    source_json_sha256: str
    model: bytes
    derivation: JsonObject


def resolve_tokenizer_source(location: Path | str) -> TokenizerSource:
    """Resolve a Gemma directory, tokenizer JSON, or packaged text artifact."""
    path = Path(location).expanduser().absolute()
    if path.is_dir():
        candidate = path / "tokenizer.json"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Gemma tokenizer not found at {candidate}; expected tokenizer.json"
            )
        return TokenizerSource(candidate)
    if not path.is_file():
        raise FileNotFoundError(f"no tokenizer source at {path}")
    if path.suffix == ".json":
        return TokenizerSource(path)
    return TokenizerSource(path, embedded_key=EMBEDDED_TOKENIZER_KEY)


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"tokenizer JSON {field} must be an object")
    return value


def _vocabulary(model: Mapping[str, object]) -> tuple[tuple[str, ...], dict[str, int]]:
    raw = _require_mapping(model.get("vocab"), field="model.vocab")
    by_id: dict[int, str] = {}
    by_piece: dict[str, int] = {}
    for piece, raw_index in raw.items():
        if not isinstance(piece, str):
            raise ValueError("tokenizer JSON vocabulary pieces must be strings")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ValueError("tokenizer JSON vocabulary ids must be non-negative integers")
        if raw_index in by_id:
            raise ValueError(f"tokenizer JSON repeats vocabulary id {raw_index}")
        by_id[raw_index] = piece
        by_piece[piece] = raw_index
    if sorted(by_id) != list(range(len(by_id))):
        raise ValueError("tokenizer JSON vocabulary ids must be dense")
    return tuple(by_id[index] for index in range(len(by_id))), by_piece


def _added_tokens(document: Mapping[str, object]) -> tuple[dict[int, str], set[int]]:
    raw = document.get("added_tokens", ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("tokenizer JSON added_tokens must be an array")
    result: dict[int, str] = {}
    special_ids: set[int] = set()
    for item in raw:
        entry = _require_mapping(item, field="added_tokens entry")
        index = entry.get("id")
        content = entry.get("content")
        special = entry.get("special", False)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(content, str)
            or not isinstance(special, bool)
        ):
            raise ValueError("tokenizer JSON added token ids/content/policy are invalid")
        if index in result and result[index] != content:
            raise ValueError(f"tokenizer JSON repeats added token id {index}")
        result[index] = content
        if special:
            special_ids.add(index)
    return result, special_ids


def _post_processor_prepends_bos(document: Mapping[str, object]) -> bool:
    processor = document.get("post_processor")
    if not isinstance(processor, Mapping):
        return False
    single = processor.get("single")
    if not isinstance(single, Sequence) or isinstance(single, str | bytes) or not single:
        return False
    first = single[0]
    if not isinstance(first, Mapping):
        return False
    token = first.get("SpecialToken")
    return isinstance(token, Mapping) and token.get("id") == "<bos>"


def derive_tokenizer_model(tokenizer_json: bytes) -> tuple[bytes, dict[str, object]]:
    """Derive a SentencePiece BPE ModelProto and immutable wrapper metadata."""
    try:
        document = json.loads(tokenizer_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("tokenizer JSON is invalid") from exc
    document = _require_mapping(document, field="root")
    model = _require_mapping(document.get("model"), field="model")
    if model.get("type") != "BPE":
        raise ValueError(f"unsupported tokenizer model type: {model.get('type')!r}")
    base_pieces, by_piece = _vocabulary(model)
    added, declared_special_ids = _added_tokens(document)
    appended = {index: content for index, content in added.items() if index >= len(base_pieces)}
    if sorted(appended) != list(range(len(base_pieces), len(base_pieces) + len(appended))):
        raise ValueError("tokenizer JSON appended token ids must be dense")
    for index, content in added.items():
        if index < len(base_pieces) and base_pieces[index] != content:
            raise ValueError("tokenizer JSON added tokens disagree with the vocabulary")
        if index >= len(base_pieces) and content in by_piece:
            raise ValueError("tokenizer JSON appended token repeats a vocabulary piece")
    pieces = base_pieces + tuple(appended[index] for index in sorted(appended))
    by_piece.update(
        {piece: index for index, piece in enumerate(pieces[len(base_pieces) :], len(base_pieces))}
    )

    byte_ids = [index for index, piece in enumerate(pieces) if _BYTE_PIECE.fullmatch(piece)]
    if len(byte_ids) != 256 or byte_ids != list(range(byte_ids[0], byte_ids[0] + 256)):
        raise ValueError("expected one contiguous 256-piece byte-fallback block")
    first_normal = byte_ids[-1] + 1
    last_normal = len(base_pieces) - 1
    while last_normal >= first_normal and _RESERVED_SHAPE.fullmatch(base_pieces[last_normal]):
        last_normal -= 1

    unk_piece = model.get("unk_token") or "<unk>"
    if not isinstance(unk_piece, str) or unk_piece not in by_piece:
        raise ValueError("tokenizer JSON has no valid unknown token")
    required_specials = ("<pad>", "<eos>", "<bos>", unk_piece)
    missing = [piece for piece in required_specials if piece not in by_piece]
    if missing:
        raise ValueError(f"tokenizer JSON is missing special token {missing[0]!r}")

    # Protobuf generates these attributes dynamically from descriptors and
    # does not expose them to static analysis.
    proto = sentencepiece_model.ModelProto()  # type: ignore[attr-defined]
    for index, piece in enumerate(pieces):
        entry = proto.pieces.add()
        entry.piece = piece
        if _BYTE_PIECE.fullmatch(piece):
            entry.type = _BYTE
        elif piece == unk_piece:
            entry.type = _UNKNOWN
        elif index in added or _WHITESPACE_RUN.fullmatch(piece):
            entry.type = _USER_DEFINED
        elif first_normal <= index <= last_normal:
            entry.type = _NORMAL
            entry.score = -float(index - first_normal)
        else:
            entry.type = _NORMAL
            entry.score = -float(len(pieces) + index)

    normalizer = proto.normalizer_spec
    normalizer.name = "identity"
    normalizer.precompiled_charsmap = b""
    normalizer.add_dummy_prefix = False
    normalizer.remove_extra_whitespaces = False

    trainer = proto.trainer_spec
    trainer.model_type = sentencepiece_model.TrainerSpec.BPE  # type: ignore[attr-defined]
    trainer.vocab_size = len(pieces)
    trainer.input_format = "tsv"
    trainer.split_by_unicode_script = True
    trainer.split_by_whitespace = False
    trainer.split_by_number = True
    trainer.split_digits = True
    trainer.allow_whitespace_only_pieces = True
    trainer.byte_fallback = bool(model.get("byte_fallback", True))
    trainer.pad_id = by_piece["<pad>"]
    trainer.eos_id = by_piece["<eos>"]
    trainer.bos_id = by_piece["<bos>"]
    trainer.unk_id = by_piece[unk_piece]
    trainer.pad_piece = "<pad>"
    trainer.eos_piece = "<eos>"
    trainer.bos_piece = "<bos>"
    trainer.unk_piece = unk_piece
    # SentencePiece otherwise decodes the UNKNOWN id to its default replacement
    # glyph. The tokenizer JSON contract decodes that id to the literal token.
    trainer.unk_surface = unk_piece

    required_special_ids = {by_piece[piece] for piece in required_specials}

    metadata: dict[str, object] = {
        "derivation_version": TOKENIZER_DERIVATION_VERSION,
        "serialization_policy": TOKENIZER_SERIALIZATION_POLICY,
        "vocab_size": len(pieces),
        "model_vocab_size": len(base_pieces),
        "normal_block": [first_normal, last_normal],
        "special_ids": {piece: by_piece[piece] for piece in required_specials},
        "special_token_ids": sorted(declared_special_ids | required_special_ids),
        "added_tokens": {
            content: index for index, content in sorted(added.items(), key=lambda item: item[0])
        },
        "post_processor_prepends_bos": _post_processor_prepends_bos(document),
        # SentencePiece requires UNKNOWN to remain non-matchable. The wrapper
        # restores literal-token behavior from this explicit ledger.
        "wrapper_literal_tokens": (
            {unk_piece: by_piece[unk_piece]} if by_piece[unk_piece] in added else {}
        ),
    }
    return proto.SerializeToString(), metadata


def _derive_tokenizer_bytes(source_json: bytes) -> TokenizerDerivation:
    model, derivation = derive_tokenizer_model(source_json)
    return TokenizerDerivation(
        source_json_sha256=_sha256(source_json),
        model=model,
        derivation=cast(JsonObject, derivation),
    )


def derive_tokenizer(location: Path | str | TokenizerSource) -> TokenizerDerivation:
    """Derive the tokenizer in memory from its source JSON without any persistence."""
    source = (
        location if isinstance(location, TokenizerSource) else resolve_tokenizer_source(location)
    )
    return _derive_tokenizer_bytes(source.read_json_bytes())
