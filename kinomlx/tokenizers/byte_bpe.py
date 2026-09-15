"""Dependency-free Qwen ByteLevel-BPE execution from derived JSON tables.

The source contract follows the Qwen tokenizer JSON consumed by the pinned
Qwen implementation.
"""

import itertools
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import cast

from kinomlx._json import JsonObject, JsonValue

from .unicode_scan import QWEN_SPLIT_PATTERN, split_qwen_unicode


def _mapping(value: JsonValue | None, *, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _string(value: JsonValue | None, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _boolean(value: JsonValue | None, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _integer(value: JsonValue | None, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _token_content(value: JsonValue | None, *, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    raise ValueError(f"{field} must name a nonempty token")


def _bytes_to_unicode() -> tuple[dict[int, str], dict[str, int]]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values.extend(range(ord("\u00a1"), ord("\u00ac") + 1))
    byte_values.extend(range(ord("\u00ae"), ord("\u00ff") + 1))
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value in byte_values:
            continue
        byte_values.append(value)
        codepoints.append(256 + extra)
        extra += 1
    encoder = dict(zip(byte_values, map(chr, codepoints), strict=True))
    return encoder, {character: value for value, character in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _bytes_to_unicode()
_BPE_CACHE_CAPACITY = 10_000
_BPE_CACHE_MAX_KEY_LENGTH = 256
QWEN_TOKENIZER_TABLE_SCHEMA_VERSION = 2
QWEN_TOKENIZER_TABLE_POLICY = "qwen-bytelevel-bpe-nfc-v2"


@dataclass(frozen=True)
class AddedToken:
    """One literal token extracted before Unicode normalization."""

    token_id: int
    content: str
    special: bool


@dataclass(frozen=True)
class ByteBPETables:
    """Validated semantic tables needed by the Qwen tokenizer."""

    vocab: dict[str, int]
    merges: tuple[tuple[str, str], ...]
    added_tokens: tuple[AddedToken, ...]
    pad_token: str
    pad_token_id: int
    eos_token: str
    eos_token_id: int

    @classmethod
    def from_source(
        cls,
        tokenizer: JsonObject,
        tokenizer_config: JsonObject,
    ) -> ByteBPETables:
        """Validate and canonicalize the supported Qwen tokenizer JSON."""
        normalizer = _mapping(tokenizer.get("normalizer"), field="normalizer")
        if normalizer != {"type": "NFC"}:
            raise ValueError("Qwen tokenizer requires the NFC normalizer")

        pre = _mapping(tokenizer.get("pre_tokenizer"), field="pre_tokenizer")
        if pre.get("type") != "Sequence":
            raise ValueError("Qwen tokenizer requires the Qwen pre-tokenizer sequence")
        raw_steps = pre.get("pretokenizers")
        if not isinstance(raw_steps, list) or len(raw_steps) != 2:
            raise ValueError("Qwen tokenizer requires two pre-tokenizer steps")
        split = _mapping(raw_steps[0], field="pre_tokenizer.pretokenizers[0]")
        byte_level = _mapping(raw_steps[1], field="pre_tokenizer.pretokenizers[1]")
        pattern = _mapping(split.get("pattern"), field="pre_tokenizer split pattern")
        if (
            split.get("type") != "Split"
            or pattern.get("Regex") != QWEN_SPLIT_PATTERN
            or split.get("behavior") != "Isolated"
            or split.get("invert") is not False
        ):
            raise ValueError("unsupported Qwen Unicode split configuration")
        # Offset trimming changes alignment metadata, not token IDs. Both Qwen3
        # and Qwen3-VL sources are supported; this runtime only exposes IDs.
        if not isinstance(byte_level.get("trim_offsets"), bool) or byte_level != {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": byte_level.get("trim_offsets"),
            "use_regex": False,
        }:
            raise ValueError("unsupported Qwen ByteLevel pre-tokenizer")

        post = _mapping(tokenizer.get("post_processor"), field="post_processor")
        if post != {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }:
            raise ValueError("unsupported Qwen ByteLevel post-processor")
        decoder = _mapping(tokenizer.get("decoder"), field="decoder")
        # Decoder byte conversion does not consume pre-tokenization options.
        if decoder not in (
            {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": False,
                "use_regex": False,
            },
        ):
            raise ValueError("unsupported Qwen ByteLevel decoder")

        model = _mapping(tokenizer.get("model"), field="model")
        required_model_values: dict[str, JsonValue] = {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        }
        for field, expected in required_model_values.items():
            if model.get(field) != expected:
                raise ValueError(f"unsupported Qwen tokenizer model field {field}")

        raw_vocab = _mapping(model.get("vocab"), field="model.vocab")
        vocab: dict[str, int] = {}
        ids: set[int] = set()
        for token, raw_id in raw_vocab.items():
            token_id = _integer(raw_id, field=f"model.vocab[{token!r}]")
            if token_id in ids:
                raise ValueError(f"Qwen tokenizer repeats vocabulary id {token_id}")
            vocab[token] = token_id
            ids.add(token_id)

        raw_merges = model.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("model.merges must be an array")
        merges: list[tuple[str, str]] = []
        seen_merges: set[tuple[str, str]] = set()
        for index, raw_merge in enumerate(raw_merges):
            if isinstance(raw_merge, str):
                parts = raw_merge.split(" ")
            elif isinstance(raw_merge, list) and len(raw_merge) == 2:
                parts = raw_merge
            else:
                raise ValueError(f"model.merges[{index}] must contain two tokens")
            if len(parts) != 2 or not all(isinstance(part, str) and part for part in parts):
                raise ValueError(f"model.merges[{index}] must contain two nonempty strings")
            pair = cast(tuple[str, str], tuple(parts))
            if pair in seen_merges:
                raise ValueError(f"Qwen tokenizer repeats merge pair {pair!r}")
            missing_operands = tuple(part for part in pair if part not in vocab)
            if missing_operands:
                raise ValueError(
                    "Qwen tokenizer merge operand is absent from vocabulary: "
                    f"{missing_operands[0]!r} in {pair!r}"
                )
            if pair[0] + pair[1] not in vocab:
                raise ValueError(f"Qwen tokenizer merge result is absent from vocabulary: {pair!r}")
            merges.append(pair)
            seen_merges.add(pair)

        raw_added = tokenizer.get("added_tokens")
        if not isinstance(raw_added, list):
            raise ValueError("added_tokens must be an array")
        added: list[AddedToken] = []
        added_contents: set[str] = set()
        for index, raw_token in enumerate(raw_added):
            value = _mapping(raw_token, field=f"added_tokens[{index}]")
            content = _string(value.get("content"), field=f"added_tokens[{index}].content")
            token_id = _integer(value.get("id"), field=f"added_tokens[{index}].id")
            flags = {
                field: _boolean(
                    value.get(field),
                    field=f"added_tokens[{index}].{field}",
                )
                for field in ("single_word", "lstrip", "rstrip", "normalized", "special")
            }
            if any(flags[field] for field in ("single_word", "lstrip", "rstrip", "normalized")):
                raise ValueError("Qwen tokenizer requires unstripped literal added tokens")
            special = flags["special"]
            if token_id in ids or content in vocab or content in added_contents:
                raise ValueError(f"Qwen tokenizer repeats added token {content!r}")
            ids.add(token_id)
            added_contents.add(content)
            added.append(AddedToken(token_id=token_id, content=content, special=special))

        token_ids = dict(vocab)
        token_ids.update((token.content, token.token_id) for token in added)
        pad_token = _token_content(tokenizer_config.get("pad_token"), field="pad_token")
        eos_token = _token_content(tokenizer_config.get("eos_token"), field="eos_token")
        try:
            pad_token_id = token_ids[pad_token]
            eos_token_id = token_ids[eos_token]
        except KeyError as exc:
            raise ValueError(
                f"configured Qwen token is absent from tokenizer: {exc.args[0]!r}"
            ) from exc
        if tokenizer_config.get("clean_up_tokenization_spaces") not in {None, False}:
            raise ValueError("Qwen tokenizer does not support cleanup tokenization spaces")
        return cls(
            vocab=vocab,
            merges=tuple(merges),
            added_tokens=tuple(added),
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            eos_token=eos_token,
            eos_token_id=eos_token_id,
        )

    @classmethod
    def from_payload(cls, payload: JsonObject) -> ByteBPETables:
        """Load the canonical tokenizer-table representation."""
        if (
            payload.get("schema_version") != QWEN_TOKENIZER_TABLE_SCHEMA_VERSION
            or payload.get("policy") != QWEN_TOKENIZER_TABLE_POLICY
        ):
            raise ValueError("unsupported Qwen tokenizer tables")
        source: JsonObject = {
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": QWEN_SPLIT_PATTERN},
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": True,
                        "use_regex": False,
                    },
                ],
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": False,
                "use_regex": False,
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": False,
                "byte_fallback": False,
                "ignore_merges": False,
                "vocab": payload.get("vocab"),
                "merges": payload.get("merges"),
            },
            "added_tokens": payload.get("added_tokens"),
        }
        config: JsonObject = {
            "pad_token": payload.get("pad_token"),
            "eos_token": payload.get("eos_token"),
            "clean_up_tokenization_spaces": False,
        }
        tables = cls.from_source(source, config)
        if payload.get("pad_token_id") != tables.pad_token_id:
            raise ValueError("Qwen tokenizer tables has the wrong pad token id")
        if payload.get("eos_token_id") != tables.eos_token_id:
            raise ValueError("Qwen tokenizer tables has the wrong EOS token id")
        return tables

    def to_payload(self) -> JsonObject:
        """Return the deterministic table representation."""
        return {
            "schema_version": QWEN_TOKENIZER_TABLE_SCHEMA_VERSION,
            "policy": QWEN_TOKENIZER_TABLE_POLICY,
            "vocab": dict(self.vocab),
            "merges": [[left, right] for left, right in self.merges],
            "added_tokens": [
                {
                    "id": token.token_id,
                    "content": token.content,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": token.special,
                }
                for token in self.added_tokens
            ],
            "pad_token": self.pad_token,
            "pad_token_id": self.pad_token_id,
            "eos_token": self.eos_token,
            "eos_token_id": self.eos_token_id,
        }


class QwenByteBPETokenizer:
    """Execute the validated Qwen ByteLevel-BPE tables without Tokenizers."""

    def __init__(self, tables: ByteBPETables) -> None:
        self.tables = tables
        self.pad_token_id = tables.pad_token_id
        self.eos_token_id = tables.eos_token_id
        self._merge_ranks = {pair: rank for rank, pair in enumerate(tables.merges)}
        self._id_to_token = {token_id: token for token, token_id in tables.vocab.items()}
        self._added_by_id = {token.token_id: token for token in tables.added_tokens}
        self._added_by_content = tuple(
            sorted(tables.added_tokens, key=lambda token: (-len(token.content), token.token_id))
        )
        self._bpe_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            self._bpe_cache.move_to_end(token)
            return cached
        pieces = list(token)
        previous = list(range(-1, len(pieces) - 1))
        following = [*range(1, len(pieces)), -1]
        ranks = self._merge_ranks
        # Rank first, then original position: equal-ranked merges are leftmost
        # first. Linked neighbors let each merge update only its two new pairs,
        # avoiding a full-word rescan for every distinct merge rank.
        pending = [
            (rank, index, index + 1)
            for index, pair in enumerate(itertools.pairwise(pieces))
            if (rank := ranks.get(pair)) is not None
        ]
        heapify(pending)
        while pending:
            rank, left, right = heappop(pending)
            if following[left] != right or ranks.get((pieces[left], pieces[right])) != rank:
                continue  # A neighboring merge invalidated this queued pair.
            pieces[left] += pieces[right]
            pieces[right] = ""
            successor = following[right]
            following[left] = successor
            following[right] = -1
            if successor >= 0:
                previous[successor] = left
            for start in (previous[left], left):
                if start < 0:
                    continue
                end = following[start]
                if end >= 0 and (new_rank := ranks.get((pieces[start], pieces[end]))) is not None:
                    heappush(pending, (new_rank, start, end))
        word = tuple(piece for piece in pieces if piece)
        if len(token) <= _BPE_CACHE_MAX_KEY_LENGTH:
            self._bpe_cache[token] = word
            self._bpe_cache.move_to_end(token)
            if len(self._bpe_cache) > _BPE_CACHE_CAPACITY:
                self._bpe_cache.popitem(last=False)
        return word

    def _encode_plain(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", text)
        encoded: list[int] = []
        for piece in split_qwen_unicode(normalized):
            byte_token = "".join(_BYTE_ENCODER[value] for value in piece.encode("utf-8"))
            for token in self._bpe(byte_token):
                try:
                    encoded.append(self.tables.vocab[token])
                except KeyError as exc:
                    raise ValueError(f"Qwen tokenizer vocabulary has no token {token!r}") from exc
        return encoded

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode one string with literal added-token extraction."""
        if not isinstance(text, str):
            raise TypeError("Qwen tokenizer input must be a string")
        result: list[int] = []
        plain_start = 0
        index = 0
        while index < len(text):
            matched = next(
                (
                    token
                    for token in self._added_by_content
                    if text.startswith(token.content, index)
                ),
                None,
            )
            if matched is None:
                index += 1
                continue
            if plain_start < index:
                result.extend(self._encode_plain(text[plain_start:index]))
            result.append(matched.token_id)
            index += len(matched.content)
            plain_start = index
        if plain_start < len(text):
            result.extend(self._encode_plain(text[plain_start:]))
        return tuple(result)

    def decode(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode IDs for diagnostics with the configured ByteLevel decoder."""
        output: list[str] = []
        byte_values: list[int] = []

        def flush() -> None:
            if not byte_values:
                return
            output.append(bytes(byte_values).decode("utf-8", errors="replace"))
            byte_values.clear()

        for token_id in token_ids:
            added = self._added_by_id.get(token_id)
            if added is not None:
                flush()
                if not (skip_special_tokens and added.special):
                    output.append(added.content)
                continue
            try:
                token = self._id_to_token[token_id]
            except KeyError as exc:
                raise ValueError(f"Qwen tokenizer has no id {token_id}") from exc
            try:
                byte_values.extend(_BYTE_DECODER[character] for character in token)
            except KeyError as exc:
                raise ValueError(
                    f"Qwen tokenizer token is not valid ByteLevel text: {token!r}"
                ) from exc
        flush()
        return "".join(output)


__all__ = [
    "QWEN_TOKENIZER_TABLE_POLICY",
    "QWEN_TOKENIZER_TABLE_SCHEMA_VERSION",
    "AddedToken",
    "ByteBPETables",
    "QwenByteBPETokenizer",
]
