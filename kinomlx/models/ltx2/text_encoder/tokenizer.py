"""SentencePiece-backed Gemma tokenizer for prompt encoding."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from .tokenizer_cache import (
    TokenizerCache,
    ensure_tokenizer_cache,
    load_tokenizer_derivation,
)

_log = logging.getLogger(__name__)
_VISIBLE_WHITESPACE = frozenset({" ", "\n", "\t"})
_CONTROL_NAMES = {
    "\r": "CARRIAGE RETURN",
    "\x85": "NEXT LINE",
}
_MAX_DISTINCT_WARNINGS = 12
_MAX_OFFSETS_PER_CHARACTER = 8


def _is_invisible_prompt_character(character: str) -> bool:
    category = unicodedata.category(character)
    if category == "Cf":
        return True
    if category == "Cc" and character not in {"\n", "\t"}:
        return True
    return character.isspace() and character not in _VISIBLE_WHITESPACE


def _warn_invisible_prompt_characters(prompt: str) -> None:
    offsets_by_character: dict[str, list[int]] = {}
    counts: dict[str, int] = {}
    for offset, character in enumerate(prompt):
        if not _is_invisible_prompt_character(character):
            continue
        counts[character] = counts.get(character, 0) + 1
        offsets = offsets_by_character.setdefault(character, [])
        if len(offsets) < _MAX_OFFSETS_PER_CHARACTER:
            offsets.append(offset)
    if not counts:
        return

    details: list[str] = []
    for character in list(counts)[:_MAX_DISTINCT_WARNINGS]:
        codepoint = f"U+{ord(character):04X}"
        name = _CONTROL_NAMES.get(
            character,
            unicodedata.name(character, "UNNAMED CONTROL CHARACTER"),
        )
        rendered_offsets = ",".join(str(value) for value in offsets_by_character[character])
        if counts[character] > len(offsets_by_character[character]):
            rendered_offsets += ",..."
        details.append(
            f"{codepoint} {name} (count={counts[character]}, offsets={rendered_offsets})"
        )
    if len(counts) > _MAX_DISTINCT_WARNINGS:
        details.append(f"and {len(counts) - _MAX_DISTINCT_WARNINGS} more distinct characters")
    _log.warning(
        "Prompt contains invisible Unicode characters: %s. Prompt text is unchanged.",
        "; ".join(details),
    )


class GemmaTokenizer:
    """Encode plain prompt text to BOS-prefixed MLX token arrays."""

    def __init__(
        self,
        source: Path | str | TokenizerCache,
        *,
        cache_root: Path | str | None = None,
    ) -> None:
        import sentencepiece

        cache = (
            source
            if isinstance(source, TokenizerCache)
            else ensure_tokenizer_cache(source, cache_root=cache_root)
        )
        self._processor = sentencepiece.SentencePieceProcessor(model_file=str(cache.model_path))
        metadata = load_tokenizer_derivation(cache)
        special = metadata.get("special_ids")
        added = metadata.get("added_tokens")
        special_token_ids = metadata.get("special_token_ids")
        literal = metadata.get("wrapper_literal_tokens")
        prepends_bos = metadata.get("post_processor_prepends_bos")
        if not isinstance(special, dict):
            raise ValueError(f"tokenizer cache has no special-token ids: {cache.metadata_path}")
        if not isinstance(added, dict):
            raise ValueError(f"tokenizer cache has no added-token ledger: {cache.metadata_path}")
        if not isinstance(special_token_ids, list):
            raise ValueError(f"tokenizer cache has no special-token ledger: {cache.metadata_path}")
        if not isinstance(literal, dict):
            raise ValueError(f"tokenizer cache has no literal-token ledger: {cache.metadata_path}")
        if not isinstance(prepends_bos, bool):
            raise ValueError(f"tokenizer cache has no BOS policy: {cache.metadata_path}")
        try:
            self.bos_id = int(special["<bos>"])
            self.eos_id = int(special["<eos>"])
            self.pad_id = int(special["<pad>"])
            self._added_token_ids = frozenset(int(index) for index in added.values())
            self._special_token_ids = frozenset(int(index) for index in special_token_ids)
            self._literal_tokens = {str(token): int(index) for token, index in literal.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"tokenizer cache metadata is invalid: {cache.metadata_path}") from exc
        if not set(self._literal_tokens.values()) <= self._added_token_ids:
            raise ValueError(
                f"tokenizer cache literal tokens are not added tokens: {cache.metadata_path}"
            )
        self._always_prepend_bos = prepends_bos
        self._splitter = (
            re.compile(
                "("
                + "|".join(
                    re.escape(token)
                    for token in sorted(self._literal_tokens, key=len, reverse=True)
                )
                + ")"
            )
            if self._literal_tokens
            else None
        )

    def _encode_text(self, text: str) -> list[int]:
        splitter = getattr(self, "_splitter", None)
        if splitter is None:
            return list(self._processor.encode(text))
        literal_tokens = getattr(self, "_literal_tokens", {})
        ids: list[int] = []
        for segment in splitter.split(text):
            if not segment:
                continue
            literal = literal_tokens.get(segment)
            if literal is not None:
                ids.append(literal)
            else:
                ids.extend(self._processor.encode(segment))
        return ids

    def encode(
        self,
        prompt: str,
        *,
        max_length: int = 1024,
        pad_to_max: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Return left-padded stock tokens, or compact tokens when requested."""
        ids = self.encode_ids(prompt, max_length=max_length)
        real_length = len(ids)
        if pad_to_max:
            pad_length = max_length - real_length
            ids = [self.pad_id] * pad_length + ids
            mask = [0] * pad_length + [1] * real_length
        else:
            mask = [1] * real_length
        return (
            mx.array(ids, dtype=mx.int32)[None, :],
            mx.array(mask, dtype=mx.int32)[None, :],
        )

    def encode_ids(self, prompt: str, *, max_length: int = 1024) -> list[int]:
        """Return the pure-Python token ids before optional left padding."""
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        normalized_prompt = (prompt or "").strip()
        _warn_invisible_prompt_characters(normalized_prompt)
        ids = self._encode_text(normalized_prompt)
        if getattr(self, "_always_prepend_bos", False):
            ids = [self.bos_id, *ids]
        ids = ids[:max_length]
        if not ids or ids[0] != self.bos_id:
            ids = [self.bos_id, *ids][:max_length]
        return ids

    def decode_ids(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode ids with the special-token policy recorded in the cache."""
        values = [int(index) for index in ids]
        if skip_special_tokens:
            values = [index for index in values if index not in self._special_token_ids]
        return str(self._processor.decode(values))


__all__ = ["GemmaTokenizer"]
