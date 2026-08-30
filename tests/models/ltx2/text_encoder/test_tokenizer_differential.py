"""Real tokenizer.json differentials for the shared LTX tokenizer station."""

from __future__ import annotations

import hashlib
import json
import random
import string
from pathlib import Path

import pytest

from kinomlx.models.ltx2.text_encoder import GemmaTokenizer
from kinomlx.models.ltx2.text_encoder.tokenizer_cache import (
    ensure_tokenizer_cache,
    resolve_tokenizer_source,
)
from kinomlx.settings import Settings

tokenizers = pytest.importorskip("tokenizers")
Tokenizer = tokenizers.Tokenizer

_CORPUS_SHA256 = "c21abf4b03e48ef9fda8eacdc77f370843e69aad12d16843f0858724571c7830"
_LITERAL_SPECIALS = ("<bos>", "<eos>", "<pad>", "<unk>")


def _random_text(rng: random.Random, kind: int) -> str:
    if kind == 0:
        alphabet = string.ascii_letters + string.digits + string.punctuation + " \t\n"
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 220)))
    if kind == 1:
        blocks = (
            (0x00A1, 0x024F),
            (0x0370, 0x052F),
            (0x3040, 0x30FF),
            (0x4E00, 0x9FFF),
            (0xAC00, 0xD7A3),
            (0x1F300, 0x1F64F),
        )
        return "".join(
            chr(rng.randrange(start, end + 1))
            for start, end in (rng.choice(blocks) for _ in range(rng.randrange(0, 90)))
        )
    if kind == 2:
        fields = ("frame", "motion", "light", "camera", "subject", "42")
        gaps = (" ", "  ", "\t", "\n", "\n\n")
        return "".join(rng.choice(fields) + rng.choice(gaps) for _ in range(rng.randrange(1, 24)))
    tokens = (
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<mask>",
        "<unused7>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<|tool>",
        "<|video|>",
    )
    return " ".join(
        rng.choice(("before", "after", "scene", "at dusk", rng.choice(tokens)))
        for _ in range(rng.randrange(1, 18))
    )


def _corpus() -> tuple[str, ...]:
    curated = [
        "",
        " ",
        "plain prompt",
        "  leading and trailing  ",
        "multiple   interior spaces",
        "tabs\tand\nnewlines",
        "caf\u00e9 na\u00efve Z\u00fcrich r\u00e9sum\u00e9",
        "\u65e5\u672c\u8a9e \u4e2d\u6587 \ud55c\uad6d\uc5b4",
        "\u041f\u0440\u0438\u0432\u0435\u0442 \u03ba\u03cc\u03c3\u03bc\u03b5",
        "emoji \U0001f3ac \U0001f680 \U0001f30a",
        "combining e\u0301 a\u0300 n\u0303",
        "punctuation !?.,;: [] {} () <> / \\ |",
        "3.14159265358979 and 2026-08-21T12:34:56Z",
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "before <bos> after",
        "before <eos> after",
        "before <pad> after",
        "before <unk> after",
        "<start_of_turn>user\nhello<end_of_turn>",
        "word " * 1400,
        "a" * 3000,
    ]
    rng = random.Random(20260821)
    curated.extend(_random_text(rng, index % 4) for index in range(2048))
    return tuple(curated)


def _corpus_sha256(corpus: tuple[str, ...]) -> str:
    payload = json.dumps(corpus, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _snapshot_with(root: Path, relative: str) -> Path | None:
    if not root.is_dir():
        return None
    return next(
        (snapshot for snapshot in sorted(root.iterdir()) if (snapshot / relative).is_file()),
        None,
    )


def _real_source(generation: str) -> Path:
    hf_home = Settings.from_env().hf_home
    if generation == "2.3":
        snapshot = _snapshot_with(
            hf_home / "hub/models--google--gemma-3-12b-it/snapshots",
            "tokenizer.json",
        )
        if snapshot is None:
            pytest.skip("Gemma 3 tokenizer.json is not cached")
        return snapshot
    if generation == "2.5":
        relative = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        snapshot = _snapshot_with(
            hf_home / "hub/models--Lightricks--LTX-2.5/snapshots",
            relative,
        )
        if snapshot is None:
            pytest.skip("LTX-2.5 text artifact is not cached")
        return snapshot / relative
    raise AssertionError(f"unknown test generation {generation}")


def _reference_ids(reference, text: str, *, bos_id: int, max_length: int) -> list[int]:
    ids = reference.encode((text or "").strip(), add_special_tokens=True).ids[:max_length]
    if not ids or ids[0] != bos_id:
        ids = [bos_id, *ids][:max_length]
    return ids


@pytest.mark.requires_weights
@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_real_tokenizer_json_matches_ids_masks_and_decode(
    generation: str,
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    assert _corpus_sha256(corpus) == _CORPUS_SHA256
    source = _real_source(generation)
    source_json = resolve_tokenizer_source(source).read_json_bytes()
    reference = Tokenizer.from_str(source_json.decode("utf-8"))
    cache = ensure_tokenizer_cache(source, cache_root=tmp_path / "cache")
    candidate = GemmaTokenizer(cache)

    for text in corpus:
        expected = _reference_ids(reference, text, bos_id=candidate.bos_id, max_length=1024)
        actual = candidate.encode_ids(text)
        assert actual == expected
        assert candidate.decode_ids(actual, skip_special_tokens=False) == reference.decode(
            expected,
            skip_special_tokens=False,
        )
        assert candidate.decode_ids(actual, skip_special_tokens=True) == reference.decode(
            expected,
            skip_special_tokens=True,
        )

    for text in corpus[:128]:
        for max_length in (1, 8, 64, 1024):
            expected = _reference_ids(
                reference,
                text,
                bos_id=candidate.bos_id,
                max_length=max_length,
            )
            expected_ids = [candidate.pad_id] * (max_length - len(expected)) + expected
            expected_mask = [0] * (max_length - len(expected)) + [1] * len(expected)
            actual_ids, actual_mask = candidate.encode(text, max_length=max_length)
            assert actual_ids.tolist() == [expected_ids]
            assert actual_mask.tolist() == [expected_mask]


@pytest.mark.requires_weights
def test_derived_ltx23_changes_only_literal_special_token_prompts(tmp_path: Path) -> None:
    gemma3 = _real_source("2.3")
    cache = ensure_tokenizer_cache(gemma3, cache_root=tmp_path / "cache")
    candidate = GemmaTokenizer(cache)

    import sentencepiece

    stock = sentencepiece.SentencePieceProcessor(model_file=str(gemma3 / "tokenizer.model"))
    ordinary = [text for text in _corpus() if not any(token in text for token in _LITERAL_SPECIALS)]
    for text in ordinary:
        expected = [candidate.bos_id, *stock.encode((text or "").strip())][:1024]
        assert candidate.encode_ids(text) == expected

    for token in _LITERAL_SPECIALS:
        old_ids = [candidate.bos_id, *stock.encode(token)][:1024]
        assert candidate.encode_ids(token) != old_ids
