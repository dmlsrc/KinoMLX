from __future__ import annotations

import logging

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.text_encoder import GemmaTokenizer
from kinomlx.models.ltx2.text_encoder.encoder import _trim_left_padding


class _Processor:
    def encode(self, text: str) -> list[int]:
        assert text == "short prompt"
        return [10, 11]


def _tokenizer() -> GemmaTokenizer:
    tokenizer = object.__new__(GemmaTokenizer)
    tokenizer._processor = _Processor()  # type: ignore[attr-defined]
    tokenizer.bos_id = 2
    tokenizer.eos_id = 1
    tokenizer.pad_id = 0
    return tokenizer


def test_tokenizer_defaults_to_reference_left_padding() -> None:
    input_ids, mask = _tokenizer().encode(" short prompt ", max_length=5)
    assert input_ids.tolist() == [[0, 0, 2, 10, 11]]
    assert mask.tolist() == [[0, 0, 1, 1, 1]]


def test_tokenizer_supports_explicit_unpadded_fast_path() -> None:
    input_ids, mask = _tokenizer().encode(
        "short prompt",
        max_length=5,
        pad_to_max=False,
    )
    assert input_ids.tolist() == [[2, 10, 11]]
    assert mask.tolist() == [[1, 1, 1]]


def test_padded_gemma_states_are_trimmed_before_connectors() -> None:
    states = (mx.arange(10).reshape(1, 5, 2),)
    mask = mx.array([[0, 0, 1, 1, 1]], dtype=mx.int32)
    compact, compact_mask = _trim_left_padding(states, mask)
    mx.eval(*compact, compact_mask)
    assert compact[0].tolist() == [[[4, 5], [6, 7], [8, 9]]]
    assert compact_mask.tolist() == [[1, 1, 1]]


class _RecordingProcessor:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.seen.append(text)
        return [ord(character) for character in text]


@pytest.mark.parametrize(
    ("prompt", "codepoint", "offset"),
    [
        ("\ufeffvisible", "U+FEFF", "offsets=0"),
        ("a\u200bb", "U+200B", "offsets=1"),
        ("a\u00a0b", "U+00A0", "offsets=1"),
        ("a\r\nb", "U+000D", "offsets=1"),
    ],
)
def test_invisible_prompt_characters_warn_once_without_mutation(
    caplog: pytest.LogCaptureFixture,
    prompt: str,
    codepoint: str,
    offset: str,
) -> None:
    processor = _RecordingProcessor()
    tokenizer = object.__new__(GemmaTokenizer)
    tokenizer._processor = processor  # type: ignore[attr-defined]
    tokenizer.bos_id = 2
    tokenizer.eos_id = 1
    tokenizer.pad_id = 0

    with caplog.at_level(logging.WARNING, logger="kinomlx.models.ltx2.text_encoder.tokenizer"):
        input_ids, _mask = tokenizer.encode(prompt, pad_to_max=False)

    records = [record for record in caplog.records if "invisible Unicode" in record.message]
    assert len(records) == 1
    assert codepoint in records[0].message
    assert offset in records[0].message
    assert processor.seen == [prompt.strip()]
    assert input_ids.tolist() == [[2, *(ord(character) for character in prompt.strip())]]


@pytest.mark.parametrize(
    "prompt",
    [
        "clean ASCII\nwith\ttabs",
        "visible cafe\u0301 CJK \u6587 emoji \U0001f3ac",
    ],
)
def test_visible_prompt_text_does_not_warn(
    caplog: pytest.LogCaptureFixture,
    prompt: str,
) -> None:
    processor = _RecordingProcessor()
    tokenizer = object.__new__(GemmaTokenizer)
    tokenizer._processor = processor  # type: ignore[attr-defined]
    tokenizer.bos_id = 2
    tokenizer.eos_id = 1
    tokenizer.pad_id = 0

    with caplog.at_level(logging.WARNING, logger="kinomlx.models.ltx2.text_encoder.tokenizer"):
        tokenizer.encode(prompt, pad_to_max=False)

    assert not caplog.records
