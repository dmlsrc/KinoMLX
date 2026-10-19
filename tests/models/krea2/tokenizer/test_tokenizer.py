"""Exact Krea prompt layout and dependency-free Qwen tokenizer contracts."""

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from kinomlx.models.krea2.catalog import Krea2Variant, variant_spec
from kinomlx.models.krea2.metadata import load_json_object
from kinomlx.models.krea2.tokenizer import (
    KREA_MAX_PROMPT_CHARACTERS,
    KREA_PREFIX,
    KREA_PREFIX_TOKENS,
    KREA_SUFFIX,
    KREA_SUFFIX_IDS,
    KREA_SUFFIX_TOKENS,
    ByteBPETables,
    QwenByteBPETokenizer,
    prepare_krea2_token_batch,
)
from kinomlx.tokenizers.unicode_scan import (
    QWEN_SPLIT_PATTERN,
    split_qwen_unicode,
)
from tests._snapshots import pinned_snapshot

from .._text_conditioning_fixtures import TURBO_REVISION

_CORPUS_SHA256 = "440989b919b76a49abe9c157e86115ed867d1b017246eaeb12b2b9393e15c2b3"
_PREFIX_SHA256 = "f3bbfde2b5167d430d6d0fe4edf9531815f57bf1badbc54ff84080dfef8b663e"
_SUFFIX_SHA256 = "3e67c686cbae4cbea835d9e7a92a27f0cd3fbe1d82e1f52429d919fd373ad199"


class _TemplateTokenizer:
    pad_token_id = 151643

    def __init__(
        self,
        *,
        prefix_ids: tuple[int, ...] | None = None,
        suffix_ids: tuple[int, ...] = KREA_SUFFIX_IDS,
    ) -> None:
        self.prefix_ids = prefix_ids or tuple(range(KREA_PREFIX_TOKENS))
        self.suffix_ids = suffix_ids

    def encode(self, text: str) -> tuple[int, ...]:
        if text == KREA_PREFIX:
            return self.prefix_ids
        if text == KREA_SUFFIX:
            return self.suffix_ids
        if not text.startswith(KREA_PREFIX):
            raise AssertionError("template builder consulted an unexpected text surface")
        prompt = text.removeprefix(KREA_PREFIX)
        return (*range(KREA_PREFIX_TOKENS), *(1000 + index for index in range(len(prompt))))

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("Krea must not consult the snapshot chat template")


def _real_tokenizer_root() -> Path:
    spec = variant_spec(Krea2Variant.TURBO)
    root = pinned_snapshot(spec.repo_id, TURBO_REVISION) / "tokenizer"
    if not (root / "tokenizer.json").is_file():
        pytest.skip("the pinned Krea 2 Turbo tokenizer is not cached")
    return root


def test_system_prefix_must_fit_the_bundled_qwen_context():
    from types import SimpleNamespace

    from kinomlx.models.krea2.tokenizer.template import KREA_MAX_QWEN_TOKENS

    tokenizer = SimpleNamespace(encode=lambda text: tuple(range(KREA_MAX_QWEN_TOKENS)))
    with pytest.raises(ValueError, match="Qwen context length"):
        prepare_krea2_token_batch(tokenizer, ("scene",), system_prompt="custom")


@pytest.mark.parametrize(
    "instruction", ["", "Describe the scene.", "Detail every visible object. " * 70]
)
def test_custom_system_prompt_recalculates_prefix_without_changing_conditioning_length(instruction):
    from kinomlx.models.krea2.tokenizer.template import regular_text_prefix

    root = _real_tokenizer_root()
    tokenizer = QwenByteBPETokenizer(
        ByteBPETables.from_source(
            load_json_object(root / "tokenizer.json"),
            load_json_object(root / "tokenizer_config.json"),
        )
    )
    prefix = tokenizer.encode(regular_text_prefix(instruction))
    prompts = ("\nA landscape.", "visible object " * 700)
    batch = prepare_krea2_token_batch(tokenizer, prompts, system_prompt=instruction)
    assert batch.prefix_tokens == len(prefix)
    assert batch.qwen_length == 512 + len(prefix)
    assert all(row[: len(prefix)] == prefix for row in batch.input_ids)
    assert all(len(row) == 512 for row in batch.conditioning_input_ids)
    assert all(row[-5:] == KREA_SUFFIX_IDS for row in batch.input_ids)
    assert batch.conditioning_input_ids[0][: len(tokenizer.encode(prompts[0]))] == tokenizer.encode(
        prompts[0]
    )
    assert batch.conditioning_position_ids[0][0][0] == len(prefix)
    assert batch.conditioning_position_ids[0][1][-1] == 511 + len(prefix)
    default = prepare_krea2_token_batch(tokenizer, prompts)
    assert batch.identity != default.identity


def _corpus() -> tuple[str, ...]:
    curated = [
        "",
        " ",
        "plain text",
        "  leading and trailing  ",
        "multiple   interior spaces",
        "tabs\tand\r\nnewlines",
        "caf\u00e9 cafe\u0301 na\u00efve",
        "\u65e5\u672c\u8a9e \u4e2d\u6587 \ud55c\uad6d\uc5b4",
        "\u041f\u0440\u0438\u0432\u0435\u0442 \u03ba\u03cc\u03c3\u03bc\u03b5",
        "\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0939\u093f\u0928\u094d\u0926\u0940",
        "emoji \U0001f469\u200d\U0001f4bb \U0001f1fa\U0001f1f8 \U0001f3ac",
        "punctuation !?.,;: [] {} () <> / \\ |",
        "3.14159265358979 and 2026-08-31T12:34:56Z",
        "I'M we've he'll can't I'D",
        "'\u00dfe '\u00dfen Strau\u00df'\u00dfe",
        "'\u017fx '\u1e9eE '\ufb05en '\ufb06en '\u1e97en",
        "unicode separators \u0085 \u00a0 \u2028 \u2029",
        "C0 separators !\u001c! !\u001d! !\u001e! !\u001f!",
        "digits \u0661\u0662\u0663 \u0967\u0968\u0969 \uff11\uff12\uff13",
        "Hebrew \u05e9\u05dc\u05d5\u05dd with marks \u05e9\u05b8\u05c1",
        "combining \u0301\u0327 before and after a\u0301\u0327",
        "apostrophes '\u00df '\u017f '\u1e9e '\ufb05 '\ufb06 '\u1e97 geht'\u00df '\u00dft",
        "<|im_start|>user\nhello<|im_end|>\n",
        "before <think> after </think>",
        "word " * 800,
    ]
    atoms = (
        "frame",
        "motion",
        "light",
        "42",
        "\u03a9",
        "\u0416",
        "\u0645",
        "\u3042",
        "\u4e2d",
        "\ud55c",
        "\U0001f30a",
        "\U0001f680",
        "e\u0301",
        "!!!",
        "'RE",
        "<|im_end|>",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
    )
    rng = random.Random(20260831)
    for _case in range(1024):
        count = rng.getrandbits(8) % 80
        curated.append("".join(atoms[rng.getrandbits(16) % len(atoms)] for _item in range(count)))
    return tuple(curated)


def _valid_source() -> tuple[dict[str, object], dict[str, object]]:
    tokenizer: dict[str, object] = {
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
            "vocab": {"a": 0, "b": 1, "ab": 2, "<pad>": 3, "<eos>": 4},
            "merges": [["a", "b"]],
        },
        "added_tokens": [
            {
                "id": 5,
                "content": "<special>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
    }
    config: dict[str, object] = {
        "pad_token": "<pad>",
        "eos_token": "<eos>",
        "clean_up_tokenization_spaces": False,
    }
    return tokenizer, config


def test_exact_template_uses_middle_padding_suffix_and_cumulative_positions() -> None:
    tokenizer = _TemplateTokenizer()
    batch = prepare_krea2_token_batch(tokenizer, ("xy", "z" * 507))  # type: ignore[arg-type]

    assert batch.batch_size == 2
    assert batch.qwen_length == 546
    assert all(len(row) == 546 for row in batch.input_ids)
    assert batch.main_valid_lengths == (36, 541)
    assert batch.input_ids[0][-KREA_SUFFIX_TOKENS:] == KREA_SUFFIX_IDS
    assert batch.input_ids[1][-KREA_SUFFIX_TOKENS:] == KREA_SUFFIX_IDS
    assert batch.attention_mask[0][36:541] == (False,) * 505
    assert all(batch.attention_mask[1])
    assert batch.position_ids[0][0][36:541] == (1,) * 505
    assert batch.position_ids[0][0][-KREA_SUFFIX_TOKENS:] == (36, 37, 38, 39, 40)
    assert batch.position_ids[0][1] == batch.position_ids[1][1] == batch.position_ids[2][1]
    assert len(batch.conditioning_input_ids[0]) == 512
    assert len(batch.conditioning_attention_mask[0]) == 512
    assert len(batch.conditioning_position_ids[0][0]) == 512


def test_exact_template_truncates_main_block_before_preserved_suffix() -> None:
    tokenizer = _TemplateTokenizer()
    prompt = "x" * 900
    batch = prepare_krea2_token_batch(tokenizer, (prompt,))  # type: ignore[arg-type]

    assert batch.main_valid_lengths == (541,)
    assert batch.input_ids[0][-KREA_SUFFIX_TOKENS:] == KREA_SUFFIX_IDS
    assert batch.input_ids[0][540] == 1000 + 506
    assert all(batch.attention_mask[0])
    assert batch.position_ids[0][0] == tuple(range(546))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("'\u00dfe", ("'\u00dfe",)),
        ("'\u00dfen", ("'\u00dfen",)),
        ("'\u017fx", ("'\u017f", "x")),
        ("'\u1e9eE", ("'\u1e9eE",)),
        ("'\ufb05en", ("'\ufb05en",)),
        ("'\ufb06en", ("'\ufb06en",)),
        ("'\u1e97en", ("'\u1e97en",)),
        ("!\u001c!", ("!\u001c!",)),
        ("!\u001d!", ("!\u001d!",)),
        ("!\u001e!", ("!\u001e!",)),
        ("!\u001f!", ("!\u001f!",)),
        ("\u0967\u0968\u0969", ("\u0967", "\u0968", "\u0969")),
        ("\u05e9\u05dc\u05d5\u05dd", ("\u05e9\u05dc\u05d5\u05dd",)),
        ("  a", (" ", " a")),
        ("\r\nx", ("\r\n", "x")),
    ],
)
def test_unicode_scanner_matches_selected_qwen_regex_cases(
    text: str,
    expected: tuple[str, ...],
) -> None:
    assert split_qwen_unicode(text) == expected


def test_tokenizer_validator_rejects_missing_merge_operand() -> None:
    tokenizer, config = _valid_source()
    del tokenizer["model"]["vocab"]["b"]  # type: ignore[index]

    with pytest.raises(ValueError, match="merge operand"):
        ByteBPETables.from_source(tokenizer, config)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["single_word", "lstrip", "rstrip", "normalized", "special"])
def test_tokenizer_validator_requires_every_added_token_flag(field: str) -> None:
    tokenizer, config = _valid_source()
    del tokenizer["added_tokens"][0][field]  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        ByteBPETables.from_source(tokenizer, config)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    [
        "normalizer",
        "split-pattern",
        "split-invert",
        "byte-level",
        "post-processor",
        "decoder",
        "model-field",
        "duplicate-vocab-id",
        "duplicate-merge",
        "missing-merge-result",
        "added-token-flag",
        "cleanup",
    ],
)
def test_tokenizer_validator_rejects_representative_unsupported_contracts(case: str) -> None:
    tokenizer, config = _valid_source()
    tokenizer = deepcopy(tokenizer)
    config = deepcopy(config)
    if case == "normalizer":
        tokenizer["normalizer"] = {"type": "NFD"}
    elif case == "split-pattern":
        tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"] = {"Regex": ".+"}  # type: ignore[index]
    elif case == "split-invert":
        tokenizer["pre_tokenizer"]["pretokenizers"][0]["invert"] = True  # type: ignore[index]
    elif case == "byte-level":
        tokenizer["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = True  # type: ignore[index]
    elif case == "post-processor":
        tokenizer["post_processor"]["trim_offsets"] = True  # type: ignore[index]
    elif case == "decoder":
        tokenizer["decoder"]["use_regex"] = False  # type: ignore[index]
    elif case == "model-field":
        tokenizer["model"]["byte_fallback"] = True  # type: ignore[index]
    elif case == "duplicate-vocab-id":
        tokenizer["model"]["vocab"]["b"] = 0  # type: ignore[index]
    elif case == "duplicate-merge":
        tokenizer["model"]["merges"].append(["a", "b"])  # type: ignore[index,union-attr]
    elif case == "missing-merge-result":
        del tokenizer["model"]["vocab"]["ab"]  # type: ignore[index]
    elif case == "added-token-flag":
        tokenizer["added_tokens"][0]["normalized"] = True  # type: ignore[index]
    elif case == "cleanup":
        config["clean_up_tokenization_spaces"] = True

    with pytest.raises(ValueError, match="requires|unsupported|repeats|absent|cleanup"):
        ByteBPETables.from_source(tokenizer, config)  # type: ignore[arg-type]


def test_bpe_cache_is_bounded_and_skips_oversized_keys() -> None:
    tokenizer_document, config = _valid_source()
    tokenizer = QwenByteBPETokenizer(
        ByteBPETables.from_source(tokenizer_document, config)  # type: ignore[arg-type]
    )
    for index in range(10_001):
        tokenizer._bpe(f"token-{index}")

    assert len(tokenizer._bpe_cache) == 10_000
    assert "token-0" not in tokenizer._bpe_cache
    assert "token-10000" in tokenizer._bpe_cache
    oversized = "x" * 257
    tokenizer._bpe(oversized)
    assert oversized not in tokenizer._bpe_cache


@pytest.mark.parametrize(
    ("merges", "text", "expected"),
    [
        ((("a", "a"),), "aaaaa", ("aa", "aa", "a")),
        ((("a", "b"), ("b", "c")), "abc", ("ab", "c")),
        ((("b", "c"), ("a", "b")), "abc", ("a", "bc")),
        ((("a", "b"), ("ab", "c"), ("abc", "a")), "abcab", ("abc", "ab")),
        ((("abc", "a"), ("ab", "c"), ("a", "b")), "abcab", ("abca", "b")),
        # A newly available lower-rank pair must run before another occurrence
        # of the previous pair; merging all equal pairs in a pass is incorrect.
        ((("ab", "a"), ("a", "b")), "abab", ("aba", "b")),
    ],
)
def test_bpe_merges_by_rank_then_leftmost_position(merges, text, expected) -> None:
    vocab = {"a", "b", "c", "<pad>", "<eos>"}
    for left, right in merges:
        vocab.update((left, right, left + right))
    ids = {piece: index for index, piece in enumerate(sorted(vocab))}
    tokenizer = QwenByteBPETokenizer(
        ByteBPETables(ids, merges, (), "<pad>", ids["<pad>"], "<eos>", ids["<eos>"])
    )
    assert tokenizer._bpe(text) == expected
    assert tokenizer.encode(text) == tuple(ids[piece] for piece in expected)


@pytest.mark.parametrize(
    ("prompts", "max_sequence_length", "error"),
    [
        ((), 512, "at least one prompt"),
        ((1,), 512, "must be strings"),
        (("x",), KREA_SUFFIX_TOKENS, "too short"),
    ],
)
def test_template_rejects_invalid_inputs(
    prompts: tuple[object, ...],
    max_sequence_length: int,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        prepare_krea2_token_batch(  # type: ignore[arg-type]
            _TemplateTokenizer(),
            prompts,
            max_sequence_length=max_sequence_length,
        )


def test_template_rejects_prompt_before_unbounded_tokenization() -> None:
    with pytest.raises(ValueError, match="65536"):
        prepare_krea2_token_batch(
            _TemplateTokenizer(),  # type: ignore[arg-type]
            ("x" * (KREA_MAX_PROMPT_CHARACTERS + 1),),
        )


@pytest.mark.parametrize(
    ("tokenizer", "error"),
    [
        (_TemplateTokenizer(prefix_ids=tuple(range(KREA_PREFIX_TOKENS - 1))), "prefix tokens"),
        (_TemplateTokenizer(suffix_ids=KREA_SUFFIX_IDS[:-1]), "suffix tokens"),
        (_TemplateTokenizer(suffix_ids=(*KREA_SUFFIX_IDS[:-1], 0)), "suffix ids"),
    ],
)
def test_template_rejects_incompatible_tokenizer_contracts(
    tokenizer: _TemplateTokenizer,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        prepare_krea2_token_batch(tokenizer, ("x",))  # type: ignore[arg-type]


def test_real_qwen_byte_bpe_matches_tokenizers_over_fixed_generated_corpus() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    root = _real_tokenizer_root()
    corpus = _corpus()
    serialized = json.dumps(corpus, ensure_ascii=True, separators=(",", ":")).encode()
    actual_corpus_sha256 = hashlib.sha256(serialized).hexdigest()
    assert actual_corpus_sha256 == _CORPUS_SHA256, (
        "Krea tokenizer corpus drifted; review _corpus() and intentionally re-freeze "
        f"_CORPUS_SHA256 (actual {actual_corpus_sha256})"
    )

    reference = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))
    candidate = QwenByteBPETokenizer(
        ByteBPETables.from_source(
            load_json_object(root / "tokenizer.json"),
            load_json_object(root / "tokenizer_config.json"),
        )
    )
    assert len(candidate.encode(KREA_PREFIX)) == KREA_PREFIX_TOKENS
    assert candidate.encode(KREA_SUFFIX) == KREA_SUFFIX_IDS
    for text in corpus:
        expected = tuple(reference.encode(text, add_special_tokens=False).ids)
        actual = candidate.encode(text)
        assert actual == expected
        assert candidate.decode(actual, skip_special_tokens=False) == reference.decode(
            list(expected),
            skip_special_tokens=False,
        )
        assert candidate.decode(actual, skip_special_tokens=True) == reference.decode(
            list(expected),
            skip_special_tokens=True,
        )


def test_exact_template_literals_are_frozen() -> None:
    assert len(KREA_PREFIX.encode("utf-8")) == 180
    assert len(KREA_SUFFIX.encode("utf-8")) == 33
    assert KREA_PREFIX_TOKENS == 34
    assert KREA_SUFFIX_TOKENS == 5
    assert KREA_SUFFIX_IDS == (151645, 198, 151644, 77091, 198)
    assert hashlib.sha256(KREA_PREFIX.encode("utf-8")).hexdigest() == _PREFIX_SHA256
    assert hashlib.sha256(KREA_SUFFIX.encode("utf-8")).hexdigest() == _SUFFIX_SHA256
