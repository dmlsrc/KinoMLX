"""Exact Krea prefix, middle-padding, suffix, mask, and position contract."""

from dataclasses import dataclass

from kinomlx._json import JsonObject
from kinomlx.io.identity import payload_digest
from kinomlx.tokenizers.byte_bpe import QwenByteBPETokenizer

KREA_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
KREA_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
KREA_PREFIX_TOKENS = 34
KREA_SUFFIX_TOKENS = 5
KREA_SUFFIX_IDS = (151645, 198, 151644, 77091, 198)
KREA_MAX_PROMPT_CHARACTERS = 65_536
# Context length declared by the bundled Qwen3-VL text encoder.
KREA_MAX_QWEN_TOKENS = 262_144


@dataclass(frozen=True)
class Krea2TokenBatch:
    """Complete pre-Qwen token product with explicit masks and three-axis positions."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    position_ids: tuple[tuple[tuple[int, ...], ...], ...]
    main_valid_lengths: tuple[int, ...]
    max_sequence_length: int
    identity: str
    prefix_tokens: int = KREA_PREFIX_TOKENS

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)

    @property
    def qwen_length(self) -> int:
        return self.max_sequence_length + self.prefix_tokens

    @property
    def conditioning_input_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row[self.prefix_tokens :] for row in self.input_ids)

    @property
    def conditioning_attention_mask(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(row[self.prefix_tokens :] for row in self.attention_mask)

    @property
    def conditioning_position_ids(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return tuple(tuple(row[self.prefix_tokens :] for row in axis) for axis in self.position_ids)


def regular_text_prefix(system_prompt: str | None) -> str:
    """None selects training text; an empty string keeps an empty system message."""
    if system_prompt is None:
        return KREA_PREFIX
    if not isinstance(system_prompt, str):
        raise TypeError("Krea 2 system_prompt must be a string or None")
    if len(system_prompt) > KREA_MAX_PROMPT_CHARACTERS:
        raise ValueError("Krea 2 system prompt exceeds the character limit")
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n"


def _position_row(mask: tuple[bool, ...]) -> tuple[int, ...]:
    position = -1
    result: list[int] = []
    for valid in mask:
        if valid:
            position += 1
            result.append(position)
        else:
            result.append(1)
    return tuple(result)


def prepare_krea2_token_batch(
    tokenizer: QwenByteBPETokenizer,
    prompts: tuple[str, ...],
    *,
    max_sequence_length: int = 512,
    system_prompt: str | None = None,
) -> Krea2TokenBatch:
    """Build `[prefix | prompt | middle pad | suffix]` exactly as Krea was trained."""
    if not prompts:
        raise ValueError("Krea 2 text conditioning requires at least one prompt")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("Krea 2 prompts must be strings")
    oversized = next(
        (len(prompt) for prompt in prompts if len(prompt) > KREA_MAX_PROMPT_CHARACTERS),
        None,
    )
    if oversized is not None:
        raise ValueError(
            f"Krea 2 prompt has {oversized} characters; "
            f"the tokenization limit is {KREA_MAX_PROMPT_CHARACTERS}"
        )
    if max_sequence_length <= KREA_SUFFIX_TOKENS:
        raise ValueError("Krea 2 max_sequence_length is too short for the suffix")

    prefix = regular_text_prefix(system_prompt)
    prefix_ids = tokenizer.encode(prefix)
    prefix_tokens = len(prefix_ids)
    if prefix_tokens + max_sequence_length > KREA_MAX_QWEN_TOKENS:
        raise ValueError("Krea 2 system prefix and conditioning exceed the Qwen context length")
    suffix_ids = tokenizer.encode(KREA_SUFFIX)
    if system_prompt is None and prefix_tokens != KREA_PREFIX_TOKENS:
        raise ValueError(
            f"Krea 2 tokenizer produced {len(prefix_ids)} prefix tokens; "
            f"expected {KREA_PREFIX_TOKENS}"
        )
    if len(suffix_ids) != KREA_SUFFIX_TOKENS:
        raise ValueError(
            f"Krea 2 tokenizer produced {len(suffix_ids)} suffix tokens; "
            f"expected {KREA_SUFFIX_TOKENS}"
        )
    if suffix_ids != KREA_SUFFIX_IDS:
        raise ValueError(
            f"Krea 2 tokenizer produced suffix ids {suffix_ids}; expected {KREA_SUFFIX_IDS}"
        )

    main_length = max_sequence_length + prefix_tokens - KREA_SUFFIX_TOKENS
    rows: list[tuple[int, ...]] = []
    masks: list[tuple[bool, ...]] = []
    valid_lengths: list[int] = []
    for prompt in prompts:
        # Keep the trained default's exact byte stream. For an override, isolate
        # the role boundary so prompt-leading whitespace cannot merge into the
        # final prefix token and invalidate the measured prefix slice.
        main = (
            tokenizer.encode(prefix + prompt)
            if system_prompt is None
            else prefix_ids + tokenizer.encode(prompt)
        )[:main_length]
        valid_lengths.append(len(main))
        padding = main_length - len(main)
        rows.append(main + (tokenizer.pad_token_id,) * padding + suffix_ids)
        masks.append((True,) * len(main) + (False,) * padding + (True,) * len(suffix_ids))

    positions = tuple(_position_row(mask) for mask in masks)
    axes = (positions, positions, positions)
    identity_payload: JsonObject = {
        "schema_version": 1,
        "policy": "krea-prefix-middle-pad-suffix-cumulative-valid-v1",
        "input_ids": [list(row) for row in rows],
        "attention_mask": [[int(value) for value in row] for row in masks],
        "position_ids": [[list(row) for row in axis] for axis in axes],
        "max_sequence_length": max_sequence_length,
    }
    return Krea2TokenBatch(
        input_ids=tuple(rows),
        attention_mask=tuple(masks),
        position_ids=axes,
        main_valid_lengths=tuple(valid_lengths),
        max_sequence_length=max_sequence_length,
        identity=payload_digest(identity_payload),
        prefix_tokens=prefix_tokens,
    )


__all__ = [
    "KREA_MAX_PROMPT_CHARACTERS",
    "KREA_PREFIX",
    "KREA_PREFIX_TOKENS",
    "KREA_SUFFIX",
    "KREA_SUFFIX_IDS",
    "KREA_SUFFIX_TOKENS",
    "Krea2TokenBatch",
    "prepare_krea2_token_batch",
]
