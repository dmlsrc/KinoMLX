"""Klein's user-only Qwen3 template and complete padded conditioning sequence."""

import logging
from dataclasses import dataclass

import mlx.core as mx

from kinomlx.tokenizers.byte_bpe import QwenByteBPETokenizer

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Flux2TokenBatch:
    input_ids: mx.array
    attention_mask: mx.array
    position_ids: mx.array
    text_ids: mx.array


def prepare_text(tokenizer: QwenByteBPETokenizer, prompts: tuple[str, ...]) -> Flux2TokenBatch:
    """Encode the bundled Qwen3 user template with thinking disabled and right padding."""
    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Klein text requires a nonempty batch of strings")
    ids, masks = [], []
    for index, prompt in enumerate(prompts, 1):
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        encoded = tokenizer.encode(text)
        if len(encoded) > 512:
            _log.warning(
                "Klein prompt %d has %d templated tokens; truncating to 512 including the template",
                index,
                len(encoded),
            )
        row = encoded[:512]
        ids.append((*row, *((tokenizer.pad_token_id,) * (512 - len(row)))))
        masks.append((True,) * len(row) + (False,) * (512 - len(row)))
    positions = mx.broadcast_to(mx.arange(512, dtype=mx.int32)[None], (len(prompts), 512))
    zeros = mx.zeros_like(positions)
    return Flux2TokenBatch(
        mx.array(ids, dtype=mx.int32),
        mx.array(masks, dtype=mx.bool_),
        positions,
        mx.stack([zeros, zeros, zeros, positions], axis=-1),
    )
