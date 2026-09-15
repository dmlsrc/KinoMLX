"""Plain Qwen3 layer foundation; model-owned positions and hidden-state taps."""

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.modeling.layers import embedding_shell
from kinomlx.modeling.qwen import QwenDecoderLayer

from .metadata import TEXT_TAPS, Flux2TextConfig


class Flux2TextEncoder(nn.Module):
    """Allocation-light Qwen3 backbone, excluding unused logits and final norm."""

    def __init__(self, config: Flux2TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = embedding_shell()
        self.layers = [QwenDecoderLayer(config) for _ in range(TEXT_TAPS[-1])]
        self.inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Qwen3 input IDs and attention mask require matching [batch, sequence] shapes"
            )
        value = self.embed_tokens(input_ids)
        length = input_ids.shape[1]
        positions = mx.arange(length, dtype=mx.float32)
        frequencies = positions[:, None] * self.inv_freq[None, :]
        phases = mx.concatenate([frequencies, frequencies], axis=-1)[None]
        rotary = (mx.cos(phases).astype(value.dtype), mx.sin(phases).astype(value.dtype))
        mask = mx.tril(mx.ones((length, length), dtype=mx.bool_), k=0)[None, None] & attention_mask[
            :, None, None
        ].astype(mx.bool_)
        taps = []
        for index, layer in enumerate(self.layers, start=1):
            value = layer(value, mask, rotary)
            if index in TEXT_TAPS:
                taps.append(value)
        return mx.concatenate(taps, axis=-1)
