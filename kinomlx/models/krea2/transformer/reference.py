"""Compact target-to-reference attention bias independent of adapter identity."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ReferenceAttention:
    """Bias reference keys only for target queries, using linear-size storage."""

    target_start: int
    key_bias: mx.array

    def attend(self, query: mx.array, key: mx.array, value: mx.array, *, scale: float) -> mx.array:
        if self.key_bias.shape != (key.shape[2],) or not 0 < self.target_start < query.shape[2]:
            raise ValueError("reference attention bias does not match the combined sequence")
        prefix = mx.fast.scaled_dot_product_attention(
            query[:, :, : self.target_start], key, value, scale=scale
        )
        target = mx.fast.scaled_dot_product_attention(
            query[:, :, self.target_start :],
            key,
            value,
            scale=scale,
            mask=self.key_bias.astype(query.dtype)[None, None, None, :],
        )
        return mx.concatenate([prefix, target], axis=2)
