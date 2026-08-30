"""Gemma 3 text backbone with LTX-required hidden-state extraction.

The architecture is adapted from mlx-lm's Gemma 3 text model (MIT). The
generation head and KV-cache path are intentionally omitted because LTX uses
the backbone once as a prompt encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import gelu_approx, rms_norm
from kinomlx.reporting import NullReporter, Reporter

from ._layers import embedding_shell, linear_shell

GEMMA3_LAYER_TYPES = tuple(
    "full_attention" if index % 6 == 5 else "sliding_attention" for index in range(48)
)


@dataclass(frozen=True)
class Gemma3Config:
    """Gemma 3 12B text configuration used by LTX-2.3."""

    vocab_size: int = 262208
    max_position_embeddings: int = 131072
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    head_dim: int = 256
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    query_pre_attn_scalar: int = 256
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    sliding_rope_theta: float = 10000.0
    full_rope_theta: float = 1000000.0
    full_rope_scaling_factor: float = 8.0
    layer_types: tuple[str, ...] = field(default_factory=lambda: GEMMA3_LAYER_TYPES)

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.query_pre_attn_scalar,
            self.sliding_window,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Gemma dimensions and window sizes must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Gemma query heads must be divisible by KV heads")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("Gemma layer_types must match num_hidden_layers")
        invalid = set(self.layer_types) - {"sliding_attention", "full_attention"}
        if invalid:
            raise ValueError(f"unsupported Gemma attention type: {sorted(invalid)[0]}")


class Gemma3RMSNorm(nn.Module):
    """Gemma RMSNorm with source-faithful FP32 ``1 + weight`` application."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.weight = mx.zeros((0,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.dims:
            raise ValueError(f"Gemma 3 RMSNorm expects width {self.dims}, got {x.shape[-1]}")
        return rms_norm(x, self.weight, self.eps, weight_offset=1.0)


class Gemma3RotaryEmbedding:
    """Split-half rotary embedding with optional linear position scaling."""

    def __init__(self, dims: int, base: float, scaling_factor: float) -> None:
        self.scaling_factor = scaling_factor
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))

    def __call__(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        positions = positions.astype(mx.float32) / self.scaling_factor
        frequencies = positions[:, None] * self.inv_freq[None, :]
        return mx.cos(frequencies), mx.sin(frequencies)


def apply_rotary_embedding(
    queries: mx.array,
    keys: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply FP32 RoPE math, then restore the attention input dtypes."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    def rotate(x: mx.array) -> mx.array:
        low, high = mx.split(x, 2, axis=-1)
        rotated = mx.concatenate(
            [low * cos - high * sin, high * cos + low * sin],
            axis=-1,
        )
        return rotated.astype(x.dtype)

    return rotate(queries), rotate(keys)


class Gemma3Attention(nn.Module):
    """Grouped-query self-attention used by Gemma 3."""

    def __init__(
        self,
        config: Gemma3Config,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.scale = config.query_pre_attn_scalar**-0.5
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = linear_shell(bias=False)
        self.k_proj = linear_shell(bias=False)
        self.v_proj = linear_shell(bias=False)
        self.o_proj = linear_shell(bias=False)
        self.q_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps)
        if layer_type == "full_attention":
            base = config.full_rope_theta
            scaling = config.full_rope_scaling_factor
        else:
            base = config.sliding_rope_theta
            scaling = 1.0
        self.rotary = Gemma3RotaryEmbedding(config.head_dim, base, scaling)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        positions: mx.array,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        cos, sin = self.rotary(positions)
        queries, keys = apply_rotary_embedding(queries, keys, cos, sin)
        if self.num_kv_groups > 1:
            keys = mx.repeat(keys, self.num_kv_groups, axis=1)
            values = mx.repeat(values, self.num_kv_groups, axis=1)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class Gemma3MLP(nn.Module):
    """GELU-tanh gated MLP used by Gemma 3."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate_proj = linear_shell(bias=False)
        self.up_proj = linear_shell(bias=False)
        self.down_proj = linear_shell(bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Gemma 3 MLP expects width {self.hidden_size}, got {x.shape[-1]}")
        output = self.down_proj(gelu_approx(self.gate_proj(x)) * self.up_proj(x))
        if output.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Gemma 3 MLP produced width {output.shape[-1]}, expected {self.hidden_size}"
            )
        return output


class Gemma3DecoderLayer(nn.Module):
    """One four-norm Gemma 3 decoder layer."""

    def __init__(
        self,
        config: Gemma3Config,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = Gemma3Attention(config, layer_type)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        positions: mx.array,
    ) -> mx.array:
        attention = self.self_attn(self.input_layernorm(x), mask, positions)
        x = x + self.post_attention_layernorm(attention)
        feed_forward = self.mlp(self.pre_feedforward_layernorm(x))
        return x + self.post_feedforward_layernorm(feed_forward)


class Gemma3Model(nn.Module):
    """Gemma 3 text model returning the embedding plus every layer state."""

    def __init__(self, config: Gemma3Config | None = None) -> None:
        super().__init__()
        self.config = config or Gemma3Config()
        self.embed_tokens = embedding_shell()
        self.layers = [
            Gemma3DecoderLayer(
                self.config,
                layer_type,
            )
            for layer_type in self.config.layer_types
        ]
        self.norm = Gemma3RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

    def _attention_masks(
        self,
        attention_mask: mx.array,
        length: int,
    ) -> tuple[mx.array, mx.array]:
        causal = mx.tril(mx.ones((length, length), dtype=mx.bool_), 0)
        valid_keys = attention_mask[:, None, None, :].astype(mx.bool_)
        full = causal[None, None, :, :] & valid_keys
        row = mx.arange(length)[:, None]
        column = mx.arange(length)[None, :]
        window = (row - column) < self.config.sliding_window
        return full, full & window[None, None, :, :]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = True,
        reporter: Reporter | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]:
        if input_ids.ndim != 2:
            raise ValueError(f"Gemma input_ids must be rank 2, got {tuple(input_ids.shape)}")
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch, length), dtype=mx.int32)
        if tuple(attention_mask.shape) != (batch, length):
            raise ValueError("Gemma attention_mask must match input_ids")
        sink = reporter if reporter is not None else NullReporter()
        phase = "encode prompt with Gemma 3"
        sink.phase_start(phase, total=len(self.layers), unit="layer")
        try:
            hidden = self.embed_tokens(input_ids)
            scale = mx.array(self.config.hidden_size**0.5, dtype=mx.bfloat16)
            hidden = hidden * scale.astype(hidden.dtype)
            positions = mx.arange(length, dtype=mx.int32)
            full_mask, sliding_mask = self._attention_masks(attention_mask, length)
            states: list[mx.array] | None = [] if output_hidden_states else None
            for layer in self.layers:
                if states is not None:
                    states.append(hidden)
                mask = sliding_mask if layer.layer_type == "sliding_attention" else full_mask
                hidden = layer(hidden, mask, positions)
                mx.eval(hidden)
                sink.phase_advance(phase)
            hidden = self.norm(hidden)
            mx.eval(hidden)
            if states is not None:
                states.append(hidden)
            return hidden, tuple(states) if states is not None else None
        finally:
            sink.phase_end(phase)


__all__ = [
    "GEMMA3_LAYER_TYPES",
    "Gemma3Config",
    "Gemma3Model",
    "apply_rotary_embedding",
]
