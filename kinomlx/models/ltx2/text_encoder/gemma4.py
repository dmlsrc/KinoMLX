"""Allocation-light Gemma 4 text backbone for LTX-2.5 conditioning.

The topology and weight spelling follow mlx-lm's MIT-licensed Gemma 4 text
model. Numerical boundaries follow the pinned Apache Transformers Gemma 4
unified text execution used to freeze the LTX-tuned reference fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import gelu_approx, rms_norm
from kinomlx.reporting import NullReporter, Reporter

from ._layers import embedding_shell, linear_shell

GEMMA4_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(48)
)


@dataclass(frozen=True)
class Gemma4Config:
    """Implemented LTX-tuned Gemma 4 text graph."""

    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    head_dim: int = 256
    global_head_dim: int = 512
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    sliding_rope_theta: float = 10000.0
    full_rope_theta: float = 1000000.0
    full_partial_rotary_factor: float = 0.25
    layer_types: tuple[str, ...] = field(default_factory=lambda: GEMMA4_LAYER_TYPES)

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.max_position_embeddings,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.head_dim,
            self.global_head_dim,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_global_key_value_heads,
            self.sliding_window,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Gemma 4 dimensions and window sizes must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Gemma 4 query heads must be divisible by sliding KV heads")
        if self.num_attention_heads % self.num_global_key_value_heads:
            raise ValueError("Gemma 4 query heads must be divisible by global KV heads")
        if self.head_dim % 2 or self.global_head_dim % 2:
            raise ValueError("Gemma 4 attention head dimensions must be even")
        if not 0.0 < self.full_partial_rotary_factor <= 1.0:
            raise ValueError("Gemma 4 full-attention rotary factor must be in (0, 1]")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("Gemma 4 layer_types must match num_hidden_layers")
        invalid = set(self.layer_types) - {"sliding_attention", "full_attention"}
        if invalid:
            raise ValueError(f"unsupported Gemma 4 attention type: {sorted(invalid)[0]}")


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm with a direct learned scale and FP32 opmath."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.weight = mx.zeros((0,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.dims:
            raise ValueError(f"Gemma 4 RMSNorm expects width {self.dims}, got {x.shape[-1]}")
        return rms_norm(x, self.weight, self.eps)


class Gemma4RotaryEmbedding:
    """Split-half RoPE for default and proportional Gemma 4 attention."""

    def __init__(
        self,
        dims: int,
        *,
        base: float,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        active_angles = int(partial_rotary_factor * dims // 2)
        active = 1.0 / (base ** (mx.arange(0, 2 * active_angles, 2, dtype=mx.float32) / dims))
        inactive_angles = dims // 2 - active_angles
        self.inv_freq = (
            mx.concatenate([active, mx.zeros((inactive_angles,), dtype=mx.float32)])
            if inactive_angles
            else active
        )

    def __call__(
        self,
        positions: mx.array,
        *,
        output_dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        frequencies = positions.astype(mx.float32)[..., None] * self.inv_freq[None, None, :]
        embedding = mx.concatenate([frequencies, frequencies], axis=-1)
        return mx.cos(embedding).astype(output_dtype), mx.sin(embedding).astype(output_dtype)


def apply_gemma4_rotary(
    value: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """Apply Transformers-style split-half RoPE to ``(B, T, H, D)`` values."""
    first, second = mx.split(value, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    return (value * cos[:, :, None, :] + rotated * sin[:, :, None, :]).astype(value.dtype)


class Gemma4Attention(nn.Module):
    """Grouped-query attention with Gemma 4's per-layer head geometry."""

    def __init__(
        self,
        config: Gemma4Config,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.num_heads = config.num_attention_heads
        if layer_type == "full_attention":
            self.head_dim = config.global_head_dim
            self.num_kv_heads = config.num_global_key_value_heads
            base = config.full_rope_theta
            partial = config.full_partial_rotary_factor
            self.v_proj = None
        else:
            self.head_dim = config.head_dim
            self.num_kv_heads = config.num_key_value_heads
            base = config.sliding_rope_theta
            partial = 1.0
            self.v_proj = linear_shell(bias=False)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = linear_shell(bias=False)
        self.k_proj = linear_shell(bias=False)
        self.o_proj = linear_shell(bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rms_norm_eps = config.rms_norm_eps
        self.rotary = Gemma4RotaryEmbedding(
            self.head_dim,
            base=base,
            partial_rotary_factor=partial,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        positions: mx.array,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        raw_keys = self.k_proj(x).reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
        )
        raw_values = raw_keys if self.v_proj is None else self.v_proj(x).reshape(raw_keys.shape)
        queries = self.q_norm(queries)
        keys = self.k_norm(raw_keys)
        values = rms_norm(raw_values, eps=self.rms_norm_eps)
        cos, sin = self.rotary(positions, output_dtype=queries.dtype)
        queries = apply_gemma4_rotary(queries, cos, sin).transpose(0, 2, 1, 3)
        keys = apply_gemma4_rotary(keys, cos, sin).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        if self.num_kv_groups > 1:
            keys = mx.repeat(keys, self.num_kv_groups, axis=1)
            values = mx.repeat(values, self.num_kv_groups, axis=1)

        scores = queries @ keys.swapaxes(-1, -2)
        minimum = mx.array(mx.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = mx.where(mask, scores, minimum)
        probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)
        probabilities = mx.where(
            mx.any(mask, axis=-1, keepdims=True),
            probabilities,
            mx.zeros_like(probabilities),
        )
        output = probabilities @ values
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class Gemma4MLP(nn.Module):
    """GELU-tanh gated MLP used by the selected Gemma 4 graph."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate_proj = linear_shell(bias=False)
        self.up_proj = linear_shell(bias=False)
        self.down_proj = linear_shell(bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Gemma 4 MLP expects width {self.hidden_size}, got {x.shape[-1]}")
        output = self.down_proj(gelu_approx(self.gate_proj(x)) * self.up_proj(x))
        if output.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Gemma 4 MLP produced width {output.shape[-1]}, expected {self.hidden_size}"
            )
        return output


class Gemma4DecoderLayer(nn.Module):
    """One direct-scale, four-norm Gemma 4 decoder layer."""

    def __init__(
        self,
        config: Gemma4Config,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = Gemma4Attention(config, layer_type)
        self.mlp = Gemma4MLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        positions: mx.array,
    ) -> mx.array:
        residual = x
        attention = self.self_attn(self.input_layernorm(x), mask, positions)
        x = residual + self.post_attention_layernorm(attention)
        residual = x
        feed_forward = self.mlp(self.pre_feedforward_layernorm(x))
        x = residual + self.post_feedforward_layernorm(feed_forward)
        return x * self.layer_scalar


class Gemma4Model(nn.Module):
    """Gemma 4 backbone returning the 49 states consumed by LTX projection."""

    def __init__(self, config: Gemma4Config | None = None) -> None:
        super().__init__()
        self.config = config or Gemma4Config()
        self.embed_tokens = embedding_shell()
        self.layers = [
            Gemma4DecoderLayer(self.config, layer_type) for layer_type in self.config.layer_types
        ]
        self.norm = Gemma4RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

    @staticmethod
    def position_ids(attention_mask: mx.array) -> mx.array:
        """Derive the left-padding-aware positions used by Transformers."""
        return mx.maximum(mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1, 0)

    def attention_masks(self, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        """Compose external key padding with full and sliding causal masks."""
        length = attention_mask.shape[1]
        row = mx.arange(length)[:, None]
        column = mx.arange(length)[None, :]
        causal = column <= row
        valid_keys = attention_mask[:, None, None, :].astype(mx.bool_)
        full = causal[None, None, :, :] & valid_keys
        window = (row - column) < self.config.sliding_window
        return full, full & window[None, None, :, :]

    def _forward(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None,
        position_ids: mx.array | None,
        output_hidden_states: bool,
        capture_boundaries: bool,
        reporter: Reporter | None,
    ) -> tuple[
        mx.array,
        tuple[mx.array, ...] | None,
        dict[str, mx.array] | None,
    ]:
        if input_ids.ndim != 2:
            raise ValueError(f"Gemma 4 input_ids must be rank 2, got {tuple(input_ids.shape)}")
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch, length), dtype=mx.int32)
        if tuple(attention_mask.shape) != (batch, length):
            raise ValueError("Gemma 4 attention_mask must match input_ids")
        if position_ids is None:
            position_ids = self.position_ids(attention_mask)
        if tuple(position_ids.shape) != (batch, length):
            raise ValueError("Gemma 4 position_ids must match input_ids")

        sink = reporter if reporter is not None else NullReporter()
        phase = "encode prompt with Gemma 4"
        sink.phase_start(phase, total=len(self.layers), unit="layer")
        try:
            hidden = self.embed_tokens(input_ids)
            scale = mx.array(self.config.hidden_size**0.5, dtype=mx.bfloat16)
            hidden = hidden * scale.astype(hidden.dtype)
            full_mask, sliding_mask = self.attention_masks(attention_mask)
            states: list[mx.array] | None = [] if output_hidden_states else None
            boundaries: dict[str, mx.array] | None = {} if capture_boundaries else None
            if boundaries is not None:
                boundaries["embedding"] = hidden
            for index, layer in enumerate(self.layers):
                if states is not None:
                    states.append(hidden)
                mask = sliding_mask if layer.layer_type == "sliding_attention" else full_mask
                hidden = layer(hidden, mask, position_ids)
                mx.eval(hidden)
                if boundaries is not None:
                    boundaries[f"layer.{index:02d}"] = hidden
                sink.phase_advance(phase)
            hidden = self.norm(hidden)
            mx.eval(hidden)
            if states is not None:
                states.append(hidden)
            if boundaries is not None:
                boundaries["final_norm"] = hidden
            return hidden, tuple(states) if states is not None else None, boundaries
        finally:
            sink.phase_end(phase)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        output_hidden_states: bool = True,
        reporter: Reporter | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]:
        hidden, states, _boundaries = self._forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            capture_boundaries=False,
            reporter=reporter,
        )
        return hidden, states

    def forward_boundaries(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        reporter: Reporter | None = None,
    ) -> dict[str, mx.array]:
        """Run the backbone and return all 50 named D2 parity boundaries."""
        _hidden, _states, boundaries = self._forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            capture_boundaries=True,
            reporter=reporter,
        )
        if boundaries is None:
            raise AssertionError("Gemma 4 boundary capture was not initialized")
        return boundaries


__all__ = [
    "GEMMA4_LAYER_TYPES",
    "Gemma4Config",
    "Gemma4DecoderLayer",
    "Gemma4Model",
    "Gemma4RMSNorm",
    "apply_gemma4_rotary",
]
