"""Qwen language primitives shared by plain Qwen3 and Qwen3-VL."""

from typing import Protocol

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import silu
from kinomlx.modeling.layers import linear_shell


class QwenLayerConfig(Protocol):
    """Read-only common language-layer dimensions; positions stay model-owned."""

    @property
    def head_dim(self) -> int: ...

    @property
    def num_attention_heads(self) -> int: ...

    @property
    def num_key_value_heads(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def rms_norm_eps(self) -> float: ...


class QwenRMSNorm(nn.Module):
    """Qwen RMSNorm with the source FP32 reduction and low-precision affine."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.zeros((0,), dtype=mx.float32)

    def __call__(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.dims:
            raise ValueError(f"Qwen RMSNorm expects width {self.dims}, got {value.shape[-1]}")
        source_dtype = value.dtype
        widened = value.astype(mx.float32)
        variance = mx.mean(widened * widened, axis=-1, keepdims=True)
        normalized = widened * mx.rsqrt(variance + self.eps)
        return self.weight * normalized.astype(source_dtype)


def apply_rotary_embedding(
    queries: mx.array,
    keys: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply Qwen split-half rotary embeddings to query and key heads."""
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    def rotate_half(value: mx.array) -> mx.array:
        low, high = mx.split(value, 2, axis=-1)
        return mx.concatenate([-high, low], axis=-1)

    return (
        queries * cos + rotate_half(queries) * sin,
        keys * cos + rotate_half(keys) * sin,
    )


class QwenAttention(nn.Module):
    """Grouped-query causal self-attention without a generation cache."""

    def __init__(self, config: QwenLayerConfig) -> None:
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Qwen query heads must be divisible by KV heads")
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.q_proj = linear_shell(bias=False)
        self.k_proj = linear_shell(bias=False)
        self.v_proj = linear_shell(bias=False)
        self.o_proj = linear_shell(bias=False)
        self.q_norm = QwenRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(config.head_dim, config.rms_norm_eps)

    def __call__(
        self,
        value: mx.array,
        mask: mx.array | str,
        rotary: tuple[mx.array, mx.array],
    ) -> mx.array:
        batch, length, _width = value.shape
        queries = self.q_proj(value).reshape(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        )
        keys = self.k_proj(value).reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
        )
        values = self.v_proj(value).reshape(
            batch,
            length,
            self.num_kv_heads,
            self.head_dim,
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries, keys = apply_rotary_embedding(queries, keys, *rotary)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class QwenMLP(nn.Module):
    """Qwen SwiGLU feed-forward block."""

    def __init__(self, config: QwenLayerConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate_proj = linear_shell(bias=False)
        self.up_proj = linear_shell(bias=False)
        self.down_proj = linear_shell(bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        if value.shape[-1] != self.hidden_size:
            raise ValueError(f"Qwen MLP expects width {self.hidden_size}, got {value.shape[-1]}")
        return self.down_proj(silu(self.gate_proj(value)) * self.up_proj(value))


class QwenDecoderLayer(nn.Module):
    """One Qwen pre-norm attention and pre-norm MLP layer."""

    def __init__(self, config: QwenLayerConfig) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def __call__(
        self,
        value: mx.array,
        mask: mx.array | str,
        rotary: tuple[mx.array, mx.array],
    ) -> mx.array:
        value = value + self.self_attn(self.input_layernorm(value), mask, rotary)
        return value + self.mlp(self.post_attention_layernorm(value))
