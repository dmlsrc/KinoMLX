"""Shared LTX-2.x audio/video embeddings connectors."""

from __future__ import annotations

import math
from enum import Enum
from functools import lru_cache

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import gelu_approx, rms_norm

from ._layers import linear_shell


def _register_shell() -> mx.array:
    """Create an update-ready zero-sized learnable-register shell."""
    return mx.zeros((0, 0), dtype=mx.float32)


class RopeType(Enum):
    """Supported LTX connector rotary layouts."""

    SPLIT = "split"


@lru_cache(maxsize=8)
def _frequency_grid(
    theta: float,
    position_dims: int,
    inner_dim: int,
    double_precision: bool,
) -> mx.array:
    count = inner_dim // (2 * position_dims)
    if count <= 0:
        raise ValueError("connector inner_dim is too small for its position axes")
    if not double_precision:
        values = mx.power(
            mx.array(theta, dtype=mx.float32),
            mx.linspace(0.0, 1.0, count, dtype=mx.float32),
        )
        return values * (math.pi / 2)

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        values = mx.power(
            mx.array(theta, dtype=mx.float64),
            mx.linspace(
                0.0,
                1.0,
                count,
                dtype=mx.float64,
            ),
        ) * mx.array(math.pi / 2, dtype=mx.float64)
        values = values.astype(mx.float32)
        mx.eval(values)
    finally:
        mx.set_default_device(previous)
    return mx.array(values, dtype=mx.float32)


def precompute_connector_rope(
    positions: mx.array,
    *,
    inner_dim: int,
    num_heads: int,
    theta: float,
    max_positions: tuple[int, ...],
    output_dtype: mx.Dtype,
    double_precision: bool,
) -> tuple[mx.array, mx.array]:
    """Build split RoPE tensors for a ``(batch, axes, tokens)`` grid."""
    if positions.ndim != 3:
        raise ValueError("connector positions must have shape (batch, axes, tokens)")
    if positions.shape[1] != len(max_positions):
        raise ValueError("connector position axes do not match max_positions")
    if inner_dim % (2 * num_heads):
        raise ValueError("connector inner_dim must divide into even attention heads")
    indices = _frequency_grid(
        theta,
        len(max_positions),
        inner_dim,
        double_precision,
    )
    fractional = mx.stack(
        [positions[:, axis, :] / max_positions[axis] for axis in range(len(max_positions))],
        axis=-1,
    )
    frequencies = indices[None, None, None, :] * (fractional[..., None] * 2 - 1)
    frequencies = frequencies.transpose(0, 1, 3, 2).reshape(
        positions.shape[0],
        positions.shape[2],
        -1,
    )
    expected = inner_dim // 2
    if frequencies.shape[-1] > expected:
        raise ValueError("connector RoPE frequencies exceed half the inner dimension")
    if frequencies.shape[-1] < expected:
        pad = expected - frequencies.shape[-1]
        cos = mx.concatenate(
            [mx.ones((*frequencies.shape[:-1], pad)), mx.cos(frequencies)],
            axis=-1,
        )
        sin = mx.concatenate(
            [mx.zeros((*frequencies.shape[:-1], pad)), mx.sin(frequencies)],
            axis=-1,
        )
    else:
        cos = mx.cos(frequencies)
        sin = mx.sin(frequencies)
    batch, tokens, _ = cos.shape
    cos = cos.reshape(batch, tokens, num_heads, -1).transpose(0, 2, 1, 3)
    sin = sin.reshape(batch, tokens, num_heads, -1).transpose(0, 2, 1, 3)
    return cos.astype(output_dtype), sin.astype(output_dtype)


def _apply_rope(x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
    input_dtype = x.dtype
    batch, tokens, packed = x.shape
    cos, sin = rope
    heads = cos.shape[1]
    x = x.reshape(batch, tokens, heads, packed // heads).transpose(0, 2, 1, 3)
    first, second = mx.split(x, 2, axis=-1)
    x = mx.concatenate(
        [first * cos - second * sin, first * sin + second * cos],
        axis=-1,
    )
    return x.transpose(0, 2, 1, 3).reshape(batch, tokens, packed).astype(input_dtype)


class ConnectorRMSNorm(nn.Module):
    """Learned connector RMSNorm using the central FP32-opmath affine path."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.dims = dims
        self.weight = mx.zeros((0,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Normalize over the last axis with the learned gain."""
        if x.shape[-1] != self.dims:
            raise ValueError(f"connector RMSNorm expects width {self.dims}, got {x.shape[-1]}")
        return rms_norm(x, self.weight, self.eps)


class ConnectorAttention(nn.Module):
    """Self-attention with packed Q/K RMSNorm and optional head gates."""

    def __init__(
        self,
        dims: int,
        heads: int,
        head_dim: int,
        *,
        norm_eps: float,
        gated: bool,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = head_dim
        self.q_norm = ConnectorRMSNorm(dims, norm_eps)
        self.k_norm = ConnectorRMSNorm(dims, norm_eps)
        self.to_q = linear_shell(bias=True)
        self.to_k = linear_shell(bias=True)
        self.to_v = linear_shell(bias=True)
        self.to_out = linear_shell(bias=True)
        self.to_gate_logits = linear_shell(bias=True) if gated else None

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None,
        rope: tuple[mx.array, mx.array],
    ) -> mx.array:
        if x.shape[-1] != self.dims:
            raise ValueError(f"connector attention expects width {self.dims}, got {x.shape[-1]}")
        batch, tokens, dims = x.shape
        gate = None
        if self.to_gate_logits is not None:
            gate = 2.0 * mx.sigmoid(self.to_gate_logits(x))
        values = self.to_v(x)
        queries = _apply_rope(self.q_norm(self.to_q(x)), rope)
        keys = _apply_rope(self.k_norm(self.to_k(x)), rope)
        queries = queries.reshape(batch, tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch, tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch, tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        if mask is not None and mask.dtype != mx.bool_:
            mask = mask.astype(queries.dtype)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.head_dim**-0.5,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, tokens, dims)
        if gate is not None:
            output = output.reshape(batch, tokens, self.heads, self.head_dim)
            output = (output * gate[..., None]).reshape(batch, tokens, dims)
        return self.to_out(output)


class ConnectorGELU(nn.Module):
    """Linear followed by the LTX connector's approximate GELU."""

    def __init__(self, dims: int, hidden_dims: int, *, bias: bool) -> None:
        super().__init__()
        self.dims = dims
        self.hidden_dims = hidden_dims
        self.proj = linear_shell(bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.dims:
            raise ValueError(f"connector GELU expects width {self.dims}, got {x.shape[-1]}")
        projected = self.proj(x)
        if projected.shape[-1] != self.hidden_dims:
            raise ValueError(
                f"connector GELU produced width {projected.shape[-1]}, expected {self.hidden_dims}"
            )
        return gelu_approx(projected)


class ConnectorFeedForward(nn.Module):
    """Four-times expansion GELU feed-forward network."""

    def __init__(self, dims: int, *, bias: bool) -> None:
        super().__init__()
        self.dims = dims
        self.project_in = ConnectorGELU(dims, dims * 4, bias=bias)
        self.project_out = linear_shell(bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.dims:
            raise ValueError(f"connector feed-forward expects width {self.dims}, got {x.shape[-1]}")
        output = self.project_out(self.project_in(x))
        if output.shape[-1] != self.dims:
            raise ValueError(
                f"connector feed-forward produced width {output.shape[-1]}, expected {self.dims}"
            )
        return output


class BasicTransformerBlock1D(nn.Module):
    """RMSNorm-free residual connector block; Q/K are normalized internally."""

    def __init__(
        self,
        dims: int,
        heads: int,
        head_dim: int,
        *,
        norm_eps: float,
        gated_attention: bool,
        ff_bias: bool,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.norm_eps = norm_eps
        self.attn1 = ConnectorAttention(
            dims,
            heads,
            head_dim,
            norm_eps=norm_eps,
            gated=gated_attention,
        )
        self.ff = ConnectorFeedForward(dims, bias=ff_bias)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None,
        rope: tuple[mx.array, mx.array],
    ) -> mx.array:
        if x.shape[-1] != self.dims:
            raise ValueError(f"connector block expects width {self.dims}, got {x.shape[-1]}")
        normalized = rms_norm(x, eps=self.norm_eps)
        x = x + self.attn1(normalized, mask=mask, rope=rope)
        normalized = rms_norm(x, eps=self.norm_eps)
        return x + self.ff(normalized)


class Embeddings1DConnector(nn.Module):
    """Refine one modality's text features and fill the 1024-token context."""

    def __init__(
        self,
        *,
        attention_head_dim: int,
        num_attention_heads: int,
        num_layers: int,
        positional_embedding_max_pos: tuple[int, ...],
        num_learnable_registers: int | None = 128,
        positional_embedding_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        gated_attention: bool = True,
        ff_bias: bool = True,
        double_precision_rope: bool = True,
    ) -> None:
        super().__init__()
        if min(attention_head_dim, num_attention_heads, num_layers) <= 0:
            raise ValueError("connector dimensions and layer count must be positive")
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.num_learnable_registers = num_learnable_registers
        self.positional_embedding_theta = positional_embedding_theta
        self.norm_eps = norm_eps
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = [
            BasicTransformerBlock1D(
                self.inner_dim,
                num_attention_heads,
                attention_head_dim,
                norm_eps=norm_eps,
                gated_attention=gated_attention,
                ff_bias=ff_bias,
            )
            for _ in range(num_layers)
        ]
        if num_learnable_registers is not None:
            if num_learnable_registers <= 0:
                raise ValueError("num_learnable_registers must be positive or None")
            self.learnable_registers = _register_shell()

    def _append_registers(self, x: mx.array) -> mx.array:
        batch, length, dims = x.shape
        register_count = self.num_learnable_registers
        if register_count is None:
            return x
        target = max(1024, math.ceil(length / register_count) * register_count)
        repetitions = math.ceil(target / register_count)
        registers = mx.tile(self.learnable_registers, (repetitions, 1))[length:target]
        if registers.shape[0]:
            registers = mx.broadcast_to(registers[None, :, :], (batch, target - length, dims))
            x = mx.concatenate([x, registers.astype(x.dtype)], axis=1)
        return x

    def __call__(
        self,
        x: mx.array,
        attention_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        if x.ndim != 3 or x.shape[-1] != self.inner_dim:
            raise ValueError(
                f"connector expects (batch, tokens, {self.inner_dim}), got {tuple(x.shape)}"
            )
        if self.num_learnable_registers is not None:
            x = self._append_registers(x)
            block_mask = None
        else:
            block_mask = attention_mask
        positions = mx.arange(x.shape[1], dtype=mx.float32)[None, None, :]
        positions = mx.broadcast_to(positions, (x.shape[0], 1, x.shape[1]))
        rope = precompute_connector_rope(
            positions,
            inner_dim=self.inner_dim,
            num_heads=self.num_attention_heads,
            theta=self.positional_embedding_theta,
            max_positions=self.positional_embedding_max_pos,
            output_dtype=x.dtype,
            double_precision=self.double_precision_rope,
        )
        for block in self.transformer_1d_blocks:
            x = block(x, mask=block_mask, rope=rope)
        x = rms_norm(x, eps=self.norm_eps)
        output_mask = mx.zeros((x.shape[0], 1, 1, x.shape[1]), dtype=x.dtype)
        return x, output_mask


__all__ = [
    "BasicTransformerBlock1D",
    "Embeddings1DConnector",
    "RopeType",
    "precompute_connector_rope",
]
