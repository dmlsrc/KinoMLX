"""Timestep embeddings and adaptive-normalization projections for LTX-2."""

from __future__ import annotations

import math
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import silu

from ..settings import TRANSFORMER_DTYPE_CHOICES
from .attention import _linear_shell, _projection


def resolve_transformer_dtype(value: str | mx.Dtype) -> mx.Dtype:
    """Resolve the product transformer dtype without accepting legacy aliases."""
    if value in ("bfloat16", mx.bfloat16):
        return mx.bfloat16
    if value in ("float16", mx.float16):
        return mx.float16
    if value in ("float32", mx.float32):
        return mx.float32
    valid = ", ".join(TRANSFORMER_DTYPE_CHOICES)
    raise ValueError(f"transformer dtype must be one of: {valid}")


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int = 256,
    *,
    max_period: int = 10000,
) -> mx.array:
    """Create the canonical cosine-then-sine timestep features in FP32."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a one-dimensional array")
    half = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    angles = timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
    embedding = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)
    if embedding_dim % 2:
        embedding = mx.pad(embedding, [(0, 0), (0, 1)])
    return embedding


class TimestepEmbedding(nn.Module):
    """Two-layer SiLU MLP for sinusoidal timestep features."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear_1 = _linear_shell(bias=True)
        self.linear_2 = _linear_shell(bias=True)

    def __call__(self, timestep: mx.array, hidden_dtype: mx.Dtype) -> mx.array:
        value = get_timestep_embedding(timestep).astype(hidden_dtype)
        value = silu(self.linear_1(value))
        value = self.linear_2(value)
        if value.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"timestep embedding produced width {value.shape[-1]}, "
                f"expected {self.embedding_dim}"
            )
        return value


class CombinedTimestepEmbeddings(nn.Module):
    """Checkpoint-shaped owner for the timestep MLP."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(embedding_dim)

    def __call__(self, timestep: mx.array, hidden_dtype: mx.Dtype) -> mx.array:
        return self.timestep_embedder(timestep, hidden_dtype)


class AdaLayerNormSingle(nn.Module):
    """Timestep MLP plus one checkpoint-backed modulation projection."""

    def __init__(self, embedding_dim: int, num_embeddings: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.emb = CombinedTimestepEmbeddings(embedding_dim)
        self.linear = _linear_shell(bias=True)
        self._linear_weight_t: mx.array | None = None

    def __call__(
        self,
        timestep: mx.array,
        hidden_dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        embedded = self.emb(timestep, hidden_dtype)
        modulation = _projection(
            self.linear,
            silu(embedded),
            self._linear_weight_t,
        )
        return modulation, embedded

    def apply_layout(self) -> list[mx.array]:
        """Materialize a same-math transpose for the large modulation linear."""
        if self._linear_weight_t is None:
            if "weight" not in self.linear:
                raise ValueError("AdaLN linear weight is unavailable for pretranspose")
            self._linear_weight_t = mx.contiguous(self.linear.weight.T)
        arrays = [self._linear_weight_t]
        bias = cast(mx.array | None, self.linear.get("bias"))
        if bias is not None:
            arrays.append(bias)
        return arrays

    def drop_layout_source(self) -> None:
        if self._linear_weight_t is not None and "weight" in self.linear:
            del self.linear.weight


__all__ = [
    "AdaLayerNormSingle",
    "CombinedTimestepEmbeddings",
    "TimestepEmbedding",
    "get_timestep_embedding",
    "resolve_transformer_dtype",
]
