"""Timestep embeddings and adaptive-normalization projections for LTX-2."""

import math

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import silu

from .attention import _linear_shell, _projection


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


__all__ = [
    "AdaLayerNormSingle",
    "CombinedTimestepEmbeddings",
    "TimestepEmbedding",
    "get_timestep_embedding",
]
