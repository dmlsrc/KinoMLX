"""Test-owned synthetic parameter materialization for transformer graphs."""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.models.ltx2.cache.keys import flatten_to_nested
from kinomlx.models.ltx2.transformer import (
    AdaLayerNormSingle,
    Attention,
    FeedForward,
    LTXAVModel,
)


def _weight(
    input_dims: int,
    output_dims: int,
    *,
    seed: int,
) -> mx.array:
    scale = math.sqrt(1.0 / input_dims)
    return mx.random.uniform(
        low=-scale,
        high=scale,
        shape=(output_dims, input_dims),
        key=mx.random.key(seed),
    )


def _materialize_linear(
    layer: nn.Linear,
    input_dims: int,
    output_dims: int,
    *,
    seed: int,
) -> None:
    layer.weight = _weight(input_dims, output_dims, seed=seed)
    if "bias" in layer:
        layer.bias = mx.zeros((output_dims,), dtype=mx.float32)


def build_shaped_attention(
    query_dim: int,
    *,
    heads: int,
    dim_head: int,
    context_dim: int | None = None,
    apply_gated_attention: bool = False,
) -> Attention:
    """Materialize a standalone synthetic Attention module."""
    attention = Attention(
        query_dim,
        heads=heads,
        dim_head=dim_head,
        context_dim=context_dim,
        apply_gated_attention=apply_gated_attention,
    )
    inner_dim = heads * dim_head
    source_dim = query_dim if context_dim is None else context_dim
    attention.q_norm.weight = mx.ones((inner_dim,), dtype=mx.float32)
    attention.k_norm.weight = mx.ones((inner_dim,), dtype=mx.float32)
    _materialize_linear(attention.to_q, query_dim, inner_dim, seed=1)
    _materialize_linear(attention.to_k, source_dim, inner_dim, seed=2)
    _materialize_linear(attention.to_v, source_dim, inner_dim, seed=3)
    _materialize_linear(attention.to_out, inner_dim, query_dim, seed=4)
    if attention.to_gate_logits is not None:
        _materialize_linear(attention.to_gate_logits, query_dim, heads, seed=5)
    return attention


def build_shaped_feed_forward(
    dim: int,
    *,
    dim_out: int | None = None,
    mult: int = 4,
    bias: bool = True,
) -> FeedForward:
    """Materialize a standalone synthetic FeedForward module."""
    feed_forward = FeedForward(dim, dim_out=dim_out, mult=mult, bias=bias)
    inner_dim = dim * mult
    output_dim = dim if dim_out is None else dim_out
    _materialize_linear(feed_forward.project_in.proj, dim, inner_dim, seed=6)
    _materialize_linear(feed_forward.project_out, inner_dim, output_dim, seed=7)
    return feed_forward


def build_shaped_adaln(embedding_dim: int, num_embeddings: int) -> AdaLayerNormSingle:
    """Materialize a standalone synthetic AdaLayerNormSingle module."""
    adaln = AdaLayerNormSingle(embedding_dim, num_embeddings)
    embedder = adaln.emb.timestep_embedder
    _materialize_linear(embedder.linear_1, 256, embedding_dim, seed=8)
    _materialize_linear(embedder.linear_2, embedding_dim, embedding_dim, seed=9)
    _materialize_linear(
        adaln.linear,
        embedding_dim,
        num_embeddings * embedding_dim,
        seed=10,
    )
    return adaln


def build_shaped_ltx_model(
    factory: Callable[[], LTXAVModel],
    *,
    seed: int = 0,
) -> LTXAVModel:
    """Materialize a model shell from its canonical expected parameter graph."""
    model = factory()
    key = mx.random.key(seed)
    weights: dict[str, mx.array] = {}
    for name, shape in model.expected_parameter_shapes().items():
        if name.endswith(".bias") or "scale_shift_table" in name:
            weights[name] = mx.zeros(shape, dtype=mx.float32)
            continue
        if name.endswith(("q_norm.weight", "k_norm.weight")):
            weights[name] = mx.ones(shape, dtype=mx.float32)
            continue
        key, subkey = mx.random.split(key)
        scale = math.sqrt(1.0 / shape[-1]) if len(shape) >= 2 else 0.05
        weights[name] = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=shape,
            key=subkey,
        )
    model.update(flatten_to_nested(weights))
    model.set_dtype(model.compute_dtype)
    model.scale_shift_table = model.scale_shift_table.astype(mx.float32)
    model.audio_scale_shift_table = model.audio_scale_shift_table.astype(mx.float32)
    for block in model.transformer_blocks:
        for name in (
            "scale_shift_table",
            "prompt_scale_shift_table",
            "audio_scale_shift_table",
            "audio_prompt_scale_shift_table",
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_video",
        ):
            setattr(block, name, getattr(block, name).astype(mx.float32))
    return model


__all__ = [
    "build_shaped_adaln",
    "build_shaped_attention",
    "build_shaped_feed_forward",
    "build_shaped_ltx_model",
]
