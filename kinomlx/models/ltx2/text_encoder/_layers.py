"""Deterministic MLX layer shells for separately loaded text checkpoints."""

from __future__ import annotations

import mlx.core as mx

import kinomlx._mlx_nn as nn


def linear_shell(*, bias: bool) -> nn.Linear:
    """Create an update-ready empty Linear shell for checkpoint loading."""
    layer = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(layer)
    layer.weight = mx.zeros((0, 0), dtype=mx.float32)
    if bias:
        layer.bias = mx.zeros((0,), dtype=mx.float32)
    return layer


def embedding_shell() -> nn.Embedding:
    """Create an update-ready empty Embedding shell for checkpoint loading."""
    layer = nn.Embedding.__new__(nn.Embedding)
    nn.Module.__init__(layer)
    layer.weight = mx.zeros((0, 0), dtype=mx.float32)
    return layer


__all__ = ["embedding_shell", "linear_shell"]
