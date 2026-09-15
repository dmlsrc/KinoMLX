"""Allocation-light MLX layer shells for separately loaded checkpoints."""

import mlx.core as mx

import kinomlx._mlx_nn as nn


class PretransposedLinear(nn.Linear):
    """Linear with contiguous [input, output] weights."""

    def __init__(self, weight: mx.array, bias: mx.array | None = None) -> None:
        nn.Module.__init__(self)
        self.weight = weight
        if bias is not None:
            self.bias = bias

    def __call__(self, value: mx.array) -> mx.array:
        if "bias" in self and self.bias is not None:
            return mx.addmm(self.bias, value, self.weight)
        return value @ self.weight


def linear_shell(*, bias: bool, pretransposed: bool = False) -> nn.Linear:
    """Create an update-ready empty Linear shell."""
    layer_type = PretransposedLinear if pretransposed else nn.Linear
    layer = layer_type.__new__(layer_type)
    nn.Module.__init__(layer)
    layer.weight = mx.zeros((0, 0), dtype=mx.float32)
    if bias:
        layer.bias = mx.zeros((0,), dtype=mx.float32)
    return layer


def conv2d_shell(
    *,
    bias: bool,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> nn.Conv2d:
    """Create an update-ready empty channels-last Conv2d shell."""
    if groups <= 0:
        raise ValueError("Conv2d groups must be positive")

    def pair(value: int | tuple[int, int]) -> tuple[int, int]:
        return (value, value) if isinstance(value, int) else value

    layer = nn.Conv2d.__new__(nn.Conv2d)
    nn.Module.__init__(layer)
    layer.weight = mx.zeros((0, 0, 0, 0), dtype=mx.float32)
    if bias:
        layer.bias = mx.zeros((0,), dtype=mx.float32)
    layer.stride = pair(stride)
    layer.padding = pair(padding)
    layer.dilation = pair(dilation)
    layer.groups = groups
    return layer


def embedding_shell() -> nn.Embedding:
    """Create an update-ready empty Embedding shell."""
    layer = nn.Embedding.__new__(nn.Embedding)
    nn.Module.__init__(layer)
    layer.weight = mx.zeros((0, 0), dtype=mx.float32)
    return layer


__all__ = ["PretransposedLinear", "conv2d_shell", "embedding_shell", "linear_shell"]
