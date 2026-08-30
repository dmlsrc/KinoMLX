"""Explicitly typed access to the MLX neural-network surface KinoMLX uses.

MLX exposes :mod:`mlx.nn` through wildcard imports.  Those names are present at
runtime, but strict type checkers do not treat them as public re-exports.  Keep
KinoMLX's small dependency surface explicit without wrapping or replacing the
underlying MLX objects.
"""

from mlx.nn.layers.activations import gelu_approx as gelu_approx
from mlx.nn.layers.activations import relu as relu
from mlx.nn.layers.activations import silu as silu
from mlx.nn.layers.base import Module as Module
from mlx.nn.layers.containers import Sequential as Sequential
from mlx.nn.layers.convolution import Conv2d as Conv2d
from mlx.nn.layers.convolution import Conv3d as Conv3d
from mlx.nn.layers.embedding import Embedding as Embedding
from mlx.nn.layers.linear import Linear as Linear
from mlx.nn.layers.normalization import GroupNorm as GroupNorm
from mlx.nn.layers.normalization import LayerNorm as LayerNorm
from mlx.nn.layers.quantized import QuantizedLinear as QuantizedLinear

__all__ = [
    "Conv2d",
    "Conv3d",
    "Embedding",
    "GroupNorm",
    "LayerNorm",
    "Linear",
    "Module",
    "QuantizedLinear",
    "Sequential",
    "gelu_approx",
    "relu",
    "silu",
]
