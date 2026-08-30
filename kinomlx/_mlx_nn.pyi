from collections.abc import Callable
from typing import Self, TypeVar, overload

import mlx.core as mx

type _Spatial2D = int | tuple[int, int]
type _Spatial3D = int | tuple[int, int, int]

_T = TypeVar("_T")

class Module:
    @property
    def training(self) -> bool: ...
    def __init__(self) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> object: ...
    @overload
    def get(self, key: str) -> object | None: ...
    @overload
    def get(self, key: str, default: _T) -> object | _T: ...
    def parameters(self) -> dict[str, object]: ...
    def trainable_parameters(self) -> dict[str, object]: ...
    def children(self) -> dict[str, object]: ...
    def update(self, parameters: object, strict: bool = True) -> Self: ...
    def load_weights(
        self,
        file_or_weights: str | list[tuple[str, mx.array]],
        strict: bool = True,
    ) -> Self: ...
    def freeze(
        self,
        *,
        recurse: bool = True,
        keys: str | list[str] | None = None,
        strict: bool = False,
    ) -> Self: ...
    def unfreeze(
        self,
        *,
        recurse: bool = True,
        keys: str | list[str] | None = None,
        strict: bool = False,
    ) -> Self: ...
    def train(self, mode: bool = True) -> Self: ...
    def eval(self) -> Self: ...
    def set_dtype(
        self,
        dtype: mx.Dtype,
        predicate: Callable[[mx.Dtype], bool] | None = ...,
    ) -> None:
        """Cast floating parameters by default; explicit None casts every parameter."""

class Linear(Module):
    weight: mx.array
    bias: mx.array | None

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class QuantizedLinear(Module):
    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    bias: mx.array | None
    group_size: int
    bits: int
    mode: str

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Conv2d(Module):
    weight: mx.array
    bias: mx.array | None
    padding: tuple[int, int]
    stride: tuple[int, int]
    dilation: _Spatial2D
    groups: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Spatial2D,
        stride: _Spatial2D = 1,
        padding: _Spatial2D = 0,
        dilation: _Spatial2D = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Conv3d(Module):
    weight: mx.array
    bias: mx.array | None
    padding: tuple[int, int, int]
    stride: tuple[int, int, int]
    dilation: _Spatial3D

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Spatial3D,
        stride: _Spatial3D = 1,
        padding: _Spatial3D = 0,
        dilation: _Spatial3D = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Embedding(Module):
    weight: mx.array

    def __init__(self, num_embeddings: int, dims: int) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...
    def as_linear(self, x: mx.array) -> mx.array: ...

class LayerNorm(Module):
    weight: mx.array | None
    bias: mx.array | None
    eps: float
    dims: int

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class GroupNorm(Module):
    weight: mx.array | None
    bias: mx.array | None
    num_groups: int
    dims: int
    eps: float
    pytorch_compatible: bool

    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Sequential(Module):
    layers: list[Module | Callable[[mx.array], mx.array]]

    def __init__(self, *modules: Module | Callable[[mx.array], mx.array]) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def relu(x: mx.array) -> mx.array: ...
def silu(x: mx.array) -> mx.array: ...
def gelu_approx(x: mx.array) -> mx.array: ...
