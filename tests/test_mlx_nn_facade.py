"""Installed-MLX compatibility contract for the typed neural-network facade."""

from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as upstream_nn
import pytest

import kinomlx._mlx_nn as nn

_CONSTRUCTOR_PARAMETERS = {
    "Module": (),
    "Linear": ("input_dims", "output_dims", "bias"),
    "QuantizedLinear": (
        "input_dims",
        "output_dims",
        "bias",
        "group_size",
        "bits",
        "mode",
    ),
    "Conv2d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
    ),
    "Conv3d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "bias",
    ),
    "Embedding": ("num_embeddings", "dims"),
    "LayerNorm": ("dims", "eps", "affine", "bias"),
    "GroupNorm": (
        "num_groups",
        "dims",
        "eps",
        "affine",
        "pytorch_compatible",
    ),
    "Sequential": ("modules",),
    "relu": ("x",),
    "silu": ("x",),
    "gelu_approx": ("x",),
}

_CONSTRUCTOR_DEFAULTS = {
    "Linear": {"bias": True},
    "QuantizedLinear": {
        "bias": True,
        "group_size": None,
        "bits": None,
        "mode": "affine",
    },
    "Conv2d": {
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
    },
    "Conv3d": {
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "bias": True,
    },
    "LayerNorm": {"eps": 1e-5, "affine": True, "bias": True},
    "GroupNorm": {
        "eps": 1e-5,
        "affine": True,
        "pytorch_compatible": False,
    },
}


@pytest.mark.parametrize("name", nn.__all__)
def test_facade_reexports_exact_installed_mlx_symbols(name: str) -> None:
    """The facade never wraps, subclasses, or substitutes an MLX object."""
    assert getattr(nn, name) is getattr(upstream_nn, name)


@pytest.mark.parametrize(("name", "parameters"), _CONSTRUCTOR_PARAMETERS.items())
def test_facade_constructor_parameters_match_installed_mlx(
    name: str,
    parameters: tuple[str, ...],
) -> None:
    """Required facade calls survive compatible upstream signature growth."""
    signature = inspect.signature(getattr(nn, name))
    actual = signature.parameters

    assert all(parameter in actual for parameter in parameters)
    actual_order = tuple(parameter for parameter in actual if parameter in parameters)
    assert actual_order == parameters
    for parameter in parameters:
        expected_kind = (
            inspect.Parameter.VAR_POSITIONAL
            if name == "Sequential" and parameter == "modules"
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        assert actual[parameter].kind is expected_kind
    for parameter, default in _CONSTRUCTOR_DEFAULTS.get(name, {}).items():
        assert actual[parameter].default == default

    unexpected_required = {
        parameter
        for parameter, contract in actual.items()
        if parameter not in parameters
        and contract.default is inspect.Parameter.empty
        and contract.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert not unexpected_required


@pytest.mark.parametrize(
    "class_name",
    [
        "Linear",
        "QuantizedLinear",
        "Conv2d",
        "Conv3d",
        "Embedding",
        "LayerNorm",
        "GroupNorm",
        "Sequential",
    ],
)
def test_facade_call_parameter_is_named_x(class_name: str) -> None:
    signature = inspect.signature(getattr(nn, class_name).__call__)
    actual = signature.parameters

    assert "self" in actual
    assert "x" in actual
    assert tuple(parameter for parameter in actual if parameter in {"self", "x"}) == (
        "self",
        "x",
    )
    assert actual["self"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert actual["x"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    unexpected_required = {
        parameter
        for parameter, contract in actual.items()
        if parameter not in {"self", "x"}
        and contract.default is inspect.Parameter.empty
        and contract.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    assert not unexpected_required


def test_facade_optional_parameters_and_spatial_normalization_match_runtime() -> None:
    linear = nn.Linear(2, 3, bias=False)
    conv2d = nn.Conv2d(1, 2, 3, stride=2, padding=1, dilation=3, bias=False)
    conv3d = nn.Conv3d(1, 2, 3, stride=2, padding=1, dilation=3, bias=False)
    layer_norm = nn.LayerNorm(4, affine=False)
    group_norm = nn.GroupNorm(2, 4, affine=False)

    assert linear.get("bias") is None
    assert conv2d.get("bias") is None
    assert conv3d.get("bias") is None
    assert layer_norm.get("weight") is None
    assert layer_norm.get("bias") is None
    assert group_norm.get("weight") is None
    assert group_norm.get("bias") is None
    assert conv2d.stride == (2, 2)
    assert conv2d.padding == (1, 1)
    assert conv2d.dilation == 3
    assert conv3d.stride == (2, 2, 2)
    assert conv3d.padding == (1, 1, 1)
    assert conv3d.dilation == 3


def test_facade_module_training_and_set_dtype_match_runtime() -> None:
    training = inspect.getattr_static(nn.Module, "training")
    assert isinstance(training, property)
    assert training.fset is None

    module = nn.Module()
    module.float_weight = mx.ones((1,), dtype=mx.float32)
    module.integer_weight = mx.ones((1,), dtype=mx.int32)

    assert module.set_dtype(mx.float16) is None
    assert module.float_weight.dtype == mx.float16
    assert module.integer_weight.dtype == mx.int32

    assert module.set_dtype(mx.float16, predicate=None) is None
    assert module.integer_weight.dtype == mx.float16
