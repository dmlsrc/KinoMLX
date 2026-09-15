"""PEFT GLoRA shape, bias dependency, and lazy layout contracts."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.formats import parse_adapter
from kinomlx.adapters.fusion import (
    WeightedAdapterOperation,
)
from kinomlx.adapters.peft import parse_configured_adapter
from kinomlx.modeling.loading import WeightTarget
from tests.adapters._fusion import fuse_and_check


def _adapter():
    arrays = {
        "layer.glora_A.X": mx.array([[0.1], [0.2], [-0.1]]),
        "layer.glora_B.X": mx.array([[0.2], [-0.1], [0.3]]),
        "layer.glora_C.X": mx.array([[0.2], [-0.1]]),
        "layer.glora_D.X": mx.array([0.1, 0.2, 0.3]),
        "layer.glora_E.X": mx.array([-0.1, 0.2, -0.3]),
    }
    config = {
        "peft_type": "GLORA",
        "r": 2,
        "config_A_B": "vector",
        "config_C": "vector",
        "config_D_E": "vector",
    }
    return arrays, config


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_bias_uses_original_companion_weight(transposed, dtype):
    arrays, config = _adapter()
    operation = parse_configured_adapter(arrays, config).operations[0]
    original = np.array([[1, 2], [-2, 3], [0.5, -1]], dtype=np.float32)
    bias = np.array([0.2, -0.3, 0.4], dtype=np.float32)
    owner = SimpleNamespace(
        weight=mx.array(original.T if transposed else original, dtype=dtype),
        bias=mx.array(bias, dtype=dtype),
    )
    original_weight = owner.weight
    original_bias = np.array(owner.bias.astype(mx.float32))
    targets = {
        "layer.weight": WeightTarget(owner, "weight", original.shape),
        "layer.bias": WeightTarget(owner, "bias", bias.shape),
    }
    operations = {
        "layer.weight": (
            WeightedAdapterOperation(operation, 0.5, "a"),
            WeightedAdapterOperation(operation, -0.25, "b"),
        ),
        "layer.bias": (
            WeightedAdapterOperation(replace(operation, part="bias"), 0.5, "a"),
            WeightedAdapterOperation(replace(operation, part="bias"), -0.25, "b"),
        ),
    }
    fuse_and_check(
        targets, operations, transposed=frozenset({"layer.weight"}) if transposed else frozenset()
    )
    actual = owner.weight.T if transposed else owner.weight
    expected_weight = original + 0.25 * (
        original * np.array(arrays["layer.glora_A.X"]) + np.array(arrays["layer.glora_B.X"])
    )
    expected_bias = original_bias + 0.25 * (
        original_bias * np.array(arrays["layer.glora_D.X"])
        + np.array(arrays["layer.glora_E.X"])
        + (original @ np.array(arrays["layer.glora_C.X"])).ravel()
    )
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(mx.array(expected_weight, dtype=dtype).astype(mx.float32)),
        rtol=2e-7,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.array(owner.bias.astype(mx.float32)),
        np.array(mx.array(expected_bias, dtype=dtype).astype(mx.float32)),
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        np.array(original_weight.astype(mx.float32)), original.T if transposed else original
    )


def test_missing_companion_and_bad_bias_fail_before_weight_mutation():
    arrays, config = _adapter()
    op = parse_configured_adapter(arrays, config).operations[0]
    owner = SimpleNamespace(weight=mx.ones((3, 2)), bias=mx.ones((3,)))
    original = owner.weight
    with pytest.raises(ValueError, match="companion"):
        fuse_and_check(
            {"bias": WeightTarget(owner, "bias", (3,))},
            {"bias": (WeightedAdapterOperation(replace(op, part="bias"), 1.0, "a"),)},
        )
    assert owner.weight is original
    bad = replace(op, factors={**op.factors, "glora_E_X": mx.ones((2,))})
    with pytest.raises(ValueError, match="path.*shape"):
        fuse_and_check(
            {"weight": WeightTarget(owner, "weight", (3, 2))},
            {"weight": (WeightedAdapterOperation(bad, 1.0, "a"),)},
        )
    assert owner.weight is original


def test_disabled_path_is_checked_and_configuration_is_required():
    arrays, config = _adapter()
    arrays["layer.glora_C.zeros"] = mx.zeros((2, 1))
    arrays.pop("layer.glora_C.X")
    parse_configured_adapter(arrays, config)
    arrays["layer.glora_C.zeros"] = mx.ones((2, 1))
    with pytest.raises(ValueError, match="must contain only zeros"):
        parse_configured_adapter(arrays, config)
    arrays, _ = _adapter()
    with pytest.raises(ValueError, match="adapter_config"):
        reconstruct_delta(parse_adapter(arrays).operations[0], mx.ones((3, 2)))


def test_missing_or_overlapping_paths_are_rejected():
    arrays, config = _adapter()
    arrays["layer.glora_A.Xd"] = mx.ones((3, 2))
    with pytest.raises(ValueError, match="overlapping"):
        parse_configured_adapter(arrays, config)
    arrays.pop("layer.glora_A.Xd")
    arrays.pop("layer.glora_E.X")
    with pytest.raises(ValueError, match="missing"):
        parse_configured_adapter(arrays, config)
