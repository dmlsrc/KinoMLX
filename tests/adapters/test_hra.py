"""Compact MLX HRA versus independent full Householder matrices."""

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


def _reflection(vectors, gs):
    vectors = vectors.astype(np.float64)
    rotation = np.eye(vectors.shape[0])
    if gs:
        columns = []
        for vector in vectors.T:
            vector = vector.copy()
            for column in columns:
                vector -= np.dot(column, vector) * column
            columns.append(vector / np.linalg.norm(vector))
        orthogonal = np.stack(columns, axis=1)
        return rotation - 2 * orthogonal @ orthogonal.T
    vectors = vectors / np.linalg.norm(vectors, axis=0)
    for vector in vectors.T:
        rotation = rotation @ (np.eye(vectors.shape[0]) - 2 * np.outer(vector, vector))
    return rotation


@pytest.mark.parametrize("gs", [False, True])
@pytest.mark.parametrize("rank", [1, 3, 4])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_compact_hra_matches_full_rotation(gs, rank, transposed, dtype):
    rng = np.random.default_rng(18)
    vectors = rng.normal(size=(6, rank)).astype(np.float32)
    original = mx.array(rng.normal(size=(4, 6))).astype(dtype)
    base = np.array(original.astype(mx.float32))
    bias = mx.arange(4).astype(dtype)
    inputs = mx.array(rng.normal(size=(2, 6))).astype(mx.float32)
    operation = parse_configured_adapter(
        {"layer.hra_u.style": mx.array(vectors)},
        {"peft_type": "HRA", "r": rank, "apply_GS": gs},
    ).operations[0]
    assert not parse_adapter({"layer.hra_u": mx.array(vectors)}).unknown_keys
    delta = reconstruct_delta(operation, original)
    expected_delta = base @ (_reflection(vectors, gs) - np.eye(6))
    np.testing.assert_allclose(np.array(delta), expected_delta, rtol=2e-5, atol=2e-6)
    owner = SimpleNamespace(weight=original.T if transposed else original, bias=bias)
    selected = (WeightedAdapterOperation(operation, -0.7, "adapter"),)
    fuse_and_check(
        {"layer": WeightTarget(owner, "weight", (4, 6))},
        {"layer": selected},
        transposed=frozenset({"layer"}) if transposed else frozenset(),
    )
    actual = owner.weight.T if transposed else owner.weight
    expected = mx.array(base - 0.7 * expected_delta, dtype=mx.float32).astype(dtype)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.008 if dtype == mx.bfloat16 else 0.002,
        atol=2e-6,
    )
    assert owner.bias is bias
    output = inputs @ actual.astype(mx.float32).T + owner.bias.astype(mx.float32)
    expected_output = inputs @ expected.astype(mx.float32).T + bias.astype(mx.float32)
    assert mx.allclose(output, expected_output, atol=2e-4, rtol=0.008).item()


@pytest.mark.parametrize("gs", [False, True])
def test_zero_reflection_vector_fails_first_use_check(gs):
    owner = SimpleNamespace(weight=mx.ones((4, 6)))
    original = owner.weight
    operation = parse_configured_adapter(
        {"layer.hra_u": mx.zeros((6, 2))}, {"peft_type": "HRA", "r": 2, "apply_GS": gs}
    ).operations[0]
    with pytest.raises(ValueError, match="nonfinite"):
        fuse_and_check(
            {"layer": WeightTarget(owner, "weight", original.shape)},
            {"layer": (WeightedAdapterOperation(operation, 1.0, "adapter"),)},
        )


def test_hra_requires_valid_configuration_and_target():
    tensors = {"layer.hra_u": mx.ones((6, 2))}
    operation = parse_adapter(tensors).operations[0]
    with pytest.raises(ValueError, match="adapter_config"):
        reconstruct_delta(operation, mx.zeros((4, 6)))
    with pytest.raises(ValueError, match="rank"):
        parse_configured_adapter(tensors, {"peft_type": "HRA", "r": 3})
    with pytest.raises(ValueError, match="boolean"):
        parse_configured_adapter(tensors, {"peft_type": "HRA", "r": 2, "apply_GS": "yes"})
    operation = parse_configured_adapter(tensors, {"peft_type": "HRA", "r": 2}).operations[0]
    with pytest.raises(ValueError, match="input dimension"):
        reconstruct_delta(operation, mx.zeros((4, 5)))
