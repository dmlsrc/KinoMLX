"""PEFT OFT and BOFT checked against independent dense equations."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.formats import parse_adapter
from kinomlx.adapters.fusion import WeightedAdapterOperation
from kinomlx.adapters.peft import configured_operation
from kinomlx.modeling.loading import WeightTarget
from tests.adapters._fusion import fuse_and_check


def _install(operation, base, expected, *, transposed, dtype):
    original = mx.array(base).astype(dtype)
    owner = SimpleNamespace(weight=mx.contiguous(original.T if transposed else original))
    target = WeightTarget(owner, "weight", original.shape)
    # Dense reference uses exactly the rounded base that the installed layer sees.
    expected_delta = expected(np.array(original.astype(mx.float32)))
    fuse_and_check(
        {"layer": target},
        {"layer": (WeightedAdapterOperation(operation, -0.4, "adapter"),)},
        transposed=frozenset({"layer"}) if transposed else frozenset(),
    )
    actual = owner.weight.T if transposed else owner.weight
    expected_weight = (original.astype(mx.float32) - 0.4 * mx.array(expected_delta)).astype(dtype)
    assert actual.dtype == dtype
    assert mx.allclose(
        actual.astype(mx.float32), expected_weight.astype(mx.float32), atol=2e-6
    ).item()


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("mode", ["exact", "poly1", "poly2", "poly3", "poly5", "shared", "coft"])
def test_packed_oft_matches_dense_input_rotation(named, transposed, dtype, mode):
    shared = mode == "shared"
    packed = np.arange((1 if shared else 2) * 3, dtype=np.float32).reshape(-1, 3) / 31 + 0.03
    key = "layer.oft_R.style.weight" if named else "layer.oft_R.weight"
    (operation,) = parse_adapter({key: mx.array(packed)}).operations
    terms = int(mode[-1]) if mode.startswith("poly") else 5
    config = {
        "peft_type": "OFT",
        "block_share": shared,
        "coft": mode == "coft",
        "eps": 0.08,
        "use_cayley_neumann": mode != "exact",
        "num_cayley_neumann_terms": terms,
    }
    operation = configured_operation(operation, config)
    matrices = []
    for vector in packed:
        skew = np.zeros((3, 3), dtype=np.float32)
        skew[np.triu_indices(3, 1)] = vector
        skew -= skew.T.copy()
        if config["coft"]:
            skew *= min(1, config["eps"] / np.sqrt(len(packed)) / np.linalg.norm(skew))
        identity = np.eye(3, dtype=np.float32)
        if mode == "exact":
            rotation = (identity - skew) @ np.linalg.inv(identity + skew)
        else:
            coefficients = {1: [1], 2: [1, 2], 3: [1, 2, 2, 1], 5: [1, 2, 2, 2, 1]}[terms]
            rotation = sum(
                c * np.linalg.matrix_power(skew, power) for power, c in enumerate(coefficients)
            )
        matrices.append(rotation)
    if shared:
        matrices *= 2
    dense = np.zeros((6, 6), dtype=np.float32)
    for block, rotation in enumerate(matrices):
        dense[block * 3 : (block + 1) * 3, block * 3 : (block + 1) * 3] = rotation
    base = np.arange(24, dtype=np.float32).reshape(4, 6) / 29
    _install(operation, base, lambda w: w @ dense.T - w, transposed=transposed, dtype=dtype)


@pytest.mark.parametrize("stages", [1, 2, 3])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_boft_permutations_and_stage_order(stages, named, transposed, dtype):
    blocks = np.sin(np.arange(stages * 16, dtype=np.float32)).reshape(stages, 4, 2, 2) * 0.1
    scale = np.arange(6, dtype=np.float32).reshape(6, 1) / 9 + 0.5
    suffix = ".style" if named else ""
    (operation,) = parse_adapter(
        {f"layer.boft_R{suffix}": mx.array(blocks), f"layer.boft_s{suffix}": mx.array(scale)}
    ).operations
    operation = configured_operation(operation, {"peft_type": "BOFT"})
    combined = np.eye(8, dtype=np.float32)
    for stage, stage_blocks in enumerate(blocks):
        skew = (stage_blocks - stage_blocks.transpose(0, 2, 1)) * 0.5
        rotations = (np.eye(2) - skew) @ np.linalg.inv(np.eye(2) + skew)
        diagonal = np.zeros((8, 8), dtype=np.float32)
        for block, rotation in enumerate(rotations):
            diagonal[block * 2 : block * 2 + 2, block * 2 : block * 2 + 2] = rotation
        indices = np.arange(8).reshape(-1, 2 ** (stage + 1))
        indices = np.concatenate((indices[:, ::2], indices[:, 1::2]), axis=1).reshape(-1)
        permutation = np.eye(8, dtype=np.float32)[indices]
        combined = permutation @ diagonal @ permutation.T @ combined
    base = np.arange(48, dtype=np.float32).reshape(6, 8) / 79
    _install(
        operation, base, lambda w: scale * (w @ combined.T) - w, transposed=transposed, dtype=dtype
    )


def test_packed_oft_requires_configuration_and_valid_triangles():
    (operation,) = parse_adapter({"layer.oft_R.weight": mx.ones((2, 2))}).operations
    with pytest.raises(ValueError, match="adapter_config"):
        reconstruct_delta(operation, mx.ones((4, 6)))
    configured = configured_operation(operation, {"peft_type": "OFT"})
    with pytest.raises(ValueError, match="packed length"):
        reconstruct_delta(configured, mx.ones((4, 6)))


def test_peft_boft_rejects_nondividing_stages_before_mutation():
    (operation,) = parse_adapter(
        {"layer.boft_R": mx.ones((3, 2, 2, 2)), "layer.boft_s": mx.ones((6, 1))}
    ).operations
    base = mx.ones((6, 4))
    owner = SimpleNamespace(weight=base)
    with pytest.raises(ValueError, match="stages"):
        fuse_and_check(
            {"layer": WeightTarget(owner, "weight", base.shape)},
            {"layer": (WeightedAdapterOperation(operation, 1, "adapter"),)},
        )
    assert owner.weight is base


def test_zero_constrained_oft_is_finite_identity():
    (operation,) = parse_adapter({"layer.oft_R.weight": mx.zeros((2, 3))}).operations
    operation = configured_operation(operation, {"peft_type": "OFT", "coft": True, "eps": 0})
    assert mx.array_equal(reconstruct_delta(operation, mx.ones((4, 6))), mx.zeros((4, 6))).item()
