"""Independent dense equations for static adapter formats; no model weights needed."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.adapters.algebra import (
    reconstruct_delta,
)
from kinomlx.adapters.formats import parse_adapter
from kinomlx.adapters.fusion import WeightedAdapterOperation
from kinomlx.modeling.dtypes import FloatDType
from kinomlx.modeling.loading import WeightTarget
from tests.adapters._fusion import fuse_and_check


def _case(kind: str):
    rng = np.random.default_rng(404)
    base = rng.normal(0, 0.3, (8, 4)).astype(np.float32)
    down = rng.normal(0, 0.1, (2, 4)).astype(np.float32)
    up = rng.normal(0, 0.1, (8, 2)).astype(np.float32)
    scale = 3.0 / 2.0
    factors = {"lora_down.weight": down, "lora_up.weight": up, "alpha": 3.0}
    delta = scale * up @ down
    if kind == "locon":
        mid = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
        factors.update(
            {
                "lora_down.weight": down[:, :, None, None],
                "lora_up.weight": up[:, :, None, None],
                "lora_mid.weight": mid[:, :, None, None],
            }
        )
        delta = scale * up @ mid @ down
    elif kind in ("dora", "dora_input", "dora_peft"):
        magnitude = np.arange(1, 9 if kind != "dora_input" else 5, dtype=np.float32) / 5
        if kind == "dora_input":
            magnitude = magnitude[None, :]
            norm = np.linalg.norm(base + delta, axis=0, keepdims=True)
        else:
            magnitude = magnitude[:, None]
            norm = np.linalg.norm(base + delta, axis=1, keepdims=True)
        factors["lora_magnitude_vector" if kind == "dora_peft" else "dora_scale"] = magnitude
        delta = (base + delta) * magnitude / norm - base
    elif kind in ("loha", "loha_tucker"):
        up2 = rng.normal(0, 0.1, (8, 2)).astype(np.float32)
        down2 = rng.normal(0, 0.1, (2, 4)).astype(np.float32)
        factors = {
            "hada_w1_a": up,
            "hada_w1_b": down,
            "hada_w2_a": up2,
            "hada_w2_b": down2,
            "alpha": 3.0,
        }
        delta = scale * (up @ down) * (up2 @ down2)
        if kind == "loha_tucker":
            first = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
            second = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
            factors.update(
                {
                    "hada_w1_a": up.T,
                    "hada_w2_a": up2.T,
                    "hada_t1": first[:, :, None, None],
                    "hada_t2": second[:, :, None, None],
                }
            )
            delta = scale * (up @ first @ down) * (up2 @ second @ down2)
    elif kind.startswith("lokr"):
        first = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
        second = rng.normal(0, 0.1, (4, 2)).astype(np.float32)
        factors = {"lokr_w1": first, "lokr_w2": second, "alpha": 3.0}
        if kind != "lokr_dense":
            a = rng.normal(0, 0.1, (4, 2)).astype(np.float32)
            b = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
            factors.pop("lokr_w2")
            factors.update({"lokr_w2_a": a, "lokr_w2_b": b})
            second = a @ b
            if kind == "lokr_tucker":
                mid = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
                factors.update({"lokr_w2_a": a.T, "lokr_t2": mid[:, :, None, None]})
                second = a @ mid @ b
            if kind == "lokr_both":
                a = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
                b = rng.normal(0, 0.1, (2, 2)).astype(np.float32)
                factors.pop("lokr_w1")
                factors.update({"lokr_w1_a": a, "lokr_w1_b": b})
                first = a @ b
        delta = np.kron(first, second) * (1.0 if kind == "lokr_dense" else scale)
    elif kind == "glora":
        a1 = rng.normal(0, 0.1, (4, 2)).astype(np.float32)
        factors = {
            "a1.weight": a1,
            "a2.weight": down,
            "b1.weight": up,
            "b2.weight": down * 0.7,
            "alpha": 3.0,
        }
        delta = scale * (base @ a1 @ down + up @ (down * 0.7))
    elif kind in ("ia3_input", "ia3_output"):
        on_input = kind == "ia3_input"
        vector = rng.normal(0, 0.1, (4 if on_input else 8,)).astype(np.float32)
        factors = {"weight": vector, "on_input": on_input}
        delta = base * vector.reshape((1, -1) if on_input else (-1, 1))
    elif kind in ("full", "norm", "set"):
        key = {"full": "diff", "norm": "w_norm", "set": "set_weight"}[kind]
        factors = {key: delta}
        if kind == "set":
            delta = delta - base
    elif kind in ("oft", "boft"):
        stages = 2 if kind == "boft" else 1
        blocks = rng.normal(0, 0.1, (stages, 4, 2, 2)).astype(np.float32)
        skew = blocks - blocks.swapaxes(-1, -2)
        identity = np.eye(2)
        rotation = (identity + skew) @ np.linalg.inv(identity - skew)
        factors = {"oft_blocks": blocks if kind == "boft" else blocks[0], "alpha": 0.0}
        if kind == "oft":
            delta = (rotation[0].swapaxes(-1, -2) @ base.reshape(4, 2, 4)).reshape(8, 4) - base
        else:
            # Explicit basis permutations keep this oracle independent of the
            # product's reshape-and-swap butterfly implementation.
            adapted = base.astype(np.float64)
            for stage in range(stages):
                step = 2**stage
                order = []
                for start in range(0, 8, 2 * step):
                    for offset in range(step):
                        order.extend((start + offset, start + offset + step))
                block_diagonal = np.zeros((8, 8))
                for block in range(4):
                    block_diagonal[block * 2 : block * 2 + 2, block * 2 : block * 2 + 2] = rotation[
                        stage, block
                    ]
                permutation = np.eye(8)[order]
                adapted = permutation.T @ block_diagonal @ permutation @ adapted
            delta = adapted - base
    return base, factors, delta


KINDS = (
    "lora",
    "locon",
    "dora",
    "dora_input",
    "dora_peft",
    "loha",
    "loha_tucker",
    "lokr_dense",
    "lokr_factored",
    "lokr_both",
    "lokr_tucker",
    "glora",
    "ia3_input",
    "ia3_output",
    "full",
    "norm",
    "set",
    "oft",
    "boft",
)


@pytest.mark.parametrize("kind", KINDS)
def test_dense_delta_and_linear_output(kind: str) -> None:
    base, factors, expected = _case(kind)
    arrays = {
        f"layer.{key}": mx.array(value, dtype=mx.bool_ if key == "on_input" else mx.float32)
        for key, value in factors.items()
    }
    (operation,) = parse_adapter(arrays).operations
    delta = reconstruct_delta(operation, mx.array(base))
    mx.eval(delta)
    np.testing.assert_allclose(np.array(delta), expected, atol=2e-6, rtol=2e-5)
    inputs = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    output = mx.array(inputs) @ (mx.array(base) + delta).T
    mx.eval(output)
    np.testing.assert_allclose(np.array(output), inputs @ (base + expected).T, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("strength", [0.0, -0.4, 1.0])
@pytest.mark.parametrize("transposed", [False, True])
def test_installed_weight_matches_dense_equation(kind, strength, transposed) -> None:
    base, factors, delta = _case(kind)
    (operation,) = parse_adapter(
        {
            f"layer.{key}": mx.array(value, dtype=mx.bool_ if key == "on_input" else mx.float32)
            for key, value in factors.items()
        }
    ).operations
    owner = SimpleNamespace(weight=mx.array(base.T.copy() if transposed else base))
    fuse_and_check(
        {"layer": WeightTarget(owner, "weight", base.shape)},
        {"layer": (WeightedAdapterOperation(operation, strength, "adapter"),)},
        transposed=frozenset({"layer"}) if transposed else frozenset(),
    )
    actual = owner.weight.T if transposed else owner.weight
    np.testing.assert_allclose(np.array(actual), base + strength * delta, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "suffixes",
    [
        ("lora_A.weight", "lora_B.weight"),
        ("lora_down", "lora_up"),
        ("lora_A.default.weight", "lora_B.default.weight"),
    ],
)
def test_standard_factor_dialects(suffixes: tuple[str, str]) -> None:
    inventory = parse_adapter(
        {f"layer.{suffixes[0]}": mx.ones((2, 3)), f"layer.{suffixes[1]}": mx.ones((4, 2))}
    )
    assert inventory.operations[0].kind == "lora"
    assert len(inventory.operations[0].source_keys) == 2
    assert inventory.unknown_keys == ()


def test_lora_bias_and_full_bias_are_preserved() -> None:
    arrays = {
        "layer.lora_A.weight": mx.ones((2, 3)),
        "layer.lora_B.weight": mx.ones((4, 2)),
        "layer.diff_b": mx.arange(4).astype(mx.float32),
    }
    inventory = parse_adapter(arrays)
    assert len(inventory.operations) == 2
    bias = reconstruct_delta(inventory.operations[1], mx.zeros((4,)))
    assert bias.tolist() == [0.0, 1.0, 2.0, 3.0]
    arrays["layer.lora_B.bias"] = arrays.pop("layer.diff_b")
    arrays["layer.alpha"] = mx.array(4.0)
    bias = reconstruct_delta(parse_adapter(arrays).operations[1], mx.zeros((4,)))
    assert bias.tolist() == [0.0, 2.0, 4.0, 6.0]


@pytest.mark.parametrize(
    "invalid",
    [
        {"lora_A.weight": mx.ones((2, 3))},
        {"lora_A.weight": mx.ones((2, 3)), "lora_B.weight": mx.ones((4, 3))},
        {
            "lora_A.weight": mx.ones((2, 3)),
            "lora_B.weight": mx.ones((4, 2)),
            "lora_down.weight": mx.ones((2, 3)),
        },
        {"dora_scale": mx.ones((4, 1))},
        {"alpha": mx.array(2.0)},
        {"lokr_w1": mx.ones((2, 2)), "lokr_w2_a": mx.ones((2, 2))},
        {
            "lora_A.weight": mx.ones((2, 3)),
            "lora_B.weight": mx.ones((4, 2)),
            "lambda_layer": mx.ones((1,)),
        },
        {
            "lora_A.weight": mx.ones((2, 3)),
            "lora_B.weight": mx.ones((4, 2)),
            "alpha": mx.array(float("nan")),
        },
        {"lora_A.weight": mx.ones((2, 3)), "lora_B.weight": mx.full((4, 2), float("inf"))},
    ],
)
def test_malformed_formats_fail_even_before_model_mapping(invalid) -> None:
    with pytest.raises(ValueError, match="adapter"):
        parse_adapter({f"unmapped.{key}": value for key, value in invalid.items()})


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_fusion_layout_dtype_and_order(transposed, dtype) -> None:
    base, factors, _ = _case("lora")
    (operation,) = parse_adapter(
        {f"layer.{key}": mx.array(value) for key, value in factors.items()}
    ).operations
    first = WeightedAdapterOperation(operation, 0.5, "first")
    second = WeightedAdapterOperation(operation, -0.25, "second")
    owners = []
    for selected in ((first, second), (second, first)):
        initial = mx.array(base).astype(dtype)
        owner = SimpleNamespace(weight=mx.contiguous(initial.T if transposed else initial))
        target = WeightTarget(owner, "weight", base.shape)
        fuse_and_check(
            {"layer": target},
            {"layer": selected},
            transposed=frozenset({"layer"}) if transposed else frozenset(),
        )
        assert owner.weight.dtype == dtype
        assert owner.weight.shape == (base.T.shape if transposed else base.shape)
        owners.append(owner)
    assert mx.array_equal(owners[0].weight, owners[1].weight).item()


@pytest.mark.parametrize("failure", ["shape", "strength"])
def test_preflight_does_not_mutate_any_target_on_error(failure: str) -> None:
    first = SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float16))
    second = SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float16))
    first_original, second_original = first.weight, second.weight
    (good,) = parse_adapter(
        {"a.lora_A.weight": mx.ones((2, 3)), "a.lora_B.weight": mx.ones((4, 2))}
    ).operations
    values = mx.full((4, 4 if failure == "shape" else 3), 1e10 if failure == "overflow" else 1.0)
    (bad,) = parse_adapter({"b.diff": values}).operations
    with pytest.raises(
        ValueError,
        match={
            "shape": "shape mismatch",
            "overflow": "nonfinite",
            "strength": "strength must be finite",
        }[failure],
    ):
        fuse_and_check(
            {
                "a": WeightTarget(first, "weight", (4, 3)),
                "b": WeightTarget(second, "weight", (4, 3)),
            },
            {
                "a": (WeightedAdapterOperation(good, 1.0, "a"),),
                "b": (
                    WeightedAdapterOperation(
                        bad, float("inf") if failure == "strength" else 1.0, "b"
                    ),
                ),
            },
        )
    assert first.weight is first_original
    assert second.weight is second_original


def test_pointwise_vs_spatial_locon_target() -> None:
    arrays = {
        "layer.lora_down.weight": mx.ones((2, 3, 3, 3)),
        "layer.lora_up.weight": mx.ones((4, 2, 1, 1)),
    }
    (operation,) = parse_adapter(arrays).operations
    with pytest.raises(ValueError, match="shape mismatch"):
        reconstruct_delta(operation, mx.zeros((4, 3)))
    result = reconstruct_delta(operation, mx.zeros((4, 3, 3, 3)))
    assert mx.all(result == 2).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_first_use_check_rejects_nonfinite_base(dtype, value) -> None:
    initial = mx.array([[1, value], [2, 3]], dtype=dtype)
    owner = SimpleNamespace(weight=initial)
    (operation,) = parse_adapter(
        {"layer.lora_A.weight": mx.ones((1, 2)), "layer.lora_B.weight": mx.ones((2, 1))}
    ).operations
    with pytest.raises(ValueError, match="nonfinite"):
        fuse_and_check(
            {"layer": WeightTarget(owner, "weight", (2, 2))},
            {"layer": (WeightedAdapterOperation(operation, 1.0, "adapter"),)},
        )


def test_first_use_check_accepts_exact_cancellation() -> None:
    initial = mx.ones((2, 2), dtype=mx.float16)
    owner = SimpleNamespace(weight=initial)
    (operation,) = parse_adapter(
        {
            "layer.lora_A.weight": mx.array([[1e4, 1e4], [-1e4, -1e4]]),
            "layer.lora_B.weight": mx.full((2, 2), 1e4),
        }
    ).operations
    fuse_and_check(
        {"layer": WeightTarget(owner, "weight", (2, 2))},
        {"layer": (WeightedAdapterOperation(operation, 1.0, "adapter"),)},
    )
    assert mx.array_equal(owner.weight, initial).item()


def test_first_use_check_rejects_bfloat16_cast_overflow() -> None:
    initial = mx.full((2, 2), float.fromhex("0x1.fep+127"), dtype=mx.bfloat16)
    owner = SimpleNamespace(weight=initial)
    (operation,) = parse_adapter(
        {
            "layer.lora_A.weight": mx.full((1, 2), 1e18),
            "layer.lora_B.weight": mx.full((2, 1), 9e17),
        }
    ).operations
    with pytest.raises(ValueError, match="nonfinite"):
        fuse_and_check(
            {"layer": WeightTarget(owner, "weight", (2, 2))},
            {"layer": (WeightedAdapterOperation(operation, 1.0, "adapter"),)},
        )


@pytest.mark.parametrize("kind", ["lora", "loha", "lokr_factored"])
def test_saved_zero_alpha_uses_unit_intrinsic_scale(kind: str) -> None:
    base, factors, expected = _case(kind)
    factors["alpha"] = 0.0
    operation = parse_adapter(
        {f"layer.{key}": mx.array(value) for key, value in factors.items()}
    ).operations[0]
    actual = reconstruct_delta(operation, mx.array(base))
    scale = 2.0 / 3.0
    assert mx.allclose(actual, mx.array(expected) * scale, atol=2e-6, rtol=2e-5).item()


def test_plain_lora_saved_zero_alpha_moves_installed_weight_at_unit_scale() -> None:
    base, factors, expected = _case("lora")
    factors["alpha"] = 0.0
    operation = parse_adapter(
        {f"layer.{key}": mx.array(value) for key, value in factors.items()}
    ).operations[0]
    owner = SimpleNamespace(weight=mx.array(base))

    fuse_and_check(
        {"layer": WeightTarget(owner, "weight", base.shape)},
        {"layer": (WeightedAdapterOperation(operation, 1.0, "adapter"),)},
    )

    unit_scale_delta = expected * (2.0 / 3.0)
    np.testing.assert_allclose(np.asarray(owner.weight), base + unit_scale_delta, atol=3e-6)


def test_multiple_replacement_operations_are_rejected_before_mutation() -> None:
    owner = SimpleNamespace(weight=mx.ones((2, 2)))
    original = owner.weight
    first = parse_adapter({"layer.set_weight": mx.zeros((2, 2))}).operations[0]
    second = parse_adapter({"layer.set_weight": mx.full((2, 2), 2.0)}).operations[0]
    with pytest.raises(ValueError, match="multiple replacement operations"):
        fuse_and_check(
            {"layer": WeightTarget(owner, "weight", (2, 2))},
            {
                "layer": (
                    WeightedAdapterOperation(first, 1.0, "first"),
                    WeightedAdapterOperation(second, 1.0, "second"),
                )
            },
        )
    assert owner.weight is original


def test_half_targets_accumulate_plain_lora_in_half_and_others_in_fp32() -> None:
    base, factors, _ = _case("lora")
    (operation,) = parse_adapter(
        {f"layer.{key}": mx.array(value) for key, value in factors.items()}
    ).operations
    first = WeightedAdapterOperation(operation, 0.5, "first")
    second = WeightedAdapterOperation(operation, -0.25, "second")
    up = mx.array(factors["lora_up.weight"])
    down = mx.array(factors["lora_down.weight"])
    unit_scale = 3.0 / 2.0  # alpha 3 over rank 2

    half = mx.array(base).astype(mx.float16)
    owner = SimpleNamespace(weight=half)
    fuse_and_check(
        {"layer": WeightTarget(owner, "weight", base.shape)},
        {"layer": (first, second)},
        accumulation=FloatDType.FLOAT16,
    )
    up16, down16 = up.astype(mx.float16), down.astype(mx.float16)
    expected_half = mx.addmm(
        mx.addmm(half, up16, down16, alpha=0.5 * unit_scale), up16, down16, alpha=-0.25 * unit_scale
    )
    assert owner.weight.dtype == mx.float16
    assert mx.array_equal(owner.weight, expected_half).item()

    # The default keeps the FP32 chain for half targets too.
    default_owner = SimpleNamespace(weight=half)
    fuse_and_check(
        {"layer": WeightTarget(default_owner, "weight", base.shape)}, {"layer": (first, second)}
    )
    expected_default = mx.addmm(
        mx.addmm(half.astype(mx.float32), up, down, alpha=0.5 * unit_scale),
        up,
        down,
        alpha=-0.25 * unit_scale,
    ).astype(mx.float16)
    assert mx.array_equal(default_owner.weight, expected_default).item()

    wide = mx.array(base).astype(mx.bfloat16)
    owner_wide = SimpleNamespace(weight=wide)
    fuse_and_check(
        {"layer": WeightTarget(owner_wide, "weight", base.shape)},
        {"layer": (first, second)},
        accumulation=FloatDType.FLOAT16,
    )
    expected_wide = mx.addmm(
        mx.addmm(wide.astype(mx.float32), up, down, alpha=0.5 * unit_scale),
        up,
        down,
        alpha=-0.25 * unit_scale,
    ).astype(mx.bfloat16)
    assert mx.array_equal(owner_wide.weight, expected_wide).item()

    # A magnitude-decomposed term keeps the FP32 chain even on a half target.
    dora_base, dora_factors, _ = _case("dora")
    (dora,) = parse_adapter(
        {f"layer.{key}": mx.array(value) for key, value in dora_factors.items()}
    ).operations
    selected = (WeightedAdapterOperation(dora, 1.0, "dora"),)
    rounded = mx.array(dora_base).astype(mx.float16)
    reference = SimpleNamespace(weight=rounded.astype(mx.float32))
    fuse_and_check(
        {"layer": WeightTarget(reference, "weight", dora_base.shape)}, {"layer": selected}
    )
    owner_half = SimpleNamespace(weight=rounded)
    fuse_and_check(
        {"layer": WeightTarget(owner_half, "weight", dora_base.shape)},
        {"layer": selected},
        accumulation=FloatDType.FLOAT16,
    )
    assert mx.array_equal(owner_half.weight, reference.weight.astype(mx.float16)).item()


def test_half_admission_rejects_factors_beyond_float16_range() -> None:
    from kinomlx.adapters.fusion import fuse_weight, half_accumulation_fallbacks

    (wide,) = parse_adapter(
        {
            "w.lora_A.weight": mx.full((1, 3), 1e-5, dtype=mx.float32),
            "w.lora_B.weight": mx.full((2, 1), 1e5, dtype=mx.float32),
        }
    ).operations
    (narrow,) = parse_adapter(
        {
            "w.lora_A.weight": mx.full((1, 3), 0.5, dtype=mx.float32),
            "w.lora_B.weight": mx.full((2, 1), 2.0, dtype=mx.float32),
        }
    ).operations
    wide_ops = (WeightedAdapterOperation(wide, 1.0, "w"),)
    narrow_ops = (WeightedAdapterOperation(narrow, 1.0, "n"),)
    assert half_accumulation_fallbacks({"wide": wide_ops}) == frozenset({"wide"})
    assert not half_accumulation_fallbacks({"narrow": narrow_ops})
    assert not half_accumulation_fallbacks({})

    # The narrow range is real: the FP32 chain fuses the wide factors to 1.0, the
    # half path overflows on the factor cast, which is why callers gate on admission.
    base = mx.zeros((2, 3), dtype=mx.float16)
    wide_fp32 = fuse_weight(base, wide_ops, accumulation=FloatDType.FLOAT32)
    wide_fp16 = fuse_weight(base, wide_ops, accumulation=FloatDType.FLOAT16)
    mx.eval(wide_fp32, wide_fp16)
    assert mx.all(wide_fp32 == 1.0).item()
    assert not mx.all(mx.isfinite(wide_fp16)).item()


def test_half_fallback_is_per_target_and_skips_native_half_factors(monkeypatch) -> None:
    from kinomlx.adapters import fusion
    from kinomlx.adapters.validation import AdapterWeightCheck

    def operation(up, down, dtype):
        (op,) = parse_adapter(
            {
                "w.lora_A.weight": mx.full((1, 2), down, dtype=dtype),
                "w.lora_B.weight": mx.full((2, 1), up, dtype=dtype),
            }
        ).operations
        return (WeightedAdapterOperation(op, 1.0, "a"),)

    groups = {
        "wide": operation(1e5, 1e-5, mx.float32),
        "narrow": operation(0.1, 0.3, mx.float32),
    }
    owners = {key: SimpleNamespace(weight=mx.ones((2, 2), dtype=mx.float16)) for key in groups}
    targets = {key: WeightTarget(owner, "weight", (2, 2)) for key, owner in owners.items()}
    expected = {
        key: fusion.fuse_weight(
            owner.weight, groups[key], accumulation=("float32" if key == "wide" else "float16")
        )
        for key, owner in owners.items()
    }
    fusion.fuse_adapter_operations(
        targets, groups, pending=AdapterWeightCheck(), accumulation=FloatDType.FLOAT16
    )
    for key, owner in owners.items():
        assert mx.array_equal(owner.weight, expected[key]).item()

    half = operation(0.1, 0.3, mx.float16)

    def unexpected(*args, **kwargs):
        raise AssertionError("native half factors must not be reduced")

    monkeypatch.setattr(fusion.mx, "max", unexpected)
    assert not fusion.half_accumulation_fallbacks({"half": half})
