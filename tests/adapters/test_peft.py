"""PEFT scaling and serialized-factor regression cases."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.formats import parse_adapter
from kinomlx.adapters.fusion import WeightedAdapterOperation
from kinomlx.adapters.peft import configured_operation, parse_configured_adapter
from kinomlx.modeling.loading import WeightTarget
from tests.adapters._fusion import fuse_and_check


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("legacy_rank", [None, 7.0])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_adalora_pruned_rank_uses_original_denominator(named, legacy_rank, transposed, dtype):
    down = np.arange(12, dtype=np.float32).reshape(2, 6) / 71
    up = np.arange(8, dtype=np.float32).reshape(4, 2) / 53
    singular = np.array([[0.4], [-0.7]], dtype=np.float32)
    suffix = ".style" if named else ""
    factors = {
        f"layer.lora_A{suffix}": mx.array(down),
        f"layer.lora_B{suffix}": mx.array(up),
        f"layer.lora_E{suffix}": mx.array(singular),
    }
    if legacy_rank is not None:
        factors[f"layer.ranknum{suffix}"] = mx.array(legacy_rank)
    (operation,) = parse_adapter(factors).operations
    config = {
        "peft_type": "ADALORA",
        "init_r": 6,
        "lora_alpha": 3,
        "rank_pattern": {"layer.lora_E": [True, False, True, False, False, False]},
    }
    operation = configured_operation(operation, config)
    denominator = 6 if legacy_rank is None else legacy_rank
    delta = (up @ (down * singular)) * 3 / (denominator + 1e-5)
    original = mx.array(np.arange(24, dtype=np.float32).reshape(4, 6) / 100).astype(dtype)
    owner = SimpleNamespace(weight=mx.contiguous(original.T if transposed else original))
    target = WeightTarget(owner, "weight", original.shape)
    fuse_and_check(
        {"layer": target},
        {"layer": (WeightedAdapterOperation(operation, -0.4, "adapter"),)},
        transposed=frozenset({"layer"}) if transposed else frozenset(),
    )
    actual = owner.weight.T if transposed else owner.weight
    expected = (original.astype(mx.float32) - mx.array(delta) * 0.4).astype(dtype)
    assert mx.allclose(actual.astype(mx.float32), expected.astype(mx.float32), atol=1e-7).item()


def test_adalora_rank_zero_is_an_explicit_noop():
    (operation,) = parse_adapter(
        {
            "layer.lora_A": mx.zeros((0, 6)),
            "layer.lora_B": mx.zeros((4, 0)),
            "layer.lora_E": mx.zeros((0, 1)),
        }
    ).operations
    operation = configured_operation(
        operation, {"peft_type": "ADALORA", "init_r": 6, "lora_alpha": 3}
    )
    assert mx.array_equal(reconstruct_delta(operation, mx.ones((4, 6))), mx.zeros((4, 6))).item()


def test_adalora_rank_pattern_must_match_serialized_factors():
    (operation,) = parse_adapter(
        {
            "layer.lora_A": mx.ones((2, 6)),
            "layer.lora_B": mx.ones((4, 2)),
            "layer.lora_E": mx.ones((2, 1)),
        }
    ).operations
    with pytest.raises(ValueError, match="rank_pattern"):
        configured_operation(
            operation,
            {
                "peft_type": "ADALORA",
                "init_r": 6,
                "rank_pattern": {"layer.lora_E": [True] * 6},
            },
        )


@pytest.mark.parametrize("axis", ["input", "output"])
@pytest.mark.parametrize("named", [False, True])
def test_peft_ia3_uses_identity_one_not_lycoris_delta_zero(axis, named):
    base = np.arange(24, dtype=np.float32).reshape(4, 6) / 29
    shape = (1, 6) if axis == "input" else (4, 1)
    vector = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 7 + 0.5
    (operation,) = parse_adapter(
        {f"layer.ia3_l{'.style' if named else ''}": mx.array(vector)}
    ).operations
    delta = reconstruct_delta(operation, mx.array(base))
    assert mx.allclose(delta, mx.array(base * (vector - 1)), atol=1e-7).item()
    identity = replace(operation, factors={"ia3_vector": mx.ones(shape)})
    assert mx.array_equal(reconstruct_delta(identity, mx.array(base)), mx.zeros(base.shape)).item()


def test_peft_alpha_pattern_uses_first_match_and_rank_pattern():
    (operation,) = parse_adapter(
        {
            "base_model.model.layers.3.attn.lora_A.weight": mx.ones((2, 6)),
            "base_model.model.layers.3.attn.lora_B.weight": mx.ones((4, 2)),
        }
    ).operations
    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "rank_pattern": {r"layers\.\d+\.attn": 2},
        "alpha_pattern": {r"layers\.\d+\.attn": 3, "attn": 9},
    }
    configured = configured_operation(operation, config)
    assert configured.alpha == 3
    with pytest.raises(ValueError, match="configured rank"):
        configured_operation(operation, {**config, "rank_pattern": {}})


def test_unsupported_nested_regex_is_rejected_without_matching():
    (operation,) = parse_adapter(
        {"layer.lora_A": mx.ones((2, 6)), "layer.lora_B": mx.ones((4, 2))}
    ).operations
    with pytest.raises(ValueError, match="complex regex"):
        configured_operation(operation, {"peft_type": "LORA", "alpha_pattern": {"(a+)+$": 3}})


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("strength", [-0.4, 0.0, 1.0])
def test_saved_full_module_and_bias_are_replacements(named, strength):
    wrapper = ".modules_to_save.style" if named else ""
    weights = {
        f"base_model.model.layer{wrapper}.weight": mx.full((4, 6), 0.3),
        f"base_model.model.layer{wrapper}.bias": mx.full((4,), 0.7),
    }
    inventory = parse_configured_adapter(
        weights, {"peft_type": "LORA", "modules_to_save": ["layer"]}
    )
    assert len(inventory.operations) == 2
    owner = SimpleNamespace(weight=mx.full((4, 6), 0.2), bias=mx.full((4,), 0.4))
    targets = {
        part: WeightTarget(owner, part, getattr(owner, part).shape) for part in ("weight", "bias")
    }
    fuse_and_check(
        targets,
        {
            op.part: (WeightedAdapterOperation(op, strength, "adapter"),)
            for op in inventory.operations
        },
    )
    assert mx.allclose(owner.weight, mx.full((4, 6), 0.2 + strength * 0.1)).item()
    assert mx.allclose(owner.bias, mx.full((4,), 0.4 + strength * 0.3)).item()
    assert {key for op in inventory.operations for key in op.source_keys} == set(weights)


@pytest.mark.parametrize("bias_mode", ["all", "lora_only"])
def test_saved_base_bias_is_distinct_from_lora_b_bias(bias_mode):
    weights = {
        "layer.lora_A.weight": mx.ones((2, 6)),
        "layer.lora_B.weight": mx.ones((4, 2)),
        "layer.base_layer.bias": mx.full((4,), 0.7),
    }
    inventory = parse_configured_adapter(weights, {"peft_type": "LORA", "bias": bias_mode})
    bias = next(op for op in inventory.operations if op.part == "bias")
    assert bias.prefix == "layer"
    assert bias.kind == "set"
    assert mx.allclose(reconstruct_delta(bias, mx.full((4,), 0.2)), mx.full((4,), 0.5)).item()
    # An actual lora_B bias stays scaled LoRA algebra, even under bias=all.
    del weights["layer.base_layer.bias"]
    weights["layer.lora_B.bias"] = mx.ones((4,))
    inventory = parse_configured_adapter(
        weights, {"peft_type": "LORA", "bias": bias_mode, "lora_alpha": 6}
    )
    bias = next(op for op in inventory.operations if op.part == "bias")
    assert bias.kind == "lora"
    assert mx.array_equal(reconstruct_delta(bias, mx.zeros((4,))), mx.full((4,), 3)).item()


def test_bare_weights_are_not_assumed_to_be_saved_modules():
    with pytest.raises(ValueError, match="no complete"):
        parse_configured_adapter({"layer.weight": mx.ones((4, 6))}, {"peft_type": "LORA"})


def test_saved_module_and_factor_adapter_names_must_agree():
    weights = {
        "layer.modules_to_save.first.weight": mx.ones((4, 6)),
        "other.lora_A.second.weight": mx.ones((2, 6)),
        "other.lora_B.second.weight": mx.ones((4, 2)),
    }
    with pytest.raises(ValueError, match="multiple named"):
        parse_configured_adapter(weights, {"peft_type": "LORA", "modules_to_save": ["layer"]})


def test_named_loha_and_lokr_exports_are_recognized():
    for factors, family in (
        (
            {
                "hada_w1_a": mx.ones((4, 2)),
                "hada_w1_b": mx.ones((2, 6)),
                "hada_w2_a": mx.ones((4, 2)),
                "hada_w2_b": mx.ones((2, 6)),
            },
            "LOHA",
        ),
        ({"lokr_w1": mx.ones((2, 2)), "lokr_w2": mx.ones((2, 3))}, "LOKR"),
    ):
        inventory = parse_configured_adapter(
            {f"layer.{key}.style": value for key, value in factors.items()}, {"peft_type": family}
        )
        assert len(inventory.operations) == 1
        assert inventory.unknown_keys == ()
        assert reconstruct_delta(inventory.operations[0], mx.ones((4, 6))).shape == (4, 6)


def test_adalora_scalar_factor_has_a_structural_error():
    with pytest.raises(ValueError, match="two-dimensional"):
        parse_adapter(
            {
                "layer.lora_A": mx.array(0.1),
                "layer.lora_B": mx.ones((4, 2)),
                "layer.lora_E": mx.ones((2, 1)),
            }
        )


def test_trained_bias_and_lora_b_bias_compose_without_double_scaling():
    weights = {
        "layer.lora_A.weight": mx.ones((2, 6)),
        "layer.lora_B.weight": mx.ones((4, 2)),
        "layer.lora_B.bias": mx.full((4,), 0.1),
        "layer.base_layer.bias": mx.full((4,), 0.7),
    }
    inventory = parse_configured_adapter(
        weights, {"peft_type": "LORA", "bias": "all", "lora_alpha": 6}
    )
    bias = next(op for op in inventory.operations if op.part == "bias")
    assert len(inventory.operations) == 2
    assert mx.allclose(reconstruct_delta(bias, mx.full((4,), 0.2)), mx.full((4,), 0.8)).item()
    assert {key for op in inventory.operations for key in op.source_keys} == set(weights)


def test_dora_config_allows_a_separate_saved_full_module():
    inventory = parse_configured_adapter(
        {"layer.weight": mx.ones((4, 6))},
        {"peft_type": "LORA", "modules_to_save": ["layer"], "use_dora": True},
    )
    assert inventory.operations[0].kind == "set"


def test_explicit_alpha_retains_rank_stabilized_convention():
    (operation,) = parse_adapter(
        {
            "layer.lora_A.weight": mx.ones((4, 6)),
            "layer.lora_B.weight": mx.ones((8, 4)),
            "layer.alpha": mx.array(3.0),
        }
    ).operations
    operation = configured_operation(
        operation, {"peft_type": "LORA", "r": 4, "lora_alpha": 17, "use_rslora": True}
    )
    assert operation.alpha == 6


def test_peft_dense_lokr_keeps_configured_rank_scaling():
    weights = {"layer.lokr_w1": mx.ones((2, 2)), "layer.lokr_w2": mx.ones((2, 3))}
    inventory = parse_configured_adapter(weights, {"peft_type": "LOKR", "r": 8, "alpha": 3})
    assert mx.array_equal(
        reconstruct_delta(inventory.operations[0], mx.ones((4, 6))), mx.full((4, 6), 3 / 8)
    ).item()
    # LyCORIS all-dense exports do not encode a factor rank to normalize by.
    (plain,) = parse_adapter({**weights, "layer.alpha": mx.array(3.0)}).operations
    assert mx.array_equal(reconstruct_delta(plain, mx.ones((4, 6))), mx.ones((4, 6))).item()


@pytest.mark.parametrize("nonfinite", [False, True])
def test_lazy_nonplain_adapter_parses_once_and_validates_values(monkeypatch, nonfinite):
    from kinomlx.adapters import peft

    weights = {
        "layer.hada_w1_a": mx.ones((4, 2)),
        "layer.hada_w1_b": mx.ones((2, 6)),
        "layer.hada_w2_a": mx.ones((4, 2)),
        "layer.hada_w2_b": mx.full((2, 6), float("nan") if nonfinite else 1.0),
    }
    parse = peft.parse_configured_adapter
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return parse(*args, **kwargs)

    monkeypatch.setattr(peft, "parse_configured_adapter", counted)
    if nonfinite:
        with pytest.raises(ValueError, match="nonfinite"):
            peft.parse_lazy_adapter(weights, {})
    else:
        inventory = peft.parse_lazy_adapter(weights, {})
        assert inventory.operations[0].kind == "loha"
    assert calls == [{"validate_values": False}]
