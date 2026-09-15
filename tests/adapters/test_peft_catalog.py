"""Configured PEFT families share shape, dtype, placement, and fusion contracts."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.fusion import WeightedAdapterOperation
from kinomlx.adapters.peft import parse_configured_adapter
from kinomlx.modeling.loading import WeightTarget
from tests.adapters._fusion import fuse_and_check

_FAMILIES = (
    "hira",
    "delora",
    "c3a",
    "miss",
    "road",
    "gralora",
    "peanut",
    "deft",
    "shira",
    "supertuning",
    "vera",
    "pvera",
    "randlora",
    "vblora",
    "unilora",
    "tinylora",
    "waveft",
    "bdlora",
    "qalora",
)


def _values(*shape):
    size = 1
    for dimension in shape:
        size *= dimension
    return (mx.arange(size, dtype=mx.float32).reshape(shape) % 11 - 5) / 23


def _case(family):
    config = {"peft_type": family.upper(), "r": 2}
    down, up = _values(2, 4), _values(8, 2)
    shared = {}
    if family == "hira":
        factors = {"A": down, "B": up}
    elif family == "delora":
        factors = {"A": down, "B": up, "lambda": mx.array([0.4]), "w_norm": mx.ones(4)}
    elif family == "c3a":
        config["block_size"] = 2
        factors = {"kernel": _values(4, 2, 2)}
    elif family == "miss":
        config["init_weights"] = "bat"
        factors = {"block": _values(4, 2, 2)}
    elif family == "road":
        config.update(variant="road_4", group_size=4)
        factors = {"theta": _values(16), "alpha": _values(16) + 1}
    elif family == "gralora":
        config.update(r=4, gralora_k=2, hybrid_r=0, alpha=3)
        factors = {"A": _values(2, 2, 4), "B": _values(2, 4, 4)}
    elif family == "peanut":
        config.update(depth=1, act_fn="gelu")
        factors = {
            "A.weight": _values(2, 8),
            "B.weight": up,
            "encoders.style.0.weight": _values(2, 2),
            "decoders.style.0.weight": _values(2, 2),
        }
    elif family == "deft":
        config.update(decomposition_method="qr", alpha=3)
        factors = {"P": up, "R": down}
    elif family == "shira":
        factors = {"weight": _values(3), "indices": mx.array([[0, 7, 7], [2, 3, 3]])}
    elif family == "supertuning":
        config.update(lora_alpha=3)
        factors = {
            "values": _values(3),
            "indices": mx.array([1, 31, 31]),
            "lora_A.weight": down,
            "lora_B.weight": up,
        }
    elif family in ("vera", "pvera"):
        count = 2 if family == "vera" else 4
        factors = {"lambda_d": _values(count), "lambda_b": _values(8)}
        shared = {"A": _values(count, 6), "B": _values(10, 2)}
    elif family == "randlora":
        config.update(r=3, randlora_alpha=2.1)
        factors = {"lambda": _values(3, 2), "gamma": _values(2, 4)}
        shared = {"A": _values(3, 1, 6), "B": _values(10, 3, 3)}
    elif family == "vblora":
        config.update(num_vectors=5, vector_length=2, topk=3)
        factors = {"logits_A": _values(2, 2, 5), "logits_B": _values(4, 2, 5)}
        shared = {"vector_bank": _values(5, 2)}
    elif family == "tinylora":
        config.update(u=3, save_projection=True, weight_tying=1.0)
        factors = {"A": down, "B": up, "P": _values(3, 2, 2)}
        shared = {"v.0": _values(3)}
    elif family == "waveft":
        config.update(wavelet_family="db4", n_frequency=6)
        factors = {"spectrum": _values(6)}
    elif family in ("bdlora", "qalora"):
        config.update(peft_type="LORA", lora_alpha=3)
        if family == "bdlora":
            config["use_bdlora"] = {"nblocks": 2, "target_modules_bd_a": ["layer"]}
        else:
            config.update(use_qalora=True, qalora_group_size=2)
        return {"layer.lora_A": _values(2, 2), "layer.lora_B": up}, config
    else:
        config.update(theta_d_length=11, save_indices=True)
        factors = {
            "indices_A": mx.arange(8).reshape(2, 4),
            "indices_B": mx.arange(16).reshape(8, 2) % 11,
            "scales_A": down,
            "scales_B": up,
        }
        shared = {"theta_d": _values(11)}
    return {
        **{f"layer.{family}_{key}": value for key, value in factors.items()},
        **{f"base_model.{family}_{key}": value for key, value in shared.items()},
    }, config


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("strength", [-0.5, 0.0, 1.0])
def test_configured_families_install_in_both_layouts(family, transposed, dtype, strength):
    weights, config = _case(family)
    inventory = parse_configured_adapter(weights, config)
    operation = inventory.operations[0]
    logical = _values(8, 4).astype(dtype)
    owner = SimpleNamespace(weight=mx.contiguous(logical.T) if transposed else logical)
    original = owner.weight
    expected = (
        logical.astype(mx.float32) + strength * reconstruct_delta(operation, logical)
    ).astype(dtype)
    fuse_and_check(
        {"weight": WeightTarget(owner, "weight", (8, 4))},
        {"weight": (WeightedAdapterOperation(operation, strength, "synthetic"),)},
        transposed=frozenset({"weight"}) if transposed else frozenset(),
    )
    actual = owner.weight.T if transposed else owner.weight
    assert actual.dtype == dtype
    assert mx.allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), atol=1e-6, rtol=1e-6
    ).item()
    assert mx.array_equal(original.T if transposed else original, logical).item()
    accounted = [key for op in inventory.operations for key in op.source_keys] + list(
        inventory.shared_keys
    )
    assert sorted(accounted) == sorted(weights)
    assert len(accounted) == len(set(accounted))


@pytest.mark.parametrize("family", _FAMILIES)
def test_bad_shape_does_not_mutate_a_live_weight(family):
    weights, config = _case(family)
    operation = parse_configured_adapter(weights, config).operations[0]
    owner = SimpleNamespace(weight=mx.ones((3, 1)))
    original = owner.weight
    with pytest.raises(
        ValueError, match="shape|width|cover|target|divide|rank|vector_length|indices"
    ):
        fuse_and_check(
            {"weight": WeightTarget(owner, "weight", original.shape)},
            {"weight": (WeightedAdapterOperation(operation, 1.0, "synthetic"),)},
        )
    assert owner.weight is original


@pytest.mark.parametrize("family", _FAMILIES)
def test_unknown_semantic_tensor_is_not_dropped(family):
    weights, config = _case(family)
    weights["layer.unknown_training_gate"] = mx.array(1.0)
    with pytest.raises(ValueError, match="unsupported semantic tensor"):
        parse_configured_adapter(weights, config)


@pytest.mark.parametrize("dtype", [mx.int32, mx.int64, mx.float32])
@pytest.mark.parametrize("indices", [[-1, 2, 3], [0, 2, 2147483648]])
def test_sparse_indices_are_validated_before_scatter(dtype, indices):
    if dtype == mx.int32 and max(indices) > 2147483647:
        return
    weights, config = _case("supertuning")
    weights["layer.supertuning_indices"] = mx.array(indices, dtype=dtype)
    with pytest.raises(ValueError, match="indices"):
        parse_configured_adapter(weights, config)


def test_sparse_coordinates_cannot_be_fractional():
    weights, config = _case("shira")
    weights["layer.shira_indices"] = mx.array([[0.5, 7, 7], [2, 3, 3]])
    with pytest.raises(ValueError, match="non-integral"):
        parse_configured_adapter(weights, config)


def test_shared_bases_must_be_present_and_agree_with_local_copies():
    weights, config = _case("randlora")
    weights["layer.randlora_A"] = weights["base_model.randlora_A"]
    parse_configured_adapter(weights, config)
    weights["layer.randlora_A"] = weights["layer.randlora_A"] + 1
    with pytest.raises(ValueError, match="projections disagree"):
        parse_configured_adapter(weights, config)
    weights, config = _case("vera")
    weights.pop("base_model.vera_A")
    with pytest.raises(ValueError, match="save_projection"):
        parse_configured_adapter(weights, config)


def test_sampling_and_missing_indices_are_explicit_runtime_boundaries():
    weights, config = _case("pvera")
    config["sample_at_inference"] = True
    with pytest.raises(ValueError, match="per-forward sampling"):
        parse_configured_adapter(weights, config)
    weights, config = _case("unilora")
    weights.pop("layer.unilora_indices_A")
    with pytest.raises(ValueError, match="save_indices"):
        parse_configured_adapter(weights, config)


def test_ambiguous_vblora_topk_requires_explicit_indices():
    weights, config = _case("vblora")
    weights["layer.vblora_logits_A"] = mx.zeros((2, 2, 5))
    with pytest.raises(ValueError, match="tied topk cutoff"):
        parse_configured_adapter(weights, config)


def test_beft_and_norm_tuning_are_bias_and_replacement_operations():
    inventory = parse_configured_adapter(
        {"layer.beft_bias": mx.ones((1, 8))}, {"peft_type": "BEFT"}
    )
    assert mx.array_equal(
        reconstruct_delta(inventory.operations[0], mx.zeros(8)), mx.ones(8)
    ).item()
    weights = {
        "layer.ln_tuning_layers.style.weight": _values(8),
        "layer.ln_tuning_layers.style.bias": _values(8),
    }
    inventory = parse_configured_adapter(weights, {"peft_type": "LN_TUNING"})
    assert len(inventory.operations) == 2
    for operation in inventory.operations:
        assert operation.kind == "set"
        assert mx.array_equal(reconstruct_delta(operation, mx.ones(8)), _values(8) - 1).item()


def test_legacy_oft_does_not_alias_the_current_packed_format():
    config = {"peft_type": "OFT", "block_share": False}
    weights = {"layer.oft_r": _values(2, 2, 2), "layer.oft_s": mx.ones((8, 1))}
    operation = parse_configured_adapter(weights, config).operations[0]
    assert mx.all(mx.isfinite(reconstruct_delta(operation, _values(8, 4)))).item()
    weights["layer.oft_R.weight"] = _values(2, 1)
    with pytest.raises(ValueError, match="extra factors"):
        parse_configured_adapter(weights, config)
    bad = replace(operation, factors={"oft_legacy": _values(3, 2, 2), "oft_scale": mx.ones((8, 1))})
    with pytest.raises(ValueError, match="cover|block"):
        reconstruct_delta(bad, _values(8, 4))


def test_hira_is_a_multiplicative_weight_delta() -> None:
    weights, config = _case("hira")
    operation = parse_configured_adapter(weights, config).operations[0]
    base = _values(8, 4)
    pair = weights["layer.hira_B"] @ weights["layer.hira_A"]
    assert mx.allclose(reconstruct_delta(operation, base), base * pair).item()
