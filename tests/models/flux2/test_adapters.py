"""Klein row placement controls against independent source-oriented equations."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.fusion import WeightedAdapterOperation
from kinomlx.adapters.peft import parse_configured_adapter
from kinomlx.modeling.loading import WeightTarget
from kinomlx.models.flux2.adapters.keys import (
    AdapterBinding,
    Flux2AdapterKeys,
    place_operation,
)
from kinomlx.models.flux2.shapes import transformer_shapes
from tests.adapters._fusion import fuse_and_check


def _source_case(kind):
    rng = np.random.default_rng(813)
    base = rng.normal(size=(12, 6)).astype(np.float32)
    up = rng.normal(size=(12, 2)).astype(np.float32) * 0.1
    down = rng.normal(size=(2, 6)).astype(np.float32) * 0.1
    factors = {"lora_A.weight": down, "lora_B.weight": up}
    delta = up @ down
    if kind.startswith("lokr"):
        first = rng.normal(size=(3, 2)).astype(np.float32) * 0.1
        left = rng.normal(size=(4, 2)).astype(np.float32) * 0.1
        right = rng.normal(size=(2, 3)).astype(np.float32) * 0.1
        second = left @ right
        factors = {"lokr_w1": first}
        if kind == "lokr_dense":
            factors["lokr_w2"] = second
        else:
            factors.update(lokr_w2_a=left, lokr_w2_b=right, alpha=np.array(3.0))
        delta = np.kron(first, second) * (1.0 if kind == "lokr_dense" else 1.5)
    elif kind == "loha":
        up2 = rng.normal(size=(12, 2)).astype(np.float32) * 0.1
        down2 = rng.normal(size=(2, 6)).astype(np.float32) * 0.1
        factors = {"hada_w1_a": up, "hada_w1_b": down, "hada_w2_a": up2, "hada_w2_b": down2}
        delta *= up2 @ down2
    elif kind == "ia3":
        vector = np.linspace(0.5, 1.5, 12, dtype=np.float32).reshape(-1, 1)
        factors = {"ia3_l": vector}
        delta = base * (vector - 1)
    elif kind == "hira":
        factors = {"hira_A": down, "hira_B": up}
        delta = base * (up @ down)
    if kind in ("dora_input", "dora_output"):
        axis = 0 if kind == "dora_input" else 1
        magnitude = np.linspace(0.7, 1.4, base.shape[1 - axis], dtype=np.float32)
        magnitude = magnitude[None, :] if axis == 0 else magnitude[:, None]
        factors["dora_scale"] = magnitude
        adapted = base + delta
        delta = adapted * magnitude / np.linalg.norm(adapted, axis=axis, keepdims=True) - base
    weights = {f"source.{key}": mx.array(value) for key, value in factors.items()}
    config = {"peft_type": "HIRA", "r": 2} if kind == "hira" else {}
    return base, delta, parse_configured_adapter(weights, config).operations[0]


@pytest.mark.parametrize(
    "kind",
    ["lora", "lokr_dense", "lokr_factored", "loha", "dora_input", "dora_output", "ia3", "hira"],
)
@pytest.mark.parametrize("layout", ["direct", "qkv", "adaln"])
@pytest.mark.parametrize("transposed", [False, True])
def test_source_equations_survive_klein_row_placement(kind, layout, transposed):
    base, delta, operation = _source_case(kind)
    if layout == "qkv":
        binding = AdapterBinding(("q", "k", "v"))
        bases, deltas = np.split(base, 3), np.split(delta, 3)
    elif layout == "adaln":
        binding = AdapterBinding(("final",), swap_halves=True)
        bases = [np.concatenate(np.split(base, 2)[::-1])]
        deltas = [np.concatenate(np.split(delta, 2)[::-1])]
    else:
        binding = AdapterBinding(("direct",))
        bases, deltas = [base], [delta]
    logical = {
        f"{name}.weight": mx.array(value)
        for name, value in zip(binding.targets, bases, strict=True)
    }
    placed = place_operation(operation, binding, logical.__getitem__)
    targets, operations = {}, {}
    for (name, op), expected in zip(placed, deltas, strict=True):
        np.testing.assert_allclose(
            np.asarray(reconstruct_delta(op, logical[name])), expected, rtol=3e-5, atol=2e-6
        )
        value = logical[name].T if transposed else logical[name]
        targets[name] = WeightTarget(SimpleNamespace(weight=value), "weight", logical[name].shape)
        operations[name] = (WeightedAdapterOperation(op, 0.7, "source"),)
    fuse_and_check(
        targets, operations, transposed=frozenset(targets) if transposed else frozenset()
    )
    for target, expected_base, expected_delta in zip(targets.values(), bases, deltas, strict=True):
        actual = target.owner.weight.T if transposed else target.owner.weight
        np.testing.assert_allclose(
            np.asarray(actual), expected_base + 0.7 * expected_delta, rtol=3e-5, atol=2e-6
        )


def test_graph_has_all_canonical_and_flattened_aliases():
    from kinomlx.models.flux2.adapters.categories import target_categories

    keys = Flux2AdapterKeys()
    for name in transformer_shapes():
        module = name.removesuffix(".weight")
        expected = AdapterBinding((module,))
        assert keys.resolve("base_model.model.transformer." + module) == expected
        assert keys.resolve("lora_unet_" + module.replace(".", "_")) == expected
    assert keys.resolve("diffusion_model.double_blocks.0.img_attn.qkv") == AdapterBinding(
        tuple(f"transformer_blocks.0.attn.to_{part}" for part in ("q", "k", "v"))
    )
    assert keys.resolve("final_layer.adaLN_modulation.1") == AdapterBinding(
        ("norm_out.linear",), swap_halves=True
    )
    assert keys.resolve("double_blocks.8.img_attn.qkv") is None
    expected = AdapterBinding(("transformer_blocks.0.attn.to_q",))
    assert keys.resolve("sformer_blocks.0.attn.to_q") == expected
    assert keys.resolve("lora_unet_sformer_blocks_0_attn_to_q") == expected
    assert "norm" not in target_categories("norm_out.linear.weight")
    assert "modulation" in target_categories("norm_out.linear.weight")
    assert "norm" in target_categories("transformer_blocks.0.attn.norm_q.weight")
