"""Krea adapter placement, physical layouts, receipts, and invocation controls."""

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.krea2.adapters.loading as adapter_loading
from kinomlx.adapters.live import live_adapters
from kinomlx.adapters.scaling import UNKNOWN_SCALE_WARNING
from kinomlx.adapters.schedule import AdapterSchedule, parse_sigma_window
from kinomlx.adapters.selection import AdapterSelection
from kinomlx.io.safetensors import save_weights
from kinomlx.models.krea2.adapters.keys import Krea2AdapterKeys
from kinomlx.models.krea2.adapters.loading import (
    apply_krea2_adapters,
    prepare_krea2_adapters,
)
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.catalog import Krea2Variant
from kinomlx.models.krea2.sampling import Krea2PreparedBranch, Krea2Recipe, denoise_krea2
from kinomlx.models.krea2.transformer.loading import transformer_weight_targets
from kinomlx.models.krea2.transformer.model import Krea2Transformer
from kinomlx.models.krea2.types import Krea2Request
from tests.adapters._snapshot import link_as_snapshot

from .transformer._fixtures import populate_transformer, tiny_transformer_config


def _model(*, reference=False, dtype=mx.float32):
    model = Krea2Transformer(
        replace(tiny_transformer_config(), intermediate_size=48), reference_denoise=reference
    )
    populate_transformer(model, transformer_weight_targets(model), dtype=dtype)
    mx.eval(model.parameters())
    return model


def _apply_krea2_adapters(model, selections, *, variant):
    plan = prepare_krea2_adapters(model.config, tuple(selections))
    return apply_krea2_adapters(model, plan, variant=variant)


def _adapter(path: Path, prefix: str, *, out=16, inputs=32, metadata=None):
    save_weights(
        path,
        {
            f"{prefix}.lora_A.weight": mx.full((2, inputs), 0.1),
            f"{prefix}.lora_B.weight": mx.full((out, 2), 0.2),
        },
        metadata,
    )
    return AdapterSelection(path, 1.0)


def _peft_file(tmp_path, weights, config):
    path = tmp_path / "adapter_model.safetensors"
    save_weights(path, weights)
    path.with_name("adapter_config.json").write_text(json.dumps(config))
    return AdapterSelection(path, 1.0)


def test_shared_projection_is_accounted_once_across_two_targets(tmp_path):
    model = _model()
    weights = {
        "base_model.vera_A": mx.full((2, 32), 0.1),
        "base_model.vera_B": mx.full((16, 2), 0.2),
    }
    originals = []
    for name in ("to_k", "to_v"):
        prefix = f"transformer_blocks.0.attn.{name}"
        weights[f"{prefix}.vera_lambda_d"] = mx.ones(2)
        weights[f"{prefix}.vera_lambda_b"] = mx.ones(16)
        originals.append(getattr(model.transformer_blocks[0].attn, name).weight)
    selected = _peft_file(tmp_path, weights, {"peft_type": "VERA", "r": 2, "save_projection": True})
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert receipt.complete_targets == receipt.placed_targets == 2
    assert receipt.coverage == 1.0
    accounted = [key for item in receipt.placements for key in item.source_keys]
    assert sorted(accounted) == sorted(weights)
    assert len(accounted) == len(set(accounted))
    (shared,) = (item for item in receipt.placements if item.kind == "shared_basis")
    assert len(shared.source_keys) == 2
    for name, original in zip(("to_k", "to_v"), originals, strict=True):
        assert mx.allclose(
            getattr(model.transformer_blocks[0].attn, name).weight, original + 0.04
        ).item()


@pytest.mark.parametrize("family", ["ROAD", "BEFT", "IA3"])
def test_trained_bias_composes_with_output_transform_once(tmp_path, family):
    model = _model()
    original_weight, original_bias = model.img_in.weight, model.img_in.bias
    out = original_bias.size
    saved_bias = mx.full((out,), 0.3)
    weights = {"img_in.base_layer.bias": saved_bias}
    config = {"peft_type": family, "bias": "all"}
    if family == "ROAD":
        weights.update(
            {"img_in.road_theta": mx.zeros(out), "img_in.road_alpha": mx.full((out,), 2.0)}
        )
        config.update(variant="road_2", group_size=4)
    elif family == "IA3":
        weights["img_in.ia3_l"] = mx.full((out, 1), 2.0)
    else:
        weights["img_in.beft_bias"] = mx.full((1, out), 0.3)
    selected = replace(_peft_file(tmp_path, weights, config), strength=0.5)
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert receipt.coverage == 1.0
    assert mx.allclose(model.img_in.bias, (original_bias + 0.6) / 2).item()
    assert mx.allclose(
        model.img_in.weight, original_weight * (1.0 if family == "BEFT" else 1.5)
    ).item()
    assert sorted(key for item in receipt.placements for key in item.source_keys) == sorted(weights)


def test_declared_training_only_sampler_state_is_accounted_without_fusion(tmp_path):
    model = _model()
    prefix = "transformer_blocks.0.attn.to_k"
    weights = {
        f"{prefix}.lora_A": mx.ones((2, 32)),
        f"{prefix}.lora_B": mx.zeros((16, 2)),
        f"{prefix}.lora_monteclora_sampler.style.std_prior": mx.ones(2),
    }
    selected = _peft_file(tmp_path, weights, {"peft_type": "LORA", "r": 2, "monteclora_config": {}})
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert receipt.complete_targets == receipt.placed_targets == 1
    (unused,) = (item for item in receipt.placements if item.kind == "training_only")
    assert unused.status == "unused_inference"
    assert sorted(key for item in receipt.placements for key in item.source_keys) == sorted(weights)


@pytest.mark.parametrize(
    "prefix",
    [
        "transformer_blocks.0.attn.to_k",
        "transformer.transformer_blocks.0.attn.to_k",
        "base_model.model.transformer_blocks.0.attn.to_k",
        "diffusion_model.blocks.0.attn.wk",
        "model.diffusion_model.blocks.0.attn.wk",
        "base_model.model.diffusion_model.blocks.0.attn.wk",
        "lora_unet_blocks_0_attn_wk",
        "lora_transformer_transformer_blocks_0_attn_to_k",
        "lycoris_blocks_0_attn_wk",
    ],
)
def test_key_dialects_load_real_safetensors(tmp_path, prefix) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    selected = _adapter(tmp_path / "adapter.safetensors", prefix)
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert receipt.placed_targets == 1
    assert receipt.coverage == 1.0
    assert set(receipt.placements[0].source_keys) == {
        f"{prefix}.lora_A.weight",
        f"{prefix}.lora_B.weight",
    }
    assert mx.allclose(model.transformer_blocks[0].attn.to_k.weight, original + 0.04).item()


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_feed_forward_updates_baked_or_reference_layout(tmp_path, reference, dtype) -> None:
    model = _model(reference=reference, dtype=dtype)
    original = model.transformer_blocks[0].ff.down.weight
    selected = _adapter(
        tmp_path / "adapter.safetensors", "diffusion_model.blocks.0.mlp.down", out=32, inputs=48
    )
    _apply_krea2_adapters(model, (selected,), variant="raw")
    updated = model.transformer_blocks[0].ff.down.weight
    assert updated.shape == ((32, 48) if reference else (48, 32))
    assert updated.dtype == dtype
    assert mx.allclose(
        updated, (original.astype(mx.float32) + 0.04).astype(dtype), atol=1e-8, rtol=1e-6
    ).item()


def test_all_official_source_modules_have_canonical_destinations() -> None:
    keys = Krea2AdapterKeys(tiny_transformer_config())
    assert keys.resolve("diffusion_model.first", "bias").target_key == "img_in.bias"
    assert (
        keys.resolve("diffusion_model.last.linear", "weight").target_key
        == "final_layer.linear.weight"
    )
    assert keys.resolve("diffusion_model.txtmlp.0", "weight").target_key == "txt_in.norm.weight"
    assert (
        keys.resolve("diffusion_model.txtfusion.layerwise_blocks.0.attn.wk", "weight").target_key
        == "text_fusion.layerwise_blocks.0.attn.to_k.weight"
    )
    assert (
        keys.resolve("diffusion_model.blocks.0.mod.lin", "weight").target_key
        == "transformer_blocks.0.scale_shift_table"
    )
    assert keys.resolve("diffusion_model.blocks.0.mod.lin", "weight").reshape_rows == 6
    assert (
        keys.resolve("transformer_blocks.0.attn.to_out", "weight").target_key
        == "transformer_blocks.0.attn.to_out.0.weight"
    )
    assert keys.resolve("diffusion_model.blocks.0.attn.wq", "bias") is None


def test_native_modulation_dense_delta_uses_canonical_rows(tmp_path) -> None:
    model = _model()
    original = model.transformer_blocks[0].scale_shift_table
    path = tmp_path / "adapter.safetensors"
    save_weights(
        path,
        {"diffusion_model.blocks.0.mod.lin.diff": mx.arange(original.size, dtype=mx.float32)},
    )
    (receipt,) = _apply_krea2_adapters(model, (AdapterSelection(path, 0.5),), variant="turbo")
    expected = original + mx.arange(original.size, dtype=mx.float32).reshape(original.shape) * 0.5
    assert receipt.placements[0].target == "transformer_blocks.0.scale_shift_table"
    assert mx.allclose(model.transformer_blocks[0].scale_shift_table, expected).item()


@pytest.mark.parametrize(
    ("prefix", "target_name"),
    [
        (
            "transformer.text_fusion.layerwise_blocks.0.attn.to_q",
            "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        ),
        (
            "lora_unet_txtfusion_layerwise_blocks_0_attn_wq",
            "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        ),
        ("lora_unet_first", "img_in.weight"),
        ("lora_unet_txtmlp_1", "txt_in.linear_1.weight"),
        ("lora_unet_txtmlp_3", "txt_in.linear_2.weight"),
        ("lora_unet_last_linear", "final_layer.linear.weight"),
    ],
)
def test_all_linear_exports_cover_text_fusion_and_outer_projections(tmp_path, prefix, target_name):
    model = _model()
    target = transformer_weight_targets(model)[target_name]
    original = getattr(target.owner, target.attribute)
    path = tmp_path / "adapter.safetensors"
    save_weights(
        path,
        {
            f"{prefix}.lora_down.weight": mx.full((2, target.shape[1]), 0.1),
            f"{prefix}.lora_up.weight": mx.full((target.shape[0], 2), 0.2),
            f"{prefix}.alpha": mx.array(2.0),
        },
    )
    (receipt,) = _apply_krea2_adapters(model, (AdapterSelection(path, 1),), variant="turbo")
    assert receipt.coverage == 1.0
    assert receipt.placements[0].target == target_name
    assert mx.allclose(getattr(target.owner, target.attribute), original + 0.04).item()


def test_bias_delta_and_variant_facts_are_not_discarded(tmp_path) -> None:
    model = _model()
    original = model.img_in.bias
    path = tmp_path / "adapter.safetensors"
    save_weights(
        path,
        {"diffusion_model.first.diff_b": mx.full((32,), 0.5)},
        {"ss_base_model_version": "krea-2-raw"},
    )
    (receipt,) = _apply_krea2_adapters(model, (AdapterSelection(path, 0.5),), variant="turbo")
    assert receipt.to_metadata()["variant_pairing"] == "recommended-raw-to-turbo"
    assert receipt.declared_variant == "raw"
    assert mx.allclose(model.img_in.bias, original + 0.25).item()


@pytest.mark.parametrize("status", ["unmapped", "excluded", "zero"])
def test_zero_effect_is_diagnostic_and_receipt_is_complete(tmp_path, caplog, status) -> None:
    caplog.set_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading")
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    prefix = "unrelated.layer" if status == "unmapped" else "diffusion_model.blocks.0.attn.wk"
    selected = _adapter(tmp_path / "adapter.safetensors", prefix)
    selected = replace(
        selected,
        strength=0.0 if status == "zero" else 1.0,
        exclude=("attn",) if status == "excluded" else (),
    )
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert "0% effective placement" in caplog.text
    assert "Types: lora" in caplog.text
    assert receipt.effective is False
    assert receipt.adapter_types == ("lora",)
    assert receipt.to_metadata()["adapter_types"] == ["lora"]
    if status == "unmapped":
        assert receipt.warnings == ("1 parsed adapter operations did not map to Krea 2 targets",)
    assert mx.array_equal(model.transformer_blocks[0].attn.to_k.weight, original).item()
    assert sum(len(item.source_keys) for item in receipt.placements) == 2


def test_zero_strength_reuses_structural_receipt_without_reloading_values(
    tmp_path,
    monkeypatch,
) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    selected = replace(
        _adapter(tmp_path / "adapter.safetensors", "diffusion_model.blocks.0.attn.wk"),
        strength=0.0,
    )
    plan = prepare_krea2_adapters(model.config, (selected,))

    def unexpected_value_load(_path):
        pytest.fail("a zero-strength adapter must not reload factor values during application")

    monkeypatch.setattr(adapter_loading, "load_weights", unexpected_value_load)
    (receipt,) = apply_krea2_adapters(model, plan, variant="turbo")
    assert receipt.effective is False
    assert receipt.coverage == 1.0
    assert model.transformer_blocks[0].attn.to_k.weight is original


def test_zero_strength_still_rejects_nonfinite_adapter_values(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.safetensors"
    prefix = "diffusion_model.blocks.0.attn.wk"
    save_weights(
        path,
        {
            f"{prefix}.lora_A.weight": mx.full((2, 32), float("nan")),
            f"{prefix}.lora_B.weight": mx.ones((16, 2)),
        },
    )

    with pytest.raises(ValueError, match="nonfinite"):
        prepare_krea2_adapters(
            tiny_transformer_config(),
            (AdapterSelection(path, strength=0.0),),
        )


def test_adapter_inspection_errors_name_path_and_reason(tmp_path: Path) -> None:
    missing = tmp_path / "missing.safetensors"
    with pytest.raises(FileNotFoundError, match=r"Krea 2 adapter does not exist: .*missing"):
        prepare_krea2_adapters(
            tiny_transformer_config(),
            (AdapterSelection(missing, 1.0),),
        )

    invalid = tmp_path / "notes.txt"
    invalid.write_text("not safetensors")
    with pytest.raises(ValueError, match=r"invalid Krea 2 adapter .*notes\.txt: .*safetensors"):
        prepare_krea2_adapters(
            tiny_transformer_config(),
            (AdapterSelection(invalid, 1.0),),
        )


def test_adapter_fusion_log_explains_deferred_arithmetic(tmp_path: Path, caplog) -> None:
    model = _model()
    selected = _adapter(
        tmp_path / "adapter.safetensors",
        "diffusion_model.blocks.0.attn.wk",
    )

    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        _apply_krea2_adapters(model, (selected,), variant="turbo")

    assert any(
        "fusion arithmetic is deferred to the first denoising step" in record.getMessage()
        for record in caplog.records
    )


def test_adapter_change_after_structural_preflight_is_rejected(tmp_path) -> None:
    model = _model()
    path = tmp_path / "adapter.safetensors"
    selected = _adapter(path, "diffusion_model.blocks.0.attn.wk")
    plan = prepare_krea2_adapters(model.config, (selected,))
    _adapter(path, "diffusion_model.blocks.0.attn.wk", metadata={"changed": "yes"})

    with pytest.raises(ValueError, match="changed after structural preflight"):
        apply_krea2_adapters(model, plan, variant="turbo")


def test_alias_collision_does_not_change_any_weight(tmp_path) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    arrays = {}
    for prefix in ("diffusion_model.blocks.0.attn.wk", "transformer_blocks.0.attn.to_k"):
        arrays[f"{prefix}.lora_down.weight"] = mx.ones((2, 32))
        arrays[f"{prefix}.lora_up.weight"] = mx.ones((16, 2))
    path = tmp_path / "collision.safetensors"
    save_weights(path, arrays)
    with pytest.raises(ValueError, match="multiple operations"):
        _apply_krea2_adapters(model, (AdapterSelection(path, 1.0),), variant="turbo")
    assert model.transformer_blocks[0].attn.to_k.weight is original


def test_malformed_second_adapter_leaves_model_unchanged(tmp_path) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    first = _adapter(tmp_path / "first.safetensors", "diffusion_model.blocks.0.attn.wk")
    second = _adapter(tmp_path / "second.safetensors", "diffusion_model.blocks.1.attn.wq", out=31)
    with pytest.raises(ValueError, match="shape mismatch"):
        _apply_krea2_adapters(model, (first, second), variant="turbo")
    assert model.transformer_blocks[0].attn.to_k.weight is original


def test_fresh_model_has_no_adapter_residue(tmp_path) -> None:
    first, second = _model(), _model()
    original = second.transformer_blocks[0].attn.to_k.weight
    selected = _adapter(tmp_path / "adapter.safetensors", "diffusion_model.blocks.0.attn.wk")
    _apply_krea2_adapters(first, (selected,), variant="turbo")
    assert mx.allclose(first.transformer_blocks[0].attn.to_k.weight, original + 0.04).item()
    assert second.transformer_blocks[0].attn.to_k.weight is original


@pytest.mark.parametrize("named", [False, True])
def test_peft_saved_module_replaces_weights_and_bias_in_a_fresh_lease(tmp_path, named):
    model = _model()
    untouched = model.transformer_blocks[0].attn.to_k.weight
    wrapper = ".modules_to_save.style" if named else ""
    path = tmp_path / "adapter_model.safetensors"
    original_weight, original_bias = model.img_in.weight, model.img_in.bias
    save_weights(
        path,
        {
            f"base_model.model.img_in{wrapper}.weight": mx.full(original_weight.shape, 0.3),
            f"base_model.model.img_in{wrapper}.bias": mx.full(original_bias.shape, 0.7),
        },
    )
    path.with_name("adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "modules_to_save": ["img_in"]})
    )
    (receipt,) = _apply_krea2_adapters(model, (AdapterSelection(path, 1),), variant="turbo")
    assert receipt.coverage == 1.0
    assert receipt.placed_targets == 2
    assert mx.allclose(model.img_in.weight, mx.full(original_weight.shape, 0.3)).item()
    assert mx.allclose(model.img_in.bias, mx.full(original_bias.shape, 0.7)).item()
    assert model.transformer_blocks[0].attn.to_k.weight is untouched


def test_peft_companion_alpha_and_rank_stabilization(tmp_path) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    selected = _adapter(
        tmp_path / "adapter_model.safetensors", "base_model.model.transformer_blocks.0.attn.to_k"
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 8,
                "use_rslora": True,
                "base_model_name_or_path": "krea/Krea-2-Raw",
            }
        )
    )
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    expected = original + 0.04 * 8 / (2**0.5)
    assert mx.allclose(model.transformer_blocks[0].attn.to_k.weight, expected).item()
    assert receipt.variant_pairing == "recommended-raw-to-turbo"


@pytest.mark.parametrize("component_prefix", ["", "transformer."])
def test_embedded_diffusers_export_uses_peft_alpha_over_rank(tmp_path, component_prefix):
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 8,
        "base_model_name_or_path": "krea/Krea-2-Raw",
    }
    selected = _adapter(
        tmp_path / "pytorch_lora_weights.safetensors",
        "transformer.transformer_blocks.0.attn.to_k",
        metadata={
            "lora_adapter_metadata": json.dumps(
                {component_prefix + key: value for key, value in config.items()}
            )
        },
    )
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert mx.allclose(model.transformer_blocks[0].attn.to_k.weight, original + 0.04 * 8 / 2).item()
    assert receipt.variant_pairing == "recommended-raw-to-turbo"


def test_peft_export_without_configuration_warns_and_applies_unscaled(tmp_path, caplog) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    selected = _adapter(
        tmp_path / "adapter_model.safetensors", "base_model.model.transformer_blocks.0.attn.to_k"
    )
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert mx.allclose(model.transformer_blocks[0].attn.to_k.weight, original + 0.04).item()
    assert receipt.warnings == (UNKNOWN_SCALE_WARNING,)
    assert receipt.scale_source is None
    assert f"adapter_model.safetensors: {UNKNOWN_SCALE_WARNING}" in caplog.text
    assert not [message for message in caplog.messages if ": scale " in message]


def test_snapshot_linked_peft_export_applies_its_configured_scale(tmp_path, caplog) -> None:
    model = _model()
    original = model.transformer_blocks[0].attn.to_k.weight
    export = tmp_path / "export"
    export.mkdir()
    _adapter(
        export / "adapter_model.safetensors", "base_model.model.transformer_blocks.0.attn.to_k"
    )
    (export / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 8})
    )
    weights, _config = link_as_snapshot(
        tmp_path / "hub",
        export / "adapter_model.safetensors",
        export / "adapter_config.json",
        folder="lora",
    )
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        (receipt,) = _apply_krea2_adapters(
            model, (AdapterSelection(weights, 0.5),), variant="turbo"
        )
    assert mx.allclose(
        model.transformer_blocks[0].attn.to_k.weight, original + 0.5 * 0.04 * 4
    ).item()
    assert receipt.scale_source == "adapter_config.json"
    assert receipt.intrinsic_scales == (4.0,)
    assert receipt.to_metadata()["intrinsic_scales"] == [4.0]
    assert receipt.warnings == ()
    assert (
        "Krea 2 adapter adapter_model.safetensors: scale 4 from adapter_config.json "
        "(lora_alpha=8, r=2); effective 2 at strength 0.5" in caplog.messages
    )


@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("strength", [-0.4, 0.0, 1.0])
def test_peft_ia3_applies_output_bias_but_not_input_bias(tmp_path, on_input, strength):
    model = _model()
    weight, bias = model.img_in.weight, model.img_in.bias
    shape = (1, weight.shape[1]) if on_input else (weight.shape[0], 1)
    vector = mx.arange(shape[0] * shape[1], dtype=mx.float32).reshape(shape) / 71 + 0.75
    path = tmp_path / "adapter_model.safetensors"
    save_weights(path, {"base_model.model.img_in.ia3_l": vector})
    path.with_name("adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "IA3",
                "feedforward_modules": ["img_in"] if on_input else [],
            }
        )
    )
    (receipt,) = _apply_krea2_adapters(model, (AdapterSelection(path, strength),), variant="turbo")
    assert mx.allclose(
        model.img_in.weight, weight * (1 + (vector - 1) * strength), atol=1e-7
    ).item()
    expected_bias = bias if on_input else bias * (1 + (vector.reshape(-1) - 1) * strength)
    assert mx.allclose(model.img_in.bias, expected_bias, atol=1e-7).item()
    assert receipt.placed_targets == (1 if on_input else 2)
    assert sum(len(item.source_keys) for item in receipt.placements) == 1
    if not on_input:
        assert next(
            item for item in receipt.placements if item.target == "img_in.bias"
        ).derived_from


def test_krea_cli_and_request_use_shared_record_conventions() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "A geometric still life",
            "--adapter",
            "one.safetensors",
            "--lora",
            "two.safetensors",
            "--adapter-strength",
            "0.7",
            "--adapter-exclude",
            "text_fusion",
        ]
    )
    request = Krea2Request(prompt=args.prompt, adapters=tuple(args.adapters))
    selections = request.resolved_adapters()
    assert [item.strength for item in selections] == [1.0, 0.7]
    assert [item.exclude for item in selections] == [(), ("text_fusion",)]
    alias_args = build_parser().parse_args(
        [
            "--prompt",
            "A geometric still life",
            "--lora",
            "one.safetensors",
            "--lora-strength",
            "0.7",
            "--lora-exclude",
            "text_fusion",
        ]
    )
    assert alias_args.adapters == [
        {"path": Path("one.safetensors"), "strength": 0.7, "exclude": ("text_fusion",)}
    ]
    with pytest.raises(TypeError, match="adapter_strengths"):
        Krea2Request(prompt="A geometric still life", adapter_strengths=(1.0,))
    with pytest.raises(ValueError, match="exclude"):
        Krea2Request(
            prompt="A geometric still life",
            adapters=({"path": "one.safetensors", "exclude": ["audio"]},),
        )


def test_adapter_log_preserves_readable_snapshot_symlink(tmp_path: Path, caplog) -> None:
    model = _model()
    blob = tmp_path / ("a" * 64)
    _adapter(blob, "transformer_blocks.0.attn.to_k")
    readable = tmp_path / "identity-edit.safetensors"
    readable.symlink_to(blob)
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        (receipt,) = _apply_krea2_adapters(
            model, (AdapterSelection(readable, 1.0),), variant="turbo"
        )
    assert "Adapter result 1/1: identity-edit.safetensors" in caplog.text
    assert f"Path: {readable}" in caplog.text
    assert f"File: {blob.name}" not in caplog.text
    assert receipt.path == blob


@pytest.mark.parametrize("edit_exclusions", [(), ("to_k",)])
def test_two_loras_apply_independent_strengths_without_averaging(
    tmp_path: Path, caplog, edit_exclusions
) -> None:
    model = _model()
    first = replace(
        _adapter(tmp_path / "first.safetensors", "transformer_blocks.0.attn.to_k"),
        strength=0.7,
        exclude=edit_exclusions,
    )
    second = replace(
        _adapter(tmp_path / "second.safetensors", "transformer_blocks.0.attn.to_k"), strength=0.2
    )
    original = model.transformer_blocks[0].attn.to_k.weight
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        receipts = _apply_krea2_adapters(model, (first, second), variant="turbo")
    delta = mx.full((16, 2), 0.2) @ mx.full((2, 32), 0.1)
    expected = original + (0 if edit_exclusions else 0.7) * delta + 0.2 * delta
    assert mx.allclose(model.transformer_blocks[0].attn.to_k.weight, expected, atol=1e-7).item()
    assert [receipt.strength for receipt in receipts] == [0.7, 0.2]
    messages = [record.getMessage() for record in caplog.records]
    for index, selection in enumerate((first, second), 1):
        heading = f"Adapter result {index}/2: {selection.path.name}"
        offset = messages.index(heading)
        assert "Model: Krea 2 Turbo | Types:" in messages[offset + 1]
        assert f"Strength: {selection.strength:g}" in messages[offset + 1]
        assert messages[offset + 2] == f"  Path: {selection.path}"


def test_scheduled_adapter_leaves_its_target_weight_unchanged(tmp_path):
    model = _model()
    selected = replace(
        _adapter(tmp_path / "gate.safetensors", "transformer_blocks.0.attn.to_k"),
        schedule=AdapterSchedule((1.0, 1.0, 0.0, 0.0)),
    )
    original = mx.array(model.transformer_blocks[0].attn.to_k.weight)
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")

    assert mx.array_equal(model.transformer_blocks[0].attn.to_k.weight, original).item()
    assert receipt.effective
    assert receipt.step_strengths == (1.0, 1.0, 0.0, 0.0)
    assert receipt.to_metadata()["application"] == "live"
    controller = live_adapters(model)
    assert controller is not None
    assert [term.target for term in controller.terms] == ["transformer_blocks.0.attn.to_k.weight"]


def test_fused_and_scheduled_adapters_share_one_target(tmp_path):
    model = _model()
    fused = replace(
        _adapter(tmp_path / "fused.safetensors", "transformer_blocks.0.attn.to_k"), strength=0.5
    )
    gated = replace(
        _adapter(tmp_path / "gated.safetensors", "transformer_blocks.0.attn.to_k"),
        schedule=AdapterSchedule((1.0, 0.0)),
    )
    original = mx.array(model.transformer_blocks[0].attn.to_k.weight)
    receipts = _apply_krea2_adapters(model, (fused, gated), variant="turbo")

    delta = mx.full((16, 2), 0.2) @ mx.full((2, 32), 0.1)
    # Only the fused adapter reaches the weight; the scheduled one rides on activations.
    assert mx.allclose(
        model.transformer_blocks[0].attn.to_k.weight, original + 0.5 * delta, atol=1e-7
    ).item()
    assert [receipt.step_strengths for receipt in receipts] == [(), (1.0, 0.0)]
    assert len(live_adapters(model).terms) == 1


def test_a_constant_step_array_fuses_exactly_like_one_strength(tmp_path):
    # "1,1,1,1" is the obvious spelling of "every step" once the flag is known,
    # and it asks for nothing a constant cannot express.
    scheduled_model, plain_model = _model(), _model()
    constant = replace(
        _adapter(tmp_path / "constant.safetensors", "transformer_blocks.0.attn.to_k"),
        schedule=AdapterSchedule((0.7, 0.7, 0.7, 0.7)),
    )
    plain = replace(
        _adapter(tmp_path / "plain.safetensors", "transformer_blocks.0.attn.to_k"), strength=0.7
    )
    (receipt,) = _apply_krea2_adapters(scheduled_model, (constant,), variant="turbo")
    _apply_krea2_adapters(plain_model, (plain,), variant="turbo")

    assert live_adapters(scheduled_model) is None
    assert mx.array_equal(
        scheduled_model.transformer_blocks[0].attn.to_k.weight,
        plain_model.transformer_blocks[0].attn.to_k.weight,
    ).item()
    # The array carried the magnitude; the selection's own strength is still 1.
    assert constant.strength == 1.0
    assert receipt.step_strengths == (0.7, 0.7, 0.7, 0.7)
    assert receipt.to_metadata()["application"] == "fused"


def test_a_fully_zero_schedule_is_recorded_as_ineffective(tmp_path):
    model = _model()
    selected = replace(
        _adapter(tmp_path / "off.safetensors", "transformer_blocks.0.attn.to_k"),
        schedule=AdapterSchedule((0.0, 0.0)),
    )
    original = mx.array(model.transformer_blocks[0].attn.to_k.weight)
    (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")

    assert not receipt.effective
    assert mx.array_equal(model.transformer_blocks[0].attn.to_k.weight, original).item()
    assert live_adapters(model) is None


def test_a_window_owning_its_band_strengths_is_placed_under_a_zero_fallback(tmp_path, caplog):
    model = _model()
    selected = replace(
        _adapter(tmp_path / "banded.safetensors", "transformer_blocks.0.attn.to_k"),
        strength=0.0,
        sigma_window=parse_sigma_window("0.93@1,0.6:0.93@0.4"),
    )
    original = mx.array(model.transformer_blocks[0].attn.to_k.weight)
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        (receipt,) = _apply_krea2_adapters(model, (selected,), variant="turbo")

    assert receipt.effective
    # A window rides on activations, so the weight itself must not move.
    assert mx.array_equal(model.transformer_blocks[0].attn.to_k.weight, original).item()
    assert receipt.to_metadata()["sigma_window"] == {
        "bands": [
            {"low": 0.93, "high": 1.0, "strength": 1.0},
            {"low": 0.6, "high": 0.93, "strength": 0.4},
        ]
    }
    assert live_adapters(model) is not None
    messages = [record.getMessage() for record in caplog.records]
    assert any("Strength: 1 " in message for message in messages)
    assert any(
        "Sigma Window: 0.93 to 1 at 1; 0.6 to 0.93 at 0.4" in message for message in messages
    )


def test_scheduled_receipt_logs_its_step_strengths(tmp_path, caplog):
    model = _model()
    selected = replace(
        _adapter(tmp_path / "gate.safetensors", "transformer_blocks.0.attn.to_k"),
        schedule=AdapterSchedule((1.0, 0.25, 0.0)),
    )
    with caplog.at_level(logging.INFO, logger="kinomlx.models.krea2.adapters.loading"):
        _apply_krea2_adapters(model, (selected,), variant="turbo")
    assert any("Step Strengths: 1, 0.25, 0" in record.getMessage() for record in caplog.records)


def test_a_format_without_an_activation_form_refuses_a_schedule(tmp_path):
    model = _model()
    prefix = "transformer_blocks.0.attn.to_k"
    path = tmp_path / "dora.safetensors"
    save_weights(
        path,
        {
            f"{prefix}.lora_A.weight": mx.full((2, 32), 0.1),
            f"{prefix}.lora_B.weight": mx.full((16, 2), 0.2),
            f"{prefix}.dora_scale": mx.full((16, 1), 1.0),
        },
    )
    selected = AdapterSelection(path, 1.0, schedule=AdapterSchedule((1.0, 0.0)))
    with pytest.raises(ValueError, match="no activation form"):
        _apply_krea2_adapters(model, (selected,), variant="turbo")


def test_denoise_advances_the_schedule_once_per_step(tmp_path):
    model = _model()
    selected = replace(
        _adapter(tmp_path / "gate.safetensors", "transformer_blocks.0.attn.to_k"),
        schedule=AdapterSchedule((1.0, 0.5, 0.0, 0.0)),
    )
    _apply_krea2_adapters(model, (selected,), variant="turbo")
    (term,) = live_adapters(model).terms

    observed = []

    class RecordingPort:
        live_adapters = live_adapters(model)

        def __call__(self, tokens, **_kwargs):
            observed.append(term.strength)
            return SimpleNamespace(sample=mx.zeros_like(tokens))

    recipe = Krea2Recipe(
        variant=Krea2Variant.TURBO,
        steps=4,
        guidance=0.0,
        mu=1.15,
        effective_mu=1.15,
        cfg_enabled=False,
        matches_variant_defaults=False,
    )
    branch = Krea2PreparedBranch(
        image_tokens=mx.zeros((1, 2, 4)),
        encoder_hidden_states=mx.zeros((1, 1, 8)),
        position_ids=mx.zeros((1, 3, 3)),
        attention_mask=None,
    )
    denoise_krea2(RecordingPort(), branch, recipe)
    assert observed == [1.0, 0.5, 0.0, 0.0]
