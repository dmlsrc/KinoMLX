"""LTX LoRA stage-strength and target-knockout contracts."""

import json
import logging
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.adapters.fusion import fuse_weight
from kinomlx.adapters.lora import LoRAConfig
from kinomlx.adapters.scaling import UNKNOWN_SCALE_WARNING
from kinomlx.models.ltx2.adapters import loading as lora_module
from kinomlx.models.ltx2.adapters.keys import lora_key_categories
from kinomlx.models.ltx2.adapters.loading import (
    community_adapter_operations,
    fuse_community_loras_into_model,
)
from kinomlx.models.ltx2.transformer import LTXAVModel
from kinomlx.models.ltx2.weights.layout import install_layout_weight
from tests.adapters._snapshot import link_as_snapshot

from ..transformer._synthetic import build_shaped_ltx_model

_LOADING_LOGGER = "kinomlx.models.ltx2.adapters.loading"
_PEFT_TO_K = "base_model.model.transformer_blocks.0.attn1.to_k"


def _live_model(dtype: mx.Dtype) -> LTXAVModel:
    return build_shaped_ltx_model(
        lambda: LTXAVModel(
            num_layers=1,
            video_heads=1,
            video_head_dim=8,
            audio_heads=1,
            audio_head_dim=4,
            video_context_dim=8,
            audio_context_dim=4,
            compute_dtype=dtype,
        )
    )


def _save_adapter(path: Path, *, video_scale: float, audio_scale: float) -> None:
    mx.save_safetensors(
        str(path),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": (
                mx.ones((1, 2), dtype=mx.float32)
            ),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": (
                mx.full((2, 1), video_scale, dtype=mx.float32)
            ),
            "diffusion_model.transformer_blocks.0.audio_ff.net.2.lora_A.weight": (
                mx.ones((1, 2), dtype=mx.float32)
            ),
            "diffusion_model.transformer_blocks.0.audio_ff.net.2.lora_B.weight": (
                mx.full((2, 1), audio_scale, dtype=mx.float32)
            ),
        },
    )


def test_lora_categories_cover_branch_module_projection_and_control() -> None:
    assert {
        "cross",
        "attn",
        "audio_to_video_attn",
        "to_v",
        "cross_control",
        "distill_control",
    } <= lora_key_categories("transformer_blocks.0.audio_to_video_attn.to_v.weight")
    assert {
        "video",
        "adaln",
        "prompt_adaln",
        "prompt_scale_shift",
        "cross_control",
        "distill_control",
    } <= lora_key_categories("prompt_adaln_single.linear.weight")


def test_live_model_lora_rebinds_pretransposed_slots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _live_model(mx.float32)
    ff = model.transformer_blocks[0].ff
    ff.project_out.weight = mx.zeros_like(ff.project_out.weight)
    attention = model.transformer_blocks[0].attn1
    attention.to_q.weight = mx.zeros_like(attention.to_q.weight)
    attention.to_k.weight = mx.zeros_like(attention.to_k.weight)
    install_layout_weight(
        model,
        "transformer_blocks.0.ff.project_out.weight_t",
        mx.contiguous(ff.project_out.weight.T),
    )
    adapter = tmp_path / "adapter.safetensors"
    install_layout_weight(
        model, "transformer_blocks.0.attn1.to_q.weight_t", mx.contiguous(attention.to_q.weight.T)
    )
    adaln = model.adaln_single
    adaln.linear.weight = mx.zeros_like(adaln.linear.weight)
    adaln_shape = tuple(adaln.linear.weight.shape)
    install_layout_weight(
        model, "adaln_single.linear.weight_t", mx.contiguous(adaln.linear.weight.T)
    )
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.ones((1, 32)),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.ones((8, 1)),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones((1, 8)),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones((8, 1)),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": mx.ones((1, 8)),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": mx.ones((8, 1)),
            "diffusion_model.adaln_single.linear.lora_A.weight": mx.ones((1, adaln_shape[1])),
            "diffusion_model.adaln_single.linear.lora_B.weight": mx.ones((adaln_shape[0], 1)),
        },
    )
    monkeypatch.setattr(
        LTXAVModel,
        "parameters",
        lambda self: pytest.fail("live LoRA fusion must not flatten the model"),
    )
    fuse_community_loras_into_model(
        model,
        [LoRAConfig(adapter, strength=0.5)],
    )

    assert ff._project_out_weight_t is not None
    assert mx.array_equal(ff._project_out_weight_t, mx.full((32, 8), 0.5)).item()
    assert "weight" not in ff.project_out
    assert attention._to_q_weight_t is not None
    assert mx.array_equal(attention._to_q_weight_t, mx.full((8, 8), 0.5)).item()
    assert "weight" not in attention.to_q
    assert mx.array_equal(attention.to_k.weight, mx.full((8, 8), 0.5)).item()
    assert adaln._linear_weight_t is not None
    assert mx.array_equal(
        adaln._linear_weight_t,
        mx.full((adaln_shape[1], adaln_shape[0]), 0.5),
    ).item()
    assert "weight" not in adaln.linear


def test_live_model_lora_skips_non_weight_leaves_and_fuses_placeable_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _live_model(mx.float16)
    attention = model.transformer_blocks[0].attn1
    attention.to_k.weight = mx.zeros_like(attention.to_k.weight)
    original_resolve = lora_module._resolve_live_module
    incompatible = {
        "transformer_blocks.0.ff.project_out",
        "transformer_blocks.0.attn1.to_q",
    }

    def resolve_with_non_weight_leaves(target: object, path: str) -> object | None:
        if path in incompatible:
            return object()
        return original_resolve(target, path)

    monkeypatch.setattr(lora_module, "_resolve_live_module", resolve_with_non_weight_leaves)
    adapter = tmp_path / "partial-live.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.ones((1, 32)),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.ones((8, 1)),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones((1, 8)),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones((8, 1)),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": mx.ones((1, 8)),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": mx.ones((8, 1)),
        },
    )

    (receipt,) = fuse_community_loras_into_model(
        model,
        [LoRAConfig(adapter)],
    )

    assert mx.array_equal(attention.to_k.weight, mx.ones_like(attention.to_k.weight)).item()
    assert receipt.complete_targets == 3
    assert receipt.placed_targets == 1
    assert receipt.structural_coverage == pytest.approx(1 / 3)
    assert receipt.skipped_reasons == (("missing_target", 2),)
    assert receipt.effective is True
    assert receipt.warnings == (
        "2 adapter targets are absent from this LTX-2 model",
        "low target coverage (33.3%)",
    )


@pytest.mark.parametrize("kind", ["lokr", "loha"])
def test_live_ltx_shared_fuser_still_rejects_unvalidated_formats(tmp_path, kind):
    model = _live_model(mx.float16)
    path = tmp_path / "unsupported.safetensors"
    prefix = "diffusion_model.transformer_blocks.0.ff.net.2"
    factors = (
        {"lokr_w1": mx.ones((2, 2)), "lokr_w2": mx.ones((4, 16))}
        if kind == "lokr"
        else {
            "hada_w1_a": mx.ones((8, 1)),
            "hada_w1_b": mx.ones((1, 32)),
            "hada_w2_a": mx.ones((8, 1)),
            "hada_w2_b": mx.ones((1, 32)),
        }
    )
    mx.save_safetensors(str(path), {f"{prefix}.{key}": value for key, value in factors.items()})
    with pytest.raises(ValueError, match="ordinary LoRA weight adapters only"):
        fuse_community_loras_into_model(model, [LoRAConfig(path)])


def test_live_fp16_fusion_is_lazy_without_weights_and_checks_after_use(tmp_path, monkeypatch):
    from kinomlx.adapters.validation import AdapterWeightError, pending_weight_check
    from kinomlx.reporting import NullReporter

    model = _live_model(mx.float16)
    ff = model.transformer_blocks[0].ff
    path = tmp_path / "overflow.safetensors"
    prefix = "diffusion_model.transformer_blocks.0.ff.net.2"
    mx.save_safetensors(
        str(path),
        {
            f"{prefix}.lora_A.weight": mx.full((1, 32), 100000.0),
            f"{prefix}.lora_B.weight": mx.ones((8, 1)),
        },
    )
    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", lambda *args: pytest.fail("fusion evaluated weights"))
        fuse_community_loras_into_model(model, [LoRAConfig(path)])
    mx.eval(ff.project_out.weight)
    pending = pending_weight_check(model)
    assert pending is not None
    with pytest.raises(AdapterWeightError, match="ff.project_out.weight"):
        pending.run(context="LTX-2 step 1", reporter=NullReporter())


def test_prepared_targets_count_in_receipts_but_are_not_fused_or_checked_again(
    tmp_path: Path,
) -> None:
    from kinomlx.adapters.validation import pending_weight_check

    model = _live_model(mx.bfloat16)
    ff = model.transformer_blocks[0].ff
    audio_ff = model.transformer_blocks[0].audio_ff
    before = mx.array(ff.project_out.weight)
    audio_before = mx.array(audio_ff.project_out.weight)
    path = tmp_path / "adapter.safetensors"
    video_out, video_in = ff.project_out.weight.shape
    audio_out, audio_in = audio_ff.project_out.weight.shape
    prefix = "diffusion_model.transformer_blocks.0"
    mx.save_safetensors(
        str(path),
        {
            f"{prefix}.ff.net.2.lora_A.weight": mx.ones((1, video_in), dtype=mx.float32),
            f"{prefix}.ff.net.2.lora_B.weight": mx.full((video_out, 1), 2.0, dtype=mx.float32),
            f"{prefix}.audio_ff.net.2.lora_A.weight": mx.ones((1, audio_in), dtype=mx.float32),
            f"{prefix}.audio_ff.net.2.lora_B.weight": mx.full(
                (audio_out, 1), 3.0, dtype=mx.float32
            ),
        },
    )

    receipts = fuse_community_loras_into_model(
        model,
        [LoRAConfig(path)],
        prepared_targets={
            "transformer_blocks.0.ff.project_out": tuple(ff.project_out.weight.shape)
        },
    )

    assert receipts[0].placed_targets == 2
    assert receipts[0].effective
    # The prepared video target keeps its packed-route weight; audio fuses live.
    assert mx.array_equal(ff.project_out.weight, before).item()
    assert not mx.array_equal(audio_ff.project_out.weight, audio_before).item()
    pending = pending_weight_check(model)
    assert pending is not None
    assert set(pending.targets) == {"transformer_blocks.0.audio_ff.project_out.weight"}


def test_pre_pack_adapter_operations_match_the_live_fuser(tmp_path: Path) -> None:
    from kinomlx.adapters.fusion import fuse_weight
    from kinomlx.models.ltx2.adapters.loading import community_adapter_operations

    path = tmp_path / "adapter.safetensors"
    _save_adapter(path, video_scale=2.0, audio_scale=3.0)
    base = "transformer_blocks.0.ff.project_out"
    operations = community_adapter_operations(
        [LoRAConfig(path, strength=0.5)], target_shapes={base: (2, 2)}
    )
    assert set(operations) == {f"{base}.weight"}
    weight = mx.arange(4, dtype=mx.float32).reshape(2, 2).astype(mx.bfloat16)

    fused = fuse_weight(weight, operations[f"{base}.weight"])
    assert fused.dtype == mx.bfloat16
    # strength 0.5 * B (2.0) @ A (1.0) over rank 1 adds 1.0 to every element.
    assert mx.array_equal(fused, weight + 1).item()
    assert community_adapter_operations([LoRAConfig(path)], target_shapes={}) == {}
    assert community_adapter_operations([], target_shapes={base: (2, 2)}) == {}


def test_unprepared_packed_adapter_target_is_reported_as_unplaced(tmp_path):
    import kinomlx._mlx_nn as nn

    model = _live_model(mx.float16)
    ff = model.transformer_blocks[0].ff
    ff.project_out = nn.QuantizedLinear.from_linear(ff.project_out, group_size=32, bits=8)
    path = tmp_path / "partial.safetensors"
    prefix = "diffusion_model.transformer_blocks.0"
    mx.save_safetensors(
        str(path),
        {
            f"{prefix}.ff.net.2.lora_A.weight": mx.ones((1, 16)),
            f"{prefix}.ff.net.2.lora_B.weight": mx.ones((8, 1)),
            f"{prefix}.audio_ff.net.2.lora_A.weight": mx.ones((1, 16)),
            f"{prefix}.audio_ff.net.2.lora_B.weight": mx.ones((4, 1)),
        },
    )
    (receipt,) = fuse_community_loras_into_model(model, [LoRAConfig(path)])
    assert receipt.complete_targets == 2
    assert ("missing_target", 1) in receipt.skipped_reasons


def test_half_accumulation_falls_back_to_fp32_for_wide_factors(tmp_path: Path, caplog) -> None:
    import logging

    from mlx.utils import tree_flatten

    from kinomlx.adapters.validation import AdapterWeightCheck

    path = tmp_path / "wide.safetensors"
    mx.save_safetensors(
        str(path),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": (
                mx.full((1, 32), 1e-5, dtype=mx.float32)
            ),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": (
                mx.full((8, 1), 1e5, dtype=mx.float32)
            ),
        },
    )
    model = _live_model(mx.float16)
    pending = AdapterWeightCheck()
    with caplog.at_level(logging.WARNING, logger="kinomlx.adapters.fusion"):
        fuse_community_loras_into_model(
            model, [LoRAConfig(path)], pending=pending, accumulation="float16"
        )
    assert "exceed the float16 range" in caplog.text
    values = [value for _key, value in tree_flatten(model.parameters())]
    mx.eval(values)
    assert all(mx.all(mx.isfinite(value)).item() for value in values)


def test_audio_side_av_cross_attention_table_classifies_as_audio() -> None:
    """The AV table's stream tag has no trailing underscore for ``audio_`` to find."""
    assert "audio" in lora_key_categories("transformer_blocks.0.scale_shift_table_a2v_ca_audio")
    assert "video" in lora_key_categories("transformer_blocks.0.scale_shift_table_a2v_ca_video")


def test_live_model_lora_fuses_bare_adaln_tables_and_names_the_shape(
    tmp_path: Path,
    caplog,
) -> None:
    """A pair targeting a bare modulation table places, and the receipt says what it is.

    LTX keeps its adaLN tables as bare parameters, so those targets carry no
    ``.weight`` segment and resolve to no weight-owning leaf. They used to count as
    missing while every ordinary target fused, which applied such an adapter's
    attention half and silently discarded its conditioning half.
    """
    import logging

    model = _live_model(mx.float32)
    block = model.transformer_blocks[0]
    tables = {
        "transformer_blocks.0.scale_shift_table": (block, "scale_shift_table", 0.25),
        "transformer_blocks.0.scale_shift_table_a2v_ca_audio": (
            block,
            "scale_shift_table_a2v_ca_audio",
            0.5,
        ),
        "scale_shift_table": (model, "scale_shift_table", 0.75),
    }
    before = {name: mx.array(getattr(owner, attr)) for name, (owner, attr, _) in tables.items()}

    weights: dict[str, mx.array] = {}
    for name, (owner, attr, delta) in tables.items():
        rows, dim = getattr(owner, attr).shape
        weights[f"diffusion_model.{name}.lora_A.weight"] = mx.ones((1, dim), dtype=mx.float32)
        weights[f"diffusion_model.{name}.lora_B.weight"] = mx.full(
            (rows, 1), delta, dtype=mx.float32
        )
    weights["diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight"] = mx.ones((1, 32))
    weights["diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight"] = mx.ones((8, 1))
    adapter = tmp_path / "adaln-tables.safetensors"
    mx.save_safetensors(str(adapter), weights)

    with caplog.at_level(logging.INFO, logger="kinomlx.models.ltx2.adapters.loading"):
        (receipt,) = fuse_community_loras_into_model(model, [LoRAConfig(adapter)])

    for name, (owner, attr, delta) in tables.items():
        expected = before[name] + delta
        assert mx.allclose(getattr(owner, attr), expected).item(), name

    assert receipt.placed_targets == 4
    assert receipt.complete_targets == 4
    assert receipt.bare_parameter_targets == 3
    assert receipt.skipped_reasons == ()
    assert receipt.warnings == ()
    assert receipt.to_metadata()["bare_parameter_targets"] == 3
    assert "is not a plain LoRA" in caplog.text


def _pair(prefix: str, *, width: int = 8, up: float = 1.0) -> dict[str, mx.array]:
    return {
        f"{prefix}.lora_A.weight": mx.ones((1, width), dtype=mx.float32),
        f"{prefix}.lora_B.weight": mx.full((width, 1), up, dtype=mx.float32),
    }


def _zeroed_to_k_model() -> tuple[LTXAVModel, mx.array]:
    model = _live_model(mx.float32)
    attention = model.transformer_blocks[0].attn1
    attention.to_k.weight = mx.zeros_like(attention.to_k.weight)
    return model, attention.to_k.weight


def _to_k(model: LTXAVModel) -> mx.array:
    return model.transformer_blocks[0].attn1.to_k.weight


def _peft_export(folder: Path, weights: dict[str, mx.array], config: dict[str, object]) -> Path:
    folder.mkdir(parents=True)
    mx.save_safetensors(str(folder / "adapter_model.safetensors"), weights)
    (folder / "adapter_config.json").write_text(json.dumps(config))
    return folder / "adapter_model.safetensors"


def test_peft_export_applies_the_scale_its_snapshot_configuration_declares(
    tmp_path: Path, caplog
) -> None:
    model, zero = _zeroed_to_k_model()
    export = _peft_export(
        tmp_path / "export", _pair(_PEFT_TO_K), {"peft_type": "LORA", "r": 1, "lora_alpha": 2}
    )
    weights, _config = link_as_snapshot(
        tmp_path / "hub", export, export.with_name("adapter_config.json"), folder="RL-Lora"
    )

    with caplog.at_level(logging.INFO, logger=_LOADING_LOGGER):
        (receipt,) = fuse_community_loras_into_model(
            model, [LoRAConfig(weights, strength=0.5)], model_generation="2.3"
        )

    # strength 0.5 x lora_alpha / r (2) x B @ A (1) in every element.
    assert mx.array_equal(_to_k(model), mx.ones_like(zero)).item()
    assert receipt.placed_targets == 1
    assert receipt.path == weights.resolve()
    assert receipt.scale_source == "adapter_config.json"
    assert receipt.intrinsic_scales == (2.0,)
    assert receipt.to_metadata()["scale_source"] == "adapter_config.json"
    assert receipt.warnings == ()
    assert "Adapter result 1/1: adapter_model.safetensors" in caplog.messages
    assert (
        "LTX-2.3 adapter adapter_model.safetensors: scale 2 from adapter_config.json "
        "(lora_alpha=2, r=1); effective 1 at strength 0.5" in caplog.messages
    )


def test_pre_pack_operations_carry_the_peft_scale(tmp_path: Path) -> None:
    export = _peft_export(
        tmp_path / "export",
        _pair("base_model.model.transformer_blocks.0.ff.net.2", width=2, up=2.0),
        {"peft_type": "LORA", "r": 1, "lora_alpha": 3},
    )
    base = "transformer_blocks.0.ff.project_out"
    operations = community_adapter_operations(
        [LoRAConfig(export, strength=0.5)], target_shapes={base: (2, 2)}
    )
    (operation,) = operations[f"{base}.weight"]
    assert operation.operation.alpha == 3.0
    weight = mx.arange(4, dtype=mx.float32).reshape(2, 2).astype(mx.bfloat16)
    # strength 0.5 x lora_alpha / r (3) x B (2.0) @ A (1.0) adds 3.0 to every element.
    assert mx.array_equal(fuse_weight(weight, operations[f"{base}.weight"]), weight + 3).item()


def test_peft_export_without_configuration_warns_and_keeps_direct_strength(
    tmp_path: Path, caplog
) -> None:
    model, zero = _zeroed_to_k_model()
    path = tmp_path / "adapter_model.safetensors"
    mx.save_safetensors(str(path), _pair(_PEFT_TO_K))

    with caplog.at_level(logging.INFO, logger=_LOADING_LOGGER):
        (receipt,) = fuse_community_loras_into_model(model, [LoRAConfig(path, strength=0.5)])

    assert mx.array_equal(_to_k(model), mx.full(zero.shape, 0.5)).item()
    assert receipt.warnings == (UNKNOWN_SCALE_WARNING,)
    assert receipt.scale_source is None
    assert receipt.intrinsic_scales == ()
    assert f"LTX-2 adapter adapter_model.safetensors: {UNKNOWN_SCALE_WARNING}" in caplog.messages
    assert not [message for message in caplog.messages if ": scale " in message]


def test_alpha_tensors_and_loose_alpha_metadata_keep_direct_strength(
    tmp_path: Path, caplog
) -> None:
    model, zero = _zeroed_to_k_model()
    prefix = "diffusion_model.transformer_blocks.0.attn1.to_k"
    path = tmp_path / "community.safetensors"
    mx.save_safetensors(
        str(path),
        {**_pair(prefix), f"{prefix}.alpha": mx.array(4.0)},
        metadata={"lora_alpha": "384", "lora_rank": "384"},
    )

    with caplog.at_level(logging.INFO, logger=_LOADING_LOGGER):
        (receipt,) = fuse_community_loras_into_model(model, [LoRAConfig(path, strength=0.5)])

    assert mx.array_equal(_to_k(model), mx.full(zero.shape, 0.5)).item()
    assert receipt.scale_source is None
    assert receipt.warnings == ()
    assert not [message for message in caplog.messages if ": scale " in message]


def test_embedded_pipeline_configuration_sets_the_ltx_scale(tmp_path: Path) -> None:
    model, zero = _zeroed_to_k_model()
    path = tmp_path / "pytorch_lora_weights.safetensors"
    config = {"peft_type": "LORA", "r": 1, "lora_alpha": 3}
    mx.save_safetensors(
        str(path),
        _pair("diffusion_model.transformer_blocks.0.attn1.to_k"),
        metadata={
            "lora_adapter_metadata": json.dumps(
                {f"transformer.{key}": value for key, value in config.items()}
            )
        },
    )

    (receipt,) = fuse_community_loras_into_model(model, [LoRAConfig(path, strength=0.5)])

    assert mx.array_equal(_to_k(model), mx.full(zero.shape, 1.5)).item()
    assert receipt.scale_source == "lora_adapter_metadata"
    assert receipt.intrinsic_scales == (3.0,)
