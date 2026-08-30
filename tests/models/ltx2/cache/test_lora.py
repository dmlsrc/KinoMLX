"""LTX LoRA stage-strength and target-knockout contracts."""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.lora.loading import LoRAConfig, lora_configs_for_stage
from kinomlx.models.ltx2.cache import lora as lora_module
from kinomlx.models.ltx2.cache.keys.lora import lora_key_categories
from kinomlx.models.ltx2.cache.lora import (
    fuse_community_loras,
    fuse_community_loras_into_model,
    normalize_lora_for_cache,
)
from kinomlx.models.ltx2.transformer import LTXAVModel

from ..transformer._synthetic import build_shaped_ltx_model


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


def test_multiple_loras_use_independent_stage_strengths_and_knockouts(
    tmp_path: Path,
) -> None:
    style_path = tmp_path / "style.safetensors"
    motion_path = tmp_path / "motion.safetensors"
    _save_adapter(style_path, video_scale=1.0, audio_scale=10.0)
    _save_adapter(motion_path, video_scale=10.0, audio_scale=2.0)
    configs = [
        LoRAConfig(
            style_path,
            stage_1_strength=0.5,
            stage_2_strength=1.0,
            exclude=("audio",),
        ),
        LoRAConfig(
            motion_path,
            stage_1_strength=1.0,
            stage_2_strength=0.25,
            exclude=("video",),
        ),
    ]
    base = {
        "transformer_blocks.0.ff.project_out.weight": mx.zeros((2, 2)),
        "transformer_blocks.0.audio_ff.project_out.weight": mx.zeros((2, 2)),
    }

    stage_1 = fuse_community_loras(
        dict(base),
        lora_configs_for_stage(configs, 1),
        in_place=False,
    )
    stage_2 = fuse_community_loras(
        dict(base),
        lora_configs_for_stage(configs, 2),
        in_place=False,
    )

    assert mx.allclose(
        stage_1["transformer_blocks.0.ff.project_out.weight"],
        mx.full((2, 2), 0.5),
    ).item()
    assert mx.allclose(
        stage_1["transformer_blocks.0.audio_ff.project_out.weight"],
        mx.full((2, 2), 2.0),
    ).item()
    assert mx.allclose(
        stage_2["transformer_blocks.0.ff.project_out.weight"],
        mx.full((2, 2), 1.0),
    ).item()
    assert mx.allclose(
        stage_2["transformer_blocks.0.audio_ff.project_out.weight"],
        mx.full((2, 2), 0.5),
    ).item()


def test_unknown_lora_knockout_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown LoRA exclude categories"):
        normalize_lora_for_cache(
            {},
            {},
            exclude=("sideways",),
        )


def test_unknown_knockout_is_rejected_before_adapter_file_load(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.safetensors"
    config = LoRAConfig(missing, exclude=("sideways",))

    with pytest.raises(ValueError, match="Unknown LoRA exclude categories"):
        fuse_community_loras({}, [config])


def test_partial_wrong_model_adapter_fuses_and_records_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter_path = tmp_path / "partial.safetensors"
    weights: dict[str, mx.array] = {}
    for block in range(5):
        prefix = f"diffusion_model.transformer_blocks.{block}.ff.net.2"
        weights[f"{prefix}.lora_A.weight"] = mx.ones((1, 2))
        weights[f"{prefix}.lora_B.weight"] = mx.ones((2, 1))
    mx.save_safetensors(str(adapter_path), weights)
    base_key = "transformer_blocks.0.ff.project_out.weight"
    base = {base_key: mx.zeros((2, 2))}
    config = LoRAConfig(adapter_path)

    receipts = []
    with caplog.at_level(logging.WARNING, logger="kinomlx.lora.fusion"):
        fused = fuse_community_loras(
            dict(base),
            [config],
            in_place=False,
            model_generation="2.3",
            receipt_collector=receipts,
        )
    assert mx.array_equal(fused[base_key], mx.ones((2, 2))).item()
    assert any("coverage 20%" in record.getMessage() for record in caplog.records)
    assert receipts[0].structural_coverage == pytest.approx(0.2)
    assert receipts[0].placed_targets == 1


def test_zero_placement_receipt_records_mismatch_skips_and_no_effect_warning(
    tmp_path: Path,
) -> None:
    blob_sha = "a" * 64
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    adapter_path = blob_dir / blob_sha
    serialized_path = blob_dir / f"{blob_sha}.safetensors"
    prefix = "diffusion_model.transformer_blocks.4.ff.net.2"
    mx.save_safetensors(
        str(serialized_path),
        {
            f"{prefix}.lora_A.weight": mx.ones((1, 2)),
            f"{prefix}.lora_B.weight": mx.ones((2, 1)),
        },
        metadata={"model_version": "2.3.0-community"},
    )
    serialized_path.replace(adapter_path)
    receipts = []

    base_weight = mx.zeros((2, 2))
    fused = fuse_community_loras(
        {"transformer_blocks.0.ff.project_out.weight": base_weight},
        [LoRAConfig(adapter_path, strength=0.5)],
        in_place=False,
        model_generation="2.5",
        receipt_collector=receipts,
    )

    assert fused["transformer_blocks.0.ff.project_out.weight"] is base_weight
    receipt = receipts[0]
    assert receipt.fingerprint == f"sha256:{blob_sha}"
    assert receipt.declared_model_generation == "2.3"
    assert receipt.base_model_generation == "2.5"
    assert receipt.generation_mismatch is True
    assert receipt.strength == 0.5
    assert receipt.complete_targets == 1
    assert receipt.placed_targets == 0
    assert receipt.structural_coverage == 0.0
    assert receipt.skipped_reasons == (("missing_target", 1),)
    assert receipt.warning is True


def test_excluded_targets_do_not_reduce_lora_coverage(tmp_path: Path) -> None:
    adapter_path = tmp_path / "branches.safetensors"
    _save_adapter(adapter_path, video_scale=1.0, audio_scale=2.0)
    base_key = "transformer_blocks.0.ff.project_out.weight"

    fused = fuse_community_loras(
        {base_key: mx.zeros((2, 2))},
        [LoRAConfig(adapter_path, exclude=("audio",))],
        in_place=False,
    )

    assert mx.array_equal(fused[base_key], mx.ones((2, 2))).item()


def test_all_excluded_targets_are_receipted_noop_without_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter_path = tmp_path / "branches.safetensors"
    _save_adapter(adapter_path, video_scale=1.0, audio_scale=2.0)
    base = {
        "transformer_blocks.0.ff.project_out.weight": mx.zeros((2, 2)),
        "transformer_blocks.0.audio_ff.project_out.weight": mx.zeros((2, 2)),
    }
    receipts = []

    with caplog.at_level(logging.WARNING):
        fused = fuse_community_loras(
            base,
            [LoRAConfig(adapter_path, exclude=("video", "audio"))],
            in_place=False,
            receipt_collector=receipts,
        )

    assert fused is not base
    assert fused.keys() == base.keys()
    assert all(fused[key] is value for key, value in base.items())
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.knockouts == ("audio", "video")
    assert receipt.complete_targets == 2
    assert receipt.placed_targets == 0
    assert receipt.structural_coverage == 0.0
    assert receipt.skipped_reasons == (("excluded", 2),)
    assert receipt.warning is False
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_lora_logs_mapping_and_fusion_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter_path = tmp_path / "diagnostic.safetensors"
    _save_adapter(adapter_path, video_scale=1.0, audio_scale=2.0)
    base_key = "transformer_blocks.0.ff.project_out.weight"

    with (
        caplog.at_level(
            logging.INFO,
            logger="kinomlx.models.ltx2.cache.lora",
        ),
        caplog.at_level(logging.INFO, logger="kinomlx.lora.fusion"),
    ):
        fuse_community_loras(
            {base_key: mx.zeros((2, 2))},
            [LoRAConfig(adapter_path, exclude=("audio",))],
            in_place=False,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "diagnostic.safetensors" in message and "1 excluded" in message and "0 unmapped" in message
        for message in messages
    )
    assert any(
        "placed 1/1 targets" in message and "0 shape mismatches" in message for message in messages
    )


def test_ltx_lora_uses_direct_strength_not_alpha_rank(tmp_path: Path) -> None:
    base_key = "transformer_blocks.0.ff.project_out.weight"
    adapter_prefix = "diffusion_model.transformer_blocks.0.ff.net.2"
    adapter_path = tmp_path / "alpha.safetensors"
    mx.save_safetensors(
        str(adapter_path),
        {
            f"{adapter_prefix}.lora_A.weight": mx.ones((1, 2)),
            f"{adapter_prefix}.lora_B.weight": mx.ones((2, 1)),
            f"{adapter_prefix}.alpha": mx.array(8.0),
        },
    )
    base = {base_key: mx.zeros((2, 2))}

    fused = fuse_community_loras(
        base,
        [LoRAConfig(adapter_path, strength=0.5)],
        in_place=False,
    )

    assert mx.array_equal(fused[base_key], mx.full((2, 2), 0.5)).item()


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
    specs = (("project_out", "pretranspose"),)
    ff.apply_layouts(specs)
    ff.drop_layout_sources(specs)
    adapter = tmp_path / "adapter.safetensors"
    attention_specs = (("to_q", "pretranspose"),)
    attention.apply_layouts(attention_specs)
    attention.drop_layout_sources(attention_specs)
    adaln = model.adaln_single
    adaln.linear.weight = mx.zeros_like(adaln.linear.weight)
    adaln_shape = tuple(adaln.linear.weight.shape)
    adaln.apply_layout()
    adaln.drop_layout_source()
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
    validation_cache = {
        "transformer_blocks.0.ff.project_out.weight": mx.zeros_like(
            model.transformer_blocks[0].ff.project_out.weight
        ),
        "transformer_blocks.0.attn1.to_q.weight": mx.zeros_like(attention.to_q.weight),
        "transformer_blocks.0.attn1.to_k.weight": mx.zeros_like(attention.to_k.weight),
    }
    validation_peaks = dict.fromkeys(validation_cache, 0.0)
    monkeypatch.setattr(lora_module, "load_cache_weights", lambda _path: validation_cache)
    monkeypatch.setattr(
        lora_module,
        "load_transformer_fp16_ranges",
        lambda _path: validation_peaks,
    )
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
        transformer_cache_path=tmp_path / "prepared-transformer.safetensors",
    )

    assert mx.array_equal(attention.to_k.weight, mx.ones_like(attention.to_k.weight)).item()
    assert validation_cache == {}
    assert validation_peaks == {}
    assert receipt.complete_targets == 3
    assert receipt.placed_targets == 1
    assert receipt.structural_coverage == pytest.approx(1 / 3)
    assert receipt.skipped_reasons == (("missing_target", 2),)
    assert receipt.warning is True


def test_live_fp16_lora_validates_consumed_cache_copies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _live_model(mx.float16)
    ff = model.transformer_blocks[0].ff
    ff.project_out.weight = mx.zeros_like(ff.project_out.weight)
    base_key = "transformer_blocks.0.ff.project_out.weight"
    validation_cache = {base_key: mx.zeros_like(ff.project_out.weight)}
    validation_peaks = {base_key: 0.0}
    cache_path = tmp_path / "prepared-transformer.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.ones(
                (1, ff.project_out.weight.shape[1])
            ),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.ones(
                (ff.project_out.weight.shape[0], 1)
            ),
        },
    )
    monkeypatch.setattr(
        "kinomlx.models.ltx2.cache.lora.load_cache_weights",
        lambda _path: validation_cache,
    )
    monkeypatch.setattr(
        "kinomlx.models.ltx2.cache.lora.load_transformer_fp16_ranges",
        lambda _path: validation_peaks,
    )

    fuse_community_loras_into_model(
        model,
        [LoRAConfig(adapter)],
        transformer_cache_path=cache_path,
    )

    assert validation_cache == {}
    assert validation_peaks == {}
    assert mx.array_equal(ff.project_out.weight, mx.ones_like(ff.project_out.weight)).item()


def test_live_fp16_lora_requires_explicit_prepared_cache_path(
    tmp_path: Path,
) -> None:
    model = _live_model(mx.float16)
    adapter = tmp_path / "adapter.safetensors"
    ff = model.transformer_blocks[0].ff
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.ones(
                (1, ff.project_out.weight.shape[1])
            ),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.ones(
                (ff.project_out.weight.shape[0], 1)
            ),
        },
    )

    with pytest.raises(ValueError, match="transformer_cache_path is required"):
        fuse_community_loras_into_model(model, [LoRAConfig(adapter)])
