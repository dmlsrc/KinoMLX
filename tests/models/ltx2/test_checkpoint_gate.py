"""Generation-neutral transformer metadata and constructor gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kinomlx.models.ltx2.metadata as metadata_module


def _transformer_config() -> dict[str, object]:
    return {
        "_class_name": "AVTransformer3DModel",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_type": "default",
        "caption_channels": 3840,
        "cross_attention_dim": 4096,
        "double_self_attention": False,
        "dropout": 0.0,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-6,
        "norm_num_groups": 32,
        "num_attention_heads": 32,
        "num_embeds_ada_norm": 1000,
        "num_layers": 48,
        "num_vector_embeds": None,
        "only_cross_attention": False,
        "cross_attention_norm": True,
        "out_channels": 128,
        "upcast_attention": False,
        "use_linear_projection": False,
        "qk_norm": "rms_norm",
        "standardization_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "timestep_scale_multiplier": 1000,
        "av_ca_timestep_scale_multiplier": 1000.0,
        "causal_temporal_positioning": True,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "use_audio_video_cross_attention": True,
        "share_ff": False,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "audio_positional_embedding_max_pos": [20],
        "av_cross_ada_norm": True,
        "use_embeddings_connector": True,
        "connector_attention_head_dim": 128,
        "connector_num_attention_heads": 32,
        "connector_num_layers": 8,
        "connector_positional_embedding_max_pos": [4096],
        "connector_num_learnable_registers": 128,
        "connector_norm_output": True,
        "use_middle_indices_grid": True,
        "apply_gated_attention": True,
        "connector_apply_gated_attention": True,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
        "caption_proj_input_norm": False,
        "connector_learnable_registers_std": 1,
        "caption_proj_before_connector": True,
        "audio_connector_attention_head_dim": 64,
        "audio_connector_num_attention_heads": 32,
        "cross_attention_adaln": True,
        "rope_type": "split",
        "frequencies_precision": "float64",
        "text_encoder_norm_type": "PER_TOKEN_RMS",
    }


def _patch_metadata(
    monkeypatch,
    *,
    version: str | None,
    transformer: dict[str, object],
    tensor_generation: str | None = None,
) -> None:
    metadata = {"config": json.dumps({"transformer": transformer})}
    if version is not None:
        metadata["model_version"] = version
    monkeypatch.setattr(metadata_module, "read_metadata", lambda path: metadata)
    if tensor_generation is None:
        ff_bias = transformer.get("ff_bias", True) is True
        keyframes = transformer.get("use_keyframes_abs_pos_embedding", False) is True
    else:
        ff_bias = tensor_generation == "2.3"
        keyframes = tensor_generation == "2.5"
    header: dict[str, object] = {
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": {},
    }
    if ff_bias:
        header["model.diffusion_model.transformer_blocks.0.ff.net.0.proj.bias"] = {}
    if keyframes:
        header["model.diffusion_model.keyframes_abs_pos_embedding"] = {}
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)


def test_checkpoint_gate_accepts_pinned_23_constructor(monkeypatch) -> None:
    transformer = _transformer_config()
    _patch_metadata(monkeypatch, version="2.3.0", transformer=transformer)

    parsed = metadata_module.checkpoint_config(Path("renamed.safetensors"))

    assert parsed.model_generation == "2.3"
    assert parsed.declared_model_version == "2.3.0"
    assert parsed.transformer.ff_bias is True
    assert parsed.transformer.audio_ff_bias is True
    assert parsed.transformer.use_keyframes_abs_pos_embedding is False
    assert parsed.video_vae is None


def test_checkpoint_gate_accepts_case_variant_enum_spelling(monkeypatch) -> None:
    transformer = _transformer_config()
    transformer["text_encoder_norm_type"] = "per_token_rms"
    transformer["frequencies_precision"] = "FLOAT64"
    _patch_metadata(monkeypatch, version="2.3.0", transformer=transformer)

    parsed = metadata_module.checkpoint_config(Path("community.safetensors"))

    assert parsed.model_generation == "2.3"


def test_checkpoint_gate_accepts_25_constructor_without_filename_or_version_gate(
    monkeypatch,
) -> None:
    transformer = _transformer_config()
    transformer["ff_bias"] = False
    transformer["use_keyframes_abs_pos_embedding"] = True
    _patch_metadata(monkeypatch, version="community-metadata", transformer=transformer)

    parsed = metadata_module.checkpoint_config(Path("anything.safetensors"))

    assert parsed.model_generation == "2.5"
    assert parsed.declared_model_version == "community-metadata"
    assert parsed.transformer.cache_identity() != ()


def test_checkpoint_gate_records_absent_semantic_version(monkeypatch) -> None:
    _patch_metadata(monkeypatch, version=None, transformer=_transformer_config())

    parsed = metadata_module.checkpoint_config(Path("model.safetensors"))

    assert parsed.model_generation == "2.3"
    assert parsed.declared_model_version is None


def test_checkpoint_gate_uses_tensor_graph_when_variant_metadata_is_stale(monkeypatch) -> None:
    transformer = _transformer_config()
    _patch_metadata(
        monkeypatch,
        version="2.3.0",
        transformer=transformer,
        tensor_generation="2.5",
    )

    parsed = metadata_module.checkpoint_config(Path("community.safetensors"))

    assert parsed.model_generation == "2.5"
    assert parsed.declared_model_version == "2.3.0"
    assert parsed.transformer.ff_bias is False
    assert parsed.transformer.use_keyframes_abs_pos_embedding is True


@pytest.mark.parametrize(
    ("ff_bias", "keyframe_embedding"),
    [(False, False), (True, True)],
)
def test_checkpoint_gate_rejects_unimplemented_constructor_combinations(
    ff_bias: bool,
    keyframe_embedding: bool,
    monkeypatch,
) -> None:
    transformer = _transformer_config()
    transformer["ff_bias"] = ff_bias
    transformer["use_keyframes_abs_pos_embedding"] = keyframe_embedding
    _patch_metadata(monkeypatch, version="2.5.0", transformer=transformer)

    with pytest.raises(ValueError, match="constructor combination"):
        metadata_module.checkpoint_config(Path("model.safetensors"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frequencies_precision", "float32"),
        ("positional_embedding_theta", 9999.0),
        ("positional_embedding_max_pos", [16, 1024, 1024]),
        ("audio_positional_embedding_max_pos", [16]),
        ("timestep_scale_multiplier", 999),
        ("av_ca_timestep_scale_multiplier", 1.0),
        ("norm_eps", 1e-5),
        ("apply_gated_attention", False),
        ("use_middle_indices_grid", False),
    ],
)
def test_checkpoint_gate_rejects_changed_transformer_constants(
    key: str,
    value: object,
    monkeypatch,
) -> None:
    transformer = _transformer_config()
    transformer[key] = value
    _patch_metadata(monkeypatch, version="2.3.0", transformer=transformer)
    with pytest.raises(ValueError, match=key):
        metadata_module.checkpoint_config(Path("model.safetensors"))


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_checkpoint_gate_records_supported_inference_for_missing_transformer_constant(
    generation: str,
    monkeypatch,
) -> None:
    transformer = _transformer_config()
    if generation == "2.5":
        transformer["ff_bias"] = False
        transformer["use_keyframes_abs_pos_embedding"] = True
    transformer.pop("frequencies_precision")
    _patch_metadata(monkeypatch, version=f"{generation}.0", transformer=transformer)
    parsed = metadata_module.checkpoint_config(Path("model.safetensors"))
    assert parsed.model_generation == generation
    assert "frequencies_precision" in parsed.transformer.inferred_fields


def test_checkpoint_gate_ignores_unknown_non_consumed_constructor_field(monkeypatch) -> None:
    transformer = _transformer_config()
    baseline = dict(transformer)
    transformer["future_attention_mode"] = "surprise"
    _patch_metadata(monkeypatch, version="2.5.0", transformer=transformer)
    with_unknown = metadata_module.checkpoint_config(Path("model.safetensors"))
    _patch_metadata(monkeypatch, version="2.5.0", transformer=baseline)
    without_unknown = metadata_module.checkpoint_config(Path("model.safetensors"))
    assert with_unknown.transformer.config_digest == without_unknown.transformer.config_digest


def test_checkpoint_gate_ignores_receipt_only_constructor_metadata(monkeypatch) -> None:
    transformer = _transformer_config()
    baseline = dict(transformer)
    transformer.update(
        {
            "_class_name": "CommunityWrappedTransformer",
            "dropout": 0.25,
            "connector_learnable_registers_std": 99,
        }
    )
    _patch_metadata(monkeypatch, version="2.3.0", transformer=transformer)
    with_baggage = metadata_module.checkpoint_config(Path("model.safetensors"))
    _patch_metadata(monkeypatch, version="2.3.0", transformer=baseline)
    baseline_config = metadata_module.checkpoint_config(Path("model.safetensors"))

    assert with_baggage.transformer.config_digest == baseline_config.transformer.config_digest


def test_checkpoint_gate_selects_static_prompt_adaln_arrangement(monkeypatch) -> None:
    transformer = _transformer_config()
    transformer["use_prompt_adaln_single"] = False
    _patch_metadata(monkeypatch, version="2.5.0", transformer=transformer)
    parsed = metadata_module.checkpoint_config(Path("model.safetensors"))
    assert not parsed.transformer.use_prompt_adaln_single


def test_checkpoint_gate_labels_unimplemented_mixed_prompt_adaln_tensors(monkeypatch) -> None:
    transformer = _transformer_config()
    transformer["ff_bias"] = False
    transformer["use_keyframes_abs_pos_embedding"] = True
    _patch_metadata(monkeypatch, version="2.5.0", transformer=transformer)
    header = {
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": {},
        "model.diffusion_model.keyframes_abs_pos_embedding": {},
        "model.diffusion_model.audio_patchify_proj.weight": {},
        "model.diffusion_model.prompt_adaln_single.linear.weight": {},
    }
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)

    with pytest.raises(
        ValueError,
        match="LTX-2.5 transformer compatibility.*mixed video/audio prompt AdaLN",
    ):
        metadata_module.checkpoint_config(Path("community.safetensors"))
