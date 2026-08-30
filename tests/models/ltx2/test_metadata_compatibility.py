"""Consumed-target LTX-2 compatibility contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import kinomlx.models.ltx2.metadata as metadata_module
from kinomlx.models.ltx2.audio_vae.config import AudioVAEConfig
from kinomlx.models.ltx2.metadata import TransformerConstructorConfig
from kinomlx.models.ltx2.video_vae.config import LTX23_VIDEO_VAE_CONFIG


def _entry(shape: tuple[int, ...] = (1,), *, dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def _transformer(generation: str) -> TransformerConstructorConfig:
    is_25 = generation == "2.5"
    return TransformerConstructorConfig(
        model_generation=generation,
        declared_model_version=f"{generation}.community",
        num_layers=48,
        video_in_channels=128,
        video_out_channels=128,
        video_heads=32,
        video_head_dim=128,
        audio_heads=32,
        audio_head_dim=64,
        audio_out_channels=128,
        video_context_dim=4096,
        audio_context_dim=2048,
        caption_channels=3840,
        video_max_pos=(20, 2048, 2048),
        audio_max_pos=(20,),
        positional_embedding_theta=10000.0,
        timestep_scale_multiplier=1000.0,
        av_ca_timestep_scale_multiplier=1000.0,
        norm_eps=1e-6,
        ff_bias=not is_25,
        audio_ff_bias=True,
        use_keyframes_abs_pos_embedding=is_25,
        use_prompt_adaln_single=True,
        config_digest="synthetic-constructor",
    )


def _transformer_header(config: TransformerConstructorConfig, *, prefix: str) -> dict[str, object]:
    header = {}
    for target, shape in metadata_module.transformer_parameter_shapes(config).items():
        source = target
        source = source.replace(".ff.project_in.proj.", ".ff.net.0.proj.")
        source = source.replace(".ff.project_out.", ".ff.net.2.")
        source = source.replace(".audio_ff.project_in.proj.", ".audio_ff.net.0.proj.")
        source = source.replace(".audio_ff.project_out.", ".audio_ff.net.2.")
        source = source.replace(".to_out.", ".to_out.0.")
        header[prefix + source] = _entry(shape)
    header["community.notes.tensor"] = _entry((7,))
    return header


@pytest.mark.parametrize(
    ("generation", "prefix"),
    [("2.3", "model.diffusion_model."), ("2.5", "")],
)
def test_transformer_applies_the_same_complete_target_policy_to_both_generations(
    generation: str,
    prefix: str,
    monkeypatch,
) -> None:
    config = _transformer(generation)
    monkeypatch.setattr(
        metadata_module, "read_header", lambda path: _transformer_header(config, prefix=prefix)
    )

    metadata_module.validate_transformer_header(Path("community-pack.safetensors"), config)


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_transformer_rejects_a_missing_consumed_anchor_with_generation_label(
    generation: str,
    monkeypatch,
) -> None:
    config = _transformer(generation)
    header = _transformer_header(config, prefix="model.diffusion_model.")
    header.pop("model.diffusion_model.patchify_proj.weight")
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)

    with pytest.raises(ValueError, match=rf"LTX-{generation} transformer compatibility"):
        metadata_module.validate_transformer_header(Path("community-pack.safetensors"), config)


def _gemma4_header() -> dict[str, object]:
    text = {"hidden_size": 3840, "num_hidden_layers": 48, "future_metadata": "ignored"}
    return {
        "__metadata__": {"gemma_config": json.dumps({"text_config": text})},
        "model.embed_tokens.weight": _entry((262144, 3840)),
        "model.layers.0.input_layernorm.weight": _entry((3840,)),
        "model.layers.47.post_feedforward_layernorm.weight": _entry((3840,)),
        "model.norm.weight": _entry((3840,)),
        "tokenizer_json": _entry((3,), dtype="U8"),
        "community.unused.weight": _entry((9,)),
    }


def test_gemma4_accepts_missing_unused_baggage_and_unknown_metadata(monkeypatch) -> None:
    header = _gemma4_header()
    metadata = json.loads(header["__metadata__"]["gemma_config"])
    metadata["text_config"].update(
        {
            "dtype": "float32",
            "bos_token_id": 123,
            "initializer_range": 9.0,
            "use_cache": False,
        }
    )
    header["__metadata__"]["gemma_config"] = json.dumps(metadata)
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)

    config = metadata_module.inspect_text_encoder(
        Path("community-text.safetensors"),
        model_generation="2.5",
    )

    assert config.family == "gemma4-12b-ltx"
    assert config.tokenizer_json_bytes == 3


def test_gemma4_rejects_missing_consumed_tokenizer_with_generation_label(monkeypatch) -> None:
    header = _gemma4_header()
    header.pop("tokenizer_json")
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)

    with pytest.raises(ValueError, match="LTX-2.5 text encoder compatibility"):
        metadata_module.inspect_text_encoder(
            Path("community-text.safetensors"),
            model_generation="2.5",
        )


def test_gemma3_directory_uses_the_same_consumed_config_policy(tmp_path: Path) -> None:
    root = tmp_path / "gemma-community"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "gemma3_text",
                    "hidden_size": 3840,
                    "num_hidden_layers": 48,
                    "future_metadata": "ignored",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").touch()

    config = metadata_module.inspect_text_encoder(root, model_generation="2.3")

    assert config.family == "gemma3-12b-it"
    assert config.hidden_size == 3840


def test_text_projection_accepts_wrapper_alias_and_extra_tensor(monkeypatch) -> None:
    header = {
        "wrapper.text_embedding_projection.video_aggregate_embed.weight": _entry((4096, 188160)),
        "wrapper.text_embedding_projection.video_aggregate_embed.bias": _entry((4096,)),
        "wrapper.text_embedding_projection.audio_aggregate_embed.weight": _entry((2048, 188160)),
        "wrapper.text_embedding_projection.audio_aggregate_embed.bias": _entry((2048,)),
        "extra.weight": _entry(),
    }
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)

    config = metadata_module.inspect_text_projection(
        Path("community-projection.safetensors"),
        model_generation="2.3",
        hidden_size=3840,
        num_hidden_layers=48,
    )

    assert config.video_projection_dim == 4096


def test_conv_video_vae_accepts_extra_namespaces_but_rejects_wrong_scale(monkeypatch) -> None:
    header = {
        "encoder.one": _entry(),
        "decoder.one": _entry(),
        "per_channel_statistics.mean": _entry(),
        "community.extra": _entry(),
    }
    monkeypatch.setattr(metadata_module, "read_header", lambda path: header)
    monkeypatch.setattr(
        metadata_module.VideoVAEConfig,
        "from_checkpoint",
        classmethod(lambda cls, path: LTX23_VIDEO_VAE_CONFIG),
    )
    metadata_module.inspect_video_vae(
        Path("community-video.safetensors"),
        model_generation="2.3",
    )

    monkeypatch.setattr(
        metadata_module.VideoVAEConfig,
        "from_checkpoint",
        classmethod(lambda cls, path: replace(LTX23_VIDEO_VAE_CONFIG, patch_size=8)),
    )
    with pytest.raises(ValueError, match="LTX-2.5 video VAE compatibility.*compression scale"):
        metadata_module.inspect_video_vae(
            Path("community-video.safetensors"),
            model_generation="2.5",
        )


def test_audio_upscaler_and_duration_tolerate_unconsumed_baggage(monkeypatch) -> None:
    monkeypatch.setattr(
        metadata_module.AudioVAEConfig,
        "from_checkpoint",
        classmethod(lambda cls, path: AudioVAEConfig()),
    )
    monkeypatch.setattr(
        metadata_module.BWEVocoderConfig,
        "from_checkpoint",
        classmethod(
            lambda cls, path: SimpleNamespace(
                input_sample_rate=16000,
                output_sample_rate=48000,
                mel_bins=64,
            )
        ),
    )
    monkeypatch.setattr(
        metadata_module,
        "read_header",
        lambda path: {
            "audio_vae.one": _entry(),
            "vocoder.one": _entry(),
            "initial_conv.weight": _entry(),
            "final_conv.weight": _entry(),
            "duration_head.one": _entry(),
            "extra.weight": _entry(),
        },
    )
    upscaler = {
        "_class_name": "CommunityWrappedUpsampler",
        "spatial_upsample": True,
        "temporal_upsample": False,
        "future_metadata": "ignored",
    }
    duration = {
        "transformer": {
            "cross_attention_dim": 4096,
            "audio_cross_attention_dim": 2048,
            "future_metadata": "ignored",
        },
        "duration_head": {"future_metadata": "ignored"},
    }
    configs = iter((upscaler, duration))
    monkeypatch.setattr(
        metadata_module,
        "read_metadata",
        lambda path: {"config": json.dumps(next(configs))},
    )

    metadata_module.inspect_audio_vae(
        Path("community-audio.safetensors"),
        model_generation="2.3",
    )
    upscaler_config = metadata_module.inspect_latent_upscaler(
        Path("community-upscaler.safetensors"),
        expected_kind="spatial",
        model_generation="2.5",
    )
    duration_config = metadata_module.inspect_duration_head(
        Path("community-duration.safetensors"),
        model_generation="2.5",
    )

    assert upscaler_config.in_channels == 128
    assert duration_config.video_context_dim == 4096


@pytest.mark.parametrize("model_generation", ["2.3", "2.5"])
@pytest.mark.parametrize(
    ("inspector", "component"),
    [
        ("video", "video VAE"),
        ("audio", "audio VAE"),
    ],
)
def test_component_constructor_failures_name_the_checked_generation(
    monkeypatch,
    model_generation: str,
    inspector: str,
    component: str,
) -> None:
    config_type = (
        metadata_module.VideoVAEConfig if inspector == "video" else metadata_module.AudioVAEConfig
    )
    monkeypatch.setattr(
        config_type,
        "from_checkpoint",
        classmethod(
            lambda cls, path: (_ for _ in ()).throw(ValueError("bad constructor topology"))
        ),
    )
    inspect = (
        metadata_module.inspect_video_vae
        if inspector == "video"
        else metadata_module.inspect_audio_vae
    )

    with pytest.raises(
        ValueError,
        match=rf"LTX-{model_generation} {component} compatibility.*bad constructor topology",
    ):
        inspect(Path("community.safetensors"), model_generation=model_generation)
