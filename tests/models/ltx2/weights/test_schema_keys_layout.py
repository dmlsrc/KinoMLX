"""Cache identity, key routing, and baked-layout contracts."""

import mlx.core as mx

from kinomlx.models.ltx2.weights.keys import (
    convert_checkpoint_key,
    convert_pytorch_key_to_mlx,
    flatten_to_nested,
    weight_family_for_key,
)
from kinomlx.models.ltx2.weights.layout import bake_conv_layout, layout_weight_key
from kinomlx.models.ltx2.weights.policy import (
    ensure_ff_pretranspose_for_dtype,
)


def test_transformer_key_conversion_covers_video_and_audio_rules() -> None:
    assert (
        convert_pytorch_key_to_mlx("transformer_blocks.0.attn1.to_out.0.weight")
        == "transformer_blocks.0.attn1.to_out.weight"
    )
    assert (
        convert_pytorch_key_to_mlx("transformer_blocks.1.ff.net.0.proj.weight")
        == "transformer_blocks.1.ff.project_in.proj.weight"
    )
    assert (
        convert_pytorch_key_to_mlx("transformer_blocks.1.ff.net.2.bias")
        == "transformer_blocks.1.ff.project_out.bias"
    )
    audio_key = "transformer_blocks.2.audio_ff.net.2.weight"
    assert convert_pytorch_key_to_mlx(audio_key) is None
    assert (
        convert_pytorch_key_to_mlx(audio_key, include_audio=True)
        == "transformer_blocks.2.audio_ff.project_out.weight"
    )
    assert (
        convert_checkpoint_key("model.diffusion_model.transformer_blocks.0.ff.net.2.weight")
        == "transformer_blocks.0.ff.project_out.weight"
    )
    assert convert_checkpoint_key("vae.decoder.weight") is None


def test_connector_keys_are_never_misrouted_into_transformer() -> None:
    key = "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"
    assert convert_pytorch_key_to_mlx(key, include_audio=True) is None
    generic = "embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"
    assert convert_pytorch_key_to_mlx(generic, include_audio=True) is None
    assert (
        convert_checkpoint_key(
            f"model.diffusion_model.{generic}",
            include_audio=True,
        )
        is None
    )


def test_auxiliary_weight_family_routing_is_exact() -> None:
    assert weight_family_for_key("vae.decoder.conv.weight") == "video_vae"
    assert weight_family_for_key("audio_vae.decoder.conv.weight") == "audio_vae"
    assert weight_family_for_key("vocoder.generator.conv.weight") == "vocoder"
    assert weight_family_for_key("text_embedding_projection.video.weight") == "connector"
    assert (
        weight_family_for_key("model.diffusion_model.video_embeddings_connector.registers")
        == "connector"
    )
    assert weight_family_for_key("model.diffusion_model.scale_shift_table") is None


def test_flatten_to_nested_converts_numeric_module_maps_to_lists() -> None:
    value = mx.array([1.0])
    nested = flatten_to_nested(
        {
            "transformer_blocks.0.weight": value,
            "transformer_blocks.1.weight": value + 1,
        }
    )
    blocks = nested["transformer_blocks"]
    assert isinstance(blocks, list)
    assert len(blocks) == 2
    assert mx.array_equal(blocks[0]["weight"], value).item()


def test_video_and_audio_conv_layouts_are_materialized_channels_last() -> None:
    video = mx.arange(2 * 3 * 2 * 2 * 2).reshape(2, 3, 2, 2, 2)
    audio = mx.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    video_baked = bake_conv_layout("video_vae", "vae.conv.weight", video)
    audio_baked = bake_conv_layout("audio_vae", "audio_vae.conv.weight", audio)
    assert video_baked.shape == (2, 2, 2, 2, 3)
    assert mx.array_equal(video_baked, video.transpose(0, 2, 3, 4, 1)).item()
    assert audio_baked.shape == (2, 2, 2, 3)


def test_vocoder_layout_is_left_for_structure_aware_loader() -> None:
    weight = mx.arange(24).reshape(2, 3, 4)
    assert bake_conv_layout("vocoder", "vocoder.conv.weight", weight) is weight


def test_layout_key_selection_covers_ff_attention_audio_and_adaln() -> None:
    common = {
        "video_ff_layout_specs": (("project_out", "pretranspose"),),
        "video_ff_layout_layers": (3,),
        "video_attn_layout_specs": (("to_q", "pretranspose"),),
        "video_attn_layout_layers": (3,),
        "audio_ff_layout_specs": (("project_in", "pretranspose"),),
        "audio_ff_layout_layers": (3,),
        "audio_attn_layout_specs": (("to_out", "pretranspose"),),
        "audio_attn_layout_layers": (3,),
        "adaln_pretranspose": True,
    }
    assert (
        layout_weight_key("transformer_blocks.3.ff.project_out.weight", **common)
        == "transformer_blocks.3.ff.project_out.weight_t"
    )
    assert (
        layout_weight_key("transformer_blocks.3.attn1.to_q.weight", **common)
        == "transformer_blocks.3.attn1.to_q.weight_t"
    )
    assert (
        layout_weight_key("transformer_blocks.3.audio_ff.project_in.proj.weight", **common)
        == "transformer_blocks.3.audio_ff.project_in.proj.weight_t"
    )
    assert (
        layout_weight_key("transformer_blocks.3.audio_attn1.to_out.weight", **common)
        == "transformer_blocks.3.audio_attn1.to_out.weight_t"
    )
    assert (
        layout_weight_key("adaln_single.linear.weight", **common) == "adaln_single.linear.weight_t"
    )
    assert layout_weight_key("transformer_blocks.2.ff.project_out.weight", **common) is None


def test_fp16_transformer_auto_adds_both_ff_pretransposes() -> None:
    existing = (("project_out", "pretranspose"),)
    normalized = ensure_ff_pretranspose_for_dtype(existing, mx.float16)
    assert normalized == (
        ("project_in", "pretranspose"),
        ("project_out", "pretranspose"),
    )
    assert ensure_ff_pretranspose_for_dtype(existing, mx.bfloat16) == existing
