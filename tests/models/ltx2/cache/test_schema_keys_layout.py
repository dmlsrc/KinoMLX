"""Cache identity, key routing, and baked-layout contracts."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from kinomlx.models.ltx2.cache.keys import (
    convert_checkpoint_key,
    convert_pytorch_key_to_mlx,
    flatten_to_nested,
    weight_family_for_key,
)
from kinomlx.models.ltx2.cache.layout import (
    bake_conv_layout_for_family,
    ensure_ff_pretranspose_for_dtype,
    layout_cache_key,
)
from kinomlx.models.ltx2.cache.policy import (
    DEFAULT_TRANSFORMER_LAYOUT_LAYERS,
    DEFAULT_VIDEO_FF_LAYOUT_SPECS,
)
from kinomlx.models.ltx2.cache.schema import (
    COMPONENT_CACHE_SCHEMA_VERSION,
    FAMILY_CACHE_SCHEMA_VERSION,
    TRANSFORMER_CACHE_SCHEMA_VERSION,
    component_cache_paths,
    component_cache_payload,
    family_directory_payload,
    payload_digest,
    transformer_cache_paths,
    transformer_cache_payload,
    weight_family_cache_paths,
)


def _common_payload_options() -> dict[str, object]:
    return {
        "include_audio": True,
        "video_ff_layout_specs": (("project_out", "pretranspose"),),
        "video_ff_layout_layers": (0, 47),
        "video_attn_layout_specs": (),
        "video_attn_layout_layers": (),
    }


def test_transformer_payload_preserves_schema_and_dtype_rules(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})

    default = transformer_cache_payload(
        source,
        transformer_dtype=mx.bfloat16,
        **_common_payload_options(),
    )
    fp16 = transformer_cache_payload(
        source,
        transformer_dtype="float16",
        **_common_payload_options(),
    )
    redundant_fp16 = transformer_cache_payload(
        source,
        transformer_dtype="float16",
        video_ff_dtype=mx.float16,
        **_common_payload_options(),
    )

    assert default["schema_version"] == TRANSFORMER_CACHE_SCHEMA_VERSION == 3
    assert "transformer_dtype" not in default
    assert fp16["transformer_dtype"] == "float16"
    assert redundant_fp16 == fp16
    assert default["video_ff_layout_specs"] == [{"target": "project_out", "layout": "pretranspose"}]
    assert default["video_ff_layout_layers"] == [0, 47]


def test_transformer_cache_path_uses_canonical_payload_digest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "odd name.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})
    cache_file, metadata_file, payload = transformer_cache_paths(
        source,
        tmp_path / "cache",
        **_common_payload_options(),
    )
    assert cache_file.name == "transformer.safetensors"
    assert metadata_file.name == "metadata.json"
    assert cache_file.parent.name == f"odd_name-{payload_digest(payload)}"


def test_default_layout_policy_expands_layers_and_mirrors_audio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})

    payload = transformer_cache_payload(source, include_audio=True)

    expected_specs = [{"target": "project_out", "layout": "pretranspose"}]
    expected_layers = list(DEFAULT_TRANSFORMER_LAYOUT_LAYERS)
    assert payload["video_ff_layout_specs"] == expected_specs
    assert payload["video_ff_layout_layers"] == expected_layers
    assert payload["audio_ff_layout_specs"] == expected_specs
    assert payload["audio_ff_layout_layers"] == expected_layers
    assert payload["video_attn_layout_specs"] == []
    assert "audio_attn_layout_specs" not in payload


def test_implicit_and_explicit_all_layers_share_cache_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})
    cache_root = tmp_path / "cache"

    implicit = transformer_cache_paths(
        source,
        cache_root,
        include_audio=True,
    )
    explicit = transformer_cache_paths(
        source,
        cache_root,
        include_audio=True,
        video_ff_layout_specs=DEFAULT_VIDEO_FF_LAYOUT_SPECS,
        video_ff_layout_layers=DEFAULT_TRANSFORMER_LAYOUT_LAYERS,
        audio_ff_layout_specs=DEFAULT_VIDEO_FF_LAYOUT_SPECS,
        audio_ff_layout_layers=DEFAULT_TRANSFORMER_LAYOUT_LAYERS,
    )

    assert implicit == explicit


def test_explicit_empty_audio_layout_disables_default_mirror(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})

    payload = transformer_cache_payload(
        source,
        include_audio=True,
        audio_ff_layout_specs=(),
        audio_ff_layout_layers=(),
        audio_attn_layout_specs=(),
        audio_attn_layout_layers=(),
    )

    assert "audio_ff_layout_specs" not in payload
    assert "audio_ff_layout_layers" not in payload


def test_fp16_ff_override_adds_both_layouts_and_fp32_does_not(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})

    fp16_ff = transformer_cache_payload(
        source,
        include_audio=False,
        video_ff_dtype=mx.float16,
    )
    fp32 = transformer_cache_payload(
        source,
        include_audio=False,
        transformer_dtype=mx.float32,
    )

    assert fp16_ff["video_ff_layout_specs"] == [
        {"target": "project_in", "layout": "pretranspose"},
        {"target": "project_out", "layout": "pretranspose"},
    ]
    assert fp16_ff["video_ff_layout_layers"] == list(DEFAULT_TRANSFORMER_LAYOUT_LAYERS)
    assert fp32["video_ff_layout_specs"] == [{"target": "project_out", "layout": "pretranspose"}]


def test_implicit_targeted_quant_layers_are_canonicalized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})

    payload = transformer_cache_payload(
        source,
        include_audio=False,
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
    )

    assert payload["video_ff_quantize_layers"] == list(DEFAULT_TRANSFORMER_LAYOUT_LAYERS)


def test_whole_transformer_quantization_canonicalizes_away_layouts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})

    payload = transformer_cache_payload(
        source,
        include_audio=True,
        audio_ff_layout_specs=(("project_out", "pretranspose"),),
        transformer_cache_quantize="mxfp8-blocks",
    )

    assert payload["video_ff_layout_specs"] == []
    assert payload["video_ff_layout_layers"] == []
    assert "audio_ff_layout_specs" not in payload


def test_source_signature_change_invalidates_family_directory_key(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.safetensors"
    source.write_bytes(b"first")
    before = payload_digest(family_directory_payload(source))
    source.write_bytes(b"second version")
    after = payload_digest(family_directory_payload(source))
    assert before != after


def test_family_path_and_sidecar_use_component_local_schema(tmp_path: Path) -> None:
    source = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(source), {"x": mx.array([1.0])})
    cache_file, metadata_file, payload = weight_family_cache_paths(
        source,
        tmp_path / "cache",
        "video_vae",
    )
    assert cache_file.name == "video_vae.safetensors"
    assert metadata_file.name == "video_vae.metadata.json"
    assert payload["kind"] == "video_vae"
    assert payload["schema_version"] == FAMILY_CACHE_SCHEMA_VERSION == 3
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert payload_digest(payload) == payload_digest(json.loads(encoded))


def test_component_cache_identity_covers_generation_without_source_schema_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "renamed community artifact.safetensors"
    source.write_bytes(b"component")
    common = {
        "source_component": "model.diffusion_model",
        "source_fingerprint": "sha256:artifact",
        "model_generation": "2.5",
        "constructor_identity": (
            ("ff_bias", "false"),
            ("keyframes_abs_pos", "true"),
        ),
        "dtype_policy": (("weights", "bfloat16"),),
        "layout_policy": (("project_out", "pretranspose"),),
    }

    cache_file, metadata_file, payload = component_cache_paths(
        source,
        tmp_path / "cache",
        **common,
    )

    assert payload["schema_version"] == COMPONENT_CACHE_SCHEMA_VERSION == 2
    assert payload["source_component"] == "model.diffusion_model"
    assert payload["model_generation"] == "2.5"
    assert "key_schema_digest" not in payload
    assert "generation" not in payload
    assert "model_version" not in payload
    assert "flavor" not in payload
    assert cache_file.name == "component.safetensors"
    assert metadata_file.name == "metadata.json"
    assert cache_file.parent.name.endswith(payload_digest(payload))

    changed_constructor = component_cache_payload(
        source,
        **{
            **common,
            "constructor_identity": (
                ("ff_bias", "true"),
                ("keyframes_abs_pos", "false"),
            ),
        },
    )
    changed_generation = component_cache_payload(
        source,
        **{**common, "model_generation": "2.3"},
    )
    changed_namespace = component_cache_payload(
        source,
        **{**common, "source_component": "video_embeddings_connector"},
    )
    changed_fingerprint = component_cache_payload(
        source,
        **{**common, "source_fingerprint": "sha256:unseen-community-artifact"},
    )
    assert (
        len(
            {
                payload_digest(payload),
                payload_digest(changed_constructor),
                payload_digest(changed_generation),
                payload_digest(changed_namespace),
                payload_digest(changed_fingerprint),
            }
        )
        == 5
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
    video_baked = bake_conv_layout_for_family(
        "video_vae",
        {"vae.conv.weight": video, "vae.bias": mx.zeros((2,))},
    )
    audio_baked = bake_conv_layout_for_family(
        "audio_vae",
        {"audio_vae.conv.weight": audio},
    )
    assert video_baked["vae.conv.weight"].shape == (2, 2, 2, 2, 3)
    assert mx.array_equal(
        video_baked["vae.conv.weight"],
        mx.contiguous(video.transpose(0, 2, 3, 4, 1)),
    ).item()
    assert audio_baked["audio_vae.conv.weight"].shape == (2, 2, 2, 3)


def test_vocoder_layout_is_left_for_structure_aware_loader() -> None:
    weight = mx.arange(24).reshape(2, 3, 4)
    baked = bake_conv_layout_for_family(
        "vocoder",
        {"vocoder.conv.weight": weight},
    )
    assert mx.array_equal(baked["vocoder.conv.weight"], weight).item()


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
        layout_cache_key("transformer_blocks.3.ff.project_out.weight", **common)
        == "transformer_blocks.3.ff.project_out.weight_t"
    )
    assert (
        layout_cache_key("transformer_blocks.3.attn1.to_q.weight", **common)
        == "transformer_blocks.3.attn1.to_q.weight_t"
    )
    assert (
        layout_cache_key("transformer_blocks.3.audio_ff.project_in.proj.weight", **common)
        == "transformer_blocks.3.audio_ff.project_in.proj.weight_t"
    )
    assert (
        layout_cache_key("transformer_blocks.3.audio_attn1.to_out.weight", **common)
        == "transformer_blocks.3.audio_attn1.to_out.weight_t"
    )
    assert (
        layout_cache_key("adaln_single.linear.weight", **common) == "adaln_single.linear.weight_t"
    )
    assert layout_cache_key("transformer_blocks.2.ff.project_out.weight", **common) is None


def test_fp16_transformer_auto_adds_both_ff_pretransposes() -> None:
    existing = (("project_out", "pretranspose"),)
    normalized = ensure_ff_pretranspose_for_dtype(existing, mx.float16)
    assert normalized == (
        ("project_in", "pretranspose"),
        ("project_out", "pretranspose"),
    )
    assert ensure_ff_pretranspose_for_dtype(existing, mx.bfloat16) == existing
