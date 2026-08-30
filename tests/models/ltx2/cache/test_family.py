"""End-to-end fake-checkpoint tests for split cache build and load."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from kinomlx.lora.loading import LoRAConfig
from kinomlx.models.ltx2.cache import (
    DEFAULT_TRANSFORMER_LAYOUT_LAYERS,
    LAYOUT_KEY_PREFIX,
    QUANT_KEY_PREFIX,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
    TransformerBlockStreamer,
    ensure_transformer_cache,
    ensure_weight_family_caches,
    load_transformer_cache,
)
from kinomlx.models.ltx2.cache.lora import (
    fuse_community_loras,
    normalize_lora_for_cache,
)
from kinomlx.models.ltx2.cache.schema import (
    transformer_fp16_ranges_path,
    weight_family_cache_paths,
)
from kinomlx.models.ltx2.cache.storage import (
    cache_shard_path,
    existing_cache_shards,
    load_cache_weights,
    load_transformer_fp16_ranges,
)
from kinomlx.reporting import RecordingReporter


def _save_fake_bundle(
    path: Path,
    *,
    transformer_dtype: mx.Dtype = mx.bfloat16,
) -> None:
    mx.save_safetensors(
        str(path),
        {
            "vae.decoder.conv.weight": mx.arange(48).reshape(2, 3, 2, 2, 2),
            "vae.decoder.conv.bias": mx.zeros((2,)),
            "audio_vae.decoder.conv.weight": mx.arange(24).reshape(2, 3, 2, 2),
            "vocoder.generator.conv.weight": mx.arange(24).reshape(2, 3, 4),
            "text_embedding_projection.video.weight": mx.ones((2, 3)),
            "model.diffusion_model.video_embeddings_connector.registers": mx.ones((2, 3)),
            "model.diffusion_model.scale_shift_table": mx.ones((2, 4), dtype=mx.float32),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.arange(32)
            .reshape(4, 8)
            .astype(transformer_dtype),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.bias": mx.arange(4).astype(
                transformer_dtype
            ),
            "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight": mx.arange(16)
            .reshape(4, 4)
            .astype(transformer_dtype),
        },
    )


def _transformer_options() -> dict[str, object]:
    return {
        "cache_mode": "auto",
        "include_audio": False,
        "video_ff_layout_specs": (("project_out", "pretranspose"),),
        "video_ff_layout_layers": (0,),
        "video_attn_layout_specs": (),
        "video_attn_layout_layers": (),
    }


def test_family_caches_build_once_with_native_conv_layouts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bundle.safetensors"
    _save_fake_bundle(checkpoint)
    reporter = RecordingReporter()
    first = ensure_weight_family_caches(
        checkpoint,
        families=("connector", "video_vae", "audio_vae", "vocoder"),
        cache_mode="auto",
        cache_root=tmp_path / "cache",
        reporter=reporter,
    )
    assert first.rebuilt
    assert first.loaded_count == 6
    assert set(first.cache_paths) == {
        "connector",
        "video_vae",
        "audio_vae",
        "vocoder",
    }
    video = mx.load(str(first.cache_paths["video_vae"]))
    audio = mx.load(str(first.cache_paths["audio_vae"]))
    vocoder = mx.load(str(first.cache_paths["vocoder"]))
    assert video["vae.decoder.conv.weight"].shape == (2, 2, 2, 2, 3)
    assert audio["audio_vae.decoder.conv.weight"].shape == (2, 2, 2, 3)
    assert vocoder["vocoder.generator.conv.weight"].shape == (2, 3, 4)
    assert reporter.events[0][0:2] == ("start", "build weight family caches")
    assert reporter.events[-1][0:2] == ("end", "build weight family caches")

    second = ensure_weight_family_caches(
        checkpoint,
        families=("video_vae", "connector"),
        cache_mode="auto",
        cache_root=tmp_path / "cache",
    )
    assert not second.rebuilt
    assert second.loaded_count == 0
    assert second.cache_paths["video_vae"] == first.cache_paths["video_vae"]


def test_component_local_video_family_cache_routes_local_prefixes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "video-component.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "encoder.conv.weight": mx.arange(48).reshape(2, 3, 2, 2, 2),
            "decoder.conv.bias": mx.zeros((2,)),
            "per_channel_statistics.mean-of-means": mx.zeros((2,)),
            "community.training_payload": mx.ones((1,)),
        },
    )

    result = ensure_weight_family_caches(
        checkpoint,
        families=("video_vae",),
        source_component="video_vae",
        cache_mode="auto",
        cache_root=tmp_path / "cache",
    )

    cached = mx.load(str(result.cache_paths["video_vae"]))
    assert result.loaded_count == 3
    assert set(cached) == {
        "encoder.conv.weight",
        "decoder.conv.bias",
        "per_channel_statistics.mean-of-means",
    }
    assert cached["encoder.conv.weight"].shape == (2, 2, 2, 2, 3)


def test_family_schema_mismatch_rebuilds_only_requested_invalid_family(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bundle.safetensors"
    _save_fake_bundle(checkpoint)
    first = ensure_weight_family_caches(
        checkpoint,
        families=("video_vae", "connector"),
        cache_mode="auto",
        cache_root=tmp_path / "cache",
    )
    sidecar = first.cache_paths["video_vae"].with_suffix(".metadata.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    second = ensure_weight_family_caches(
        checkpoint,
        families=("video_vae", "connector"),
        cache_mode="auto",
        cache_root=tmp_path / "cache",
    )
    assert second.rebuilt
    assert second.loaded_count == 2
    repaired = json.loads(sidecar.read_text(encoding="utf-8"))
    assert repaired["schema_version"] == 3


def test_absent_requested_family_fails_instead_of_caching_empty_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "video-only.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {"vae.decoder.conv.bias": mx.zeros((2,))},
    )
    with pytest.raises(ValueError, match="no tensors.*audio_vae"):
        ensure_weight_family_caches(
            checkpoint,
            families=("audio_vae",),
            cache_mode="auto",
            cache_root=tmp_path / "cache",
        )


def test_absent_family_preflight_does_not_publish_earlier_family(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "video-only.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {"vae.decoder.conv.bias": mx.zeros((2,))},
    )
    cache_root = tmp_path / "cache"
    video_cache, video_metadata, _payload = weight_family_cache_paths(
        checkpoint,
        cache_root,
        "video_vae",
    )

    with pytest.raises(ValueError, match="no tensors.*audio_vae"):
        ensure_weight_family_caches(
            checkpoint,
            families=("video_vae", "audio_vae"),
            cache_mode="auto",
            cache_root=cache_root,
        )

    assert not video_cache.exists()
    assert not video_metadata.exists()


def test_default_transformer_layout_is_built_and_canonicalized(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bundle.safetensors"
    _save_fake_bundle(checkpoint)

    result = ensure_transformer_cache(
        checkpoint,
        cache_mode="auto",
        cache_root=tmp_path / "cache",
        include_audio=False,
    )
    cached = load_cache_weights(result.cache_path)
    metadata = json.loads((result.cache_path.parent / "metadata.json").read_text(encoding="utf-8"))

    assert f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight_t" in cached
    assert metadata["video_ff_layout_layers"] == list(DEFAULT_TRANSFORMER_LAYOUT_LAYERS)


def test_targeted_fp16_ff_builds_both_pretransposed_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "fp16-ff.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": (
                mx.eye(32, dtype=mx.bfloat16)
            ),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.eye(
                32, dtype=mx.bfloat16
            ),
        },
    )

    result = ensure_transformer_cache(
        checkpoint,
        cache_mode="auto",
        cache_root=tmp_path / "cache",
        include_audio=False,
        video_ff_dtype=mx.float16,
    )
    cached = load_cache_weights(result.cache_path)
    metadata = json.loads((result.cache_path.parent / "metadata.json").read_text(encoding="utf-8"))

    assert {
        f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_in.proj.weight_t",
        f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight_t",
    } <= set(cached)
    assert all(value.dtype == mx.float16 for value in cached.values())
    assert metadata["video_ff_layout_specs"] == [
        {"target": "project_in", "layout": "pretranspose"},
        {"target": "project_out", "layout": "pretranspose"},
    ]


def test_fp16_transformer_cache_bakes_layout_dtype_and_preserves_fp32(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bundle.safetensors"
    _save_fake_bundle(checkpoint)
    reporter = RecordingReporter()
    result = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        transformer_dtype="float16",
        reporter=reporter,
        shard_limit_bytes=64,
        **_transformer_options(),
    )
    assert result.rebuilt
    assert len(existing_cache_shards(result.cache_path)) >= 2
    cached = load_cache_weights(result.cache_path)
    layout_key = f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight_t"
    assert cached[layout_key].shape == (8, 4)
    assert cached[layout_key].dtype == mx.float16
    assert cached["transformer_blocks.0.ff.project_out.bias"].dtype == mx.float16
    assert cached["scale_shift_table"].dtype == mx.float32
    ranges = load_transformer_fp16_ranges(result.cache_path)
    assert ranges[layout_key] == 31.0
    assert ranges["transformer_blocks.0.ff.project_out.bias"] == 3.0
    assert ranges["transformer_blocks.0.attn1.to_out.weight"] == 15.0
    assert "scale_shift_table" not in ranges
    metadata = json.loads((result.cache_path.parent / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["transformer_dtype"] == "float16"
    assert metadata["video_ff_layout_specs"] == [
        {"target": "project_in", "layout": "pretranspose"},
        {"target": "project_out", "layout": "pretranspose"},
    ]
    assert reporter.events[0][0:2] == ("start", "build transformer cache")
    assert reporter.events[-1][0:2] == ("end", "build transformer cache")

    reused = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        transformer_dtype="float16",
        **_transformer_options(),
    )
    assert not reused.rebuilt
    assert reused.cache_path == result.cache_path

    transformer_fp16_ranges_path(result.cache_path).unlink()
    rebuilt_without_ranges = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        transformer_dtype="float16",
        **_transformer_options(),
    )
    assert rebuilt_without_ranges.rebuilt

    transformer_fp16_ranges_path(result.cache_path).write_text("not json", encoding="utf-8")
    rebuilt_with_invalid_ranges = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        transformer_dtype="float16",
        **_transformer_options(),
    )
    assert rebuilt_with_invalid_ranges.rebuilt


def test_native_fp16_cache_publishes_and_requires_realized_ranges(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "native-fp16.safetensors"
    _save_fake_bundle(checkpoint, transformer_dtype=mx.float16)
    result = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )

    ranges = load_transformer_fp16_ranges(result.cache_path)
    assert ranges["transformer_blocks.0.attn1.to_out.weight"] == 15.0
    transformer_fp16_ranges_path(result.cache_path).unlink()

    rebuilt = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    assert rebuilt.rebuilt
    assert transformer_fp16_ranges_path(rebuilt.cache_path).is_file()


def test_bf16_cache_without_range_sidecar_rebuilds_once(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bf16.safetensors"
    _save_fake_bundle(checkpoint)
    result = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    assert load_transformer_fp16_ranges(result.cache_path) == {}
    transformer_fp16_ranges_path(result.cache_path).unlink()

    rebuilt = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    assert rebuilt.rebuilt
    assert load_transformer_fp16_ranges(rebuilt.cache_path) == {}


def test_transformer_schema_mismatch_forces_clean_rebuild(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bundle.safetensors"
    _save_fake_bundle(checkpoint)
    first = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    metadata_file = first.cache_path.parent / "metadata.json"
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    metadata_file.write_text(json.dumps(payload), encoding="utf-8")
    stale = cache_shard_path(first.cache_path, 99)
    mx.save_safetensors(str(stale), {"stale": mx.ones((1,))})

    rebuilt = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    assert rebuilt.rebuilt
    assert not stale.exists()
    repaired = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert repaired["schema_version"] == 3


def test_failed_transformer_build_leaves_no_partial_shards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "no-transformer.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {"vae.decoder.conv.bias": mx.zeros((1,))},
    )
    with pytest.raises(ValueError, match="no compatible transformer"):
        ensure_transformer_cache(
            checkpoint,
            cache_root=tmp_path / "cache",
            **_transformer_options(),
        )
    assert not list((tmp_path / "cache").rglob("transformer-*.safetensors"))


def test_transformer_build_counts_only_skipped_diffusion_tensors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint = tmp_path / "mixed.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.scale_shift_table": mx.ones((2, 4)),
            "model.diffusion_model.audio_adaln_single.linear.weight": mx.ones((4, 4)),
            "vae.decoder.conv.bias": mx.zeros((1,)),
        },
    )
    with caplog.at_level(
        logging.INFO,
        logger="kinomlx.models.ltx2.cache.building",
    ):
        ensure_transformer_cache(
            checkpoint,
            cache_root=tmp_path / "cache",
            **_transformer_options(),
        )
    assert "1 quantized, 1 skipped" not in caplog.text
    assert "0 quantized, 1 skipped" in caplog.text


class _ProjectIn(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)


class _FeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.project_in = _ProjectIn(width)
        self.project_out = nn.Linear(width, width)
        self._project_in_weight_t = None
        self._project_out_weight_t = None


class _Block(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.ff = _FeedForward(width)
        self.idx = -1


class _Transformer(nn.Module):
    def __init__(self, blocks: int = 1, width: int = 32) -> None:
        super().__init__()
        self.transformer_blocks = [_Block(width) for _ in range(blocks)]


def test_cache_binders_require_graph_introspection_by_default(tmp_path: Path) -> None:
    cache_file = tmp_path / "partial-fixture.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {"transformer_blocks.0.ff.project_out.weight": mx.eye(32, dtype=mx.bfloat16)},
    )

    with pytest.raises(TypeError, match="expected_parameter_shapes"):
        load_transformer_cache(_Transformer(), cache_file)
    with pytest.raises(TypeError, match="expected_parameter_shapes"):
        TransformerBlockStreamer(cache_file, expected_model=_Transformer())


def test_targeted_quantized_cache_builds_and_loads_quantized_linear(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "quant.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.arange(32 * 32)
            .reshape(32, 32)
            .astype(mx.bfloat16)
            / 100,
            "model.diffusion_model.transformer_blocks.0.ff.net.2.bias": mx.zeros(
                (32,), dtype=mx.bfloat16
            ),
        },
    )
    result = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
        video_ff_quantize_layers=(0,),
        **_transformer_options(),
    )
    cached = load_cache_weights(result.cache_path)
    quant_base = f"{QUANT_KEY_PREFIX}transformer_blocks.0.ff.project_out"
    assert f"{quant_base}.weight" in cached
    assert f"{quant_base}.scales" in cached
    assert "transformer_blocks.0.ff.project_out.bias" in cached

    model = _Transformer()
    loaded, layout_count, quant_count = load_transformer_cache(
        model,
        result.cache_path,
        require_graph=False,
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
    )
    assert loaded == len(cached)
    assert layout_count == 0
    assert quant_count >= 2
    assert isinstance(model.transformer_blocks[0].ff.project_out, nn.QuantizedLinear)
    output = model.transformer_blocks[0].ff.project_out(mx.ones((1, 32), dtype=mx.bfloat16))
    mx.eval(output)
    assert output.shape == (1, 32)
    assert mx.all(mx.isfinite(output)).item()
    assert mx.any(output != 0).item()


def test_pretransposed_whole_block_quant_mode_uses_nontranspose_linear(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "quant.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.eye(
                32, dtype=mx.bfloat16
            ),
        },
    )
    result = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        transformer_cache_quantize=(TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE),
        **_transformer_options(),
    )
    model = _Transformer()
    load_transformer_cache(
        model,
        result.cache_path,
        require_graph=False,
        transformer_cache_quantize=(TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE),
    )
    linear = model.transformer_blocks[0].ff.project_out
    assert not isinstance(linear, nn.QuantizedLinear)
    assert linear.__class__.__name__ == "_PretransposedQuantizedLinear"
    value = mx.arange(32, dtype=mx.bfloat16).reshape(1, 32)
    assert mx.allclose(linear(value), value, rtol=0.15, atol=0.15).item()


def test_full_loader_restores_base_linear_after_quantized_cache(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "quant.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.eye(
                32, dtype=mx.bfloat16
            ),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.bias": mx.zeros(
                (32,), dtype=mx.bfloat16
            ),
        },
    )
    quantized = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
        video_ff_quantize_layers=(0,),
        **_transformer_options(),
    )
    unquantized = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    model = _Transformer()
    load_transformer_cache(
        model,
        quantized.cache_path,
        require_graph=False,
        video_ff_quantize_specs=(("project_out", "mxfp8"),),
    )
    assert isinstance(
        model.transformer_blocks[0].ff.project_out,
        nn.QuantizedLinear,
    )
    load_transformer_cache(model, unquantized.cache_path, require_graph=False)
    assert isinstance(model.transformer_blocks[0].ff.project_out, nn.Linear)


def test_full_loader_clears_stale_layout_before_normal_cache(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "layout.safetensors"
    source_weight = mx.arange(32 * 32).reshape(32, 32).astype(mx.bfloat16)
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": (source_weight),
            "model.diffusion_model.transformer_blocks.0.ff.net.2.bias": mx.zeros(
                (32,), dtype=mx.bfloat16
            ),
        },
    )
    layout = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **_transformer_options(),
    )
    normal_options = {
        **_transformer_options(),
        "video_ff_layout_specs": (),
        "video_ff_layout_layers": (),
    }
    normal = ensure_transformer_cache(
        checkpoint,
        cache_root=tmp_path / "cache",
        **normal_options,
    )
    model = _Transformer()

    load_transformer_cache(model, layout.cache_path, require_graph=False)
    feed_forward = model.transformer_blocks[0].ff
    assert feed_forward._project_out_weight_t is not None
    assert "weight" not in feed_forward.project_out

    load_transformer_cache(model, normal.cache_path, require_graph=False)
    assert feed_forward._project_out_weight_t is None
    assert mx.array_equal(feed_forward.project_out.weight, source_weight).item()


def test_quantized_cache_requires_weight_and_scales(tmp_path: Path) -> None:
    cache_file = tmp_path / "incomplete.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {
            f"{QUANT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight": (
                mx.zeros((32, 1), dtype=mx.uint32)
            )
        },
    )
    with pytest.raises(ValueError, match="missing scales"):
        load_transformer_cache(
            _Transformer(),
            cache_file,
            require_graph=False,
            video_ff_quantize_specs=(("project_out", "mxfp8"),),
        )


def test_full_loader_rejects_cache_missing_a_model_block(tmp_path: Path) -> None:
    cache_file = tmp_path / "partial.safetensors"
    mx.save_safetensors(
        str(cache_file),
        {
            "transformer_blocks.0.ff.project_out.weight": mx.eye(32, dtype=mx.bfloat16),
        },
    )

    with pytest.raises(ValueError, match="missing block weights for layer 1"):
        load_transformer_cache(
            _Transformer(blocks=2),
            cache_file,
            require_graph=False,
        )


def test_whole_block_and_targeted_quant_modes_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "quant.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {
            "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": mx.eye(
                32, dtype=mx.bfloat16
            )
        },
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        ensure_transformer_cache(
            checkpoint,
            cache_root=tmp_path / "cache",
            transformer_cache_quantize=(TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE),
            video_ff_quantize_specs=(("project_out", "mxfp8"),),
            **_transformer_options(),
        )
    with pytest.raises(ValueError, match="targeted FF dtype"):
        ensure_transformer_cache(
            checkpoint,
            cache_root=tmp_path / "cache",
            transformer_cache_quantize=(TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE),
            video_ff_dtype=mx.float16,
            **_transformer_options(),
        )


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([0.0])
        self.idx = -1


def test_block_streamer_reloads_evicted_weights_from_shards(tmp_path: Path) -> None:
    cache_file = tmp_path / "transformer.safetensors"
    mx.save_safetensors(
        str(cache_shard_path(cache_file, 0)),
        {"transformer_blocks.0.weight": mx.array([1.0])},
    )
    mx.save_safetensors(
        str(cache_shard_path(cache_file, 1)),
        {"transformer_blocks.1.weight": mx.array([2.0])},
    )
    streamer = TransformerBlockStreamer(cache_file)
    block_zero = streamer.bind(_TinyBlock(), 0, evict_block_idx=1)
    block_one = streamer.bind(_TinyBlock(), 1, evict_block_idx=0)
    assert float(block_zero.weight.item()) == 1.0
    assert float(block_one.weight.item()) == 2.0
    assert block_one.idx == 1
    streamer.close()


def test_block_streamer_clears_layout_before_normal_rebind(
    tmp_path: Path,
) -> None:
    source_weight = mx.arange(32 * 32).reshape(32, 32).astype(mx.bfloat16)
    layout_cache = tmp_path / "layout.safetensors"
    normal_cache = tmp_path / "normal.safetensors"
    mx.save_safetensors(
        str(layout_cache),
        {
            f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight_t": (
                mx.contiguous(source_weight.T)
            )
        },
    )
    mx.save_safetensors(
        str(normal_cache),
        {"transformer_blocks.0.ff.project_out.weight": source_weight},
    )
    block = _Block()
    layout_streamer = TransformerBlockStreamer(layout_cache)
    normal_streamer = TransformerBlockStreamer(normal_cache)

    layout_streamer.bind(block, 0)
    assert block.ff._project_out_weight_t is not None
    assert "weight" not in block.ff.project_out

    normal_streamer.bind(block, 0)
    assert block.ff._project_out_weight_t is None
    assert mx.array_equal(block.ff.project_out.weight, source_weight).item()

    layout_streamer.close()
    normal_streamer.close()


def test_community_lora_fuses_into_pretransposed_cache_slot(
    tmp_path: Path,
) -> None:
    cache_key = f"{LAYOUT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight_t"
    base = {cache_key: mx.zeros((3, 2), dtype=mx.float16)}
    adapter_path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(adapter_path),
        {
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.array(
                [[1.0, 2.0, 3.0]]
            ),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.array([[4.0], [5.0]]),
        },
    )
    fused = fuse_community_loras(
        base,
        [LoRAConfig(adapter_path, strength=1.0)],
    )
    expected = mx.array([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]])
    assert fused is base
    assert mx.array_equal(fused[cache_key], expected.astype(mx.float16)).item()


def test_lora_mapping_rejects_quantized_cache_target() -> None:
    quant_key = f"{QUANT_KEY_PREFIX}transformer_blocks.0.ff.project_out.weight"
    base = {quant_key: mx.zeros((1,), dtype=mx.uint32)}
    adapter = {
        "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight": mx.ones((1, 2)),
        "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight": mx.ones((2, 1)),
    }
    with pytest.raises(RuntimeError, match="quantized cache"):
        normalize_lora_for_cache(base, adapter)
