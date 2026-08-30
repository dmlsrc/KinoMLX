from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import kinomlx.models.ltx2.cache.transformer as transformer_cache_module
from kinomlx.models.ltx2.cache import (
    LAYOUT_KEY_PREFIX,
    TransformerBlockStreamer,
    TransformerCacheResult,
    load_transformer_cache,
    load_transformer_weights_cached,
    load_transformer_weights_cached_streaming,
)
from kinomlx.models.ltx2.cache.keys import flatten_to_nested
from kinomlx.models.ltx2.transformer import LTXAVModel, Modality, X0Model
from kinomlx.models.ltx2.transformer import model as model_module

from ._synthetic import build_shaped_ltx_model


def _small_model(dtype: mx.Dtype, *, layers: int = 1, shaped: bool = True) -> LTXAVModel:
    def factory() -> LTXAVModel:
        return LTXAVModel(
            num_layers=layers,
            video_heads=2,
            video_head_dim=4,
            audio_heads=2,
            audio_head_dim=2,
            video_in_channels=3,
            video_out_channels=3,
            audio_in_channels=2,
            audio_out_channels=2,
            video_context_dim=8,
            audio_context_dim=4,
            video_max_pos=(8, 8, 8),
            audio_max_pos=(8,),
            compute_dtype=dtype,
            double_precision_rope=True,
        )

    return build_shaped_ltx_model(factory) if shaped else factory()


def _modalities(tokens: int = 2) -> tuple[Modality, Modality]:
    video_positions = mx.stack(
        (
            mx.stack((mx.arange(tokens), mx.arange(1, tokens + 1)), axis=-1),
            mx.stack((mx.arange(tokens), mx.arange(1, tokens + 1)), axis=-1),
            mx.stack((mx.arange(tokens), mx.arange(1, tokens + 1)), axis=-1),
        ),
        axis=0,
    )[None, ...]
    audio_positions = mx.stack(
        (mx.arange(tokens), mx.arange(1, tokens + 1)),
        axis=-1,
    )[None, None, ...]
    video = Modality(
        latent=mx.arange(tokens * 3).reshape(1, tokens, 3).astype(mx.float32) / 10,
        context=mx.arange(16).reshape(1, 2, 8).astype(mx.float32) / 20,
        timesteps=mx.full((1, tokens), 0.5),
        sigma=mx.array([0.5]),
        positions=video_positions,
        context_mask=mx.ones((1, 2), dtype=mx.int32),
    )
    audio = Modality(
        latent=mx.arange(tokens * 2).reshape(1, tokens, 2).astype(mx.float32) / 10,
        context=mx.arange(8).reshape(1, 2, 4).astype(mx.float32) / 20,
        timesteps=mx.full((1, tokens), 0.5),
        sigma=mx.array([0.5]),
        positions=audio_positions,
        context_mask=mx.ones((1, 2), dtype=mx.int32),
    )
    return video, audio


def test_default_transformer_uses_allocation_light_48_block_shells() -> None:
    model = LTXAVModel()
    arrays = [value for _key, value in tree_flatten(model.parameters())]
    assert len(model.transformer_blocks) == 48
    assert sum(value.nbytes for value in arrays) < 100_000
    assert tuple(model.patchify_proj.weight.shape) == (0, 0)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_joint_forward_preserves_selected_transformer_dtype(
    dtype: mx.Dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sdpa = mx.fast.scaled_dot_product_attention
    sdpa_dtypes: list[tuple[mx.Dtype, mx.Dtype, mx.Dtype, mx.Dtype | None]] = []

    def recording_sdpa(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        *,
        scale: float,
        mask: mx.array | None = None,
    ) -> mx.array:
        sdpa_dtypes.append(
            (
                query.dtype,
                key.dtype,
                value.dtype,
                None if mask is None else mask.dtype,
            )
        )
        return original_sdpa(query, key, value, scale=scale, mask=mask)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", recording_sdpa)
    mx.random.seed(7)
    model = _small_model(dtype)
    video, audio = _modalities()
    video_output, audio_output = model(video, audio)
    mx.eval(video_output, audio_output)
    assert tuple(video_output.shape) == tuple(video.latent.shape)
    assert tuple(audio_output.shape) == tuple(audio.latent.shape)
    assert video_output.dtype == dtype
    assert audio_output.dtype == dtype
    assert model.patchify_proj.weight.dtype == dtype
    assert model.transformer_blocks[0].scale_shift_table.dtype == mx.float32
    prepared = model._video_preprocessor.prepare(model._cast_modality(video), audio)
    assert prepared.positional_embeddings[0].dtype == mx.float32
    assert prepared.cross_positional_embeddings[0].dtype == mx.float32
    assert sdpa_dtypes
    assert any(mask_dtype is not None for *_dtypes, mask_dtype in sdpa_dtypes)
    assert all(
        query_dtype == key_dtype == value_dtype == dtype
        and (mask_dtype is None or mask_dtype == dtype)
        for query_dtype, key_dtype, value_dtype, mask_dtype in sdpa_dtypes
    )
    assert mx.all(mx.isfinite(video_output)).item()
    assert mx.all(mx.isfinite(audio_output)).item()


def test_audio_only_forward_does_not_require_video() -> None:
    mx.random.seed(9)
    model = _small_model(mx.bfloat16)
    _video, audio = _modalities()
    video_output, audio_output = model(None, audio)
    mx.eval(audio_output)
    assert video_output is None
    assert tuple(audio_output.shape) == tuple(audio.latent.shape)
    assert audio_output.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(audio_output)).item()


def test_disabled_modality_preserves_reference_cross_attention_directions() -> None:
    class RecordingAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def __call__(
            self,
            query: mx.array,
            *,
            context: mx.array,
            pe: tuple[mx.array, mx.array],
            k_pe: tuple[mx.array, mx.array],
        ) -> mx.array:
            del context, pe, k_pe
            self.calls += 1
            return mx.zeros_like(query)

    model = _small_model(mx.float32)
    block = model.transformer_blocks[0]
    audio_to_video = RecordingAttention()
    video_to_audio = RecordingAttention()
    block.audio_to_video_attn = audio_to_video
    block.video_to_audio_attn = video_to_audio
    video, audio = _modalities()

    disabled_video, active_audio = model(replace(video, enabled=False), audio)
    mx.eval(disabled_video, active_audio)
    assert disabled_video is not None
    assert active_audio is not None
    assert audio_to_video.calls == 0
    assert video_to_audio.calls == 1

    active_video, disabled_audio = model(video, replace(audio, enabled=False))
    mx.eval(active_video, disabled_audio)
    assert active_video is not None
    assert disabled_audio is not None
    assert audio_to_video.calls == 1
    assert video_to_audio.calls == 1


def test_compiled_block_groups_match_eager_forward() -> None:
    mx.random.seed(13)
    model = _small_model(mx.float32, layers=2)
    video, audio = _modalities()
    expected = model(video, audio)
    mx.eval(*expected)

    model.compile_block_groups = 1
    actual = model(video, audio)
    mx.eval(*actual)

    assert mx.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5).item()
    compiled = object.__getattribute__(model, "_compiled_transformer_block_groups")
    assert len(compiled) == 2


def test_compiled_block_groups_keep_low_memory_eval_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model(mx.float32, layers=2)
    model.compile_block_groups = 1
    model._eval_frequency = 8
    evaluations = []
    monkeypatch.setattr(
        LTXAVModel,
        "_eval_args",
        lambda self, video, audio: evaluations.append((video, audio)),
    )
    video, audio = _modalities()

    model(video, audio)

    assert len(evaluations) == 2


def test_compiled_block_groups_preserve_disabled_modality_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model(mx.float32)
    model.compile_block_groups = 1
    monkeypatch.setattr(
        model_module,
        "_compile_transformer_block_group",
        lambda blocks: pytest.fail("disabled modalities must bypass compilation"),
    )
    video, audio = _modalities()

    video_output, audio_output = model(replace(video, enabled=False), audio)
    mx.eval(video_output, audio_output)

    assert video_output is not None
    assert audio_output is not None


def test_cross_scale_shift_uses_own_timesteps_and_gate_uses_other_sigma() -> None:
    mx.random.seed(11)
    model = _small_model(mx.float32)
    video, audio = _modalities()
    baseline = model._video_preprocessor.prepare(video, audio)
    own_timestep_changed = model._video_preprocessor.prepare(
        replace(video, timesteps=mx.full(video.timesteps.shape, 0.25)),
        audio,
    )
    other_sigma_changed = model._video_preprocessor.prepare(
        video,
        replace(audio, sigma=mx.array([0.25])),
    )
    assert mx.any(
        baseline.cross_scale_shift_timestep != own_timestep_changed.cross_scale_shift_timestep
    ).item()
    assert mx.array_equal(
        baseline.cross_gate_timestep,
        own_timestep_changed.cross_gate_timestep,
    ).item()
    assert mx.array_equal(
        baseline.cross_scale_shift_timestep,
        other_sigma_changed.cross_scale_shift_timestep,
    ).item()
    assert mx.any(baseline.cross_gate_timestep != other_sigma_changed.cross_gate_timestep).item()


def test_model_parameter_tree_round_trips_through_transformer_cache(tmp_path) -> None:
    mx.random.seed(19)
    source = _small_model(mx.float32, layers=2)
    cache = tmp_path / "transformer.safetensors"
    weights = dict(tree_flatten(source.parameters()))
    mx.save_safetensors(str(cache), weights)
    target = _small_model(mx.float32, layers=2, shaped=False)
    loaded, layouts, quantized = load_transformer_cache(target, cache)
    assert (loaded, layouts, quantized) == (len(weights), 0, 0)
    video, audio = _modalities()
    expected = source(video, audio)
    actual = target(video, audio)
    mx.eval(*expected, *actual)
    assert mx.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5).item()


@pytest.mark.parametrize("compile_group_size", [None, 1])
def test_bounded_streaming_rotates_one_slot_and_reloads_for_next_forward(
    tmp_path,
    compile_group_size: int | None,
) -> None:
    mx.random.seed(23)
    source = _small_model(mx.float32, layers=4)
    weights = dict(tree_flatten(source.parameters()))
    cache = tmp_path / "streaming-transformer.safetensors"
    mx.save_safetensors(str(cache), weights)

    target = _small_model(mx.float32, layers=4, shaped=False)
    non_block = {
        key: value for key, value in weights.items() if not key.startswith("transformer_blocks.")
    }
    target.update(flatten_to_nested(non_block))
    target.transformer_blocks = target.transformer_blocks[:1]
    target.transformer_compile_group_size = compile_group_size
    streamer = TransformerBlockStreamer(cache)
    target.transformer_block_streamer = streamer
    bound: list[tuple[int, int | None]] = []
    original_bind = streamer.bind

    def recording_bind(
        block: nn.Module,
        block_idx: int,
        *,
        evict_block_idx: int | None = None,
    ) -> nn.Module:
        bound.append((block_idx, evict_block_idx))
        return original_bind(
            block,
            block_idx,
            evict_block_idx=evict_block_idx,
        )

    streamer.bind = recording_bind
    video, audio = _modalities()
    expected = source(video, audio)
    mx.eval(*expected)
    try:
        first = target(video, audio)
        mx.eval(*first)
        second = target(video, audio)
        mx.eval(*second)
        assert target.num_blocks == 4
        assert X0Model(target).num_blocks == 4
        assert bound == [
            (0, None),
            (1, 0),
            (2, 1),
            (3, 2),
            (0, None),
            (1, 0),
            (2, 1),
            (3, 2),
        ]
        for actual in (first, second):
            assert mx.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5).item()
            assert mx.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5).item()
    finally:
        target.close_streamer()


@pytest.mark.parametrize("compile_group_size", [None, 1, 2])
def test_heterogeneous_layout_streaming_matches_eager_across_window_rebind(
    tmp_path,
    compile_group_size: int | None,
) -> None:
    """A structural slot swap must retrace within the compiled FP32 tolerance."""
    mx.random.seed(29)
    source = _small_model(mx.float32, layers=4)
    weights = dict(tree_flatten(source.parameters()))
    cache_weights = dict(weights)
    # resident=2 creates windows [0, 1] and [2, 3]. Both slots change
    # structure at the boundary: normal->pretransposed and the reverse.
    for block_index in (1, 2):
        weight_key = f"transformer_blocks.{block_index}.ff.project_out.weight"
        weight = cache_weights.pop(weight_key)
        layout_key = f"{LAYOUT_KEY_PREFIX}transformer_blocks.{block_index}.ff.project_out.weight_t"
        cache_weights[layout_key] = mx.contiguous(weight.T)
    cache = tmp_path / "heterogeneous-streaming-transformer.safetensors"
    mx.save_safetensors(str(cache), cache_weights)

    target = _small_model(mx.float32, layers=4, shaped=False)
    non_block = {
        key: value
        for key, value in cache_weights.items()
        if not key.startswith(("transformer_blocks.", LAYOUT_KEY_PREFIX))
    }
    target.update(flatten_to_nested(non_block))
    target.transformer_blocks = target.transformer_blocks[:2]
    target.transformer_compile_group_size = compile_group_size
    target.transformer_block_streamer = TransformerBlockStreamer(cache)

    video, audio = _modalities()
    expected = source(video, audio)
    mx.eval(*expected)
    try:
        actual = target(video, audio)
        mx.eval(*actual)
        for actual_stream, expected_stream in zip(actual, expected, strict=True):
            if compile_group_size is None:
                assert mx.array_equal(actual_stream, expected_stream).item()
                continue
            delta = (actual_stream - expected_stream).astype(mx.float32)
            expected_norm = float(mx.linalg.norm(expected_stream.astype(mx.float32)).item())
            max_abs = float(mx.max(mx.abs(delta)).item())
            relative_l2 = float(mx.linalg.norm(delta).item()) / expected_norm
            # mx.compile changes only the final FP32 accumulation order here.
            assert max_abs <= 2**-23
            assert relative_l2 <= 1e-7
        if compile_group_size is not None:
            compiled = object.__getattribute__(target, "_compiled_transformer_block_groups")
            assert compiled
            assert not object.__getattribute__(
                target,
                "_transformer_block_compile_disabled",
            )
    finally:
        target.close_streamer()


@pytest.mark.parametrize("actual_layers", [3, 5])
def test_streaming_loader_rejects_wrong_logical_depth_before_model_truncation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    actual_layers: int,
) -> None:
    source = _small_model(mx.float32, layers=4)
    weights = dict(tree_flatten(source.parameters()))
    if actual_layers == 3:
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("transformer_blocks.3.")
        }
    else:
        for key, value in list(weights.items()):
            if key.startswith("transformer_blocks.3."):
                suffix = key.removeprefix("transformer_blocks.3.")
                weights[f"transformer_blocks.4.{suffix}"] = value
    cache = tmp_path / "wrong-depth.safetensors"
    mx.save_safetensors(str(cache), weights)
    monkeypatch.setattr(
        transformer_cache_module,
        "ensure_transformer_cache",
        lambda *_args, **_kwargs: TransformerCacheResult(cache, False, 0, 0),
    )
    target = _small_model(mx.float32, layers=4, shaped=False)

    with pytest.raises(
        ValueError,
        match="parameter graph mismatch",
    ):
        load_transformer_weights_cached_streaming(
            target,
            tmp_path / "unused.safetensors",
            cache_mode="auto",
            cache_root=tmp_path / "cache",
            include_audio=True,
            resident_blocks=1,
            transformer_dtype=mx.float32,
        )

    assert len(target.transformer_blocks) == 4
    assert target.transformer_block_streamer is None


def test_streaming_loader_rejects_missing_later_block_tensor_before_mutation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _small_model(mx.float32, layers=4)
    weights = dict(tree_flatten(source.parameters()))
    missing_key = "transformer_blocks.3.ff.project_out.bias"
    weights.pop(missing_key)
    cache = tmp_path / "missing-tensor.safetensors"
    mx.save_safetensors(str(cache), weights)
    monkeypatch.setattr(
        transformer_cache_module,
        "ensure_transformer_cache",
        lambda *_args, **_kwargs: TransformerCacheResult(cache, False, 0, 0),
    )
    target = _small_model(mx.float32, layers=4, shaped=False)

    with pytest.raises(
        ValueError,
        match="parameter graph mismatch.*transformer_blocks.3.ff.project_out.bias",
    ):
        load_transformer_weights_cached_streaming(
            target,
            tmp_path / "unused.safetensors",
            cache_mode="auto",
            cache_root=tmp_path / "cache",
            include_audio=True,
            resident_blocks=1,
            transformer_dtype=mx.float32,
        )

    assert len(target.transformer_blocks) == 4
    assert target.transformer_block_streamer is None


def test_streaming_loader_installs_top_level_adaln_layout_and_matches_eager(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx.random.seed(31)
    source = _small_model(mx.float32, layers=2)
    weights = dict(tree_flatten(source.parameters()))
    adaln_key = "adaln_single.linear.weight"
    adaln_weight = weights.pop(adaln_key)
    layout_key = f"{LAYOUT_KEY_PREFIX}adaln_single.linear.weight_t"
    weights[layout_key] = mx.contiguous(adaln_weight.T)
    cache = tmp_path / "adaln-layout.safetensors"
    mx.save_safetensors(str(cache), weights)
    monkeypatch.setattr(
        transformer_cache_module,
        "ensure_transformer_cache",
        lambda *_args, **_kwargs: TransformerCacheResult(cache, False, 0, 0),
    )
    target = _small_model(mx.float32, layers=2, shaped=False)

    load_transformer_weights_cached_streaming(
        target,
        tmp_path / "unused.safetensors",
        cache_mode="auto",
        cache_root=tmp_path / "cache",
        include_audio=True,
        resident_blocks=1,
        transformer_dtype=mx.float32,
    )
    video, audio = _modalities()
    expected = source(video, audio)
    try:
        actual = target(video, audio)
        mx.eval(*expected, *actual)
        assert target.adaln_single._linear_weight_t is not None
        assert mx.array_equal(target.adaln_single._linear_weight_t, adaln_weight.T).item()
        assert mx.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5).item()
        assert mx.allclose(actual[1], expected[1], rtol=1e-5, atol=1e-5).item()
    finally:
        target.close_streamer()


def test_cached_load_rejects_model_compute_dtype_mismatch(tmp_path) -> None:
    model = _small_model(mx.bfloat16)
    options = {
        "transformer_dtype": mx.float16,
        "cache_mode": "auto",
        "cache_root": tmp_path / "cache",
        "include_audio": True,
    }
    with pytest.raises(ValueError, match="model compute dtype"):
        load_transformer_weights_cached(
            model,
            tmp_path / "missing.safetensors",
            **options,
        )
    with pytest.raises(ValueError, match="model compute dtype"):
        load_transformer_weights_cached_streaming(
            model,
            tmp_path / "missing.safetensors",
            resident_blocks=1,
            **options,
        )


def test_x0_model_uses_per_token_timesteps_not_scalar_sigma() -> None:
    class Velocity(nn.Module):
        num_blocks = 7

        def __call__(self, video: Modality, audio: Modality | None):
            return mx.full(video.latent.shape, -2.453125, dtype=mx.bfloat16), None

    video, _audio = _modalities(tokens=2)
    video = replace(
        video,
        latent=mx.full(video.latent.shape, -1.640625, dtype=mx.bfloat16),
        timesteps=mx.full(video.timesteps.shape, 0.090820007, dtype=mx.float32),
        sigma=mx.array([0.9]),
    )
    model = X0Model(Velocity())
    denoised, audio = model(video)
    mx.eval(denoised)
    assert model.num_blocks == 7
    assert audio is None
    assert denoised.dtype == mx.bfloat16
    assert mx.array_equal(
        denoised,
        mx.full(video.latent.shape, -1.4140625, dtype=mx.bfloat16),
    ).item()


def test_x0_model_keeps_lora_receipts_out_of_module_state() -> None:
    class Velocity(nn.Module):
        def __call__(self, video: Modality, audio: Modality | None):
            return video.latent, None

    model = X0Model(Velocity())

    assert model.lora_receipts == ()
    assert "lora_receipts" not in model
