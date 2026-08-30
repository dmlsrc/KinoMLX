"""LTX-2.5 diffusion video decoder graph and binding tests."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx.utils import tree_flatten

from kinomlx.models.ltx2.video_vae.config import (
    DiffusionVideoDecoderConfig,
    VideoVAEBlock,
    VideoVAEConfig,
)
from kinomlx.models.ltx2.video_vae.diffusion_decoder import (
    NativeDiffusionVideoDecoder,
    NeighborhoodAttention3D,
    load_diffusion_video_decoder_weights,
    patchify_spatial,
    plan_attention_tiles,
    unpatchify_spatial,
)
from kinomlx.models.ltx2.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalChunkConfig,
    TilingConfig,
    decode_streaming,
)


def _res() -> VideoVAEBlock:
    return VideoVAEBlock("res_x", num_layers=1)


def _compress(name: str, multiplier: int) -> VideoVAEBlock:
    return VideoVAEBlock(name, multiplier=multiplier)


def _mini_config() -> VideoVAEConfig:
    diffusion = DiffusionVideoDecoderConfig(
        head_dim=8,
        stage_channels=(64, 32, 16, 16, 8),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((3, 3, 3),) * 4,
        upsample_strides=((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        upsample_channel_reductions=(2, 2, 1, 2),
        stage5_kernel=(3, 3, 3),
        patch_size=2,
        t_emb_dim=32,
        spatial_compression_ratio=16,
        temporal_compression_ratio=8,
    )
    return VideoVAEConfig(
        encoder_blocks=(
            _res(),
            _compress("compress_space_res", 2),
            _compress("compress_time_res", 2),
            _compress("compress_all_res", 2),
            _compress("compress_all_res", 1),
        ),
        decoder_blocks=(),
        encoder_base_channels=2,
        decoder_base_channels=8,
        latent_channels=8,
        patch_size=2,
        timestep_conditioning=True,
        decoder_kind="diffusion-na",
        diffusion_decoder=diffusion,
        latent_log_var="constant",
        latent_log_var_value=-1.0,
    )


def _nested_mapping() -> dict[str, object]:
    config = _mini_config()
    diffusion = config.diffusion_decoder
    assert diffusion is not None
    return {
        "_class_name": "CausalDiffusionVAE",
        "encoder": {
            "_class_name": "Encoder",
            "dims": 3,
            "in_channels": 3,
            "out_channels": 8,
            "blocks": [
                ["res_x", {"num_layers": 1}],
                ["compress_space_res", {"multiplier": 2}],
                ["compress_time_res", {"multiplier": 2}],
                ["compress_all_res", {"multiplier": 2}],
                ["compress_all_res", {"multiplier": 1}],
            ],
            "patch_size": 2,
            "latent_log_var": "constant",
            "latent_log_var_value": -1.0,
            "norm_layer": "pixel_norm",
            "base_channels": 2,
            "spatial_padding_mode": "zeros",
        },
        "decoder": {
            "_class_name": "NADiffusionDecoder",
            "in_channels": 8,
            "out_channels": 3,
            "patch_size": 2,
            "head_dim": diffusion.head_dim,
            "stage_channels": list(diffusion.stage_channels),
            "stage_depths": list(diffusion.stage_depths),
            "stage_kernels": [
                *[list(kernel) for kernel in diffusion.stage_kernels],
                list(diffusion.stage5_kernel),
            ],
            "upsamples": [
                [list(stride), reduction]
                for stride, reduction in zip(
                    diffusion.upsample_strides,
                    diffusion.upsample_channel_reductions,
                    strict=True,
                )
            ],
            "stage5_kernel": list(diffusion.stage5_kernel),
            "t_emb_dim": diffusion.t_emb_dim,
            "timestep_scale_multiplier": 1000.0,
            "default_num_inference_steps": 1,
        },
        "model_output_type": "x0",
    }


def test_nested_metadata_selects_the_diffusion_graph() -> None:
    config = VideoVAEConfig.from_mapping(_nested_mapping())

    assert config.decoder_kind == "diffusion-na"
    assert config.decoder_blocks == ()
    assert tuple(config.encoder_scale) == (8, 16, 16)
    assert config.decoder_scale == config.encoder_scale
    assert config.diffusion_decoder is not None
    assert config.diffusion_decoder.stage5_kernel == (3, 3, 3)
    assert config.inferred_fields == ("vae.signal_domain",)
    assert config.diffusion_decoder.inferred_fields == ()


def test_patchify_round_trip_preserves_ltx_packing() -> None:
    source = mx.arange(1 * 2 * 3 * 4 * 6).reshape(1, 2, 3, 4, 6)
    packed = patchify_spatial(source, 2)
    restored = unpatchify_spatial(packed, 2)

    assert tuple(packed.shape) == (1, 8, 3, 2, 3)
    assert mx.array_equal(source, restored).item()


def test_attention_planner_bounds_total_head_score_work() -> None:
    plan = plan_attention_tiles(
        (17, 64, 96),
        (11, 11, 11),
        4,
        score_budget=2_000_000,
    )

    assert plan.tile_count > 1
    assert plan.max_score_elements <= 2_000_000
    assert all(tile <= length for tile, length in zip(plan.query_tile, plan.grid, strict=True))


def _brute_attention(module: NeighborhoodAttention3D, value: mx.array) -> mx.array:
    batch, frames, height, width, channels = value.shape
    qkv = module.qkv(value)
    query, key, projected_value = mx.split(qkv, 3, axis=-1)
    query = query.reshape(batch, frames, height, width, module.heads, module.head_dim)
    key = key.reshape(batch, frames, height, width, module.heads, module.head_dim)
    projected_value = projected_value.reshape(
        batch,
        frames,
        height,
        width,
        module.heads,
        module.head_dim,
    )
    query = module.rope(
        module.q_norm(query) * module.head_dim**-0.5,
        offsets=(0, 0, 0),
    )
    key = module.rope(module.k_norm(key), offsets=(0, 0, 0))
    kernel_t, kernel_h, kernel_w = module.kernel_size
    outputs = []
    for query_t in range(frames):
        start_t = min(max(query_t - kernel_t // 2, 0), frames - kernel_t)
        for query_h in range(height):
            start_h = min(max(query_h - kernel_h // 2, 0), height - kernel_h)
            for query_w in range(width):
                start_w = min(max(query_w - kernel_w // 2, 0), width - kernel_w)
                q = query[:, query_t, query_h, query_w].transpose(0, 1, 2)
                k = key[
                    :,
                    start_t : start_t + kernel_t,
                    start_h : start_h + kernel_h,
                    start_w : start_w + kernel_w,
                ].reshape(batch, -1, module.heads, module.head_dim)
                v = projected_value[
                    :,
                    start_t : start_t + kernel_t,
                    start_h : start_h + kernel_h,
                    start_w : start_w + kernel_w,
                ].reshape(batch, -1, module.heads, module.head_dim)
                scores = mx.einsum("bhd,bkhd->bhk", q, k)
                weights = mx.softmax(scores, axis=-1)
                output = mx.einsum("bhk,bkhd->bhd", weights, v)
                outputs.append(output.reshape(batch, channels))
    joined = mx.stack(outputs, axis=1).reshape(batch, frames, height, width, channels)
    return module.proj(joined)


def test_bounded_attention_matches_brute_inward_windows() -> None:
    mx.random.seed(17)
    stats = type("Stats", (), {"record": lambda self, plan: None})()
    module = NeighborhoodAttention3D(
        8,
        (3, 3, 3),
        head_dim=8,
        stats=stats,
        score_budget=1_000,
    )
    value = mx.random.normal((1, 3, 4, 4, 8))
    expected = _brute_attention(module, value)
    actual = module(value)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()


def test_full_decoder_is_seeded_and_single_covering_tile_is_exact() -> None:
    mx.random.seed(19)
    decoder = NativeDiffusionVideoDecoder(
        _mini_config(),
        compute_dtype=mx.float32,
        attention_score_budget=500_000,
    )
    latent = mx.random.normal((1, 8, 2, 3, 3))
    first = decoder(latent, seed=42)
    second = decoder(latent, seed=42)
    different = decoder(latent, seed=43)
    tiled = tuple(
        decode_streaming(
            latent,
            decoder,
            TilingConfig(
                SpatialTilingConfig(64, 32),
                TemporalChunkConfig(32, 8),
            ),
            seed=42,
        )
    )
    mx.eval(first, second, different, *tiled)

    assert tuple(first.shape) == (1, 3, 9, 48, 48)
    assert mx.array_equal(first, second).item()
    assert not mx.array_equal(first, different).item()
    assert len(tiled) == 1
    assert mx.array_equal(first, tiled[0]).item()


def test_short_and_narrow_latent_is_padded_internally_then_cropped() -> None:
    mx.random.seed(21)
    decoder = NativeDiffusionVideoDecoder(
        _mini_config(),
        compute_dtype=mx.float32,
        attention_score_budget=500_000,
    )
    latent = mx.random.normal((1, 8, 2, 1, 1))
    plans = []

    direct = decoder(latent, seed=42)
    tiled = tuple(
        decode_streaming(
            latent,
            decoder,
            TilingConfig(
                SpatialTilingConfig(64, 32),
                TemporalChunkConfig(32, 8),
            ),
            seed=42,
            plan_callback=plans.append,
        )
    )
    mx.eval(direct, *tiled)

    assert decoder.minimum_latent_shape == (3, 3, 3)
    assert tuple(direct.shape) == (1, 3, 9, 16, 16)
    assert len(tiled) == 1
    assert mx.array_equal(direct, tiled[0]).item()
    assert plans[-1].latent_shape == (2, 1, 1)
    assert plans[-1].decoded_shape == (9, 16, 16)


def test_actual_temporal_and_spatial_split_is_bounded_and_complete() -> None:
    mx.random.seed(23)
    decoder = NativeDiffusionVideoDecoder(
        _mini_config(),
        compute_dtype=mx.float32,
        attention_score_budget=500_000,
    )
    latent = mx.random.normal((1, 8, 3, 5, 5))
    plans = []
    chunks = tuple(
        decode_streaming(
            latent,
            decoder,
            TilingConfig(
                SpatialTilingConfig(64, 32),
                TemporalChunkConfig(16, 8),
            ),
            seed=7,
            plan_callback=plans.append,
        )
    )
    output = mx.concatenate(chunks, axis=2)
    mx.eval(output)

    assert tuple(output.shape) == (1, 3, 17, 80, 80)
    assert mx.all(mx.isfinite(output)).item()
    assert len(chunks) == 2
    assert plans[-1].total_tiles == 18
    assert plans[-1].to_dict()["decoder_kind"] == "diffusion-na"
    assert plans[-1].attention_tiling["max_score_elements"] <= 500_000


def _checkpoint_key(parameter_key: str) -> str:
    replacements = {
        "t_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
        "t_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
        "t_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
        "t_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    }
    return replacements.get(parameter_key, parameter_key)


def test_loader_requires_every_target_folds_gates_and_receipts_baggage() -> None:
    mx.random.seed(29)
    source = NativeDiffusionVideoDecoder(_mini_config(), compute_dtype=mx.float32)
    weights = {
        f"decoder.{_checkpoint_key(key)}": value
        for key, value in tree_flatten(source.parameters())
        if not key.startswith("per_channel_statistics.")
    }
    weights["per_channel_statistics.mean-of-means"] = mx.arange(8, dtype=mx.float32)
    weights["per_channel_statistics.std-of-means"] = mx.arange(1, 9, dtype=mx.float32)
    gate = mx.arange(1, 65, dtype=mx.float32) / 64
    weights["decoder.det_stages.0.0.gate_msa"] = gate
    weights["decoder.type_emb"] = mx.zeros((8,))

    target = NativeDiffusionVideoDecoder(_mini_config(), compute_dtype=mx.float32)
    receipt = load_diffusion_video_decoder_weights(target, weights)
    expected_weight = source.det_stages[0][0].attn.proj.weight * gate[:, None]
    expected_bias = source.det_stages[0][0].attn.proj.bias * gate
    mx.eval(expected_weight, expected_bias, target.parameters())

    assert receipt.folded_gate_tensors == 1
    assert receipt.ignored_decoder_tensors == ("decoder.type_emb",)
    assert mx.allclose(target.det_stages[0][0].attn.proj.weight, expected_weight).item()
    assert mx.allclose(target.det_stages[0][0].attn.proj.bias, expected_bias).item()
    assert mx.array_equal(
        target.per_channel_statistics.mean_of_means,
        weights["per_channel_statistics.mean-of-means"],
    ).item()
    assert math.isfinite(float(mx.sum(target.conv_out.weight).item()))
