"""Small structurally complete VAE decoder fixtures."""

from pathlib import Path

import mlx.core as mx

from kinomlx.models.krea2.metadata import Krea2VAEConfig
from kinomlx.models.krea2.vae.model import Krea2ImageDecoder
from tests.models.krea2._source_bindings import fixture_bindings

RAW_REVISION = "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
TURBO_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"

VAE_FIXTURE_SHA256 = {
    "denoise-trajectories.safetensors": (
        "ab32899a70958a607071fddb962be5342ff223ff48668d3c8697937707c9de95"
    ),
    "official-vae-boundaries.safetensors": (
        "6d08fd67552fc98a4ad2a1b796792a267b01217feb13d2e04cb37d116a117d74"
    ),
    "official.png": "a0e72e4b26c537ea36b6677e16be4c09f734d052775fab675b89c307beb4eaed",
    "image-report.json": "3ab0645ef6b6acf039b08afb75f955d65c4eb85d98ede7ede7901094f9a789f0",
    "run.json": "f91f520a1308c460871be6cde74bf8a11ac3e101e274c9af6199eb22f5971ecd",
    "patch-manifest.json": ("db2f7cf247e027d1290dfe53db8173f88b6466e0f9cea1fad989668d7e8e8272"),
}

VAE_BOUNDARY_NORMALIZED_RMSE_LIMITS = {
    "normalized_input": 0.0,
    "post_quant_conv": 0.0,
    "decoder_conv_in": 2e-5,
    "decoder_mid": 0.006,
    "decoder_up_0": 0.007,
    "decoder_up_1": 0.011,
    "decoder_up_2": 0.011,
    "decoder_up_3": 0.012,
    "decoder_norm_out": 0.013,
    "decoder_conv_out_pre_clamp": 0.005,
    "decoded_float": 0.005,
}
VAE_PIXEL_MAX_ABS_LIMIT = 5
VAE_PIXEL_MEAN_ABS_LIMIT = 0.3
ALTERNATE_VAE_NORMALIZED_RMSE_LIMIT = 0.004
ALTERNATE_VAE_PIXEL_MAX_ABS_LIMIT = 4
ALTERNATE_VAE_PIXEL_MEAN_ABS_LIMIT = 0.35
TILED_NORMALIZED_RMSE_LIMIT = 0.007
TILED_PIXEL_MAX_ABS_LIMIT = 16
TILED_PIXEL_MEAN_ABS_LIMIT = 0.4
TILED_SEAM_MEAN_ABS_LIMIT = 0.004
TILED_SEAM_TO_GLOBAL_MEAN_LIMIT = 1.5
# One evaluated graph per tile trades the former per-block peak for less dispatch.
# The memory contract is a measured reduction relative to full-frame decode.
TILED_TO_FULL_PEAK_MEMORY_LIMIT = 1.0


def tiny_vae_config(
    *,
    latent_mean: tuple[float, ...] = (0.0, 0.0),
    latent_std: tuple[float, ...] = (1.0, 1.0),
) -> Krea2VAEConfig:
    return Krea2VAEConfig(
        base_dim=2,
        dim_mult=(1, 2),
        input_channels=3,
        z_dim=2,
        num_res_blocks=0,
        attn_scales=(),
        latent_mean=latent_mean,
        latent_std=latent_std,
    )


def write_tiny_vae_cache(
    path: Path,
    model: Krea2ImageDecoder,
    *,
    overrides: dict[str, mx.array] | None = None,
) -> dict[str, mx.array]:
    from kinomlx.models.krea2.vae.loading import _targets

    weights: dict[str, mx.array] = {}
    for key, target in _targets(model).items():
        fill = 1.0 if key.endswith(".gamma") else 0.0
        weights[key] = mx.full(target.shape, fill, dtype=model.compute_dtype)
    if overrides is not None:
        weights.update(overrides)
    mx.save_safetensors(str(path), weights)
    return weights


def loaded_tiny_vae(
    tmp_path: Path,
    *,
    config: Krea2VAEConfig | None = None,
    compute_dtype: str | mx.Dtype = "float32",
    output_scale: int = 1,
    overrides: dict[str, mx.array] | None = None,
) -> Krea2ImageDecoder:
    from kinomlx.models.krea2.vae.loading import load_krea2_vae_weights

    model = Krea2ImageDecoder(
        tiny_vae_config() if config is None else config,
        compute_dtype=compute_dtype,
        output_scale=output_scale,
    )
    cache = tmp_path / "vae.safetensors"
    write_tiny_vae_cache(cache, model, overrides=overrides)
    load_krea2_vae_weights(
        model,
        source_bindings=fixture_bindings(model, cache, "vae"),
        expected_dtype=model.compute_dtype,
    )
    return model


__all__ = [
    "ALTERNATE_VAE_NORMALIZED_RMSE_LIMIT",
    "ALTERNATE_VAE_PIXEL_MAX_ABS_LIMIT",
    "ALTERNATE_VAE_PIXEL_MEAN_ABS_LIMIT",
    "RAW_REVISION",
    "TILED_NORMALIZED_RMSE_LIMIT",
    "TILED_PIXEL_MAX_ABS_LIMIT",
    "TILED_PIXEL_MEAN_ABS_LIMIT",
    "TILED_SEAM_MEAN_ABS_LIMIT",
    "TILED_SEAM_TO_GLOBAL_MEAN_LIMIT",
    "TILED_TO_FULL_PEAK_MEMORY_LIMIT",
    "TURBO_REVISION",
    "VAE_BOUNDARY_NORMALIZED_RMSE_LIMITS",
    "VAE_FIXTURE_SHA256",
    "VAE_PIXEL_MAX_ABS_LIMIT",
    "VAE_PIXEL_MEAN_ABS_LIMIT",
    "loaded_tiny_vae",
    "tiny_vae_config",
    "write_tiny_vae_cache",
]
