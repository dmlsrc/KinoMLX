"""Opt-in regression canary for the official LTX-2.3 checkpoint."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.cache import (
    ensure_weight_family_caches,
    weight_family_cache_paths,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.video_vae.config import VideoVAEConfig
from kinomlx.models.ltx2.video_vae.loading import load_native_video_vae
from kinomlx.settings import Settings


@pytest.mark.requires_weights
def test_real_ltx23_video_vae_round_trip() -> None:
    checkpoint = LTX2Settings.from_env().weights_path
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("KINO_WEIGHTS_PATH does not name an LTX-2.3 checkpoint file")

    bundle = None
    video = None
    latent = None
    decoded = None
    try:
        bundle = load_native_video_vae(checkpoint)
        video = mx.random.uniform(
            low=-1.0,
            high=1.0,
            shape=(1, 3, 25, 256, 256),
        ).astype(mx.bfloat16)
        latent = bundle.encoder(video)
        decoded = bundle.decoder(latent)
        mx.eval(latent, decoded)

        assert tuple(latent.shape) == (1, 128, 4, 8, 8)
        assert latent.dtype == mx.float32
        assert tuple(decoded.shape) == tuple(video.shape)
        assert decoded.dtype == mx.bfloat16
        assert mx.all(mx.isfinite(latent)).item()
        assert mx.all(mx.isfinite(decoded)).item()
        assert mx.any(latent != 0).item()
        assert mx.any(decoded != 0).item()
    finally:
        del bundle, video, latent, decoded
        gc.collect()
        mx.clear_cache()


@pytest.mark.requires_weights
def test_real_ltx23_compatible_family_cache_is_reused_and_loadable() -> None:
    infrastructure = Settings.from_env()
    checkpoint = LTX2Settings.from_env().weights_path
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("KINO_WEIGHTS_PATH does not name an LTX-2.3 checkpoint file")
    cache_file, metadata_file, _payload = weight_family_cache_paths(
        checkpoint,
        infrastructure.cache_dir,
        "video_vae",
    )
    if not cache_file.is_file() or not metadata_file.is_file():
        pytest.skip("KINO_CACHE_DIR does not contain a matching schema-v3 video VAE cache")

    result = ensure_weight_family_caches(
        checkpoint,
        families=("video_vae",),
        cache_mode="auto",
        cache_root=infrastructure.cache_dir,
    )
    assert not result.rebuilt
    assert result.cache_paths["video_vae"] == cache_file

    bundle = None
    try:
        bundle = load_native_video_vae(
            Path(cache_file),
            config=VideoVAEConfig.from_checkpoint(checkpoint),
        )
        assert bundle.encoder.conv_in.conv.weight.shape == (128, 3, 3, 3, 48)
        assert bundle.decoder.conv_in.conv.weight.shape == (1024, 3, 3, 3, 128)
    finally:
        del bundle
        gc.collect()
        mx.clear_cache()
