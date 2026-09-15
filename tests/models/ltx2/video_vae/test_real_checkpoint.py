"""Opt-in regression canary for the official LTX-2.3 checkpoint."""

import gc

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.video_vae.loading import load_native_video_vae
from tests.models.ltx2._source_weights import NO_LTX23_CHECKPOINT, ltx23_checkpoint


@pytest.mark.heavy
def test_real_ltx23_video_vae_round_trip() -> None:
    checkpoint = ltx23_checkpoint()
    if checkpoint is None:
        pytest.skip(NO_LTX23_CHECKPOINT)

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
