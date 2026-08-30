"""Opt-in canaries for the official LTX-2.3 audio stack."""

from __future__ import annotations

import gc
import math

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.audio_vae import (
    AudioDecoder,
    AudioEncoder,
    AudioVAEConfig,
    BWEVocoderConfig,
    VocoderWithBWE,
    load_audio_vae_weights,
    load_vocoder_weights,
)
from kinomlx.models.ltx2.cache import ensure_weight_family_caches
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import Settings


@pytest.mark.requires_weights
def test_real_ltx23_audio_encode_decode_and_vocode() -> None:
    settings = Settings.from_env()
    checkpoint = LTX2Settings.from_env().weights_path
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("KINO_WEIGHTS_PATH is not configured")
    caches = ensure_weight_family_caches(
        checkpoint,
        families=("audio_vae", "vocoder"),
        cache_mode="auto",
        cache_root=settings.cache_dir,
    )

    encoder = decoder = latent = mel = None
    vocoder = waveform = None
    try:
        vae_config = AudioVAEConfig.from_checkpoint(checkpoint)
        encoder = AudioEncoder(vae_config)
        decoder = AudioDecoder(vae_config)
        assert (
            load_audio_vae_weights(
                encoder,
                decoder,
                caches.cache_paths["audio_vae"],
            )
            == 102
        )
        spectrogram = mx.random.normal((1, 2, 7, vae_config.mel_bins)).astype(mx.bfloat16)
        encoded = encoder(spectrogram)
        mx.eval(encoded)
        assert tuple(encoded.shape) == (1, 8, 2, 16)
        assert mx.all(mx.isfinite(encoded)).item()

        latent = mx.random.normal(
            (1, vae_config.latent_channels, 1, vae_config.latent_mel_bins)
        ).astype(mx.bfloat16)
        mel = decoder(latent)
        mx.eval(mel)
        assert tuple(mel.shape) == (1, 2, 1, 64)
        assert mx.all(mx.isfinite(mel)).item()
        assert mx.any(mel != 0).item()
        del encoder, decoder, encoded, spectrogram, latent
        encoder = decoder = latent = None
        gc.collect()
        mx.clear_cache()

        vocoder_config = BWEVocoderConfig.from_checkpoint(checkpoint)
        vocoder = VocoderWithBWE(vocoder_config)
        assert load_vocoder_weights(vocoder, caches.cache_paths["vocoder"]) == 1227
        waveform = vocoder(mel)
        mx.eval(waveform)
        expected_length = (
            mel.shape[2]
            * math.prod(vocoder_config.vocoder.upsample_rates)
            * vocoder_config.output_sample_rate
            // vocoder_config.input_sample_rate
        )
        assert tuple(waveform.shape) == (1, 2, expected_length)
        assert waveform.dtype == mx.bfloat16
        assert mx.all(mx.isfinite(waveform)).item()
        assert mx.any(waveform != 0).item()
    finally:
        del encoder, decoder, latent, mel, vocoder, waveform
        gc.collect()
        mx.clear_cache()
