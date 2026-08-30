"""Real-weight canary for the official LTX-2.5 diffusion video VAE."""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.components import load_video_decoder
from kinomlx.models.ltx2.resources import prepare_resources
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.video_vae.tiling import TilingConfig, decode_streaming
from kinomlx.settings import Settings

_LTX25_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"


def _diffusion_vae(host: Settings) -> Path:
    return (
        host.hf_home
        / "hub/models--Lightricks--LTX-2.5/snapshots"
        / _LTX25_REVISION
        / "vae/ltx-2.5-video-vae-bf16.safetensors"
    )


@pytest.mark.requires_weights
@pytest.mark.requires_metal
def test_real_ltx25_diffusion_decoder_is_seeded_bounded_and_provider_selected() -> None:
    host = Settings.from_env()
    checkpoint = _diffusion_vae(host)
    transformer = checkpoint.parents[1] / (
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    text_encoder = checkpoint.parents[1] / (
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    )
    if not all(path.is_file() for path in (checkpoint, transformer, text_encoder)):
        pytest.skip("the pinned LTX-2.5 diffusion VAE pack is not cached")

    resources = prepare_resources(
        LTX2Settings(model_generation="2.5", video_vae_path=checkpoint),
        infrastructure=host,
    )
    latent = None
    first = None
    second = None
    mx.clear_cache()
    try:
        with load_video_decoder(resources) as decoder:
            assert decoder.decoder_kind == "diffusion-na"
            assert resources.capabilities.video_vae_kind == "diffusion-na"
            assert decoder.load_receipt is not None
            assert decoder.load_receipt.loaded_tensors == 311
            assert decoder.load_receipt.ignored_decoder_tensors == ("decoder.type_emb",)
            assert decoder.load_receipt.inferred_constructor_fields == (
                "decoder.t_emb_dim",
                "vae.signal_domain",
            )

            mx.random.seed(2_508_220)
            latent = mx.random.normal((1, 128, 2, 7, 7), dtype=mx.bfloat16)
            mx.eval(latent)
            mx.reset_peak_memory()
            first = tuple(
                decode_streaming(
                    latent,
                    decoder,
                    TilingConfig(),
                    seed=42,
                )
            )[0]
            second = tuple(
                decode_streaming(
                    latent,
                    decoder,
                    TilingConfig(),
                    seed=42,
                )
            )[0]
            mx.eval(first, second)

            assert tuple(first.shape) == (1, 3, 9, 224, 224)
            assert mx.array_equal(first, second).item()
            assert mx.all(mx.isfinite(first)).item()
            assert decoder.attention_tiling_stats.max_score_elements <= (
                decoder.attention_tiling_stats.score_budget
            )
            assert mx.get_peak_memory() < 3_000_000_000
    finally:
        del resources, latent, first, second
        gc.collect()
        mx.clear_cache()
