"""Real-weight canary for the official LTX-2.5 diffusion video VAE."""

import gc
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.components import load_video_decoder
from kinomlx.models.ltx2.video_vae.config import VideoVAEConfig
from kinomlx.models.ltx2.video_vae.kinds import VideoVAEKind
from kinomlx.models.ltx2.video_vae.tiling import TilingConfig, decode_streaming
from tests._snapshots import pinned_snapshot

_LTX25_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"


def _diffusion_vae() -> Path:
    snapshot = pinned_snapshot("Lightricks/LTX-2.5", _LTX25_REVISION)
    return snapshot / "vae/ltx-2.5-video-vae-bf16.safetensors"


@pytest.mark.heavy
@pytest.mark.requires_metal
def test_real_ltx25_diffusion_decoder_is_seeded_bounded_and_provider_selected() -> None:
    checkpoint = _diffusion_vae()
    if not checkpoint.is_file():
        pytest.skip("the pinned LTX-2.5 diffusion VAE is not cached")
    # Exercise the real provider with only its required component descriptor.
    # Full model preparation would unnecessarily build the 35 GiB transformer.
    resources = SimpleNamespace(
        video_vae_config=VideoVAEConfig.from_checkpoint(checkpoint),
        dtype_policy=SimpleNamespace(video_vae=mx.bfloat16),
        execution_policy=SimpleNamespace(),
        require=lambda kind: SimpleNamespace(source_path=checkpoint),
    )
    latent = None
    first_chunks = None
    second_chunks = None
    first = None
    second = None
    mx.clear_cache()
    try:
        with load_video_decoder(resources) as decoder:
            assert decoder.decoder_kind == VideoVAEKind.DIFFUSION
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
            # Both decodes are consumed whole: the peak-memory bound below is a
            # claim about the entire decode, not about its first chunk.
            first_chunks = list(decode_streaming(latent, decoder, TilingConfig(), seed=42))
            second_chunks = list(decode_streaming(latent, decoder, TilingConfig(), seed=42))
            first, second = first_chunks[0], second_chunks[0]
            mx.eval(first, second)

            assert tuple(first.shape) == (1, 3, 9, 224, 224)
            assert mx.array_equal(first, second).item()
            assert mx.all(mx.isfinite(first)).item()
            assert decoder.attention_tiling_stats.max_score_elements <= (
                decoder.attention_tiling_stats.score_budget
            )
            assert mx.get_peak_memory() < 3_000_000_000
    finally:
        del resources, latent, first_chunks, second_chunks, first, second
        gc.collect()
        mx.clear_cache()
