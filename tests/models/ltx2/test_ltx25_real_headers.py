"""Opt-in header-only gates for the pinned LTX-2.5 split pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from kinomlx.models.ltx2.metadata import (
    checkpoint_config,
    inspect_audio_vae,
    inspect_connectors,
    inspect_duration_head,
    inspect_latent_upscaler,
    inspect_text_encoder,
    inspect_text_projection,
    inspect_video_vae,
    validate_transformer_header,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import Settings

_LTX25_REVISION = "6c7e5e573ac1667efc83407806fe9b0b93730e60"


def _ltx25_snapshot() -> Path:
    return (
        Settings.from_env().hf_home / "hub/models--Lightricks--LTX-2.5/snapshots" / _LTX25_REVISION
    )


def _require_files(*paths: Path) -> None:
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        pytest.skip("pinned LTX-2.5 files are not cached: " + ", ".join(missing))


@pytest.mark.requires_weights
def test_real_ltx25_pack_matches_every_consumed_compatibility_anchor() -> None:
    root = _ltx25_snapshot()
    transformer = root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    text = root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    video = root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    diffusion_video = root / "vae/ltx-2.5-video-vae-bf16.safetensors"
    audio = root / "vae/ltx-2.5-audio-vae-bf16.safetensors"
    spatial = root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    temporal = (
        root / "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    )
    duration = root / "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    _require_files(
        transformer,
        text,
        video,
        diffusion_video,
        audio,
        spatial,
        temporal,
        duration,
    )

    parsed = checkpoint_config(transformer)
    validate_transformer_header(transformer, parsed.transformer)
    text_config = inspect_text_encoder(text, model_generation="2.5")
    inspect_text_projection(
        text,
        model_generation="2.5",
        hidden_size=text_config.hidden_size,
        num_hidden_layers=text_config.num_hidden_layers,
    )
    inspect_connectors(transformer, config=parsed.transformer)
    video_config = inspect_video_vae(video, model_generation="2.5")
    diffusion_video_config = inspect_video_vae(diffusion_video, model_generation="2.5")
    audio_config = inspect_audio_vae(audio, model_generation="2.5")
    spatial_config = inspect_latent_upscaler(
        spatial,
        expected_kind="spatial",
        model_generation="2.5",
    )
    temporal_config = inspect_latent_upscaler(
        temporal,
        expected_kind="temporal",
        model_generation="2.5",
    )
    duration_config = inspect_duration_head(duration, model_generation="2.5")

    assert parsed.model_generation == "2.5"
    assert text_config.hidden_size == 3840
    assert text_config.tokenizer_json_bytes == 32_169_626
    assert tuple(video_config.encoder_scale) == (8, 32, 32)
    assert video_config.decoder_kind == "native-conv3d"
    assert diffusion_video_config.decoder_kind == "diffusion-na"
    assert diffusion_video_config.encoder_scale == video_config.encoder_scale
    assert diffusion_video_config.inferred_fields == ("vae.signal_domain",)
    assert diffusion_video_config.diffusion_decoder is not None
    assert diffusion_video_config.diffusion_decoder.inferred_fields == ("decoder.t_emb_dim",)
    assert audio_config.channels == 128
    assert spatial_config.kind == "spatial"
    assert temporal_config.kind == "temporal"
    assert duration_config.video_context_dim == 4096
    assert duration_config.audio_context_dim == 2048


@pytest.mark.requires_weights
def test_real_dev_distilled_filename_alias_and_cross_generation_upscalers(
    tmp_path: Path,
) -> None:
    root = _ltx25_snapshot()
    dev = root / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    distilled = root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    spatial_25 = (
        root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )
    temporal_25 = (
        root / "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    )
    checkpoint_23 = LTX2Settings.from_env().weights_path
    if checkpoint_23 is None:
        pytest.skip("KINO_WEIGHTS_PATH is not configured")
    spatial_23 = checkpoint_23.parent / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    temporal_23 = checkpoint_23.parent / "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
    _require_files(dev, distilled, spatial_25, temporal_25, spatial_23, temporal_23)

    dev_config = checkpoint_config(dev).transformer
    distilled_config = checkpoint_config(distilled).transformer
    validate_transformer_header(dev, dev_config)
    validate_transformer_header(distilled, distilled_config)
    assert dev_config.cache_identity() == distilled_config.cache_identity()
    assert dev.resolve() != distilled.resolve()

    renamed = tmp_path / "renamed-transformer.safetensors"
    renamed.symlink_to(distilled)
    renamed_config = checkpoint_config(renamed).transformer
    validate_transformer_header(renamed, renamed_config)
    assert renamed_config.cache_identity() == distilled_config.cache_identity()

    spatial_config_23 = inspect_latent_upscaler(
        spatial_23,
        expected_kind="spatial",
        model_generation="2.3",
    )
    spatial_config_25 = inspect_latent_upscaler(
        spatial_25,
        expected_kind="spatial",
        model_generation="2.5",
    )
    assert spatial_config_23 == spatial_config_25
    assert spatial_23.resolve() != spatial_25.resolve()

    temporal_config_23 = inspect_latent_upscaler(
        temporal_23,
        expected_kind="temporal",
        model_generation="2.3",
    )
    temporal_config_25 = inspect_latent_upscaler(
        temporal_25,
        expected_kind="temporal",
        model_generation="2.5",
    )
    assert temporal_config_23 == temporal_config_25
    assert temporal_23.resolve().name == temporal_25.resolve().name
    assert temporal_23.stat().st_size == temporal_25.stat().st_size
