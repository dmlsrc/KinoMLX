"""Video VAE checkpoint architecture contracts."""

from __future__ import annotations

import pytest

from kinomlx.models.ltx2.video_vae.config import (
    LTX23_VIDEO_VAE_CONFIG,
    VideoVAEBlock,
    VideoVAEConfig,
)
from kinomlx.types import SpatioTemporalScaleFactors

from ._fixtures import mini_vae_mapping


def test_ltx23_config_has_checkpoint_geometry() -> None:
    config = LTX23_VIDEO_VAE_CONFIG
    expected = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    assert config.encoder_scale == expected
    assert config.decoder_scale == expected
    assert config.latent_channels == 128
    assert [block.num_layers for block in config.encoder_blocks if block.name == "res_x"] == [
        4,
        6,
        4,
        2,
        2,
    ]


def test_mapping_drives_miniature_architecture() -> None:
    config = VideoVAEConfig.from_mapping(mini_vae_mapping())
    expected = SpatioTemporalScaleFactors(time=4, height=8, width=8)
    assert config.encoder_scale == expected
    assert config.decoder_scale == expected
    assert config.encoder_base_channels == 2
    assert config.latent_channels == 4


def test_mapping_ignores_unconsumed_community_metadata() -> None:
    raw = mini_vae_mapping()
    raw["future_root_field"] = "ignored"
    raw["encoder_blocks"][0][1]["training_note"] = "ignored"
    raw["decoder_blocks"][0][1]["wrapper_version"] = 7

    assert VideoVAEConfig.from_mapping(raw) == VideoVAEConfig.from_mapping(mini_vae_mapping())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dims", 2, "requires dims=3"),
        ("norm_layer", "group_norm", "requires pixel_norm"),
        ("latent_log_var", "per_channel", "requires uniform"),
        ("timestep_conditioning", True, "does not use timestep conditioning"),
        ("spatial_padding_mode", "reflect", "only zero spatial padding"),
    ],
)
def test_mapping_rejects_unsupported_architecture(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = mini_vae_mapping()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        VideoVAEConfig.from_mapping(raw)


def test_config_rejects_mismatched_encoder_decoder_scales() -> None:
    raw = mini_vae_mapping()
    raw["decoder_blocks"] = [["res_x", {"num_layers": 1}]]
    with pytest.raises(ValueError, match="scale factors differ"):
        VideoVAEConfig.from_mapping(raw)


def test_config_rejects_residual_on_non_all_decoder_compression() -> None:
    raw = mini_vae_mapping()
    raw["decoder_blocks"][1][1]["residual"] = True
    with pytest.raises(ValueError, match="only for decoder compress_all"):
        VideoVAEConfig.from_mapping(raw)


def test_direct_block_construction_is_validated() -> None:
    with pytest.raises(ValueError, match="requires num_layers"):
        VideoVAEBlock("res_x")
    with pytest.raises(ValueError, match="cannot set num_layers"):
        VideoVAEBlock("compress_all", num_layers=1)
