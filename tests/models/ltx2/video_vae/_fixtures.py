"""Small native video VAE architecture used by unit tests."""

from __future__ import annotations

from typing import Any

from kinomlx.models.ltx2.video_vae.config import VideoVAEConfig


def mini_vae_mapping() -> dict[str, Any]:
    """Return a scale-4x8x8 architecture with tiny feature widths."""
    return {
        "dims": 3,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "patch_size": 2,
        "norm_layer": "pixel_norm",
        "latent_log_var": "uniform",
        "encoder_base_channels": 2,
        "decoder_base_channels": 2,
        "causal_decoder": False,
        "timestep_conditioning": False,
        "spatial_padding_mode": "zeros",
        "encoder_blocks": [
            ["res_x", {"num_layers": 1}],
            ["compress_space_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 1}],
            ["compress_time_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 1}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 1}],
        ],
        "decoder_blocks": [
            ["res_x", {"num_layers": 1}],
            ["compress_space", {"multiplier": 2}],
            ["res_x", {"num_layers": 1}],
            ["compress_time", {"multiplier": 2}],
            ["res_x", {"num_layers": 1}],
            ["compress_all", {"multiplier": 2}],
            ["res_x", {"num_layers": 1}],
        ],
    }


def mini_vae_config() -> VideoVAEConfig:
    """Parse the shared miniature architecture."""
    return VideoVAEConfig.from_mapping(mini_vae_mapping())
