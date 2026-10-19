"""Checkpoint construction for the native LTX video VAE."""

import gc
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import load_weights, read_metadata
from kinomlx.reporting import NullReporter, Reporter

from .config import VideoVAEConfig
from .decoder import Conv3dVideoDecoder, load_vae_decoder_weights
from .diffusion_decoder import (
    DiffusionVideoDecoder,
    load_diffusion_video_decoder_weights,
)
from .encoder import Conv3dVideoEncoder, load_vae_encoder_weights
from .kinds import VideoVAEKind


@dataclass(frozen=True)
class VideoVAE:
    """A matched native encoder/decoder pair and its checkpoint config."""

    config: VideoVAEConfig
    encoder: Conv3dVideoEncoder
    decoder: Conv3dVideoDecoder | DiffusionVideoDecoder


def _config_for_artifact(
    checkpoint: Path,
    config: VideoVAEConfig | None,
) -> VideoVAEConfig:
    if config is not None:
        if not isinstance(config, VideoVAEConfig):
            raise TypeError("config must be a VideoVAEConfig")
        return config

    metadata = read_metadata(checkpoint)
    if "config" in metadata:
        return VideoVAEConfig.from_checkpoint(checkpoint)

    raise ValueError(
        f"{checkpoint}: missing config.vae metadata; weight-only files do not "
        "carry constructor authority, so pass the inspected VideoVAEConfig explicitly"
    )


def load_video_vae(
    path: Path | str,
    *,
    config: VideoVAEConfig | None = None,
    compute_dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> VideoVAE:
    """Build a complete VAE from a checkpoint with constructor metadata or an explicit config."""
    checkpoint = Path(path)
    sink = reporter if reporter is not None else NullReporter()
    phase = "load video VAE"
    sink.phase_start(phase, total=4, unit="step")
    try:
        resolved_config = _config_for_artifact(checkpoint, config)
        sink.phase_advance(phase)

        weights = load_weights(checkpoint)
        sink.phase_advance(phase)
        try:
            encoder = Conv3dVideoEncoder(
                resolved_config,
                compute_dtype=compute_dtype,
            )
            load_vae_encoder_weights(encoder, weights)
            sink.phase_advance(phase)

            decoder: Conv3dVideoDecoder | DiffusionVideoDecoder
            if resolved_config.decoder_kind == VideoVAEKind.DIFFUSION:
                decoder = DiffusionVideoDecoder(
                    resolved_config,
                    compute_dtype=compute_dtype,
                )
                load_diffusion_video_decoder_weights(decoder, weights)
            else:
                decoder = Conv3dVideoDecoder(
                    resolved_config,
                    compute_dtype=compute_dtype,
                )
                load_vae_decoder_weights(decoder, weights)
            sink.phase_advance(phase)

            return VideoVAE(
                config=resolved_config,
                encoder=encoder,
                decoder=decoder,
            )
        finally:
            del weights
            gc.collect()
            mx.clear_cache()
    finally:
        sink.phase_end(phase)


__all__ = ["VideoVAE", "load_video_vae"]
