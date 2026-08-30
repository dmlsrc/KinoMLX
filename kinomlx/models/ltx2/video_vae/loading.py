"""Checkpoint construction for the native LTX video VAE."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import load_weights, read_metadata
from kinomlx.reporting import NullReporter, Reporter

from .config import VideoVAEConfig
from .decoder import NativeConv3dVideoDecoder, load_native_vae_decoder_weights
from .diffusion_decoder import (
    NativeDiffusionVideoDecoder,
    load_diffusion_video_decoder_weights,
)
from .encoder import NativeConv3dVideoEncoder, load_native_vae_encoder_weights


@dataclass(frozen=True)
class NativeVideoVAE:
    """A matched native encoder/decoder pair and its checkpoint config."""

    config: VideoVAEConfig
    encoder: NativeConv3dVideoEncoder
    decoder: NativeConv3dVideoDecoder | NativeDiffusionVideoDecoder


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
        f"{checkpoint}: missing config.vae metadata; normalized family caches do not "
        "carry constructor authority, so pass the inspected VideoVAEConfig explicitly"
    )


def load_native_video_vae(
    path: Path | str,
    *,
    config: VideoVAEConfig | None = None,
    compute_dtype: mx.Dtype = mx.bfloat16,
    reporter: Reporter | None = None,
) -> NativeVideoVAE:
    """Build a complete VAE from a checkpoint or recognized family cache."""
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
            encoder = NativeConv3dVideoEncoder(
                resolved_config,
                compute_dtype=compute_dtype,
            )
            load_native_vae_encoder_weights(encoder, weights)
            sink.phase_advance(phase)

            decoder: NativeConv3dVideoDecoder | NativeDiffusionVideoDecoder
            if resolved_config.decoder_kind == "diffusion-na":
                decoder = NativeDiffusionVideoDecoder(
                    resolved_config,
                    compute_dtype=compute_dtype,
                )
                load_diffusion_video_decoder_weights(decoder, weights)
            else:
                decoder = NativeConv3dVideoDecoder(
                    resolved_config,
                    compute_dtype=compute_dtype,
                )
                load_native_vae_decoder_weights(decoder, weights)
            sink.phase_advance(phase)

            return NativeVideoVAE(
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


__all__ = ["NativeVideoVAE", "load_native_video_vae"]
