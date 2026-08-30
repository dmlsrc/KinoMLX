"""Native MLX LTX audio VAE encoder."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import silu
from kinomlx.models.ltx2.patchifier import AudioPatchifier
from kinomlx.models.ltx2.types import AudioLatentShape
from kinomlx.reporting import NullReporter, Reporter

from .blocks import (
    CausalConv2d,
    CausalityAxis,
    Downsample2d,
    PerChannelStatistics,
    PixelNorm,
    ResBlock2d,
)
from .config import AudioVAEConfig


class _EncoderStage(TypedDict):
    res_blocks: list[ResBlock2d]
    downsample: Downsample2d | None


class AudioEncoder(nn.Module):
    """Encode stereo log-mel spectrograms to normalized mean latents."""

    def __init__(
        self,
        config: AudioVAEConfig | None = None,
        *,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config or AudioVAEConfig()
        self.compute_dtype = compute_dtype
        axis = CausalityAxis(self.config.causality_axis)
        self.per_channel_statistics = PerChannelStatistics(self.config.channels)
        self.patchifier = AudioPatchifier(
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length,
            audio_latent_downsample_factor=self.config.downsample_factor,
            is_causal=self.config.is_causal,
        )
        self.conv_in = CausalConv2d(
            self.config.input_channels,
            self.config.channels,
            causality_axis=axis,
        )
        self.down_blocks: list[_EncoderStage] = []
        block_channels = self.config.channels
        for level, multiplier in enumerate(self.config.channel_multipliers):
            output_channels = self.config.channels * multiplier
            blocks = []
            for _ in range(self.config.num_res_blocks):
                blocks.append(
                    ResBlock2d(
                        block_channels,
                        output_channels,
                        causality_axis=axis,
                    )
                )
                block_channels = output_channels
            downsample = (
                Downsample2d(
                    output_channels,
                    causality_axis=axis,
                )
                if level != len(self.config.channel_multipliers) - 1
                else None
            )
            self.down_blocks.append({"res_blocks": blocks, "downsample": downsample})
        self.mid_block_1 = ResBlock2d(
            block_channels,
            block_channels,
            causality_axis=axis,
        )
        self.mid_block_2 = ResBlock2d(
            block_channels,
            block_channels,
            causality_axis=axis,
        )
        self.norm_out = PixelNorm()
        output_channels = self.config.latent_channels * (2 if self.config.double_latent else 1)
        self.conv_out = CausalConv2d(
            block_channels,
            output_channels,
            causality_axis=axis,
        )

    def _normalize(self, latent: mx.array) -> mx.array:
        mean = latent[:, : self.config.latent_channels]
        batch, channels, frames, mel_bins = mean.shape
        shape = AudioLatentShape(batch, channels, frames, mel_bins)
        patched = self.patchifier.patchify(mean)
        return self.patchifier.unpatchify(
            self.per_channel_statistics.normalize(patched),
            shape,
        )

    def __call__(
        self,
        spectrogram: mx.array,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array:
        if spectrogram.ndim != 4 or spectrogram.shape[1] != self.config.input_channels:
            raise ValueError(
                "audio encoder expects (B, channels, time, mel_bins), got "
                f"{tuple(spectrogram.shape)}"
            )
        if spectrogram.shape[3] != self.config.mel_bins:
            raise ValueError(f"audio encoder expects {self.config.mel_bins} mel bins")
        sink = reporter if reporter is not None else NullReporter()
        phase = "audio VAE encode"
        sink.phase_start(phase, total=len(self.down_blocks) + 2, unit="stage")
        try:
            x = self.conv_in(spectrogram.astype(self.compute_dtype))
            for level in self.down_blocks:
                for block in level["res_blocks"]:
                    x = block(x)
                downsample = level["downsample"]
                if downsample is not None:
                    x = downsample(x)
                mx.eval(x)
                sink.phase_advance(phase)
            x = self.mid_block_2(self.mid_block_1(x))
            mx.eval(x)
            sink.phase_advance(phase)
            x = self.conv_out(silu(self.norm_out(x)))
            result = self._normalize(x).astype(mx.float32)
            mx.eval(result)
            sink.phase_advance(phase)
            return result
        finally:
            sink.phase_end(phase)


def create_audio_encoder_from_checkpoint(
    path: Path | str,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> AudioEncoder:
    return AudioEncoder(
        AudioVAEConfig.from_checkpoint(path),
        compute_dtype=compute_dtype,
    )


__all__ = ["AudioEncoder", "create_audio_encoder_from_checkpoint"]
