"""Native MLX LTX audio VAE decoder."""

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
    PerChannelStatistics,
    PixelNorm,
    ResBlock2d,
    Upsample2d,
)
from .config import AudioVAEConfig


class _DecoderStage(TypedDict):
    res_blocks: list[ResBlock2d]
    upsample: Upsample2d | None


class AudioDecoder(nn.Module):
    """Decode normalized ``(B, 8, T, 16)`` latents to stereo log-mel."""

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
        block_channels = self.config.channels * self.config.channel_multipliers[-1]
        self.conv_in = CausalConv2d(
            self.config.latent_channels,
            block_channels,
            causality_axis=axis,
        )
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
        self.up_blocks: list[_DecoderStage] = []
        for level in reversed(range(len(self.config.channel_multipliers))):
            output_channels = self.config.channels * self.config.channel_multipliers[level]
            blocks = []
            for _ in range(self.config.num_res_blocks + 1):
                blocks.append(
                    ResBlock2d(
                        block_channels,
                        output_channels,
                        causality_axis=axis,
                    )
                )
                block_channels = output_channels
            upsample = (
                Upsample2d(
                    output_channels,
                    causality_axis=axis,
                )
                if level != 0
                else None
            )
            self.up_blocks.append({"res_blocks": blocks, "upsample": upsample})
        self.norm_out = PixelNorm()
        self.conv_out = CausalConv2d(
            self.config.channels,
            self.config.output_channels,
            causality_axis=axis,
        )

    def _denormalize(self, sample: mx.array) -> mx.array:
        batch, channels, frames, mel_bins = sample.shape
        shape = AudioLatentShape(batch, channels, frames, mel_bins)
        patched = self.patchifier.patchify(sample)
        return self.patchifier.unpatchify(
            self.per_channel_statistics.denormalize(patched),
            shape,
        )

    def __call__(
        self,
        sample: mx.array,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array:
        expected = self.config.latent_channels
        if sample.ndim != 4 or sample.shape[1] != expected:
            raise ValueError(
                f"audio decoder expects (B, {expected}, T, F), got {tuple(sample.shape)}"
            )
        if sample.shape[3] != self.config.latent_mel_bins:
            raise ValueError(f"audio decoder expects {self.config.latent_mel_bins} latent mel bins")
        sink = reporter if reporter is not None else NullReporter()
        phase = "audio VAE decode"
        sink.phase_start(phase, total=len(self.up_blocks) + 2, unit="stage")
        try:
            sample = self._denormalize(sample.astype(self.compute_dtype))
            target_frames = sample.shape[2] * self.config.downsample_factor
            if self.config.causality_axis != "none":
                target_frames = max(target_frames - (self.config.downsample_factor - 1), 1)
            x = self.conv_in(sample)
            x = self.mid_block_2(self.mid_block_1(x))
            mx.eval(x)
            sink.phase_advance(phase)
            for level in self.up_blocks:
                for block in level["res_blocks"]:
                    x = block(x)
                upsample = level["upsample"]
                if upsample is not None:
                    x = upsample(x)
                mx.eval(x)
                sink.phase_advance(phase)
            x = self.conv_out(silu(self.norm_out(x)))
            x = x[
                :,
                : self.config.output_channels,
                :target_frames,
                : self.config.mel_bins,
            ]
            time_padding = target_frames - x.shape[2]
            mel_padding = self.config.mel_bins - x.shape[3]
            if time_padding > 0 or mel_padding > 0:
                x = mx.pad(
                    x,
                    [
                        (0, 0),
                        (0, 0),
                        (0, max(time_padding, 0)),
                        (0, max(mel_padding, 0)),
                    ],
                )
            result = x[:, :, :target_frames, : self.config.mel_bins]
            mx.eval(result)
            sink.phase_advance(phase)
            return result
        finally:
            sink.phase_end(phase)


def create_audio_decoder_from_checkpoint(
    path: Path | str,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> AudioDecoder:
    return AudioDecoder(
        AudioVAEConfig.from_checkpoint(path),
        compute_dtype=compute_dtype,
    )


__all__ = ["AudioDecoder", "create_audio_decoder_from_checkpoint"]
