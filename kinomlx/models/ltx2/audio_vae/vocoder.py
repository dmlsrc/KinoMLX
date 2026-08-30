"""LTX-2 BigVGAN vocoder and bandwidth-extension composition."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import read_metadata
from kinomlx.reporting import NullReporter, Reporter

from .vocoder_layers import Activation1d, AMPBlock1, Conv1d, ConvTranspose1d, UpSample1d
from .vocoder_stft import MelSTFT


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"vocoder {field} must be an integer")
    return value


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"vocoder {field} must be an integer sequence")
    sequence = cast(Sequence[object], value)
    return tuple(_integer(item, field) for item in sequence)


def _nested_integer_tuple(value: object, field: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"vocoder {field} must be a nested integer sequence")
    sequence = cast(Sequence[object], value)
    return tuple(_integer_tuple(item, field) for item in sequence)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"vocoder {field} must be a boolean")
    return value


@dataclass(frozen=True)
class VocoderConfig:
    """One BigVGAN generator architecture."""

    resblock_kernel_sizes: tuple[int, ...]
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_initial_channels: int
    output_sample_rate: int
    mel_bins: int = 64
    stereo: bool = True
    resblock: str = "AMP1"
    activation: str = "snakebeta"
    apply_final_activation: bool = True
    use_tanh_at_final: bool = True
    use_bias_at_final: bool = True

    def __post_init__(self) -> None:
        if not self.upsample_rates or len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("vocoder upsample rates and kernels must have equal lengths")
        if not self.resblock_kernel_sizes or len(self.resblock_kernel_sizes) != len(
            self.resblock_dilation_sizes
        ):
            raise ValueError("vocoder resblock kernels and dilations must have equal lengths")
        if any(value <= 0 for value in self.upsample_rates + self.upsample_kernel_sizes):
            raise ValueError("vocoder upsample values must be positive")
        if any(value <= 0 for value in self.resblock_kernel_sizes):
            raise ValueError("vocoder resblock kernels must be positive")
        if any(
            not values or any(value <= 0 for value in values)
            for values in self.resblock_dilation_sizes
        ):
            raise ValueError("vocoder dilation groups must be non-empty and positive")
        if min(self.upsample_initial_channels, self.output_sample_rate, self.mel_bins) <= 0:
            raise ValueError("vocoder dimensions and sample rate must be positive")
        if self.upsample_initial_channels % (2 ** len(self.upsample_rates)):
            raise ValueError("vocoder initial channels must divide across all upsample stages")
        if self.resblock != "AMP1" or self.activation != "snakebeta":
            raise ValueError("only LTX BigVGAN AMP1/SnakeBeta checkpoints are supported")
        if not self.stereo:
            raise ValueError("only stereo LTX vocoders are supported")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        output_sample_rate: int,
        apply_final_activation: bool,
    ) -> VocoderConfig:
        return cls(
            resblock_kernel_sizes=_integer_tuple(
                values.get("resblock_kernel_sizes", (3, 7, 11)),
                "resblock_kernel_sizes",
            ),
            upsample_rates=_integer_tuple(values.get("upsample_rates"), "upsample_rates"),
            upsample_kernel_sizes=_integer_tuple(
                values.get("upsample_kernel_sizes"),
                "upsample_kernel_sizes",
            ),
            resblock_dilation_sizes=_nested_integer_tuple(
                values.get(
                    "resblock_dilation_sizes",
                    ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                ),
                "resblock_dilation_sizes",
            ),
            upsample_initial_channels=_integer(
                values.get("upsample_initial_channel"),
                "upsample_initial_channel",
            ),
            output_sample_rate=output_sample_rate,
            mel_bins=_integer(values.get("num_mels", 64), "num_mels"),
            stereo=_boolean(values.get("stereo", True), "stereo"),
            resblock=str(values.get("resblock", "AMP1")),
            activation=str(values.get("activation", "snakebeta")),
            apply_final_activation=apply_final_activation,
            use_tanh_at_final=_boolean(
                values.get("use_tanh_at_final", True),
                "use_tanh_at_final",
            ),
            use_bias_at_final=_boolean(
                values.get("use_bias_at_final", True),
                "use_bias_at_final",
            ),
        )


@dataclass(frozen=True)
class BWEVocoderConfig:
    vocoder: VocoderConfig
    generator: VocoderConfig
    input_sample_rate: int
    output_sample_rate: int
    n_fft: int
    hop_length: int
    mel_bins: int

    def __post_init__(self) -> None:
        if self.output_sample_rate % self.input_sample_rate:
            raise ValueError("BWE output sample rate must be an integer multiple of input")
        if min(self.n_fft, self.hop_length, self.mel_bins) <= 0:
            raise ValueError("BWE spectral dimensions must be positive")
        expected_factor = self.hop_length * self.output_sample_rate // self.input_sample_rate
        if math.prod(self.generator.upsample_rates) != expected_factor:
            raise ValueError("BWE generator upsample factor does not match hop/sample rates")
        if self.vocoder.output_sample_rate != self.input_sample_rate:
            raise ValueError("base vocoder output sample rate must match BWE input sample rate")
        if self.generator.output_sample_rate != self.output_sample_rate:
            raise ValueError("BWE generator output sample rate must match BWE output sample rate")
        if self.vocoder.mel_bins != self.mel_bins or self.generator.mel_bins != self.mel_bins:
            raise ValueError("base and BWE vocoder mel bins must match the BWE spectral config")

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> BWEVocoderConfig:
        metadata = read_metadata(path)
        try:
            root: object = json.loads(metadata["config"])
        except KeyError as exc:
            raise ValueError(f"{path}: checkpoint metadata has no config") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: checkpoint config is invalid JSON") from exc
        vocoder_root_value = root.get("vocoder") if isinstance(root, dict) else None
        if not isinstance(vocoder_root_value, dict):
            raise ValueError("checkpoint does not contain an LTX BWE vocoder config")
        vocoder_root = cast(Mapping[str, object], vocoder_root_value)
        base_value = vocoder_root.get("vocoder", {})
        bwe_value = vocoder_root.get("bwe")
        if not isinstance(base_value, dict) or not isinstance(bwe_value, dict):
            raise ValueError("checkpoint does not contain an LTX BWE vocoder config")
        base = cast(Mapping[str, object], base_value)
        bwe = cast(Mapping[str, object], bwe_value)
        input_rate = _integer(bwe.get("input_sampling_rate"), "input_sampling_rate")
        output_rate = _integer(bwe.get("output_sampling_rate"), "output_sampling_rate")
        return cls(
            vocoder=VocoderConfig.from_mapping(
                base,
                output_sample_rate=input_rate,
                apply_final_activation=True,
            ),
            generator=VocoderConfig.from_mapping(
                bwe,
                output_sample_rate=output_rate,
                apply_final_activation=False,
            ),
            input_sample_rate=input_rate,
            output_sample_rate=output_rate,
            n_fft=_integer(bwe.get("n_fft"), "n_fft"),
            hop_length=_integer(bwe.get("hop_length"), "hop_length"),
            mel_bins=_integer(bwe.get("num_mels"), "num_mels"),
        )


class Vocoder(nn.Module):
    """One stereo BigVGAN generator."""

    def __init__(
        self,
        config: VocoderConfig,
        *,
        compute_dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.conv_pre = Conv1d(
            config.mel_bins * 2,
            config.upsample_initial_channels,
            7,
            padding=3,
        )
        self.ups = []
        self.resblocks = []
        for index, (rate, kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)
        ):
            input_channels = config.upsample_initial_channels // (2**index)
            output_channels = config.upsample_initial_channels // (2 ** (index + 1))
            self.ups.append(
                ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            for block_kernel, dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes,
                strict=True,
            ):
                self.resblocks.append(
                    AMPBlock1(
                        output_channels,
                        block_kernel,
                        dilations,
                    )
                )
        final_channels = config.upsample_initial_channels // (2 ** len(self.ups))
        self.act_post = Activation1d(final_channels)
        self.conv_post = Conv1d(
            final_channels,
            2,
            7,
            padding=3,
            bias=config.use_bias_at_final,
        )

    def __call__(
        self,
        mel: mx.array,
        *,
        reporter: Reporter | None = None,
        phase: str = "synthesize audio",
    ) -> mx.array:
        if mel.ndim != 4 or mel.shape[1] != 2 or mel.shape[3] != self.config.mel_bins:
            raise ValueError(
                f"vocoder expects (B, 2, T, {self.config.mel_bins}), got {tuple(mel.shape)}"
            )
        sink = reporter if reporter is not None else NullReporter()
        sink.phase_start(phase, total=len(self.ups) + 1, unit="stage")
        try:
            mel = mel.astype(self.compute_dtype).transpose(0, 1, 3, 2)
            batch, channels, mel_bins, frames = mel.shape
            x = self.conv_pre(mel.reshape(batch, channels * mel_bins, frames))
            for stage, upsample in enumerate(self.ups):
                x = upsample(x)
                start = stage * self.num_kernels
                outputs = [
                    self.resblocks[index](x) for index in range(start, start + self.num_kernels)
                ]
                x = mx.stack(outputs, axis=0).mean(axis=0)
                mx.eval(x)
                sink.phase_advance(phase)
            x = self.conv_post(self.act_post(x))
            mx.eval(x)
            sink.phase_advance(phase)
            if self.config.apply_final_activation:
                x = mx.tanh(x) if self.config.use_tanh_at_final else mx.clip(x, -1, 1)
                mx.eval(x)
            return x
        finally:
            sink.phase_end(phase)


class VocoderWithBWE(nn.Module):
    """FP32 base vocoder, causal STFT, BWE residual, and 3x skip resampler."""

    def __init__(self, config: BWEVocoderConfig) -> None:
        super().__init__()
        self.config = config
        self.vocoder = Vocoder(config.vocoder, compute_dtype=mx.float32)
        self.bwe_generator = Vocoder(
            config.generator,
            compute_dtype=mx.float32,
        )
        self.mel_stft = MelSTFT(
            config.n_fft,
            config.hop_length,
            config.n_fft,
            config.mel_bins,
        )
        self.resampler = UpSample1d(
            config.output_sample_rate // config.input_sample_rate,
            window_type="hann",
        )
        self.output_sample_rate = config.output_sample_rate

    def _compute_mel(self, audio: mx.array) -> mx.array:
        batch, channels, samples = audio.shape
        mel, _, _, _ = self.mel_stft.mel_spectrogram(audio.reshape(batch * channels, samples))
        return mel.reshape(batch, channels, mel.shape[1], mel.shape[2])

    def __call__(
        self,
        mel: mx.array,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array:
        input_dtype = mel.dtype
        sink = reporter if reporter is not None else NullReporter()
        phase = "vocode audio with bandwidth extension"
        sink.phase_start(phase, total=4, unit="stage")
        try:
            x = self.vocoder(
                mel.astype(mx.float32),
                reporter=sink,
                phase="synthesize base audio",
            )
            mx.eval(x)
            sink.phase_advance(phase)
            low_rate_length = x.shape[2]
            output_length = (
                low_rate_length * self.config.output_sample_rate // self.config.input_sample_rate
            )
            remainder = low_rate_length % self.config.hop_length
            if remainder:
                x = mx.pad(
                    x,
                    [(0, 0), (0, 0), (0, self.config.hop_length - remainder)],
                )
            mel_for_bwe = self._compute_mel(x).transpose(0, 1, 3, 2)
            mx.eval(mel_for_bwe)
            sink.phase_advance(phase)
            residual = self.bwe_generator(
                mel_for_bwe,
                reporter=sink,
                phase="synthesize bandwidth extension",
            )
            mx.eval(residual)
            sink.phase_advance(phase)
            del mel_for_bwe
            skip = self.resampler(x)
            mx.eval(skip)
            del x
            if tuple(residual.shape) != tuple(skip.shape):
                raise RuntimeError(
                    "bandwidth-extension residual and resampled base disagree: "
                    f"{tuple(residual.shape)} vs {tuple(skip.shape)}"
                )
            full_band = skip + residual
            del residual, skip
            result = mx.clip(full_band[:, :, :output_length], -1.0, 1.0)
            del full_band
            mx.eval(result)
            sink.phase_advance(phase)
            result = result.astype(input_dtype)
            mx.eval(result)
            return result
        finally:
            sink.phase_end(phase)


def create_vocoder_from_checkpoint(path: Path | str) -> VocoderWithBWE:
    return VocoderWithBWE(BWEVocoderConfig.from_checkpoint(path))


__all__ = [
    "BWEVocoderConfig",
    "Vocoder",
    "VocoderConfig",
    "VocoderWithBWE",
    "create_vocoder_from_checkpoint",
]
