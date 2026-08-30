"""Checkpoint-derived LTX audio VAE architecture configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kinomlx.io.safetensors import read_metadata


@dataclass(frozen=True)
class AudioVAEConfig:
    """Supported LTX-2 audio VAE and preprocessing parameters."""

    channels: int = 128
    input_channels: int = 2
    output_channels: int = 2
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    latent_channels: int = 8
    double_latent: bool = True
    mel_bins: int = 64
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 1024
    is_causal: bool = True
    causality_axis: str = "height"
    norm_type: str = "pixel"
    resolution: int = 256

    def __post_init__(self) -> None:
        dimensions = (
            self.channels,
            self.input_channels,
            self.output_channels,
            self.num_res_blocks,
            self.latent_channels,
            self.mel_bins,
            self.sample_rate,
            self.hop_length,
            self.n_fft,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("audio VAE dimensions and rates must be positive")
        if not self.channel_multipliers or any(value <= 0 for value in self.channel_multipliers):
            raise ValueError("audio VAE channel multipliers must be positive")
        if self.causality_axis not in {"none", "width", "height"}:
            raise ValueError(f"unsupported audio VAE causality axis: {self.causality_axis}")
        if self.is_causal != (self.causality_axis != "none"):
            raise ValueError(
                "audio VAE STFT and convolution causality disagree: "
                f"is_causal={self.is_causal}, causality_axis={self.causality_axis!r}"
            )
        if self.norm_type != "pixel":
            raise ValueError(f"unsupported audio VAE normalization: {self.norm_type}")
        if not self.double_latent:
            raise ValueError("audio VAE requires double_latent=True")
        if self.mel_bins % self.downsample_factor:
            raise ValueError("mel_bins must divide evenly by the VAE downsample factor")
        expected_statistics = self.latent_channels * self.latent_mel_bins
        if expected_statistics != self.channels:
            raise ValueError(
                "audio VAE latent geometry must match per-channel statistics: "
                f"{expected_statistics} != {self.channels}"
            )

    @property
    def downsample_factor(self) -> int:
        return 1 << (len(self.channel_multipliers) - 1)

    @property
    def latent_mel_bins(self) -> int:
        return self.mel_bins // self.downsample_factor

    @classmethod
    def from_checkpoint(cls, path: Path | str) -> AudioVAEConfig:
        """Read the audio VAE and preprocessing fields from checkpoint metadata."""
        metadata = read_metadata(path)
        try:
            root = json.loads(metadata["config"])
        except KeyError as exc:
            raise ValueError(f"{path}: checkpoint metadata has no config") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: checkpoint config is invalid JSON") from exc
        audio = root.get("audio_vae") if isinstance(root, dict) else None
        if not isinstance(audio, dict):
            raise ValueError(f"{path}: checkpoint has no audio_vae config")
        params = audio.get("model", {}).get("params", {})
        ddconfig = params.get("ddconfig", {})
        preprocessing = audio.get("preprocessing", {})
        stft = preprocessing.get("stft", {})
        mel = preprocessing.get("mel", {})
        if ddconfig.get("mid_block_add_attention", False):
            raise ValueError("audio VAE mid-block attention is not supported")
        if ddconfig.get("attn_resolutions", []):
            raise ValueError("audio VAE attention resolutions are not supported")
        if bool(ddconfig.get("give_pre_end", False)):
            raise ValueError("audio VAE decoder give_pre_end=True is not supported")
        if bool(ddconfig.get("tanh_out", False)):
            raise ValueError("audio VAE decoder tanh_out=True is not supported")
        if bool(ddconfig.get("downsample_time", False)):
            raise ValueError("audio VAE temporal downsampling is not supported")
        mel_bins = ddconfig.get("mel_bins") or mel.get("n_mel_channels")
        if mel_bins is None:
            raise ValueError("audio VAE config does not declare mel bins")
        sample_rate = int(params.get("sampling_rate", 16000))
        audio_preprocessing = preprocessing.get("audio", {})
        preprocessing_rate = int(audio_preprocessing.get("sampling_rate", sample_rate))
        if preprocessing_rate != sample_rate:
            raise ValueError(
                "audio VAE model and preprocessing sample rates differ: "
                f"{sample_rate} != {preprocessing_rate}"
            )
        return cls(
            channels=int(ddconfig.get("ch", 128)),
            input_channels=int(ddconfig.get("in_channels", 2)),
            output_channels=int(ddconfig.get("out_ch", 2)),
            channel_multipliers=tuple(int(value) for value in ddconfig.get("ch_mult", (1, 2, 4))),
            num_res_blocks=int(ddconfig.get("num_res_blocks", 2)),
            latent_channels=int(ddconfig.get("z_channels", 8)),
            double_latent=bool(ddconfig.get("double_z", True)),
            mel_bins=int(mel_bins),
            sample_rate=sample_rate,
            hop_length=int(stft.get("hop_length", 160)),
            n_fft=int(stft.get("filter_length", 1024)),
            is_causal=bool(stft.get("causal", True)),
            causality_axis=str(ddconfig.get("causality_axis", "height")).lower(),
            norm_type=str(ddconfig.get("norm_type", "pixel")).lower(),
            resolution=int(ddconfig.get("resolution", 256)),
        )


__all__ = ["AudioVAEConfig"]
