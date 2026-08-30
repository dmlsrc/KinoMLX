"""Data-only host configuration records shared with model contributions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from kinomlx.audio.trim_mode import parse_trim_mode
from kinomlx.settings import EnvironmentSettings

DEFAULT_OUTPUT_DIRECTORY = Path("outputs")
DEFAULT_OUTPUT_PREFIX = "kinomlx"
VSR_SPATIAL_MODE_CHOICES = ("off", "fast", "balanced", "image")
VSR_TEMPORAL_MODE_CHOICES = ("normal", "high")
CUT_DETECT_MODE_CHOICES = ("off", "simple", "hist")
AUDIO_CODEC_CHOICES = ("alac", "aac")


@dataclass(frozen=True)
class OutputConfig(EnvironmentSettings):
    """Resolved VideoToolbox output parameters."""

    path: Path | None = None
    directory: Path = field(
        default=DEFAULT_OUTPUT_DIRECTORY,
        metadata={"env": "{{KINO_OUTPUT_DIR}}"},
    )
    prefix: str = DEFAULT_OUTPUT_PREFIX
    vsr_spatial_mode: str = "off"
    target_fps: float | None = None
    vsr_temporal_mode: str = "normal"
    cut_detect_mode: str = "simple"
    cut_detect_threshold: float | None = None
    vsr_save_original: bool = False
    encode_quality: float = 0.65
    audio_codec: str = "alac"
    audio_onset_trim: str = "auto"
    save_run_log: bool = False
    save_console_log: bool = False
    save_effective_config: bool = False
    save_audio_sidecar: bool = False
    save_hdr_heic_frames: bool = False
    save_vae_frames: bool = False
    save_all_sidecars: bool = False

    def __post_init__(self) -> None:
        if self.vsr_spatial_mode not in VSR_SPATIAL_MODE_CHOICES:
            valid = ", ".join(VSR_SPATIAL_MODE_CHOICES)
            raise ValueError(f"vsr_spatial_mode must be one of: {valid}")
        if self.target_fps is not None and (
            not math.isfinite(self.target_fps) or self.target_fps <= 0
        ):
            raise ValueError("target_fps must be finite and positive")
        if self.vsr_temporal_mode not in VSR_TEMPORAL_MODE_CHOICES:
            valid = ", ".join(VSR_TEMPORAL_MODE_CHOICES)
            raise ValueError(f"vsr_temporal_mode must be one of: {valid}")
        if self.cut_detect_mode not in CUT_DETECT_MODE_CHOICES:
            valid = ", ".join(CUT_DETECT_MODE_CHOICES)
            raise ValueError(f"cut_detect_mode must be one of: {valid}")
        if self.cut_detect_threshold is not None and (
            not math.isfinite(self.cut_detect_threshold) or self.cut_detect_threshold < 0.0
        ):
            raise ValueError("cut_detect_threshold must be finite and non-negative")
        if not math.isfinite(self.encode_quality) or not 0.0 <= self.encode_quality <= 1.0:
            raise ValueError("encode_quality must be between 0 and 1")
        if self.audio_codec not in AUDIO_CODEC_CHOICES:
            valid = ", ".join(AUDIO_CODEC_CHOICES)
            raise ValueError(f"audio_codec must be one of: {valid}")
        parse_trim_mode(self.audio_onset_trim)


__all__ = [
    "AUDIO_CODEC_CHOICES",
    "CUT_DETECT_MODE_CHOICES",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_OUTPUT_PREFIX",
    "OutputConfig",
    "VSR_SPATIAL_MODE_CHOICES",
    "VSR_TEMPORAL_MODE_CHOICES",
]
