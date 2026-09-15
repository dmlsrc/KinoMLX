"""Data-only LTX-2 video output configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.audio.trim_mode import AudioOnsetTrimConstraint
from kinomlx.config_constraints import ChoiceConstraint, NumberConstraint
from kinomlx.config_records import config_record
from kinomlx.debug.metadata import sidecar_selected
from kinomlx.io.output_config import VIDEO_MP4_PATH
from kinomlx.io.output_paths import DEFAULT_OUTPUT_DIRECTORY

from .output_rules import HDR_OUTPUT

VSR_SPATIAL_MODE_CHOICES = ("off", "fast", "balanced", "image")
VSR_TEMPORAL_MODE_CHOICES = ("normal", "high")
CUT_DETECT_MODE_CHOICES = ("off", "simple", "hist")
AUDIO_CODEC_CHOICES = ("alac", "aac")

# An unset toggle in this group inherits save_all_sidecars.
SAVE_ALL_SIDECAR_FIELDS = (
    "save_run_log",
    "save_console_log",
    "save_effective_config",
    "save_audio_sidecar",
)
# Each of these writes a whole image sequence, so save-all deliberately leaves
# it out; unset means off. The schema declares the same split as an excluded
# implication and a test holds the two statements together.
INDEPENDENT_SIDECAR_FIELDS = ("save_hdr_heic_frames", "save_vae_frames")


@config_record
@dataclass(frozen=True)
class OutputConfig:
    """Resolved VideoToolbox output parameters."""

    config_table: ClassVar[str] = "[output]"

    path: Path | None = field(default=None, metadata={"config_constraints": (VIDEO_MP4_PATH,)})
    directory: Path = DEFAULT_OUTPUT_DIRECTORY
    prefix: str | None = None
    vsr_spatial_mode: str = field(
        default="off",
        metadata={
            "config_constraints": (ChoiceConstraint(VSR_SPATIAL_MODE_CHOICES),),
            "config_rules": (HDR_OUTPUT,),
        },
    )
    target_fps: float | None = field(
        default=None,
        metadata={"config_constraints": (NumberConstraint(minimum=0, exclusive_minimum=True),)},
    )
    vsr_temporal_mode: str = field(
        default="normal",
        metadata={"config_constraints": (ChoiceConstraint(VSR_TEMPORAL_MODE_CHOICES),)},
    )
    cut_detect_mode: str = field(
        default="simple",
        metadata={"config_constraints": (ChoiceConstraint(CUT_DETECT_MODE_CHOICES),)},
    )
    cut_detect_threshold: float | None = field(
        default=None, metadata={"config_constraints": (NumberConstraint(minimum=0),)}
    )
    vsr_save_original: bool = False
    encode_quality: float = field(
        default=0.65, metadata={"config_constraints": (NumberConstraint(minimum=0, maximum=1),)}
    )
    audio_codec: str = field(
        default="alac", metadata={"config_constraints": (ChoiceConstraint(AUDIO_CODEC_CHOICES),)}
    )
    audio_onset_trim: str = field(
        default="auto", metadata={"config_constraints": (AudioOnsetTrimConstraint(),)}
    )
    save_run_log: bool | None = None
    save_console_log: bool | None = None
    save_effective_config: bool | None = None
    save_audio_sidecar: bool | None = None
    save_hdr_heic_frames: bool | None = None
    save_vae_frames: bool | None = None
    save_all_sidecars: bool = False


@dataclass(frozen=True)
class SidecarSelection:
    """Every LTX sidecar toggle resolved once against the save-all setting."""

    run_log: bool
    console_log: bool
    effective_config: bool
    audio_sidecar: bool
    hdr_heic_frames: bool
    vae_frames: bool


def sidecar_selection(config: OutputConfig) -> SidecarSelection:
    """Resolve every unset toggle one time for all consumers of one output record."""
    save_all = config.save_all_sidecars
    return SidecarSelection(
        run_log=sidecar_selected(config.save_run_log, save_all=save_all),
        console_log=sidecar_selected(config.save_console_log, save_all=save_all),
        effective_config=sidecar_selected(config.save_effective_config, save_all=save_all),
        audio_sidecar=sidecar_selected(config.save_audio_sidecar, save_all=save_all),
        hdr_heic_frames=sidecar_selected(config.save_hdr_heic_frames, save_all=False),
        vae_frames=sidecar_selected(config.save_vae_frames, save_all=False),
    )


__all__ = [
    "AUDIO_CODEC_CHOICES",
    "CUT_DETECT_MODE_CHOICES",
    "INDEPENDENT_SIDECAR_FIELDS",
    "SAVE_ALL_SIDECAR_FIELDS",
    "VSR_SPATIAL_MODE_CHOICES",
    "VSR_TEMPORAL_MODE_CHOICES",
    "OutputConfig",
    "SidecarSelection",
    "sidecar_selection",
]
