"""Shared output path suffixes and image output selection, independent of tensor layouts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import PathSuffixConstraint
from kinomlx.config_records import config_record
from kinomlx.io.output_paths import DEFAULT_OUTPUT_DIRECTORY
from kinomlx.io.output_rules import EXACT_REPLACEMENT

LOSSLESS_IMAGE_PATH = PathSuffixConstraint(frozenset({".png", ".tif", ".tiff"}))
# The video writers emit one standard HEVC MP4, so an exact video path names it.
VIDEO_MP4_PATH = PathSuffixConstraint(frozenset({".mp4"}))


@config_record
@dataclass(frozen=True)
class ImageOutputConfig:
    """Select still paths, execution sidecars, and replacement policy."""

    config_table: ClassVar[str] = "[output]"

    path: Path | None = field(default=None, metadata={"config_constraints": (LOSSLESS_IMAGE_PATH,)})
    directory: Path = DEFAULT_OUTPUT_DIRECTORY
    prefix: str | None = None
    save_initial_noise: bool | None = None
    save_final_latents: bool | None = None
    save_text_conditioning: bool | None = None
    save_run_log: bool | None = None
    save_console_log: bool | None = None
    save_effective_config: bool | None = None
    save_all_sidecars: bool = False
    force: bool = field(default=False, metadata={"config_rules": (EXACT_REPLACEMENT,)})
