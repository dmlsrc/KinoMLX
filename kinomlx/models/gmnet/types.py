"""Typed requests for the public GMNet recipe surface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from kinomlx.config_constraints import PathSuffixConstraint
from kinomlx.config_records import config_record
from kinomlx.io.output_paths import DEFAULT_OUTPUT_DIRECTORY
from kinomlx.io.output_rules import EXACT_REPLACEMENT

from .output_rules import OUTPUT_FORMATS

_SDR_IMAGE = PathSuffixConstraint(frozenset({".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff"}))


@config_record
@dataclass(frozen=True)
class GMNetRequest:
    """Expand one display-referred SDR still into scene-linear HDR."""

    image: Path = field(metadata={"config_constraints": (_SDR_IMAGE,)})

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", Path(self.image).expanduser())


@config_record
@dataclass(frozen=True)
class GMNetExpandConfig:
    """Per-invocation still expansion inputs."""

    image: Path | None = field(default=None, metadata={"config_constraints": (_SDR_IMAGE,)})


@config_record
@dataclass(frozen=True)
class GMNetOutputConfig:
    """Select exact GMNet still artifacts and replacement policy.

    ``path`` is an exact primary ``.exr`` or ``.heic`` artifact. Without it,
    ``directory`` and ``prefix`` select sibling artifacts. ``None`` format
    switches choose both formats for a derived stem, or only the suffix named
    by an exact path.
    """

    config_table: ClassVar[str] = "[output]"

    path: Path | None = field(
        default=None,
        metadata={
            "config_constraints": (PathSuffixConstraint(frozenset({".exr", ".heic"})),),
            "config_rules": (OUTPUT_FORMATS,),
        },
    )
    directory: Path = DEFAULT_OUTPUT_DIRECTORY
    prefix: str | None = None
    exr: bool | None = None
    heic: bool | None = None
    save_gain_map: bool | None = None
    save_run_log: bool | None = None
    save_console_log: bool | None = None
    save_effective_config: bool | None = None
    save_all_sidecars: bool = False
    force: bool = field(default=False, metadata={"config_rules": (EXACT_REPLACEMENT,)})


__all__ = ["GMNetExpandConfig", "GMNetOutputConfig", "GMNetRequest"]
