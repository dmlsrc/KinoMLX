"""Typed requests for the public GMNet recipe surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kinomlx.settings import EnvironmentSettings


@dataclass(frozen=True)
class GMNetRequest:
    """Expand one display-referred SDR still into scene-linear HDR."""

    image: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", Path(self.image).expanduser())


@dataclass(frozen=True)
class GMNetExpandConfig:
    """Per-invocation still expansion inputs."""

    image: Path | None = None


@dataclass(frozen=True)
class GMNetOutputConfig(EnvironmentSettings):
    """Select exact GMNet still artifacts and replacement policy.

    ``path`` is an exact primary ``.exr`` or ``.heic`` artifact. Without it,
    ``directory`` and ``prefix`` select sibling artifacts. ``None`` format
    switches choose both formats for a derived stem, or only the suffix named
    by an exact path.
    """

    path: Path | None = None
    directory: Path = field(
        default=Path("outputs"),
        metadata={"env": "{{KINO_OUTPUT_DIR}}"},
    )
    prefix: str | None = None
    exr: bool | None = None
    heic: bool | None = None
    save_gain_map: bool | None = None
    save_run_log: bool | None = None
    save_console_log: bool | None = None
    save_effective_config: bool | None = None
    save_all_sidecars: bool = False
    force: bool = False


__all__ = ["GMNetExpandConfig", "GMNetOutputConfig", "GMNetRequest"]
