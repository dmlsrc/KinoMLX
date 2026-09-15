"""Application-wide generated output names, independent of model and format."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_DIRECTORY = Path("outputs")
DEFAULT_OUTPUT_PREFIX = "kinomlx"


class OutputCollisionError(FileExistsError):
    """A proposed output name is occupied by an artifact or another run."""


def timestamped_output_base(
    directory: Path | str,
    prefix: str | None,
    *,
    now: datetime | None = None,
) -> Path:
    """Build one extensionless ``<prefix>_YYYYMMDD_HHMMSS`` run stem."""
    value = (prefix or DEFAULT_OUTPUT_PREFIX).strip() or DEFAULT_OUTPUT_PREFIX
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    )
    name = sanitized.strip("._") or DEFAULT_OUTPUT_PREFIX
    timestamp = (now if now is not None else datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(directory).expanduser() / f"{name}_{timestamp}"


def reserve_output_base[T](
    base: Path,
    reserve: Callable[[Path], T],
    *,
    generated: bool,
) -> T:
    """Reserve a whole run, suffixing generated stems on name collisions only.

    The caller reserves the model's artifacts together and raises
    ``OutputCollisionError`` if any selected target is occupied. Explicit
    output paths are never renamed. Permission and other I/O errors propagate.
    """
    collision = 1
    while True:
        candidate = base if collision == 1 else base.with_name(f"{base.name}_{collision}")
        try:
            return reserve(candidate)
        except OutputCollisionError:
            if not generated:
                raise
            collision += 1


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_OUTPUT_PREFIX",
    "OutputCollisionError",
    "reserve_output_base",
    "timestamped_output_base",
]
