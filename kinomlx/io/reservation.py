"""Exclusive sidecar reservations that never masquerade as final artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class PathReservationError(RuntimeError):
    """A target cannot be exclusively reserved for publication."""


def reservation_path(target: Path | str) -> Path:
    """Return the deterministic hidden reservation peer for ``target``."""
    path = Path(target)
    return path.parent / f".{path.name}.kinomlx-reservation"


@dataclass
class PathReservation:
    """Own one hidden exclusive-create marker beside a publication target."""

    target: Path
    marker: Path
    creation_mode: int
    _active: bool = True

    @classmethod
    def acquire(cls, target: Path | str) -> PathReservation:
        """Create and own a private marker, or refuse an existing owner."""
        path = Path(target)
        marker = reservation_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666)
        except FileExistsError as exc:
            raise PathReservationError(
                f"{path} is already reserved by {marker}; another run may still "
                "own it. If no run is active, remove that marker and retry"
            ) from exc
        try:
            creation_mode = os.fstat(descriptor).st_mode & 0o777
            os.fchmod(descriptor, 0o600)
            message = f"pid={os.getpid()}\ntarget={path.name}\n".encode("ascii")
            os.write(descriptor, message)
        except BaseException:
            os.close(descriptor)
            marker.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        return cls(path, marker, creation_mode)

    @property
    def active(self) -> bool:
        """Whether this object still owns its marker."""
        return self._active

    def release(self) -> None:
        """Remove the owned marker."""
        if not self._active:
            return
        self.marker.unlink(missing_ok=True)
        self._active = False


__all__ = ["PathReservation", "PathReservationError", "reservation_path"]
