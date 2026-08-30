"""No-clobber reservation shared by generic and model-specific converters."""

from __future__ import annotations

import logging
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from kinomlx.errors import KinoMLXError
from kinomlx.io.atomic import atomic_output_path
from kinomlx.io.reservation import PathReservation, PathReservationError


class WeightOutputError(KinoMLXError, RuntimeError):
    """A conversion target cannot be safely created or replaced."""


_log = logging.getLogger(__name__)


@contextmanager
def reserved_weight_output(
    target: Path,
    *,
    source: Path,
    force: bool,
) -> Iterator[Path]:
    """Yield a private peer that publishes only after conversion verifies.

    A hidden peer is exclusively created before work starts, so an interrupted
    conversion never leaves a zero-byte file at the final artifact name. A
    forced existing file remains untouched until the verified temporary
    atomically replaces it.
    """
    if target.expanduser().absolute() == source.expanduser().absolute():
        raise WeightOutputError("conversion output must not replace its source checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        reservation = PathReservation.acquire(target)
    except PathReservationError as exc:
        raise WeightOutputError(f"cannot reserve conversion output: {exc}") from exc
    try:
        try:
            status = target.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(status.st_mode):
                raise WeightOutputError(f"conversion output {target} is not a regular file")
            if not force:
                raise WeightOutputError(
                    f"conversion output {target} exists; enable replacement to replace it"
                )
        with atomic_output_path(target, temp_suffix=".conversion.tmp") as temporary:
            yield temporary
    finally:
        try:
            reservation.release()
        except OSError as exc:
            _log.warning(
                "could not remove conversion reservation marker %s: %s",
                reservation.marker,
                exc,
            )


__all__ = ["WeightOutputError", "reserved_weight_output"]
