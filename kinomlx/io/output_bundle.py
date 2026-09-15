"""Shared reservation and rollback primitives for image output bundles."""

import logging
import stat
from pathlib import Path
from typing import Protocol, Self

from kinomlx.errors import KinoMLXError

from .output_paths import OutputCollisionError
from .reservation import PathReservation, PathReservationError

_log = logging.getLogger(__name__)


class BundleOutputError(KinoMLXError, RuntimeError):
    """An image output bundle cannot be reserved or published safely."""


class BundleCollisionError(BundleOutputError, OutputCollisionError):
    """A generated output stem can be retried."""


class RetainedTransactionError(BundleOutputError):
    """Rollback failed; retain recovery files."""


class BundlePlan(Protocol):
    @property
    def directory(self) -> Path: ...

    @property
    def force(self) -> bool: ...

    def targets(self) -> tuple[Path, ...]: ...


class BundleReservation[Plan: BundlePlan]:
    """Own hidden markers and target modes for one complete image bundle."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.modes: dict[Path, int] = {}
        self._reservations: dict[Path, PathReservation] = {}
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> Self:
        if self._active:
            raise BundleOutputError("Image output reservation is already active")
        targets = self.plan.targets()
        if len(set(targets)) != len(targets):
            raise BundleOutputError("Image output selections resolve to duplicate targets")
        self.plan.directory.mkdir(parents=True, exist_ok=True)
        try:
            for target in targets:
                try:
                    reservation = PathReservation.acquire(target)
                except PathReservationError as exc:
                    raise BundleCollisionError(f"cannot reserve output target: {exc}") from exc
                self._reservations[target] = reservation
                try:
                    status = target.lstat()
                except FileNotFoundError:
                    self.modes[target] = reservation.creation_mode
                    continue
                if not stat.S_ISREG(status.st_mode):
                    raise BundleCollisionError(
                        f"output target {target} is not a regular file; symbolic links "
                        "and other special files are unsupported"
                    )
                if not self.plan.force:
                    raise BundleCollisionError(
                        f"output target {target} exists; enable replacement to replace it"
                    )
                self.modes[target] = status.st_mode & 0o777
        except BaseException:
            self.close()
            raise
        self._active = True
        return self

    def complete(self) -> None:
        self.close()

    def close(self) -> None:
        for target, reservation in tuple(self._reservations.items()):
            try:
                reservation.release()
            except OSError as exc:
                _log.warning("could not remove image reservation marker for %s: %s", target, exc)
        self._reservations.clear()
        self._active = False

    def __exit__(self, *_exc_info: object) -> None:
        if self._active:
            self.close()


def publish_staged(temporary: dict[Path, Path], modes: dict[Path, int], transaction: Path) -> None:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for index, target in enumerate(temporary):
            prior = transaction / f"prior-{index}"
            if target.exists():
                target.rename(prior)
                backups[target] = prior
            temporary[target].chmod(modes[target])
            temporary[target].rename(target)
            published.append(target)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target in reversed(published):
            try:
                target.rename(temporary[target])
            except OSError as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        for target, prior in backups.items():
            if not prior.exists():
                continue
            try:
                prior.rename(target)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        if rollback_errors:
            raise RetainedTransactionError(
                f"Image publication failed ({exc}) and rollback could not restore "
                f"every target ({'; '.join(rollback_errors)}). Recovery files were "
                f"retained at {transaction}"
            ) from exc
        raise


def protect_sources(targets: tuple[Path, ...], sources: tuple[Path, ...]) -> None:
    """Reject lexical, symbolic, and hard-link aliases of any input."""
    for target in targets:
        target = target.expanduser()
        for source in sources:
            source = source.expanduser()
            if target.resolve() == source.resolve() or (
                target.exists() and source.exists() and target.samefile(source)
            ):
                raise BundleOutputError(f"output cannot overwrite an input: {target}")
