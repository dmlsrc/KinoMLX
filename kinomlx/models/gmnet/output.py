"""Typed, transactional EXR/HEIC/gain-map output for GMNet expansion."""

from __future__ import annotations

import logging
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kinomlx.debug.metadata import selected_execution_sidecar_paths, sidecar_selected
from kinomlx.errors import KinoMLXError
from kinomlx.io.reservation import PathReservation, PathReservationError
from kinomlx.reporting import NullReporter, Reporter

from .expand import GAIN_MAP_SIDECAR_SUFFIX, ExpansionResult
from .types import GMNetOutputConfig, GMNetRequest


class GMNetOutputError(KinoMLXError, RuntimeError):
    """A typed refusal or failure while publishing GMNet artifacts."""


class _RetainedTransactionError(GMNetOutputError):
    """Publication failed and recovery files must remain on disk."""


_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GMNetArtifactSet:
    """The files published by one successful GMNet output transaction."""

    exr: Path | None = None
    heic: Path | None = None
    gain_map: Path | None = None

    def paths(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.exr, self.heic, self.gain_map) if path is not None)

    def to_dict(self) -> dict[str, str]:
        return {
            name: str(path)
            for name, path in (
                ("exr", self.exr),
                ("heic", self.heic),
                ("gain_map", self.gain_map),
            )
            if path is not None
        }


@dataclass(frozen=True)
class GMNetOutputPlan:
    """Resolved target names that can be reserved before model inference."""

    source: Path
    artifacts: GMNetArtifactSet
    execution_sidecars: tuple[tuple[str, Path], ...]
    force: bool

    @property
    def directory(self) -> Path:
        paths = self.artifacts.paths()
        if not paths:
            raise GMNetOutputError("GMNet output plan has no artifacts")
        return paths[0].parent

    def reserve(self) -> GMNetOutputReservation:
        """Exclusively reserve absent targets and inspect forced replacements."""
        return GMNetOutputReservation(self)

    def sidecar_paths(self) -> dict[str, Path]:
        """Return selected execution sidecars by stable artifact name."""
        return dict(self.execution_sidecars)

    def targets(self) -> tuple[Path, ...]:
        """Return every primary artifact and selected sidecar target."""
        return (*self.artifacts.paths(), *(path for _name, path in self.execution_sidecars))


class GMNetOutputReservation:
    """Own hidden target markers until one transaction publishes or aborts."""

    def __init__(self, plan: GMNetOutputPlan) -> None:
        self.plan = plan
        self.modes: dict[Path, int] = {}
        self._reservations: dict[Path, PathReservation] = {}
        self._active = False

    @property
    def active(self) -> bool:
        """Whether this reservation currently owns its selected targets."""
        return self._active

    def __enter__(self) -> GMNetOutputReservation:
        if self._active:
            raise GMNetOutputError("GMNet output reservation is already active")
        targets = self.plan.targets()
        if len(set(targets)) != len(targets):
            raise GMNetOutputError("GMNet output selections resolve to duplicate targets")
        self.plan.directory.mkdir(parents=True, exist_ok=True)
        try:
            for target in targets:
                if target.resolve() == self.plan.source.resolve():
                    raise GMNetOutputError(f"output target {target} would replace the source image")
                try:
                    reservation = PathReservation.acquire(target)
                except PathReservationError as exc:
                    raise GMNetOutputError(f"cannot reserve output target: {exc}") from exc
                self._reservations[target] = reservation
                try:
                    status = target.lstat()
                except FileNotFoundError:
                    self.modes[target] = reservation.creation_mode
                    continue
                if not stat.S_ISREG(status.st_mode):
                    raise GMNetOutputError(
                        f"output target {target} is not a regular file; symbolic links "
                        "and other special files are unsupported"
                    )
                if not self.plan.force:
                    raise GMNetOutputError(
                        f"output target {target} exists; enable replacement to replace it"
                    )
                self.modes[target] = status.st_mode & 0o777
        except BaseException:
            self.close()
            raise
        self._active = True
        return self

    def complete(self) -> None:
        """Relinquish hidden markers after the transaction has replaced targets."""
        self.close()

    def close(self) -> None:
        """Remove every hidden marker still owned by this reservation."""
        for target, reservation in tuple(self._reservations.items()):
            try:
                reservation.release()
            except OSError as exc:
                _log.warning(
                    "could not remove GMNet reservation marker for %s: %s",
                    target,
                    exc,
                )
        self._reservations.clear()
        self._active = False

    def __exit__(self, *_exc_info: object) -> None:
        if self._active:
            self.close()


def plan_gmnet_output(
    request: GMNetRequest,
    config: GMNetOutputConfig,
) -> GMNetOutputPlan:
    """Resolve exact artifact paths without touching the filesystem."""
    source = request.image.expanduser()
    exact = None if config.path is None else Path(config.path).expanduser()
    if exact is not None:
        suffix = exact.suffix.lower()
        if suffix not in {".exr", ".heic"}:
            raise GMNetOutputError("exact GMNet output path must end in .exr or .heic")
        default_exr = suffix == ".exr"
        default_heic = suffix == ".heic"
        write_exr = default_exr if config.exr is None else config.exr
        write_heic = default_heic if config.heic is None else config.heic
        if suffix == ".exr" and not write_exr:
            raise GMNetOutputError("an exact .exr path requires EXR output")
        if suffix == ".heic" and not write_heic:
            raise GMNetOutputError("an exact .heic path requires HEIC output")
        base = exact.with_suffix("")
    else:
        write_exr = True if config.exr is None else config.exr
        write_heic = True if config.heic is None else config.heic
        prefix = source.stem if config.prefix is None else config.prefix
        if not prefix or Path(prefix).name != prefix:
            raise GMNetOutputError("GMNet output prefix must be one filename component")
        base = Path(config.directory).expanduser() / prefix
        suffix = ""
    save_gain_map = sidecar_selected(
        config.save_gain_map,
        save_all=config.save_all_sidecars,
    )
    if not (write_exr or write_heic or save_gain_map):
        raise GMNetOutputError("nothing to write: EXR, HEIC, and gain-map outputs are disabled")

    exr = None
    if write_exr:
        exr = exact if exact is not None and suffix == ".exr" else base.parent / f"{base.name}.exr"
    heic = None
    if write_heic:
        heic = (
            exact if exact is not None and suffix == ".heic" else base.parent / f"{base.name}.heic"
        )
    gain_map = base.parent / f"{base.name}{GAIN_MAP_SIDECAR_SUFFIX}" if save_gain_map else None
    return GMNetOutputPlan(
        source=source,
        artifacts=GMNetArtifactSet(exr=exr, heic=heic, gain_map=gain_map),
        execution_sidecars=tuple(selected_execution_sidecar_paths(base, config).items()),
        force=config.force,
    )


def _default_exr_writer(result: ExpansionResult, path: Path) -> object:
    from kinomlx.media.signals import (
        ColorPrimaries,
        ColorTransfer,
        ExrDeliverySpec,
        ExrSampleType,
    )
    from kinomlx.videotoolbox.exr import save_exr_frame

    delivery = ExrDeliverySpec(
        primaries=ColorPrimaries.REC709,
        transfer=ColorTransfer.LINEAR,
        sample_type=ExrSampleType.FLOAT16,
        color_space_tag="Linear sRGB",
    )
    return save_exr_frame(result.linear_rgb, path, delivery=delivery)


def _default_heic_writer(result: ExpansionResult, path: Path) -> object:
    from kinomlx.media.signals import ColorPrimaries
    from kinomlx.videotoolbox.heic import save_pq_heic_frame

    return save_pq_heic_frame(
        result.linear_rgb,
        path,
        primaries=ColorPrimaries.REC709,
    )


def _default_gain_writer(result: ExpansionResult, path: Path, source: Path) -> object:
    from .expand import write_gain_map_sidecar

    return write_gain_map_sidecar(path, result, source_image=source)


class GMNetOutputSink:
    """Write a complete artifact bundle, then publish it as one transaction."""

    def __init__(
        self,
        plan: GMNetOutputPlan,
        *,
        reporter: Reporter | None = None,
        exr_writer: Callable[[ExpansionResult, Path], object] | None = None,
        heic_writer: Callable[[ExpansionResult, Path], object] | None = None,
        gain_writer: Callable[[ExpansionResult, Path, Path], object] | None = None,
    ) -> None:
        self.plan = plan
        self.reporter = reporter if reporter is not None else NullReporter()
        self.exr_writer = exr_writer if exr_writer is not None else _default_exr_writer
        self.heic_writer = heic_writer if heic_writer is not None else _default_heic_writer
        self.gain_writer = gain_writer if gain_writer is not None else _default_gain_writer

    def _write_temporaries(
        self,
        result: ExpansionResult,
        transaction: Path,
    ) -> dict[Path, Path]:
        temporary: dict[Path, Path] = {}
        phase = "write GMNet outputs"
        artifacts = self.plan.artifacts
        total = len(artifacts.paths())
        self.reporter.phase_start(phase, total=total, unit="artifact")
        try:
            for target, writer in (
                (artifacts.exr, self.exr_writer),
                (artifacts.heic, self.heic_writer),
            ):
                if target is None:
                    continue
                peer = transaction / target.name
                writer(result, peer)
                if not peer.is_file():
                    raise GMNetOutputError(f"output writer did not create {peer.name}")
                temporary[target] = peer
                self.reporter.phase_advance(phase)
            if artifacts.gain_map is not None:
                peer = transaction / artifacts.gain_map.name
                self.gain_writer(result, peer, self.plan.source)
                if not peer.is_file():
                    raise GMNetOutputError(f"output writer did not create {peer.name}")
                temporary[artifacts.gain_map] = peer
                self.reporter.phase_advance(phase)
            return temporary
        finally:
            self.reporter.phase_end(phase)

    def _publish(
        self,
        temporary: dict[Path, Path],
        reservation: GMNetOutputReservation,
        transaction: Path,
    ) -> None:
        backups: dict[Path, Path] = {}
        published: list[Path] = []
        try:
            for index, target in enumerate(self.plan.artifacts.paths()):
                prior = transaction / f"prior-{index}"
                if target.exists():
                    target.rename(prior)
                    backups[target] = prior
                temporary[target].chmod(reservation.modes[target])
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
                try:
                    survivors = sorted(str(path) for path in transaction.iterdir() if path.exists())
                except OSError as listing_exc:
                    survivors = [f"{transaction} (could not list recovery files: {listing_exc})"]
                retained = ", ".join(survivors) if survivors else str(transaction)
                raise _RetainedTransactionError(
                    f"GMNet publication failed ({exc}) and rollback could not "
                    f"restore every target ({'; '.join(rollback_errors)}). Recovery "
                    f"files were retained at {transaction}: {retained}"
                ) from exc
            raise

    def write(
        self,
        result: ExpansionResult,
        *,
        reservation: GMNetOutputReservation | None = None,
    ) -> GMNetArtifactSet:
        """Write and atomically publish every artifact in the plan."""
        owns_reservation = reservation is None
        selected = self.plan.reserve() if reservation is None else reservation
        if selected.plan != self.plan:
            raise ValueError("output reservation belongs to another GMNet output plan")
        if not owns_reservation and not selected.active:
            raise ValueError("passed GMNet output reservation is not active")
        transaction: Path | None = None
        retain_transaction = False
        try:
            if owns_reservation:
                selected.__enter__()
            transaction = Path(tempfile.mkdtemp(prefix=".gmnet-output-", dir=self.plan.directory))
            temporary = self._write_temporaries(result, transaction)
            self._publish(temporary, selected, transaction)
            if owns_reservation:
                selected.complete()
            return self.plan.artifacts
        except _RetainedTransactionError:
            retain_transaction = True
            raise
        except GMNetOutputError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise GMNetOutputError(f"cannot write GMNet outputs: {exc}") from exc
        finally:
            if owns_reservation:
                selected.__exit__(None, None, None)
            if transaction is not None and not retain_transaction:
                try:
                    shutil.rmtree(transaction)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    _log.warning(
                        "could not remove completed GMNet transaction %s: %s",
                        transaction,
                        exc,
                    )


def write_gmnet_output(
    result: ExpansionResult,
    request: GMNetRequest,
    config: GMNetOutputConfig,
    *,
    reporter: Reporter | None = None,
) -> GMNetArtifactSet:
    """Resolve, reserve, and transactionally publish one expansion result."""
    plan = plan_gmnet_output(request, config)
    return GMNetOutputSink(plan, reporter=reporter).write(result)


__all__ = [
    "GMNetArtifactSet",
    "GMNetOutputConfig",
    "GMNetOutputError",
    "GMNetOutputPlan",
    "GMNetOutputReservation",
    "GMNetOutputSink",
    "plan_gmnet_output",
    "write_gmnet_output",
]
