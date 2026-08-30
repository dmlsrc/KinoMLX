"""Generic safetensors persistence for model-contributed artifacts."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from kinomlx.artifacts import TensorArtifact
from kinomlx.io.fingerprints import file_sha256
from kinomlx.io.safetensors import save_weights
from kinomlx.reporting import NullReporter, Reporter

from .metadata import sidecar_failure

if TYPE_CHECKING:
    import mlx.core as mx

_LOG = logging.getLogger(__name__)


class SidecarArtifactSink:
    """Persist requested MLX artifacts without retaining them across stages."""

    def __init__(
        self,
        paths: Mapping[str, Path],
        *,
        enabled: Collection[str],
        reporter: Reporter | None = None,
        errors: list[dict[str, str]] | None = None,
    ) -> None:
        self._paths = dict(paths)
        self._enabled = frozenset(enabled)
        missing = self._enabled - self._paths.keys()
        if missing:
            raise ValueError(f"enabled artifacts have no output path: {', '.join(sorted(missing))}")
        self.reporter = reporter if reporter is not None else NullReporter()
        self._manifest: dict[str, str] = {}
        self._fingerprints: dict[str, str] = {}
        self._fingerprint_errors: dict[str, str] = {}
        self._errors = errors if errors is not None else []

    @property
    def manifest(self) -> dict[str, str]:
        """Return sidecars that were successfully materialized."""
        return dict(self._manifest)

    @property
    def errors(self) -> list[dict[str, str]]:
        """Return requested sidecars that could not be materialized."""
        return [dict(error) for error in self._errors]

    @property
    def fingerprints(self) -> dict[str, str]:
        """Return observational identities for successfully saved sidecars."""
        return dict(self._fingerprints)

    @property
    def fingerprint_errors(self) -> dict[str, str]:
        """Return non-fatal failures from attempted sidecar fingerprinting."""
        return dict(self._fingerprint_errors)

    @contextmanager
    def _reporting_phase(self, phase: str) -> Iterator[None]:
        """Keep host reporter failures outside artifact persistence semantics."""
        try:
            self.reporter.phase_start(phase)
        except Exception as exc:
            _LOG.warning("Could not start sidecar reporting phase %r: %s", phase, exc)
        try:
            yield
        finally:
            try:
                self.reporter.phase_end(phase)
            except Exception as exc:
                _LOG.warning("Could not end sidecar reporting phase %r: %s", phase, exc)

    def _save(
        self,
        path: Path,
        arrays: Mapping[str, mx.array],
        metadata: dict[str, str],
        *,
        phase: str,
        manifest_key: str,
    ) -> None:
        with self._reporting_phase(phase):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                save_weights(path, arrays, metadata)
                # Do not let later receipt/logging failures retain stage tensors
                # through this frame's locals or exception context.
                del arrays, metadata
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failure = sidecar_failure(manifest_key, path, exc)
                error = f"{type(exc).__name__}: {exc}"
                self._errors.append(failure)
                _LOG.warning(
                    "Could not materialize %s sidecar %s: %s",
                    manifest_key,
                    path,
                    error,
                )
                return
        self._manifest[manifest_key] = str(path)
        self._fingerprint(path, manifest_key=manifest_key)

    def _fingerprint(self, path: Path, *, manifest_key: str) -> None:
        """Fingerprint without retaining the tensor-bearing save frame on failure."""
        try:
            self._fingerprints[manifest_key] = file_sha256(path)
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._fingerprint_errors[manifest_key] = error
            _LOG.warning(
                "Could not fingerprint %s sidecar %s: %s",
                manifest_key,
                path,
                error,
            )

    def save(self, artifact: TensorArtifact) -> None:
        """Persist one enabled envelope without interpreting model vocabulary."""
        if artifact.name not in self._enabled:
            return
        self._save(
            self._paths[artifact.name],
            dict(artifact.tensors),
            dict(artifact.metadata),
            phase=artifact.reporting_phase,
            manifest_key=artifact.name,
        )


__all__ = ["SidecarArtifactSink"]
