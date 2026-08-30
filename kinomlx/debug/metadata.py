"""Paths and structured metadata for reproducible generation sidecars."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from kinomlx.errors import KinoMLXError
from kinomlx.io.atomic import write_text_atomic, write_text_exclusive
from kinomlx.reporting import TimingReporter


class SidecarError(KinoMLXError, RuntimeError):
    """A requested reproducibility sidecar could not be written."""


@dataclass(frozen=True)
class ExecutionSidecarSpec:
    """One host-owned execution sidecar and its config selector."""

    artifact: str
    selector: str
    suffix: str


EXECUTION_SIDECAR_SPECS = (
    ExecutionSidecarSpec("run_log", "save_run_log", "_run.json"),
    ExecutionSidecarSpec("execution_log", "save_console_log", "_console.log"),
    ExecutionSidecarSpec("effective_config", "save_effective_config", "_config.toml"),
)


def sidecar_selected(selected: object, *, save_all: object) -> bool:
    """Return whether an explicit selector or inherited save-all enables a sidecar."""
    return selected is True or (selected is None and save_all is True)


def execution_sidecar_paths(base: Path | str) -> dict[str, Path]:
    """Return host sidecar paths for one suffix-free output stem."""
    stem = Path(base)
    return {
        spec.artifact: stem.parent / f"{stem.name}{spec.suffix}" for spec in EXECUTION_SIDECAR_SPECS
    }


def selected_execution_sidecar_paths(
    base: Path | str,
    options: object,
) -> dict[str, Path]:
    """Return host sidecars enabled by one model's output record."""
    paths = execution_sidecar_paths(base)
    return {
        spec.artifact: paths[spec.artifact]
        for spec in EXECUTION_SIDECAR_SPECS
        if sidecar_selected(
            getattr(options, spec.selector),
            save_all=getattr(options, "save_all_sidecars", False),
        )
    }


def sidecar_failure(
    artifact: str,
    path: Path,
    exc: BaseException,
) -> dict[str, str]:
    """Return one JSON-ready auxiliary-artifact failure record."""
    return {
        "artifact": artifact,
        "path": str(path),
        "error": f"{type(exc).__name__}: {exc}",
    }


def normalized_video_path(path: Path | str) -> Path:
    """Return the primary path the native output adapter will materialize."""
    resolved = Path(path)
    return resolved if resolved.suffix.lower() == ".mp4" else resolved.with_suffix(".mp4")


@dataclass(frozen=True)
class SidecarPaths:
    """Deterministic host paths plus model-contributed tensor artifacts."""

    video: Path
    run_log: Path
    execution_log: Path
    effective_config: Path
    audio_waveform: Path
    original_video: Path
    model_artifacts: tuple[tuple[str, Path], ...] = ()

    @classmethod
    def for_output(cls, path: Path | str) -> SidecarPaths:
        video = normalized_video_path(path)
        stem = video.stem
        parent = video.parent
        execution = execution_sidecar_paths(video.with_suffix(""))
        return cls(
            video=video,
            run_log=execution["run_log"],
            execution_log=execution["execution_log"],
            effective_config=execution["effective_config"],
            audio_waveform=video.with_suffix(".wav"),
            original_video=parent / f"{stem}_orig.mp4",
        )

    def with_model_artifacts(self, artifacts: dict[str, Path]) -> SidecarPaths:
        """Return paths extended by the selected model's artifact vocabulary."""
        return replace(
            self,
            model_artifacts=tuple(sorted(artifacts.items())),
        )

    def artifact_paths(self) -> dict[str, Path]:
        """Return model-contributed tensor artifact paths by stable name."""
        return dict(self.model_artifacts)

    def execution_paths(self) -> dict[str, Path]:
        """Return every host-owned execution sidecar by stable artifact name."""
        return {spec.artifact: getattr(self, spec.artifact) for spec in EXECUTION_SIDECAR_SPECS}

    def selected_execution_paths(self, options: object) -> dict[str, Path]:
        """Return execution sidecars enabled by a model's output record."""
        paths = self.execution_paths()
        return {
            spec.artifact: paths[spec.artifact]
            for spec in EXECUTION_SIDECAR_SPECS
            if sidecar_selected(
                getattr(options, spec.selector),
                save_all=getattr(options, "save_all_sidecars", False),
            )
        }

    def auxiliary_paths(self) -> dict[str, Path]:
        """Return every output-adjacent artifact except the primary video."""
        return {
            **self.artifact_paths(),
            **self.execution_paths(),
            "audio_waveform": self.audio_waveform,
            "original_video": self.original_video,
        }


def initialize_execution_log(path: Path, argv: list[str]) -> None:
    """Create a human-readable log with the exact invocation as its header."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, f"Command: {shlex.join(argv)}\n\n")
    except OSError as exc:
        raise SidecarError(f"cannot initialize execution log {path}: {exc}") from exc


def write_effective_config(
    path: Path,
    text: str,
    *,
    replace_existing: bool = False,
) -> None:
    """Publish serialized effective config, replacing only when authorized."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = write_text_atomic if replace_existing else write_text_exclusive
        writer(path, text)
    except OSError as exc:
        raise SidecarError(f"cannot write effective config {path}: {exc}") from exc


class RunRecord:
    """Incrementally write one structured run record and timing snapshot."""

    def __init__(
        self,
        path: Path,
        *,
        model: str,
        invocation: Mapping[str, object],
        argv: list[str],
        timings: TimingReporter,
        planned_outputs: Mapping[str, str],
        sidecar_errors: list[dict[str, str]] | None = None,
        preexisting_sidecars: Mapping[str, str] | None = None,
        restart: Mapping[str, object] | None = None,
    ) -> None:
        self.path = path
        self._timings = timings
        self._sidecar_errors = sidecar_errors if sidecar_errors is not None else []
        self._payload: dict[str, object] = {
            "schema_version": 1,
            "status": "started",
            "started_at": datetime.now(UTC).isoformat(),
            "model": model,
            "argv": list(argv),
            "invocation": dict(invocation),
            "planned_outputs": dict(planned_outputs),
            "preexisting_sidecars": dict(preexisting_sidecars or {}),
        }
        if restart is not None:
            self._payload["restart"] = dict(restart)
        self.write(status="started")

    def write(
        self,
        *,
        status: str,
        outputs: Mapping[str, object] | None = None,
        output_fingerprints: Mapping[str, str] | None = None,
        output_fingerprint_errors: Mapping[str, str] | None = None,
        generation: Mapping[str, object] | None = None,
        diagnostics: Mapping[str, object] | None = None,
        stale_sidecars: Mapping[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        """Persist the latest status without discarding the start metadata."""
        now = datetime.now(UTC).isoformat()
        self._payload["status"] = status
        self._payload["updated_at"] = now
        self._payload["timings"] = self._timings.to_dict()
        self._payload["sidecar_errors"] = [dict(error) for error in self._sidecar_errors]
        if status != "started":
            self._payload["finished_at"] = now
        if outputs is not None:
            self._payload["outputs"] = dict(outputs)
        if output_fingerprints is not None:
            self._payload["output_fingerprints"] = dict(output_fingerprints)
        if output_fingerprint_errors is not None:
            self._payload["output_fingerprint_errors"] = dict(output_fingerprint_errors)
        if generation is not None:
            self._payload["generation"] = dict(generation)
        if diagnostics is not None:
            self._payload["diagnostics"] = dict(diagnostics)
        if stale_sidecars is not None:
            self._payload["stale_sidecars"] = dict(stale_sidecars)
        if error is not None:
            self._payload["error"] = error
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(
                self.path,
                json.dumps(self._payload, indent=2, sort_keys=True, default=str) + "\n",
            )
        except OSError as exc:
            raise SidecarError(f"cannot write run log {self.path}: {exc}") from exc


__all__ = [
    "EXECUTION_SIDECAR_SPECS",
    "ExecutionSidecarSpec",
    "RunRecord",
    "SidecarError",
    "SidecarPaths",
    "execution_sidecar_paths",
    "initialize_execution_log",
    "normalized_video_path",
    "selected_execution_sidecar_paths",
    "sidecar_failure",
    "write_effective_config",
]
