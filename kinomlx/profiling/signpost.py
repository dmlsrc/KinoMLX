"""Apple ``os_signpost`` emission and Reporter integration.

PyObjC exposes the OSLog read side but not the macro-based signpost emit API.
The stock Python logging surface therefore cannot create Instruments Points of
Interest intervals. KinoMLX uses the adjacent tiny C shim to establish the
required native calling image, while this module owns build caching, ctypes,
structured messages, and non-throwing fallback behavior.

The shim is compiled only when profiling is explicitly selected. The installed
CLI keeps it below the configured KinoMLX cache directory; low-level callers
fall back to the ordinary user cache. Keeping it outside the installed package
means read-only wheels remain profileable and no architecture-specific binary
needs to be checked into the source tree.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from kinomlx.reporting import NullReporter, Reporter

_log = logging.getLogger(__name__)
_SOURCE = Path(__file__).with_name("_signpost.c")
_BUILD_LOCK = threading.Lock()
_DEFAULT_BUILD_DIR = Path("~/.cache/kinomlx/_native/signpost").expanduser()


class _SignpostBackend(Protocol):
    """Native operations used by :class:`SignpostEmitter`."""

    @property
    def capture_enabled(self) -> bool: ...

    def generate_id(self) -> int: ...

    def begin(self, sid: int, message: str) -> None: ...

    def end(self, sid: int, message: str) -> None: ...

    def event(self, sid: int, message: str) -> None: ...


class _NativeBackend:
    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        library.kino_signpost_id_generate.argtypes = []
        library.kino_signpost_id_generate.restype = ctypes.c_uint64
        library.kino_signpost_enabled.argtypes = []
        library.kino_signpost_enabled.restype = ctypes.c_int
        for name in ("interval_begin", "interval_end", "event"):
            function = getattr(library, f"kino_signpost_{name}")
            function.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            function.restype = None

    @property
    def capture_enabled(self) -> bool:
        return bool(self._library.kino_signpost_enabled())

    def generate_id(self) -> int:
        return int(self._library.kino_signpost_id_generate())

    def begin(self, sid: int, message: str) -> None:
        self._library.kino_signpost_interval_begin(sid, message.encode("utf-8"))

    def end(self, sid: int, message: str) -> None:
        self._library.kino_signpost_interval_end(sid, message.encode("utf-8"))

    def event(self, sid: int, message: str) -> None:
        self._library.kino_signpost_event(sid, message.encode("utf-8"))


def _library_path(build_dir: Path) -> Path:
    source = _SOURCE.read_bytes()
    identity = hashlib.sha256(source + platform.machine().encode()).hexdigest()[:16]
    return build_dir / f"_signpost-{identity}.dylib"


def _build_library(build_dir: Path) -> Path:
    """Build and cache the native shim, returning its stable path."""
    build_dir.mkdir(parents=True, exist_ok=True)
    library = _library_path(build_dir)
    if library.is_file():
        return library
    with _BUILD_LOCK:
        if library.is_file():
            return library
        temporary = library.with_name(f".{library.name}.{os.getpid()}.tmp")
        try:
            subprocess.run(
                [
                    "/usr/bin/clang",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-Wall",
                    "-Wextra",
                    "-o",
                    str(temporary),
                    str(_SOURCE),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            temporary.replace(library)
        finally:
            temporary.unlink(missing_ok=True)
    return library


def _load_native(build_dir: Path) -> _NativeBackend:
    return _NativeBackend(ctypes.CDLL(str(_build_library(build_dir))))


def _message(phase: str, **fields: object) -> str:
    payload: dict[str, object] = {"phase": phase, **fields}
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class SignpostToken:
    """Opaque identity for one active signpost interval."""

    sid: int
    phase: str


class SignpostEmitter:
    """Best-effort native intervals plus an optional monotonic sidecar log."""

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        build_dir: Path | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        backend: _SignpostBackend | None = None,
        load_native: bool = True,
    ) -> None:
        self._clock_ns = clock_ns
        self._backend = backend
        self._log_file: TextIO | None = None
        self._lock = threading.RLock()
        if self._backend is None and load_native:
            native_dir = build_dir or _DEFAULT_BUILD_DIR
            try:
                self._backend = _load_native(native_dir)
            except Exception as exc:  # observability must never break generation
                _log.warning("Native signposts are unavailable: %s", exc)
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = log_path.open("w", encoding="utf-8", buffering=1)
                self._log_file.write(
                    "# KinoMLX signpost log; monotonic_ns event phase payload_json\n"
                )
            except OSError as exc:
                _log.warning("Could not open signpost log %s: %s", log_path, exc)

    @property
    def available(self) -> bool:
        """Whether at least one native or sidecar emission path is active."""
        return self._backend is not None or self._log_file is not None

    @property
    def capture_enabled(self) -> bool:
        """Whether Instruments currently requests native signpost records."""
        if self._backend is None:
            return False
        try:
            return self._backend.capture_enabled
        except Exception:
            return False

    def _write(self, event: str, phase: str, message: str) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.write(f"{self._clock_ns()} {event} {phase} {message}\n")
        except OSError as exc:
            _log.warning("Disabling failed signpost sidecar: %s", exc)
            with suppress(OSError):
                self._log_file.close()
            self._log_file = None

    def begin(self, phase: str, **fields: object) -> SignpostToken:
        with self._lock:
            sid = 0
            if self._backend is not None:
                try:
                    sid = self._backend.generate_id()
                except Exception as exc:
                    _log.warning("Disabling failed native signposts: %s", exc)
                    self._backend = None
            message = _message(phase, **fields)
            if self._backend is not None:
                try:
                    self._backend.begin(sid, message)
                except Exception as exc:
                    _log.warning("Disabling failed native signposts: %s", exc)
                    self._backend = None
            self._write("begin", phase, message)
            return SignpostToken(sid=sid, phase=phase)

    def advance(self, token: SignpostToken, **fields: object) -> None:
        with self._lock:
            message = _message(token.phase, **fields)
            if self._backend is not None:
                try:
                    self._backend.event(token.sid, message)
                except Exception as exc:
                    _log.warning("Disabling failed native signposts: %s", exc)
                    self._backend = None
            self._write("advance", token.phase, message)

    def end(self, token: SignpostToken, **fields: object) -> None:
        with self._lock:
            message = _message(token.phase, **fields)
            if self._backend is not None:
                try:
                    self._backend.end(token.sid, message)
                except Exception as exc:
                    _log.warning("Disabling failed native signposts: %s", exc)
                    self._backend = None
            self._write("end", token.phase, message)

    def close(self) -> None:
        with self._lock:
            if self._log_file is not None:
                with suppress(OSError):
                    self._log_file.close()
                self._log_file = None


@dataclass
class _ActivePhase:
    token: SignpostToken
    completed: float = 0.0


class SignpostReporter:
    """Reporter decorator that mirrors lifecycle events into signposts.

    Unlike instrumentation inside model math, Reporter phases already delimit
    orchestration and materialization boundaries. Mirroring them preserves the
    measured execution path and makes the same installed CLI traceable without
    adding ``mx.eval`` fences or claiming module-level GPU ownership.
    """

    def __init__(
        self,
        delegate: Reporter | None = None,
        *,
        log_path: Path | None = None,
        build_dir: Path | None = None,
        emitter: SignpostEmitter | None = None,
    ) -> None:
        self._delegate = delegate if delegate is not None else NullReporter()
        self._emitter = emitter or SignpostEmitter(log_path=log_path, build_dir=build_dir)
        self._active: dict[str, _ActivePhase] = {}

    @property
    def capture_enabled(self) -> bool:
        return self._emitter.capture_enabled

    def __enter__(self) -> SignpostReporter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
    ) -> None:
        previous = self._active.pop(phase, None)
        if previous is not None:
            self._emitter.end(previous.token, completed=previous.completed, status="restarted")
        token = self._emitter.begin(phase, total=total, unit=unit)
        self._active[phase] = _ActivePhase(token)
        self._delegate.phase_start(phase, total=total, unit=unit)

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        active = self._active.get(phase)
        if active is not None:
            active.completed += advance
            self._emitter.advance(
                active.token,
                advance=advance,
                completed=active.completed,
            )
        self._delegate.phase_advance(phase, advance)

    def phase_end(self, phase: str) -> None:
        active = self._active.pop(phase, None)
        if active is not None:
            self._emitter.end(active.token, completed=active.completed, status="completed")
        self._delegate.phase_end(phase)

    def phase_peak_memory(self, phase: str, peak_memory_bytes: int) -> None:
        """Forward optional presentation diagnostics without changing signposts."""
        receiver = getattr(self._delegate, "phase_peak_memory", None)
        if callable(receiver):
            receiver(phase, peak_memory_bytes)

    def close(self) -> None:
        for phase in reversed(tuple(self._active)):
            active = self._active.pop(phase)
            self._emitter.end(active.token, completed=active.completed, status="aborted")
        self._emitter.close()


__all__ = ["SignpostEmitter", "SignpostReporter", "SignpostToken"]
