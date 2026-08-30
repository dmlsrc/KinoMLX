"""Reporter-level Instruments signpost tests."""

from __future__ import annotations

import json
from pathlib import Path

from kinomlx.profiling import SignpostEmitter, SignpostReporter
from kinomlx.reporting import RecordingReporter, Reporter


class _Backend:
    capture_enabled = True

    def __init__(self) -> None:
        self.next_id = 10
        self.events: list[tuple[str, int, dict[str, object]]] = []

    def generate_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _record(self, event: str, sid: int, message: str) -> None:
        self.events.append((event, sid, json.loads(message)))

    def begin(self, sid: int, message: str) -> None:
        self._record("begin", sid, message)

    def end(self, sid: int, message: str) -> None:
        self._record("end", sid, message)

    def event(self, sid: int, message: str) -> None:
        self._record("advance", sid, message)


class _FailingIdBackend(_Backend):
    def generate_id(self) -> int:
        raise RuntimeError("native failure")


def test_signpost_reporter_mirrors_and_delegates(tmp_path: Path) -> None:
    backend = _Backend()
    ticks = iter((100, 200, 300))
    emitter = SignpostEmitter(
        backend=backend,
        log_path=tmp_path / "events.log",
        clock_ns=lambda: next(ticks),
    )
    delegate = RecordingReporter()
    reporter = SignpostReporter(delegate, emitter=emitter)

    assert isinstance(reporter, Reporter)
    assert reporter.capture_enabled is True
    reporter.phase_start("distilled stage 1", total=8, unit="step")
    reporter.phase_advance("distilled stage 1", 2)
    reporter.phase_end("distilled stage 1")
    reporter.close()

    assert delegate.events == [
        ("start", "distilled stage 1", {"total": 8, "unit": "step"}),
        ("advance", "distilled stage 1", {"advance": 2}),
        ("end", "distilled stage 1", {}),
    ]
    assert [event[:2] for event in backend.events] == [
        ("begin", 11),
        ("advance", 11),
        ("end", 11),
    ]
    assert backend.events[1][2] == {
        "advance": 2,
        "completed": 2.0,
        "phase": "distilled stage 1",
    }
    lines = (tmp_path / "events.log").read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("100 begin distilled stage 1 ")
    assert lines[2].startswith("200 advance distilled stage 1 ")
    assert lines[3].startswith("300 end distilled stage 1 ")


def test_signpost_reporter_closes_restarted_and_aborted_phases() -> None:
    backend = _Backend()
    emitter = SignpostEmitter(backend=backend)
    reporter = SignpostReporter(emitter=emitter)

    reporter.phase_start("load")
    reporter.phase_start("load")
    reporter.phase_start("denoise")
    reporter.close()

    ends = [payload for event, _sid, payload in backend.events if event == "end"]
    assert [payload["status"] for payload in ends] == ["restarted", "aborted", "aborted"]


def test_sidecar_remains_available_without_native_backend(tmp_path: Path) -> None:
    emitter = SignpostEmitter(
        load_native=False,
        log_path=tmp_path / "events.log",
        clock_ns=lambda: 123,
    )
    assert emitter.available is True
    assert emitter.capture_enabled is False
    token = emitter.begin("encode", total=None, unit="it")
    emitter.end(token, completed=0, status="completed")
    emitter.close()
    text = (tmp_path / "events.log").read_text(encoding="utf-8")
    assert "123 begin encode" in text
    assert "123 end encode" in text


def test_native_id_failure_does_not_break_sidecar(tmp_path: Path) -> None:
    emitter = SignpostEmitter(
        backend=_FailingIdBackend(),
        log_path=tmp_path / "events.log",
        clock_ns=lambda: 123,
    )

    token = emitter.begin("load")
    emitter.end(token, status="completed")
    emitter.close()

    assert token.sid == 0
    assert "123 begin load" in (tmp_path / "events.log").read_text(encoding="utf-8")


def test_default_native_build_uses_the_persistent_user_cache(monkeypatch) -> None:
    import kinomlx.profiling.signpost as signpost_module

    build_directories = []

    def fake_load(build_dir):
        build_directories.append(build_dir)
        return _Backend()

    monkeypatch.setattr(signpost_module, "_load_native", fake_load)
    emitter = SignpostEmitter()
    emitter.close()

    assert build_directories == [Path("~/.cache/kinomlx/_native/signpost").expanduser()]
