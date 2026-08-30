"""Host-neutral progress reporting.

Runtime code reports phase progress through :class:`Reporter`; it does not
own a terminal or import Rich. The CLI supplies a Rich-backed reporter, host
applications may provide their own adapter, and tests use
:class:`RecordingReporter`.

Log messages are deliberately outside this protocol. The stdlib logging
framework already provides a host-neutral message surface; Reporter carries
live phase totals, advancement, and completion.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable


class MemorySample(TypedDict):
    """One non-synchronizing allocator observation."""

    sequence: int
    elapsed_seconds: float
    event: str
    phase: str | None
    label: str | None
    active_phases: list[str]
    active_bytes: int
    cache_bytes: int
    peak_bytes_since_previous_sample: int


class MemorySamplingError(TypedDict):
    """One failed allocator observation retained for diagnostics."""

    elapsed_seconds: float
    event: str
    phase: str | None
    label: str | None
    error: str


class PhaseMemoryPeak(TypedDict):
    phase: str
    parent_phase: str | None
    depth: int
    peak_bytes: int


class MemoryReport(TypedDict, total=False):
    """JSON-ready allocator report; empty when sampling is disabled."""

    counter_source: str
    unit: str
    synchronizes_device: bool
    peak_reset_between_samples: bool
    observed_run_peak_bytes: int
    phase_peaks: list[PhaseMemoryPeak]
    samples: list[MemorySample]
    sampling_errors: list[MemorySamplingError]


class PhaseTimingRecord(TypedDict):
    phase: str
    parent_phase: str | None
    depth: int
    total: float | None
    unit: str
    completed: float
    started_seconds: float
    duration_seconds: float
    status: str


class TimingReport(TypedDict):
    """JSON-ready snapshot of elapsed time and phase relationships."""

    total_seconds: float
    phases: list[PhaseTimingRecord]


@runtime_checkable
class Reporter(Protocol):
    """Progress operations runtime code may use."""

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
    ) -> None:
        """Record that a phase began."""

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        """Advance a phase by ``advance`` units."""

    def phase_end(self, phase: str) -> None:
        """Record that a phase ended and release its live display."""


class NullReporter:
    """No-op reporter used by library defaults and hosts choosing silence."""

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
    ) -> None:
        pass

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        pass

    def phase_end(self, phase: str) -> None:
        pass


@dataclass
class RecordingReporter:
    """Capture ordered progress events for tests and host integration."""

    events: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
    ) -> None:
        self.events.append(("start", phase, {"total": total, "unit": unit}))

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        self.events.append(("advance", phase, {"advance": advance}))

    def phase_end(self, phase: str) -> None:
        self.events.append(("end", phase, {}))


@dataclass
class _PhaseTiming:
    phase: str
    parent_phase: str | None
    depth: int
    total: float | None
    unit: str
    started_seconds: float
    completed: float = 0.0
    ended_seconds: float | None = None
    peak_memory_bytes: int = 0


class TimingReporter:
    """Record monotonic phase timings while delegating presentation events."""

    def __init__(
        self,
        delegate: Reporter | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        memory_sampler: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self._delegate = delegate if delegate is not None else NullReporter()
        self._clock = clock
        self._started = clock()
        self._phases: list[_PhaseTiming] = []
        self._active: dict[str, _PhaseTiming] = {}
        self._stack: list[_PhaseTiming] = []
        self._memory_sampler = memory_sampler
        self._memory_samples: list[MemorySample] = []
        self._memory_errors: list[MemorySamplingError] = []

    def _sample_memory(
        self,
        event: str,
        *,
        phase: str | None = None,
        label: str | None = None,
    ) -> MemorySample | None:
        sampler = self._memory_sampler
        if sampler is None:
            return None
        elapsed = self._clock() - self._started
        try:
            counters = sampler()
            sample: MemorySample = {
                "sequence": len(self._memory_samples),
                "elapsed_seconds": elapsed,
                "event": event,
                "phase": phase,
                "label": label,
                "active_phases": [timing.phase for timing in self._stack],
                "active_bytes": max(0, int(counters["active_bytes"])),
                "cache_bytes": max(0, int(counters["cache_bytes"])),
                "peak_bytes_since_previous_sample": max(
                    0,
                    int(counters["peak_bytes"]),
                ),
            }
            self._memory_samples.append(sample)
            interval_peak = sample["peak_bytes_since_previous_sample"]
            for timing in self._stack:
                timing.peak_memory_bytes = max(timing.peak_memory_bytes, interval_peak)
            return sample
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._memory_errors.append(
                {
                    "elapsed_seconds": elapsed,
                    "event": event,
                    "phase": phase,
                    "label": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

    def _publish_phase_peak(self, phase: str, peak_memory_bytes: int) -> None:
        receiver = getattr(self._delegate, "phase_peak_memory", None)
        if callable(receiver):
            receiver(phase, peak_memory_bytes)

    def memory_checkpoint(self, label: str) -> None:
        """Record an explicit orchestration-boundary allocator snapshot."""
        self._sample_memory("checkpoint", label=label)

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
    ) -> None:
        now = self._clock()
        previous = self._active.pop(phase, None)
        if previous is not None:
            previous.ended_seconds = now
            self._stack.remove(previous)
        parent = self._stack[-1].phase if self._stack else None
        timing = _PhaseTiming(
            phase=phase,
            parent_phase=parent,
            depth=len(self._stack),
            total=total,
            unit=unit,
            started_seconds=now,
        )
        self._phases.append(timing)
        self._sample_memory("phase_start", phase=phase)
        self._active[phase] = timing
        self._stack.append(timing)
        self._delegate.phase_start(phase, total=total, unit=unit)

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        timing = self._active.get(phase)
        if timing is not None:
            timing.completed += advance
        self._delegate.phase_advance(phase, advance)

    def phase_end(self, phase: str) -> None:
        timing = self._active.pop(phase, None)
        if timing is not None:
            timing.ended_seconds = self._clock()
        self._sample_memory("phase_end", phase=phase)
        if timing is not None and self._memory_sampler is not None:
            self._publish_phase_peak(phase, timing.peak_memory_bytes)
        if timing is not None:
            self._stack.remove(timing)
        self._delegate.phase_end(phase)

    def memory_to_dict(self) -> MemoryReport:
        """Return JSON-ready non-synchronizing allocator observations."""
        if self._memory_sampler is None:
            return {}
        observed_peak = max(
            (sample["peak_bytes_since_previous_sample"] for sample in self._memory_samples),
            default=0,
        )
        return {
            "counter_source": "mlx_allocator",
            "unit": "bytes",
            "synchronizes_device": False,
            "peak_reset_between_samples": True,
            "observed_run_peak_bytes": observed_peak,
            "phase_peaks": [
                {
                    "phase": timing.phase,
                    "parent_phase": timing.parent_phase,
                    "depth": timing.depth,
                    "peak_bytes": timing.peak_memory_bytes,
                }
                for timing in self._phases
            ],
            "samples": [sample.copy() for sample in self._memory_samples],
            "sampling_errors": [error.copy() for error in self._memory_errors],
        }

    def to_dict(self) -> TimingReport:
        """Return a JSON-ready snapshot without ending active phases."""
        now = self._clock()
        phases: list[PhaseTimingRecord] = []
        for timing in self._phases:
            ended = timing.ended_seconds
            phases.append(
                {
                    "phase": timing.phase,
                    "parent_phase": timing.parent_phase,
                    "depth": timing.depth,
                    "total": timing.total,
                    "unit": timing.unit,
                    "completed": timing.completed,
                    "started_seconds": timing.started_seconds - self._started,
                    "duration_seconds": (ended if ended is not None else now)
                    - timing.started_seconds,
                    "status": "completed" if ended is not None else "active",
                }
            )
        return {
            "total_seconds": now - self._started,
            "phases": phases,
        }


__all__ = [
    "MemoryReport",
    "MemorySample",
    "MemorySamplingError",
    "NullReporter",
    "PhaseMemoryPeak",
    "PhaseTimingRecord",
    "RecordingReporter",
    "Reporter",
    "TimingReport",
    "TimingReporter",
]
