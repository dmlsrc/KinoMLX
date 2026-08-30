"""Host-neutral Reporter contract and Rich adapter tests."""

from __future__ import annotations

import io
import logging

from rich.console import Console

from kinomlx.reporting import NullReporter, RecordingReporter, Reporter, TimingReporter
from kinomlx.ui import RichReporter, make_progress


def _quiet_progress():
    return make_progress(console=Console(file=io.StringIO(), width=120))


def test_reporter_protocol_matches_all_implementations() -> None:
    assert isinstance(NullReporter(), Reporter)
    assert isinstance(RecordingReporter(), Reporter)
    assert isinstance(RichReporter(progress=_quiet_progress()), Reporter)


def test_recording_reporter_preserves_event_order_and_payloads() -> None:
    reporter = RecordingReporter()
    reporter.phase_start("denoise", total=8, unit="step")
    reporter.phase_advance("denoise")
    reporter.phase_advance("denoise", 2.0)
    reporter.phase_end("denoise")
    assert reporter.events == [
        ("start", "denoise", {"total": 8, "unit": "step"}),
        ("advance", "denoise", {"advance": 1.0}),
        ("advance", "denoise", {"advance": 2.0}),
        ("end", "denoise", {}),
    ]


def test_null_reporter_is_an_unconditional_noop() -> None:
    reporter = NullReporter()
    reporter.phase_start("load", total=None)
    reporter.phase_advance("load")
    reporter.phase_end("load")


def test_timing_reporter_records_monotonic_phases_and_delegates() -> None:
    now = [10.0]
    delegate = RecordingReporter()
    reporter = TimingReporter(delegate, clock=lambda: now[0])

    now[0] = 11.5
    reporter.phase_start("denoise", total=3, unit="step")
    reporter.phase_advance("denoise", 2)
    now[0] = 14.0
    reporter.phase_end("denoise")
    now[0] = 16.0

    assert delegate.events == [
        ("start", "denoise", {"total": 3, "unit": "step"}),
        ("advance", "denoise", {"advance": 2}),
        ("end", "denoise", {}),
    ]
    assert reporter.to_dict() == {
        "total_seconds": 6.0,
        "phases": [
            {
                "phase": "denoise",
                "parent_phase": None,
                "depth": 0,
                "total": 3,
                "unit": "step",
                "completed": 2.0,
                "started_seconds": 1.5,
                "duration_seconds": 2.5,
                "status": "completed",
            }
        ],
    }


def test_timing_reporter_preserves_nested_phase_relationships() -> None:
    now = [0.0]
    reporter = TimingReporter(clock=lambda: now[0])
    reporter.phase_start("vocode")
    now[0] = 1.0
    reporter.phase_start("load vocoder")
    now[0] = 2.0
    reporter.phase_end("load vocoder")
    reporter.phase_start("run vocoder")
    now[0] = 4.0
    reporter.phase_end("run vocoder")
    now[0] = 5.0
    reporter.phase_end("vocode")

    phases = reporter.to_dict()["phases"]
    assert [(phase["phase"], phase["parent_phase"], phase["depth"]) for phase in phases] == [
        ("vocode", None, 0),
        ("load vocoder", "vocode", 1),
        ("run vocoder", "vocode", 1),
    ]


def test_timing_reporter_records_memory_at_phases_and_explicit_boundaries() -> None:
    now = [0.0]
    counters = iter(
        (
            {"active_bytes": 1, "cache_bytes": 2, "peak_bytes": 3},
            {"active_bytes": 4, "cache_bytes": 5, "peak_bytes": 6},
            {"active_bytes": 7, "cache_bytes": 8, "peak_bytes": 9},
        )
    )
    reporter = TimingReporter(
        clock=lambda: now[0],
        memory_sampler=lambda: next(counters),
    )

    reporter.memory_checkpoint("runner_start")
    now[0] = 1.0
    reporter.phase_start("denoise")
    now[0] = 2.0
    reporter.phase_end("denoise")

    assert reporter.memory_to_dict() == {
        "counter_source": "mlx_allocator",
        "unit": "bytes",
        "synchronizes_device": False,
        "peak_reset_between_samples": True,
        "observed_run_peak_bytes": 9,
        "phase_peaks": [
            {
                "phase": "denoise",
                "parent_phase": None,
                "depth": 0,
                "peak_bytes": 9,
            }
        ],
        "samples": [
            {
                "sequence": 0,
                "elapsed_seconds": 0.0,
                "event": "checkpoint",
                "phase": None,
                "label": "runner_start",
                "active_phases": [],
                "active_bytes": 1,
                "cache_bytes": 2,
                "peak_bytes_since_previous_sample": 3,
            },
            {
                "sequence": 1,
                "elapsed_seconds": 1.0,
                "event": "phase_start",
                "phase": "denoise",
                "label": None,
                "active_phases": [],
                "active_bytes": 4,
                "cache_bytes": 5,
                "peak_bytes_since_previous_sample": 6,
            },
            {
                "sequence": 2,
                "elapsed_seconds": 2.0,
                "event": "phase_end",
                "phase": "denoise",
                "label": None,
                "active_phases": ["denoise"],
                "active_bytes": 7,
                "cache_bytes": 8,
                "peak_bytes_since_previous_sample": 9,
            },
        ],
        "sampling_errors": [],
    }


def test_timing_reporter_aggregates_child_intervals_into_parent_peak() -> None:
    counters = iter(
        (
            {"active_bytes": 1, "cache_bytes": 0, "peak_bytes": 1},
            {"active_bytes": 2, "cache_bytes": 0, "peak_bytes": 5},
            {"active_bytes": 3, "cache_bytes": 0, "peak_bytes": 9},
            {"active_bytes": 4, "cache_bytes": 0, "peak_bytes": 4},
        )
    )
    reporter = TimingReporter(memory_sampler=lambda: next(counters))

    reporter.phase_start("stage")
    reporter.phase_start("child")
    reporter.phase_end("child")
    reporter.phase_end("stage")

    assert reporter.memory_to_dict()["phase_peaks"] == [
        {"phase": "stage", "parent_phase": None, "depth": 0, "peak_bytes": 9},
        {"phase": "child", "parent_phase": "stage", "depth": 1, "peak_bytes": 9},
    ]


def test_rich_reporter_prints_time_and_interval_peak_then_releases_phase(caplog) -> None:
    progress = _quiet_progress()
    counters = iter(
        (
            {"active_bytes": 1, "cache_bytes": 2, "peak_bytes": 3 * 1024**3},
            {"active_bytes": 4, "cache_bytes": 5, "peak_bytes": 6 * 1024**3},
            {"active_bytes": 4, "cache_bytes": 5, "peak_bytes": 6 * 1024**3},
        )
    )
    with (
        caplog.at_level(logging.INFO, logger="kinomlx.ui.bars"),
        RichReporter(progress=progress) as presentation,
    ):
        reporter = TimingReporter(presentation, memory_sampler=lambda: next(counters))
        reporter.phase_start("denoise", total=2, unit="step")
        reporter.phase_advance("denoise")
        reporter.phase_advance("denoise")
        reporter.phase_end("denoise")
        assert progress.tasks == []
        reporter.phase_end("denoise")
        reporter.phase_advance("missing")
    assert any(
        "denoise: 2 step in" in record.message and "peak memory 6.0 GiB" in record.message
        for record in caplog.records
    )
