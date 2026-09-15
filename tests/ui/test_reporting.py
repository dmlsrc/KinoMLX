"""Host-neutral Reporter contract and Rich adapter tests."""

import io
import logging

import pytest
from rich.console import Console

from kinomlx.reporting import NullReporter, Reporter, ScopedReporter, TimingReporter, reported_phase
from kinomlx.ui import RichReporter, make_progress
from tests._reporting import RecordingReporter


def _quiet_progress():
    return make_progress(console=Console(file=io.StringIO(), width=120))


def test_reporter_protocol_matches_all_implementations() -> None:
    assert isinstance(NullReporter(), Reporter)
    assert isinstance(RecordingReporter(), Reporter)
    assert isinstance(ScopedReporter(RecordingReporter(), "Krea 2"), Reporter)
    assert isinstance(RichReporter(progress=_quiet_progress()), Reporter)


def test_scoped_reporter_qualifies_every_phase_operation() -> None:
    delegate = RecordingReporter()
    reporter = ScopedReporter(delegate, "Krea 2")

    reporter.phase_start("denoise", total=2, unit="step")
    reporter.phase_advance("denoise")
    reporter.phase_end("denoise")

    assert delegate.events == [
        ("start", "Krea 2: denoise", {"total": 2, "unit": "step"}),
        ("advance", "Krea 2: denoise", {"advance": 1.0}),
        ("end", "Krea 2: denoise", {}),
    ]


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
                "advances": [{"advance": 2, "completed": 2.0, "duration_seconds": 0.0}],
            }
        ],
    }


def test_progress_intervals_include_work_without_double_counting_nested_phases() -> None:
    now = [0.0]
    reporter = TimingReporter(clock=lambda: now[0])
    reporter.phase_start("denoise", total=2, unit="step")
    now[0] = 1.0
    reporter.phase_start("child")
    now[0] = 2.0
    reporter.phase_end("child")
    now[0] = 3.0
    reporter.phase_advance("denoise")
    now[0] = 7.0
    reporter.phase_advance("denoise")
    reporter.phase_end("denoise")
    now[0] = 10.0

    phase = reporter.to_dict()["phases"][0]
    assert phase["duration_seconds"] == 7.0
    assert phase["advances"] == [
        {"advance": 1.0, "completed": 1.0, "duration_seconds": 3.0},
        {"advance": 1.0, "completed": 2.0, "duration_seconds": 4.0},
    ]
    phase["advances"][0]["duration_seconds"] = -1.0
    assert reporter.to_dict()["phases"][0]["advances"][0]["duration_seconds"] == 3.0


def test_reported_phase_closes_on_failure_and_preserves_cleanup_with_broken_reporter() -> None:
    reporter = RecordingReporter()
    with pytest.raises(ValueError, match="load failed"), reported_phase(reporter, "load"):
        raise ValueError("load failed")
    assert reporter.events == [
        ("start", "load", {"total": None, "unit": "it"}),
        ("end", "load", {}),
    ]

    class BrokenReporter(NullReporter):
        def phase_start(self, *_args, **_kwargs):
            raise RuntimeError("display failed")

        def phase_end(self, *_args, **_kwargs):
            raise RuntimeError("display failed")

    with pytest.raises(ValueError, match="load failed"), reported_phase(BrokenReporter(), "load"):
        raise ValueError("load failed")


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
        "denoise: 2 steps in" in record.message and "peak memory 6.0 GiB" in record.message
        for record in caplog.records
    )


def test_rich_reporter_keeps_short_load_times_readable(caplog) -> None:
    now = [0.0]
    progress = _quiet_progress()
    progress.get_time = lambda: now[0]
    with (
        caplog.at_level(logging.INFO, logger="kinomlx.ui.bars"),
        RichReporter(progress=progress) as reporter,
    ):
        reporter.phase_start("load VAE")
        now[0] = 0.0185
        reporter.phase_end("load VAE")
    assert "load VAE: completed in 18.5ms" in caplog.text


def test_phase_labels_change_presentation_without_changing_timing_identity(caplog):
    progress = _quiet_progress()
    with (
        caplog.at_level(logging.INFO, logger="kinomlx.ui.bars"),
        RichReporter(
            progress=progress, phase_labels={"encode_text": "Klein: encode prompt with Qwen3"}
        ) as presentation,
    ):
        reporter = TimingReporter(presentation)
        reporter.phase_start("encode_text")
        assert progress.tasks[0].description == "Klein: encode prompt with Qwen3"
        reporter.phase_end("encode_text")
        assert progress.tasks == []
    assert "Klein: encode prompt with Qwen3: completed in" in caplog.text
    assert reporter.to_dict()["phases"][0]["phase"] == "encode_text"


def test_phase_peak_includes_live_memory_when_allocator_peak_stays_zero() -> None:
    counters = iter(
        [
            {"active_bytes": 100, "cache_bytes": 0, "peak_bytes": 900},
            {"active_bytes": 80, "cache_bytes": 0, "peak_bytes": 0},
            {"active_bytes": 80, "cache_bytes": 0, "peak_bytes": 0},
            {"active_bytes": 120, "cache_bytes": 0, "peak_bytes": 0},
        ]
    )
    reporter = TimingReporter(memory_sampler=lambda: next(counters))
    reporter.phase_start("lazy load")
    reporter.phase_end("lazy load")
    reporter.phase_start("next")
    reporter.phase_end("next")
    peaks = reporter.memory_to_dict()["phase_peaks"]
    assert [phase["peak_bytes"] for phase in peaks] == [100, 120]
    # Preserve raw samples, and keep the earlier peak in the whole-run record.
    report = reporter.memory_to_dict()
    assert report["samples"][1]["peak_bytes_since_previous_sample"] == 0
    assert report["observed_run_peak_bytes"] == 900


@pytest.mark.parametrize("finish", ["advance", "end"])
def test_grouped_checks_keep_one_bar_and_sum_only_check_intervals(caplog, finish):
    from rich.progress import Progress

    now = [0.0]
    progress = Progress(disable=True, get_time=lambda: now[0])
    display = RichReporter(progress=progress)
    reporter = TimingReporter(display, clock=lambda: now[0])
    phase = "LTX-2: check adapter weights"
    with caplog.at_level(logging.INFO, logger="kinomlx.ui.bars"):
        reporter.phase_start("denoise", total=2, unit="step")
        for index in range(2):
            reporter.phase_start(phase, total=4, unit="tensor", aggregate_total=8)
            now[0] += 0.25
            reporter.phase_advance(phase, 4)
            reporter.phase_end(phase)
            assert len(progress.tasks) == 2
            check_task = next(t for t in progress.tasks if t.description == phase)
            assert check_task.total == 8
            assert check_task.completed == (index + 1) * 4
            now[0] += 10  # Intervening model work is not check time.
        reporter.phase_start(phase, total=2, unit="tensor")
        now[0] += 0.125
        reporter.phase_advance(phase, 2)
        reporter.phase_end(phase)
        assert not caplog.records
        if finish == "advance":
            reporter.phase_advance("denoise")
        else:
            reporter.phase_end("denoise")
    summaries = [r.message for r in caplog.records if phase in r.message]
    assert len(summaries) == 1
    assert "10 tensors in 625.0ms" in summaries[0]
    assert not any(t.description == phase for t in progress.tasks)
    records = [p for p in reporter.to_dict()["phases"] if p["phase"] == phase]
    assert [p["duration_seconds"] for p in records] == [0.25, 0.25, 0.125]
