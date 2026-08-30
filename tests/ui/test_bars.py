"""Behavioral tests for ``kinomlx.ui.bars``.

The Progress object itself is rich's responsibility - we just test
the configuration (refresh rate, custom pace column) and the
math-consistent pace formatting that's our value-add.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from kinomlx.ui.bars import (
    WallClockPaceColumn,
    WallClockRemainingColumn,
    log_phase_summary,
    make_progress,
    track_phase,
)

# ---------------------------------------------------------------------------
# WallClockPaceColumn - custom whole-phase rate math
# ---------------------------------------------------------------------------


@dataclass
class _FakeTask:
    """Stand-in for ``rich.progress.Task`` - just the attributes the column uses."""

    completed: int
    elapsed: float | None
    fields: dict[str, Any]
    total: int | None = None


def test_pace_formats_slow_as_seconds_per_unit() -> None:
    """sec_per_unit >= 1 -> ``X.X s/<unit>`` (with a space between number and unit)."""
    col = WallClockPaceColumn()
    task = _FakeTask(completed=4, elapsed=8.0, fields={"unit": "step"})
    # 8 s / 4 steps = 2.0 s/step.
    assert "2.0 s/step" in str(col.render(task))


def test_pace_formats_fast_as_units_per_second() -> None:
    """sec_per_unit < 1 -> ``X.X <unit>/s`` (with a space between number and unit)."""
    col = WallClockPaceColumn()
    task = _FakeTask(completed=40, elapsed=2.0, fields={"unit": "frame"})
    # 2 s / 40 frames = 0.05 s/frame -> 20 frames/s.
    assert "20.0 frame/s" in str(col.render(task))


def test_pace_unit_defaults_to_it_when_unset() -> None:
    col = WallClockPaceColumn()
    task = _FakeTask(completed=2, elapsed=4.0, fields={})
    assert "it" in str(col.render(task))


def test_pace_returns_measuring_when_no_progress_yet() -> None:
    col = WallClockPaceColumn()
    task = _FakeTask(completed=0, elapsed=1.5, fields={"unit": "step"})
    assert "measuring" in str(col.render(task))


def test_pace_returns_measuring_when_elapsed_is_none() -> None:
    col = WallClockPaceColumn()
    task = _FakeTask(completed=5, elapsed=None, fields={"unit": "step"})
    assert "measuring" in str(col.render(task))


@pytest.mark.parametrize(
    ("completed", "elapsed", "expected_substr"),
    [
        # Round numbers that won't surprise on fp formatting.
        (1, 5.0, "5.0 s/step"),  # 5 s/step
        (1, 1.0, "1.0 s/step"),  # exactly the boundary
        (2, 1.0, "2.0 step/s"),  # just below the boundary -> switch to step/s
        (10, 1.0, "10.0 step/s"),  # multiple steps/s
        (100, 0.5, "200.0 step/s"),  # high rate
    ],
)
def test_pace_boundary_cases(completed: int, elapsed: float, expected_substr: str) -> None:
    col = WallClockPaceColumn()
    task = _FakeTask(completed=completed, elapsed=elapsed, fields={"unit": "step"})
    assert expected_substr in str(col.render(task))


# ---------------------------------------------------------------------------
# WallClockRemainingColumn - sparse long steps must still produce an ETA
# ---------------------------------------------------------------------------


def test_eta_uses_whole_phase_wall_clock_average() -> None:
    col = WallClockRemainingColumn()
    task = _FakeTask(
        completed=2,
        elapsed=110.0,
        fields={"unit": "step"},
        total=8,
    )
    # 55 s/step * 6 remaining steps = 330 s.
    assert str(col.render(task)) == "0:05:30"


def test_eta_counts_down_between_completed_units() -> None:
    now = [0.0]
    prog = make_progress(disable=True)
    prog.get_time = lambda: now[0]
    task_id = prog.add_task("denoise", total=3, unit="step")
    now[0] = 10.0
    prog.advance(task_id)
    task = prog.tasks[0]
    col = WallClockRemainingColumn()
    pace = WallClockPaceColumn()

    assert str(col.render(task)) == "0:00:20"
    assert "10.0 s/step" in str(pace.render(task))
    now[0] = 15.0
    assert str(col.render(task)) == "0:00:15"
    assert "10.0 s/step" in str(pace.render(task))


def test_eta_reanchors_when_the_next_unit_completes() -> None:
    now = [0.0]
    prog = make_progress(disable=True)
    prog.get_time = lambda: now[0]
    task_id = prog.add_task("denoise", total=3, unit="step")
    now[0] = 10.0
    prog.advance(task_id)
    now[0] = 22.0
    prog.advance(task_id)
    task = prog.tasks[0]

    # Two steps took 22 s, so the newly observed average is 11 s/step.
    assert str(WallClockRemainingColumn().render(task)) == "0:00:11"
    assert "11.0 s/step" in str(WallClockPaceColumn().render(task))


def test_eta_reports_measuring_before_first_completed_unit() -> None:
    col = WallClockRemainingColumn()
    task = _FakeTask(completed=0, elapsed=10.0, fields={}, total=8)
    assert str(col.render(task)) == "measuring"


def test_eta_reports_unknown_when_total_is_unknown() -> None:
    col = WallClockRemainingColumn()
    task = _FakeTask(completed=3, elapsed=10.0, fields={}, total=None)
    assert str(col.render(task)) == "unknown"


def test_eta_reports_zero_when_complete() -> None:
    col = WallClockRemainingColumn()
    task = _FakeTask(completed=8, elapsed=110.0, fields={}, total=8)
    assert str(col.render(task)) == "0:00:00"


# ---------------------------------------------------------------------------
# Progress factory - verify refresh discipline
# ---------------------------------------------------------------------------


def test_make_progress_defaults_to_one_hz_refresh() -> None:
    """1 Hz is the redraw-cost floor - verify it's actually the default."""
    prog = make_progress()
    # rich stows the rate on the underlying Live display.
    assert prog.live.refresh_per_second == 1.0


def test_make_progress_honors_refresh_override() -> None:
    """Higher rates should be available for short-lived bars where 1 Hz is too slow."""
    prog = make_progress(refresh_per_second=4.0)
    assert prog.live.refresh_per_second == 4.0


def test_make_progress_returns_usable_context_manager() -> None:
    """Smoke: ``with make_progress() as p:`` works and add_task returns a task id."""
    prog = make_progress()
    # ``transient=False`` so the bar persists; we won't even render here -
    # just verify the lifecycle methods don't blow up.
    prog.disable = True  # suppress live rendering for the test
    with prog as p:
        task = p.add_task("test", total=3, unit="step")
        p.advance(task)
        p.advance(task)
        p.advance(task)


def test_track_phase_logs_completion_summary(caplog: pytest.LogCaptureFixture) -> None:
    """track_phase logs total time, iterations, and time-per-iter on block exit."""
    prog = make_progress()
    prog.disable = True  # suppress live rendering in the test
    with (
        caplog.at_level(logging.INFO, logger="kinomlx.ui.bars"),
        prog,
        track_phase(prog, "denoise", total=4, unit="step") as task,
    ):
        for _ in range(4):
            prog.advance(task)
    text = caplog.text
    assert "denoise" in text
    assert "4 step" in text  # iterations + unit
    assert "s/step" in text  # time per iteration


def test_track_phase_freezes_elapsed_time_when_the_block_exits() -> None:
    now = [10.0]
    prog = make_progress(disable=True)
    prog.get_time = lambda: now[0]
    with prog, track_phase(prog, "denoise", total=1, unit="step") as task_id:
        now[0] = 15.0
        prog.advance(task_id)
        now[0] = 17.0
    task = prog.tasks[0]
    assert task.elapsed == 7.0
    now[0] = 100.0
    assert task.elapsed == 7.0


def test_phase_summary_prints_time_and_peak_without_iterations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [0.0]
    prog = make_progress(disable=True)
    prog.get_time = lambda: now[0]
    task_id = prog.add_task("load transformer", total=None)
    now[0] = 2.5
    prog.stop_task(task_id)

    with caplog.at_level(logging.INFO, logger="kinomlx.ui.bars"):
        log_phase_summary(prog, task_id, peak_memory_bytes=1536 * 1024**2)

    assert "load transformer: completed in 2.5s, peak memory 1.5 GiB" in caplog.text
