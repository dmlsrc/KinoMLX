"""Progress bars - thin layer over ``rich.progress``.

A :class:`rich.progress.Progress` renders stacked bars as an internal
table, so cross-row column alignment, label / count / percentage /
time / ETA columns, and the ``with Progress() as p:`` context all come
for free.  We add custom columns for math-consistent wall-clock pace and
remaining time.

**Redraw discipline matters.**  macOS hardware-accelerates the
terminal - every redraw burns Terminal and WindowServer GPU cycles.
The measured reference bars throttle to 1 Hz; we match that by
default via ``refresh_per_second=1.0`` rather than rich's default
of 10 Hz.  At ~25-second stages (8-step distilled denoise x ~3 s/step)
this is 25 redraws total instead of 250 - measurable in benchmarks
(the 1 Hz throttling measured at ~5.9% wall time on the
standard run).

Usage::

    from kinomlx.ui.bars import make_progress

    with make_progress() as prog:
        stage1 = prog.add_task("stage 1", total=8, unit="step")
        for _ in range(8):
            ...
            prog.advance(stage1)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from kinomlx.ui.console import get_console

_log = logging.getLogger(__name__)

_ETA_ANCHOR_COMPLETED = "_kinomlx_eta_anchor_completed"
_ETA_ANCHOR_ELAPSED = "_kinomlx_eta_anchor_elapsed"
_ETA_SECONDS_PER_UNIT = "_kinomlx_eta_seconds_per_unit"


class _TaskFieldUpdater(Protocol):
    def __call__(self, task_id: TaskID, **fields: float) -> None: ...


def _anchored_seconds_per_unit(task: Task) -> float | None:
    """Return the whole-phase average captured at the last completed unit."""
    fields = task.fields
    anchor_completed = fields.get(_ETA_ANCHOR_COMPLETED)
    seconds_per_unit = fields.get(_ETA_SECONDS_PER_UNIT)
    if anchor_completed == task.completed and seconds_per_unit is not None:
        return float(seconds_per_unit)
    if task.elapsed is None or task.completed == 0:
        return None
    return task.elapsed / task.completed


class _WallClockProgress(Progress):
    """Capture a stable rate sample whenever progress actually advances.

    Long denoise steps may spend minutes between calls to ``advance``. Keeping
    the sample on the task lets the display count ETA down between those calls
    without treating an unfinished step as evidence that every future step is
    slower.
    """

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        super().advance(task_id, advance)
        if advance <= 0:
            return
        task = next((item for item in self.tasks if item.id == task_id), None)
        if task is None or task.elapsed is None or task.completed <= 0:
            return
        elapsed = task.elapsed
        update_fields = cast(_TaskFieldUpdater, self.update)
        update_fields(
            task_id,
            **{
                _ETA_ANCHOR_COMPLETED: task.completed,
                _ETA_ANCHOR_ELAPSED: elapsed,
                _ETA_SECONDS_PER_UNIT: elapsed / task.completed,
            },
        )


class WallClockPaceColumn(ProgressColumn):
    """Pace as ``X.X s/<unit>`` (slow) or ``X.X <unit>/s`` (fast).

    The whole-phase average is sampled whenever a unit completes and held
    steady while the next unit is running. Rich's built-in speed column uses a
    sliding window, while a continuously recomputed wall average incorrectly
    makes an unfinished long step look like evidence that every future step
    is getting slower.

    The ``unit`` (``"step"``, ``"frame"``, etc.) is read from the
    task's ``fields`` dict; defaults to ``"it"`` if unset.
    """

    def render(self, task: Task) -> Text:
        sec_per_unit = _anchored_seconds_per_unit(task)
        if sec_per_unit is None:
            return Text("measuring", style="dim")
        unit = task.fields.get("unit", "it")
        if sec_per_unit >= 1.0:
            return Text(f"{sec_per_unit:>6.1f} s/{unit}")
        return Text(f"{1.0 / sec_per_unit:>6.1f} {unit}/s")


class WallClockRemainingColumn(ProgressColumn):
    """ETA derived from the same whole-phase average as the pace column.

    Rich's stock ETA uses a 30-second sliding speed window. Production LTX
    denoise steps routinely take longer than that, so the previous sample ages
    out before the next one arrives and the ETA falls back to ``-:--:--``.
    Whole-phase wall time remains meaningful for these sparse, expensive steps.
    The estimate is anchored when a unit completes and counts down until the
    next completion, where it may adjust to the newly observed average.
    """

    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("unknown", style="dim")
        remaining = max(task.total - task.completed, 0.0)
        if remaining == 0:
            return Text("0:00:00", style="progress.remaining")
        seconds_per_unit = _anchored_seconds_per_unit(task)
        if task.elapsed is None or seconds_per_unit is None:
            return Text("measuring", style="dim")
        anchor_elapsed = task.fields.get(_ETA_ANCHOR_ELAPSED)
        elapsed_since_anchor = (
            max(task.elapsed - float(anchor_elapsed), 0.0) if anchor_elapsed is not None else 0.0
        )
        seconds = math.ceil(max(remaining * seconds_per_unit - elapsed_since_anchor, 0.0))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return Text(
            f"{hours:d}:{minutes:02d}:{seconds:02d}",
            style="progress.remaining",
        )


def make_progress(
    *,
    refresh_per_second: float = 1.0,
    console: Console | None = None,
    disable: bool | None = None,
) -> Progress:
    """Build a configured ``Progress`` ready for ``with`` use.

    ``refresh_per_second`` defaults to ``1.0`` for the redraw-cost
    reason in the module docstring.  Bump it (e.g. ``2.0``) for
    short-lived bars where 1 Hz feels sluggish, but never set it
    above what the workload actually warrants.
    """
    return _WallClockProgress(
        TextColumn("  {task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("| RUN"),
        TimeElapsedColumn(),
        TextColumn("| ETA"),
        WallClockRemainingColumn(),
        TextColumn("|"),
        WallClockPaceColumn(),
        console=console if console is not None else get_console(),
        refresh_per_second=refresh_per_second,
        transient=False,
        disable=False if disable is None else disable,
    )


@contextmanager
def track_phase(
    progress: Progress,
    description: str,
    *,
    total: float,
    unit: str = "it",
    logger: logging.Logger | None = None,
) -> Iterator[TaskID]:
    """Add a task to ``progress``, yield its id, log a summary when the block exits.

    On exit (success or error) :func:`log_phase_summary` writes one INFO line -
    total wall time, iterations, and seconds per iteration - so the milestone
    persists after the (non-transient) bar and lands in any log file. Pass
    ``logger`` to attribute it to the calling subsystem rather than
    ``kinomlx.ui.bars``.

    Usage::

        with make_progress() as prog, track_phase(prog, "denoise", total=8, unit="step") as task:
            for _ in range(8):
                ...
                prog.advance(task)
    """
    task_id = progress.add_task(description, total=total, unit=unit)
    try:
        yield task_id
    finally:
        progress.stop_task(task_id)
        log_phase_summary(progress, task_id, logger=logger)


def log_phase_summary(
    progress: Progress,
    task_id: TaskID,
    *,
    logger: logging.Logger | None = None,
    peak_memory_bytes: int | None = None,
) -> None:
    """Log one task's time, progress, pace, and optional peak memory.

    Reads the finished task's ``elapsed`` and ``completed`` straight from rich, so
    the numbers match the live ``WallClockPaceColumn``. A no-op if the task id is
    unknown; a task without iterations still records its elapsed time.
    """
    task = next((t for t in progress.tasks if t.id == task_id), None)
    if task is None:
        return
    out = logger or _log
    iters = int(task.completed)
    unit = task.fields.get("unit", "it")
    elapsed = task.elapsed or 0.0
    memory = (
        ""
        if peak_memory_bytes is None
        else f", peak memory {_human_size(max(0, peak_memory_bytes))}"
    )
    if iters <= 0:
        out.info(f"{task.description}: completed in {elapsed:.1f}s{memory}")
        return
    out.info(
        f"{task.description}: {iters} {unit} in {elapsed:.1f}s "
        f"({elapsed / iters:.3f}s/{unit}){memory}"
    )


def _human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


class RichReporter:
    """Drive one stacked Rich progress row per active runtime phase."""

    def __init__(
        self,
        *,
        progress: Progress | None = None,
        disable: bool | None = None,
    ) -> None:
        if progress is not None and disable is not None:
            raise ValueError("pass either progress or disable, not both")
        self._progress = progress if progress is not None else make_progress(disable=disable)
        self._tasks: dict[str, TaskID] = {}
        self._phase_peaks: dict[str, int] = {}

    def __enter__(self) -> RichReporter:
        self._progress.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._progress.stop()

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
    ) -> None:
        self._phase_peaks.pop(phase, None)
        self._tasks[phase] = self._progress.add_task(phase, total=total, unit=unit)

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        task_id = self._tasks.get(phase)
        if task_id is not None:
            self._progress.advance(task_id, advance)

    def phase_end(self, phase: str) -> None:
        task_id = self._tasks.pop(phase, None)
        peak_memory_bytes = self._phase_peaks.pop(phase, None)
        if task_id is not None:
            self._progress.stop_task(task_id)
            try:
                log_phase_summary(
                    self._progress,
                    task_id,
                    peak_memory_bytes=peak_memory_bytes,
                )
            finally:
                # Rich retains tasks until explicitly removed. Reporter phases
                # are live state, while the INFO summary is their durable
                # history, so completed rows must leave the active stack.
                self._progress.remove_task(task_id)

    def phase_peak_memory(self, phase: str, peak_memory_bytes: int) -> None:
        """Attach the allocator peak measured by the timing decorator."""
        if phase in self._tasks:
            self._phase_peaks[phase] = max(0, int(peak_memory_bytes))
