"""CLI presentation: Rich logging, machine output, and progress bars.

All terminal output is routed through this module; nothing else in
the package should call ``print`` directly (it clobbers live
progress bars and bypasses output-mode toggles).

Runtime modules use stdlib logging and :class:`kinomlx.reporting.Reporter`.
Only CLI-facing code imports this package to bind those host-neutral surfaces
to Rich and the terminal.
"""

from kinomlx.ui.bars import (
    RichReporter,
    WallClockPaceColumn,
    WallClockRemainingColumn,
    log_phase_summary,
    make_progress,
    track_phase,
)
from kinomlx.ui.console import get_console
from kinomlx.ui.logging import (
    configure_logging,
    configure_logging_from_settings,
    configure_machine_output,
)

__all__ = [
    "RichReporter",
    "WallClockPaceColumn",
    "WallClockRemainingColumn",
    "configure_logging",
    "configure_logging_from_settings",
    "configure_machine_output",
    "get_console",
    "log_phase_summary",
    "make_progress",
    "track_phase",
]
