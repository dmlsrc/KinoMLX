"""The shared Rich console used by CLI logging and progress bars."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

_THEME = Theme(
    {
        "logging.level.info": "cyan",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "log.time": "cyan",
    }
)

_console: Console | None = None


def get_console() -> Console:
    """Return the lazily-created stderr console shared by the CLI UI."""
    global _console
    if _console is None:
        _console = Console(stderr=True, theme=_THEME, highlight=False)
    return _console


__all__ = ["get_console"]
