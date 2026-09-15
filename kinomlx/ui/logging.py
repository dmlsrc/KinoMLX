"""Rich-backed CLI logging and machine-readable stdout protocols."""

import logging
import sys
import time
from pathlib import Path
from typing import TextIO

from rich.console import Console, ConsoleOptions, ConsoleRenderable, RenderResult
from rich.logging import RichHandler
from rich.segment import Segment
from rich.traceback import Traceback

from kinomlx.settings import Settings

from .console import get_console

_ROOT_LOGGER = "kinomlx"
_LEVEL_WIDTH = 8
_DEFAULT_TIME_FORMAT = "[%H:%M:%S]"


class _ConsoleLayout:
    """Columns taken by the Rich time and level cells before every console message."""

    prefix_width = len(time.strftime(_DEFAULT_TIME_FORMAT)) + _LEVEL_WIDTH + 2


def console_message_width() -> int:
    """Return the columns one console log message can use before Rich folds it."""
    return max(24, get_console().width - _ConsoleLayout.prefix_width)


def _strip_trailing_blanks(line: list[Segment]) -> list[Segment]:
    """Drop the whitespace-only tail of one rendered row, keeping control segments."""
    kept = list(line)
    while kept and not kept[-1].control and not kept[-1].text.strip():
        kept.pop()
    if kept and not kept[-1].control:
        last = kept[-1]
        kept[-1] = Segment(last.text.rstrip(), last.style, last.control)
    return kept


class _UnpaddedRows:
    """Render a log row's lines without the spaces Rich adds to fill the console width."""

    def __init__(self, renderable: ConsoleRenderable) -> None:
        self._renderable = renderable

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for line in console.render_lines(self._renderable, options, pad=False):
            yield from _strip_trailing_blanks(line)
            yield Segment.line()


class _ScrollbackFriendlyHandler(RichHandler):
    """A RichHandler whose rows end at their text.

    Rich pads every log row with spaces out to the full console width. When a
    terminal re-wraps its scrollback for a narrower window, that padding
    becomes blank rows between the real ones; rows that end at their text
    re-wrap without them.
    """

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        rendered = super().render(
            record=record,
            traceback=traceback,
            message_renderable=message_renderable,
        )
        return _UnpaddedRows(rendered)


def configure_machine_output(
    logger_name: str,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure an isolated, message-only logger for stdout protocols."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _level_for(verbosity: int, quiet: bool) -> int:
    """Map verbosity and quiet mode to the console logging level."""
    if quiet:
        return logging.WARNING
    if verbosity >= 1:
        return logging.DEBUG
    return logging.INFO


def configure_logging(
    verbosity: int = 0,
    *,
    quiet: bool = False,
    show_date: bool = False,
    log_file: str | Path | None = None,
    log_file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Install idempotent Rich and optional file handlers for KinoMLX."""
    console_level = _level_for(verbosity, quiet)
    time_format = "[%Y-%m-%d %H:%M:%S]" if show_date else _DEFAULT_TIME_FORMAT
    _ConsoleLayout.prefix_width = len(time.strftime(time_format)) + _LEVEL_WIDTH + 2
    console_handler = _ScrollbackFriendlyHandler(
        console=get_console(),
        show_time=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format=time_format,
    )
    console_handler.setLevel(console_level)

    logger = logging.getLogger(_ROOT_LOGGER)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    handler_levels = [console_level]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        handler_levels.append(log_file_level)

    logger.setLevel(min(handler_levels))
    return logger


def configure_logging_from_settings(
    settings: Settings,
    *,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure CLI logging from resolved settings."""
    return configure_logging(
        verbosity=1 if settings.verbose else 0,
        quiet=settings.quiet,
        log_file=log_file,
    )


__all__ = [
    "configure_logging",
    "configure_logging_from_settings",
    "configure_machine_output",
    "console_message_width",
]
