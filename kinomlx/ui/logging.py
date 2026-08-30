"""Rich-backed CLI logging and machine-readable stdout protocols."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from rich.logging import RichHandler

from .console import get_console

if TYPE_CHECKING:
    from kinomlx.settings import Settings

_ROOT_LOGGER = "kinomlx"


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
    time_format = "[%Y-%m-%d %H:%M:%S]" if show_date else "[%H:%M:%S]"
    console_handler = RichHandler(
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
]
