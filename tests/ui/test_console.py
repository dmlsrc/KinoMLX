"""Behavioral tests for ``kinomlx.ui.console`` - the rich-backed logging setup."""

from __future__ import annotations

import logging
import pathlib

import pytest

from kinomlx.settings import Settings
from kinomlx.ui.console import get_console
from kinomlx.ui.logging import (
    _level_for,
    configure_logging,
    configure_logging_from_settings,
    configure_machine_output,
)


@pytest.fixture(autouse=True)
def _restore_kinomlx_logger() -> object:
    """Isolate the global ``kinomlx`` logger state that configure_logging mutates."""
    logger = logging.getLogger("kinomlx")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    yield
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)


def test_get_console_returns_singleton() -> None:
    assert get_console() is get_console()
    assert get_console().stderr is True


def test_level_mapping() -> None:
    assert _level_for(0, quiet=False) == logging.INFO
    assert _level_for(1, quiet=False) == logging.DEBUG
    assert _level_for(2, quiet=False) == logging.DEBUG
    # quiet wins over verbosity and floors at WARNING - warnings/errors survive.
    assert _level_for(0, quiet=True) == logging.WARNING
    assert _level_for(5, quiet=True) == logging.WARNING


def test_configure_logging_default_is_info() -> None:
    logger = configure_logging()
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1


def test_configure_logging_verbose_enables_debug() -> None:
    assert configure_logging(1).level == logging.DEBUG


def test_configure_logging_from_settings_maps_ux_fields() -> None:
    assert (
        configure_logging_from_settings(Settings(verbose=True)).handlers[0].level == logging.DEBUG
    )
    logger = configure_logging_from_settings(Settings(verbose=True, quiet=True))
    assert logger.handlers[0].level == logging.WARNING


def test_quiet_keeps_warnings_and_errors() -> None:
    """The footgun fix: quiet floors at WARNING instead of muting everything."""
    configure_logging(quiet=True)
    sub = logging.getLogger("kinomlx.videotoolbox.encode")
    assert not sub.isEnabledFor(logging.INFO)  # info hushed
    assert sub.isEnabledFor(logging.WARNING)  # warnings still get through
    assert sub.isEnabledFor(logging.ERROR)


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    assert len(logging.getLogger("kinomlx").handlers) == 1


def test_subsystems_are_independently_filterable() -> None:
    """Logger name = subsystem: one can be quieted without touching another."""
    configure_logging()  # kinomlx -> INFO
    vt = logging.getLogger("kinomlx.videotoolbox")
    lora = logging.getLogger("kinomlx.lora")
    vt.setLevel(logging.WARNING)
    try:
        assert not vt.isEnabledFor(logging.INFO)
        assert lora.isEnabledFor(logging.INFO)  # unaffected
    finally:
        vt.setLevel(logging.NOTSET)


def test_records_render_through_the_handler(caplog: pytest.LogCaptureFixture) -> None:
    """A child logger's INFO record reaches the configured pipeline."""
    configure_logging()
    with caplog.at_level(logging.INFO, logger="kinomlx.videotoolbox.encode"):
        logging.getLogger("kinomlx.videotoolbox.encode").info("encode start")
    assert "encode start" in caplog.text


def test_show_date_toggles_the_timestamp() -> None:
    """Default timestamp is time-only; show_date adds the calendar date."""
    import io
    import re

    from rich.console import Console

    import kinomlx.ui.console as console_mod

    def _emit(show_date: bool) -> str:
        buf = io.StringIO()
        console_mod._console = Console(file=buf, no_color=True, width=200)
        try:
            configure_logging(show_date=show_date)
            logging.getLogger("kinomlx.sample").warning("hello")
        finally:
            console_mod._console = None  # let get_console rebuild the real one
        return buf.getvalue()

    no_date, with_date = _emit(show_date=False), _emit(show_date=True)
    assert "hello" in no_date
    assert not re.search(r"\d{4}-\d\d-\d\d", no_date)  # time-only by default
    assert re.search(r"\d{4}-\d\d-\d\d", with_date)  # date when asked


def test_log_file_records_at_its_own_level(tmp_path: pathlib.Path) -> None:
    """The file handler logs at its own level (DEBUG) even when the console is quiet."""
    log_path = tmp_path / "run.log"
    configure_logging(quiet=True, log_file=log_path)  # console WARNING, file DEBUG
    logging.getLogger("kinomlx.models.ltx2").debug("loading transformer block 3")
    text = log_path.read_text()
    assert "loading transformer block 3" in text  # DEBUG record reached the file
    assert "DEBUG" in text
    assert "kinomlx.models.ltx2" in text  # subsystem recorded on the line


def test_machine_output_is_plain_and_isolated() -> None:
    import io

    stream = io.StringIO()
    logger = configure_machine_output("kinomlx.test.machine", stream=stream)
    try:
        logger.info('{"ok": true}')
        assert stream.getvalue() == '{"ok": true}\n'
        assert logger.propagate is False
    finally:
        logger.handlers.clear()
