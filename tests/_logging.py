"""Logging state the test suite must not leak from one test into the next."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

PACKAGE_LOGGER = "kinomlx"


@contextmanager
def package_logger_state() -> Iterator[logging.Logger]:
    """Restore the package logger's level, handlers and propagation on exit.

    ``configure_logging`` installs handlers on the ``kinomlx`` logger and sets its
    level, so a test that runs the CLI under ``--quiet`` would otherwise leave the
    package at WARNING for whichever test happens to run next. All three values are
    restored, because every one of them decides whether a record reaches a capture
    handler.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    level = logger.level
    handlers = list(logger.handlers)
    propagate = logger.propagate
    try:
        yield logger
    finally:
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate
