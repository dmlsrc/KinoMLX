"""Typed operational failures at KinoMLX's public and CLI boundaries."""

import logging
import traceback
from collections.abc import Mapping


class KinoMLXError(Exception):
    """Base class for failures safe to render without a traceback."""


class RunCanceled(KeyboardInterrupt):
    """Cancellation that reports the outputs a run had already published.

    Subclasses KeyboardInterrupt so every existing handler keeps treating it as
    an interrupt; the CLI boundary renders the completed outputs with it.
    """

    def __init__(self, outputs: Mapping[str, object] | None = None) -> None:
        super().__init__("canceled")
        self.outputs: dict[str, object] = {} if outputs is None else dict(outputs)


def log_debug_exception(
    logger: logging.Logger,
    message: str,
    error: BaseException,
) -> None:
    """Log traceback text without retaining its live frame graph in a handler."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()
    logger.debug("%s\n%s", message, rendered)


__all__ = ["KinoMLXError", "RunCanceled", "log_debug_exception"]
