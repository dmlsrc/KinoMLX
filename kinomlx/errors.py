"""Typed operational failures at KinoMLX's public and CLI boundaries."""


class KinoMLXError(Exception):
    """Base class for failures safe to render without a traceback."""


__all__ = ["KinoMLXError"]
