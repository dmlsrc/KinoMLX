"""Typed VideoToolbox failures and narrow native-error classification."""

from __future__ import annotations

import objc

from kinomlx.errors import KinoMLXError


class VideoToolboxError(KinoMLXError, RuntimeError):
    """An operational failure in KinoMLX's native media bridge."""


class VideoToolboxUnavailableError(VideoToolboxError):
    """The requested native framework or device capability is unavailable."""


def is_objc_error(exc: BaseException) -> bool:
    """Whether ``exc`` is PyObjC's native bridge exception."""
    return isinstance(exc, objc.error)


def is_video_toolbox_operation_error(exc: BaseException) -> bool:
    """Whether an exception from the explicit encoder call is operational."""
    return isinstance(
        exc,
        (KinoMLXError, OSError, RuntimeError, ValueError),
    ) or is_objc_error(exc)


__all__ = [
    "VideoToolboxError",
    "VideoToolboxUnavailableError",
    "is_objc_error",
    "is_video_toolbox_operation_error",
]
