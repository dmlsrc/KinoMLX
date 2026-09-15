"""Common public recipe execution boundary, independent of model ports."""

import logging
from collections.abc import Callable

from .config_values import ConfigError
from .errors import KinoMLXError, log_debug_exception
from .memory_policy import MLXMemoryPolicy, memory_policy_scope


def execute_recipe[OutputT](
    operation: Callable[[], OutputT],
    *,
    result_type: type[OutputT],
    error_type: type[KinoMLXError],
    label: str,
    logger: logging.Logger,
    memory_policy: MLXMemoryPolicy,
) -> OutputT:
    """Validate results and render operational failures without retaining model frames."""
    failure: str | None = None
    try:
        with memory_policy_scope(memory_policy):
            output = operation()
    except ConfigError as exc:
        # A record rule refusing the request inside the recipe is the recipe
        # failing on its input, the same as any other ValueError below; it is
        # a KinoMLXError only so the CLI can render it after assembly.
        failure = f"{label} failed: {exc}"
        log_debug_exception(logger, f"{label} failure", exc)
        exc.__traceback__ = None
    except KinoMLXError, FileNotFoundError:
        raise
    except (ArithmeticError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        failure = f"{label} failed: {exc}"
        log_debug_exception(logger, f"{label} failure", exc)
        exc.__traceback__ = None
    if failure is not None:
        raise error_type(failure) from None
    if not isinstance(output, result_type):
        raise TypeError(f"a recipe must return {result_type.__name__}")
    return output
