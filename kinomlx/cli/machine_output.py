"""Isolated machine-readable CLI output."""

import json
from typing import TextIO

from kinomlx.ui.logging import configure_machine_output


def emit_json(
    payload: dict[str, object],
    *,
    stream: TextIO | None = None,
    logger_name: str = "kinomlx.cli.json_output",
) -> None:
    """Emit one compact JSON record on an isolated stdout logger."""
    logger = configure_machine_output(logger_name, stream=stream)
    logger.info("%s", json.dumps(payload, sort_keys=True, default=str))


__all__ = ["emit_json"]
