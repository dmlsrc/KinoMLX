"""Model-neutral CLI arguments, formatting, and typed error rendering."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

from kinomlx.settings import Settings

from .output import emit_json

_log = logging.getLogger(__name__)


def add_invocation_arguments(
    parser: argparse.ArgumentParser,
    *,
    choices: Iterable[str],
    model_help: str | None = None,
) -> None:
    """Add the bootstrap flags shared by every model command."""
    parser.add_argument(
        "--model",
        choices=tuple(choices),
        default=None,
        help=model_help or "Model family.",
    )
    parser.add_argument("--config", type=Path, default=None, help="TOML invocation file.")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Apply a TOML-typed key override after config and ordinary CLI flags.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully resolved invocation as round-trippable TOML and exit.",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save the fully resolved invocation as round-trippable TOML and exit; "
            "existing paths are refused."
        ),
    )
    parser.add_argument(
        "--only-non-defaults",
        action="store_true",
        help=(
            "With --print-config or --save-config, omit values equal to built-in defaults "
            "while retaining environment-derived non-defaults and values that mask a "
            "different active environment setting."
        ),
    )


def format_elapsed(seconds: float) -> str:
    """Format one non-negative wall-clock duration for human CLI output."""
    rounded_tenths = int(max(0.0, seconds) * 10.0 + 0.5)
    hours, remainder = divmod(rounded_tenths, 36_000)
    minutes, second_tenths = divmod(remainder, 600)
    rendered_seconds = second_tenths / 10.0
    if hours:
        return f"{hours}h {minutes}m {rendered_seconds:.1f}s"
    if minutes:
        return f"{minutes}m {rendered_seconds:.1f}s"
    return f"{rendered_seconds:.1f}s"


def render_error(message: str, *, json_output: bool) -> int:
    """Render one expected failure through the selected human or JSON surface."""
    if json_output:
        emit_json({"status": "error", "error": message})
    else:
        _log.error("%s", message)
    return 2


def bootstrap_json_mode(options: object, settings: Settings) -> bool:
    """Resolve the pre-assembly error format from env plus explicit CLI."""
    cli_value = getattr(options, "json_output", None)
    return settings.json_output if cli_value is None else bool(cli_value)


def bootstrap_json_from_arguments(arguments: Iterable[str], settings: Settings) -> bool:
    """Resolve JSON mode before a model-specific parser is selected."""
    selected = settings.json_output
    for argument in arguments:
        if argument == "--":
            break
        if argument == "--json":
            selected = True
        elif argument == "--no-json":
            selected = False
    return selected


def bootstrap_flag_present(arguments: Iterable[str], flags: set[str]) -> bool:
    """Return whether a bootstrap flag occurs before the option terminator."""
    for argument in arguments:
        if argument == "--":
            return False
        if argument in flags:
            return True
    return False


__all__ = [
    "add_invocation_arguments",
    "bootstrap_flag_present",
    "bootstrap_json_from_arguments",
    "bootstrap_json_mode",
    "format_elapsed",
    "render_error",
]
