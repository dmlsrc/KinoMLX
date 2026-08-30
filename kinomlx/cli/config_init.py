"""The lightweight ``kinomlx config`` utility command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from kinomlx.io.atomic import write_text_exclusive

from .common import render_error
from .config_templates import config_template, template_models

_DEFAULT_CONFIG_PATH = Path("kino-config.toml")
_log = logging.getLogger(__name__)


def build_config_parser() -> argparse.ArgumentParser:
    """Build the model-aware configuration utility parser."""
    parser = argparse.ArgumentParser(
        prog="kinomlx config",
        description="Create model-specific KinoMLX invocation files.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(
        dest="config_command",
        metavar="{init}",
        required=True,
    )
    init = commands.add_parser(
        "init",
        help="write an annotated starter configuration",
        description="Write an annotated model-specific starter configuration.",
        allow_abbrev=False,
    )
    init.add_argument(
        "--model",
        choices=template_models(),
        default="ltx2",
        help="Configuration schema to write (default: ltx2).",
    )
    init.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Destination path (default: ./kino-config.toml; existing paths are refused).",
    )
    return parser


def run_config_command(argv: list[str]) -> int:
    """Run one ``kinomlx config`` utility and return its process exit code."""
    options = build_config_parser().parse_args(argv)
    if options.config_command != "init":
        raise AssertionError(f"unhandled config command {options.config_command!r}")
    model = str(options.model)
    output = Path(options.output).expanduser()
    try:
        write_text_exclusive(output, config_template(model, destination=output))
    except FileExistsError:
        return render_error(
            f"config init refused: output already exists: {output}; "
            "choose another path or remove the existing file",
            json_output=False,
        )
    except OSError as exc:
        return render_error(
            f"config init failed for {output}: {exc}",
            json_output=False,
        )
    _log.info("Wrote %s starter configuration to %s", model, output)
    return 0


__all__ = ["build_config_parser", "run_config_command"]
