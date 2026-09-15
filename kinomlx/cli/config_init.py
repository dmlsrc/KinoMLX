"""The lightweight ``kinomlx config`` utility command."""

import argparse
import json
import logging
from pathlib import Path

from kinomlx.config import ConfigError
from kinomlx.config.inspection import configuration_inventory
from kinomlx.io.atomic import write_text_exclusive
from kinomlx.model_registry import DEFAULT_MODEL, config_registry
from kinomlx.ui import configure_machine_output

from .config_templates import config_template, template_models
from .presentation import render_error

_DEFAULT_CONFIG_PATH = Path("kino-config.toml")
_log = logging.getLogger(__name__)


def build_config_parser() -> argparse.ArgumentParser:
    """Build the model-aware configuration utility parser."""
    parser = argparse.ArgumentParser(
        prog="kinomlx config",
        description="Create and inspect model-specific KinoMLX configuration.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(
        dest="config_command",
        metavar="{init,inspect}",
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
        default=DEFAULT_MODEL,
        help=f"Configuration schema to write (default: {DEFAULT_MODEL}).",
    )
    init.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Destination path (default: ./kino-config.toml; existing paths are refused).",
    )
    inspect = commands.add_parser(
        "inspect",
        help="show registered field ownership and policies as JSON",
        description="Inspect configuration declarations without loading sources or model weights.",
        allow_abbrev=False,
    )
    inspect.add_argument("--model", choices=template_models(), default=DEFAULT_MODEL)
    inspect.add_argument(
        "--field", help="Select one canonical dotted field, such as generate.seed."
    )
    return parser


def run_config_command(argv: list[str]) -> int:
    """Run one ``kinomlx config`` utility and return its process exit code."""
    options = build_config_parser().parse_args(argv)
    if options.config_command == "inspect":
        try:
            inventory = configuration_inventory(
                config_registry().model(options.model), field=options.field
            )
        except ConfigError as exc:
            return render_error(f"config inspect: {exc}", json_output=False)
        output_logger = configure_machine_output("kinomlx.cli.config_inspect.stdout")
        output_logger.info("%s", json.dumps(inventory, indent=2))
        return 0
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
