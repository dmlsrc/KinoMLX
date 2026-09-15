"""Shared print/save behavior for canonical configuration intent."""

import argparse
import logging
from pathlib import Path

from kinomlx.config.provenance import PreparedConfig
from kinomlx.config_values import ConfigError
from kinomlx.invocation_controls import validate_control_inputs
from kinomlx.io.atomic import write_text_exclusive
from kinomlx.model_registry import config_registry
from kinomlx.ui import configure_machine_output

from .presentation import render_error

_log = logging.getLogger(__name__)


def handle_config_output(
    options: argparse.Namespace,
    *,
    model: str,
    json_output: bool,
    intent: PreparedConfig,
) -> int | None:
    """Print and/or exclusively save canonical inputs, or continue execution."""
    print_requested = bool(options.print_config)
    save_path = options.save_config
    try:
        validate_control_inputs(options)
    except ValueError as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    if not print_requested and save_path is None:
        return None

    schema = config_registry().model(model)
    try:
        text = schema.dump_intent(intent)
        if not options.only_non_defaults:
            defaults = schema.dump_config(schema.default_config())
            reference = "\n".join("# " + line if line else "#" for line in defaults.splitlines())
            text += (
                "\n# Built-in defaults for reference only; these comments are not inputs.\n"
                + reference
                + "\n"
            )
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    if save_path is not None:
        destination = Path(save_path).expanduser()
        try:
            write_text_exclusive(destination, text)
        except FileExistsError:
            return render_error(
                f"config save refused: output already exists: {destination}; "
                "choose another path or remove the existing file",
                json_output=json_output,
            )
        except OSError as exc:
            return render_error(
                f"config save failed for {destination}: {exc}",
                json_output=json_output,
            )
        _log.info("Saved %s configuration to %s", model, destination)
    if print_requested:
        output_logger = configure_machine_output("kinomlx.cli.config_output.stdout")
        output_logger.info("%s", text.rstrip("\n"))
    return 0


__all__ = ["handle_config_output"]
