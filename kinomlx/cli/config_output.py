"""Shared print/save behavior for fully resolved model configurations."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

from kinomlx.io.atomic import write_text_exclusive

from ._registry import config_registry
from .common import render_error

_log = logging.getLogger(__name__)


def handle_config_output(
    options: argparse.Namespace,
    *,
    model: str,
    resolved: Mapping[str, object],
    json_output: bool,
) -> int | None:
    """Print and/or exclusively save a resolved config, or continue execution."""
    print_requested = bool(options.print_config)
    save_path = options.save_config
    if options.only_non_defaults and not print_requested and save_path is None:
        return render_error(
            "config error: --only-non-defaults requires --print-config or --save-config",
            json_output=json_output,
        )
    if not print_requested and save_path is None:
        return None

    schema = config_registry().model(model)
    output = schema.non_default_config(resolved) if options.only_non_defaults else dict(resolved)
    text = schema.dump_config(output)
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
        _log.info("Saved resolved %s configuration to %s", model, destination)
    if print_requested:
        from kinomlx.ui import configure_machine_output

        output_logger = configure_machine_output("kinomlx.cli.config_output.stdout")
        output_logger.info("%s", text.rstrip("\n"))
    return 0


__all__ = ["handle_config_output"]
