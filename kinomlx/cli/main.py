"""Installed CLI utility router and model-command dispatcher."""

import sys

from kinomlx.config import ConfigError
from kinomlx.errors import RunCanceled
from kinomlx.settings import Settings
from kinomlx.ui import configure_logging
from kinomlx.weights.cli import run_weights_command

from ._registry import resolve_model_command, resolve_model_selection
from .arguments import build_root_parser
from .config_init import run_config_command
from .invocation import (
    bootstrap_flag_present,
    bootstrap_json_from_arguments,
)
from .presentation import (
    render_canceled,
    render_error,
)


def _run_main(arguments: list[str]) -> int:
    """Route one captured argument vector without owning model behavior."""

    configure_logging()
    if arguments and arguments[0] == "config":
        return run_config_command(arguments[1:])
    if arguments and arguments[0] == "weights":
        return run_weights_command(arguments[1:])

    try:
        error_settings = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        error_settings = Settings()
    bootstrap_json = bootstrap_json_from_arguments(arguments, error_settings)
    help_requested = bootstrap_flag_present(arguments, {"-h", "--help"})
    try:
        selection = resolve_model_selection(
            arguments,
            tolerate_errors=help_requested,
        )
    except ConfigError as exc:
        return render_error(f"config error: {exc}", json_output=bootstrap_json)
    if help_requested and not selection.explicit:
        build_root_parser().print_help()
        return 0
    return resolve_model_command(selection)(arguments)


def main(argv: list[str] | None = None) -> int:
    """Run any KinoMLX command and translate user cancellation to status 130."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        return _run_main(arguments)
    except KeyboardInterrupt as exc:
        try:
            error_settings = Settings.from_env_fields("json_output")
        except TypeError, ValueError:
            error_settings = Settings()
        return render_canceled(
            json_output=bootstrap_json_from_arguments(arguments, error_settings),
            outputs=exc.outputs if isinstance(exc, RunCanceled) else None,
        )


if __name__ == "__main__":
    raise SystemExit(main())
