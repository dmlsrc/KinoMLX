"""Model-neutral invocation flags and bootstrap argument inspection."""

import argparse
from collections.abc import Iterable

from kinomlx.invocation_controls import INVOCATION_CONTROLS, MODEL_CONTROL
from kinomlx.settings import Settings


def model_config_epilog(model: str) -> str:
    """Return the shared starter-config hint for a model-specific parser."""
    return f"Use 'kinomlx config init --model {model}' to write an annotated starter config."


def add_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    choices: Iterable[str] | None = None,
    model_help: str | None = None,
) -> None:
    """Declare the selectors shared by bootstrap and every complete model parser."""
    for control in INVOCATION_CONTROLS:
        if control.phase == "selection":
            control.add(
                parser,
                choices=tuple(choices) if control is MODEL_CONTROL and choices is not None else (),
                help=model_help if control is MODEL_CONTROL else None,
            )


def add_invocation_arguments(
    parser: argparse.ArgumentParser,
    *,
    choices: Iterable[str],
    model_help: str | None = None,
) -> None:
    """Add the invocation flags shared by every model command."""
    add_selection_arguments(parser, choices=choices, model_help=model_help)
    for control in INVOCATION_CONTROLS:
        if control.phase == "utility":
            control.add(parser)


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
    "add_selection_arguments",
    "bootstrap_flag_present",
    "bootstrap_json_from_arguments",
    "bootstrap_json_mode",
    "model_config_epilog",
]
