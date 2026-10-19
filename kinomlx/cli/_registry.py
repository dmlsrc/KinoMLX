"""Model selection and command dispatch."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from kinomlx.config import (
    ConfigError,
    load_config,
    parse_set_argument,
)
from kinomlx.model_catalog import DEFAULT_MODEL, MODEL_SPECS, model_choices
from kinomlx.models.flux2 import command as flux2_command
from kinomlx.models.gmnet import command as gmnet_command
from kinomlx.models.krea2 import command as krea2_command
from kinomlx.models.ltx2 import command as ltx2_command

from .invocation import add_selection_arguments

# The complete installed command each catalog model contributes. Each entry looks its
# command up on the model's module when it runs, so a command replaced there is the one
# dispatched; binding the functions here would silently bypass such a replacement.
_MODEL_COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "flux2": lambda arguments: flux2_command.run_flux2_command(arguments),
    "gmnet": lambda arguments: gmnet_command.run_gmnet_command(arguments),
    "krea2": lambda arguments: krea2_command.run_krea2_command(arguments),
    "ltx2": lambda arguments: ltx2_command.run_ltx2_command(arguments),
}
if _MODEL_COMMANDS.keys() != MODEL_SPECS.keys():
    raise RuntimeError("every catalog model must contribute exactly one command")


@dataclass(frozen=True)
class ModelSelection:
    """The model selected by config, ordinary CLI flags, and ``--set``."""

    name: str
    explicit: bool
    source: str


def model_summary() -> str:
    """Render the registered model catalog for root CLI discovery."""
    width = max(map(len, MODEL_SPECS))
    return "\n".join(f"  {name:<{width}}  {MODEL_SPECS[name].help}" for name in model_choices())


def _bootstrap_options(arguments: list[str]) -> argparse.Namespace:
    """Parse only model-selection inputs while leaving model flags untouched."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    add_selection_arguments(parser)
    try:
        options, _unknown = parser.parse_known_args(arguments)
    except argparse.ArgumentError as exc:
        raise ConfigError(f"model selection: {exc}") from exc
    return options


def resolve_model_selection(
    arguments: list[str],
    *,
    tolerate_errors: bool = False,
) -> ModelSelection:
    """Resolve which model an invocation selects.

    Selection follows the same precedence as a complete invocation: config,
    then ``--model``, then ordered ``--set model=...`` overrides. Absence of a
    selector preserves the compatible LTX-2 default.
    """
    try:
        options = _bootstrap_options(arguments)
    except ConfigError:
        if tolerate_errors:
            return ModelSelection(name=DEFAULT_MODEL, explicit=False, source="default")
        raise
    selected: object = None
    explicit = False
    source = "default"
    if options.config is not None:
        try:
            file_config = load_config(options.config)
        except ConfigError:
            if not tolerate_errors:
                raise
        else:
            if "model" in file_config:
                selected = file_config["model"]
                explicit = True
                source = f"config {options.config}"
    if options.model is not None:
        selected = options.model
        explicit = True
        source = "--model"
    for argument in options.set_overrides or []:
        try:
            path, value = parse_set_argument(argument)
        except ConfigError:
            if tolerate_errors:
                continue
            raise
        if path == ["model"]:
            selected = value
            explicit = True
            source = f"--set {argument!r}"
    if selected is None:
        selected = DEFAULT_MODEL
    if not isinstance(selected, str):
        if tolerate_errors:
            return ModelSelection(name=DEFAULT_MODEL, explicit=False, source="default")
        raise ConfigError(f"{source}: model must be a string, got {type(selected).__name__}")
    if selected not in model_choices():
        if tolerate_errors:
            return ModelSelection(name=DEFAULT_MODEL, explicit=False, source="default")
        supported = ", ".join(model_choices())
        raise ConfigError(f"{source}: unknown model {selected!r}; expected one of {supported}")
    return ModelSelection(name=selected, explicit=explicit, source=source)


def resolve_model_command(selection: ModelSelection | str) -> Callable[[list[str]], int]:
    """Resolve one model's complete CLI command at dispatch."""
    name = selection.name if isinstance(selection, ModelSelection) else selection
    return _MODEL_COMMANDS[name]


__all__ = [
    "ModelSelection",
    "model_summary",
    "resolve_model_command",
    "resolve_model_selection",
]
