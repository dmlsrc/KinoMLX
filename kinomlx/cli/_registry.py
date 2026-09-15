"""Model command and configuration registry."""

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from kinomlx.config import (
    ConfigError,
    load_config,
    parse_set_argument,
)
from kinomlx.invocation_controls import (
    INVOCATION_CONTROLS,
    MODEL_CONTROL,
    validate_invocation_controls,
)
from kinomlx.model_registry import DEFAULT_MODEL, MODEL_SPECS, config_registry, model_choices

from .invocation import add_selection_arguments


class _ModelCommand(Protocol):
    """Complete installed command contributed by one model package."""

    def __call__(self, arguments: list[str]) -> int: ...


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
    """Resolve model precedence without importing a model or native runtime.

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
    spec = MODEL_SPECS[name]
    return cast(
        Callable[[list[str]], int],
        _resolve_callable(spec.command_module, spec.command_function),
    )


CONFIG_CONTROL_DESTINATIONS = frozenset(
    {"help", *(control.name for control in INVOCATION_CONTROLS if control is not MODEL_CONTROL)}
)


def validate_model_parser(parser: argparse.ArgumentParser, model: str) -> None:
    """Fail when a complete public parser drifts from its model contribution."""
    schema = config_registry().model(model)
    validate_invocation_controls(parser)
    actions: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if action.dest in CONFIG_CONTROL_DESTINATIONS:
            continue
        actions.setdefault(action.dest, []).append(action)

    expected = schema.cli_destinations()
    actual = frozenset(actions)
    if expected != actual:
        missing = sorted(expected - actual)
        unregistered = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing parser destinations: " + ", ".join(missing))
        if unregistered:
            details.append("unregistered parser destinations: " + ", ".join(unregistered))
        raise ValueError(f"{model} config/CLI drift: {'; '.join(details)}")

    for field in schema.root_fields:
        if field.cli is None:
            raise TypeError(f"{model}.{field.name}: missing root CLI binding")
        root_actions = actions[field.cli_dest]
        if len(root_actions) != 1:
            raise ValueError(f"{model}.{field.name}: expected one root CLI action")
        field.cli.validate_action(root_actions[0], annotation=type(field.default))

    for table in schema.tables:
        if not table.cli_generated:
            continue
        for field in table.fields:
            if not field.surfaces.supports("cli"):
                continue
            binding = field.cli
            if binding is None:
                raise TypeError(f"{model}.{field.name}: missing CLI binding")
            table.validate_cli_actions(field, actions[field.cli_dest])

    for destination, field in schema.cli_fields().items():
        if field.cli_item is not None:
            continue  # The record declaration validates each member's help, choices and defaults.
        for action in actions[destination]:
            negative_only = isinstance(action, argparse._StoreFalseAction) or (
                bool(action.option_strings)
                and all(option.startswith("--no-") for option in action.option_strings)
            )
            accepted_help = {field.help_text(negative=negative_only)}
            if not negative_only:
                for implication in schema.implications:
                    if schema.field(implication.trigger).cli_dest == destination:
                        accepted_help.add(schema.implication_help(implication.trigger))
            if action.help not in accepted_help:
                raise ValueError(
                    f"{model} config/CLI help drift for {destination}: "
                    f"registered={sorted(accepted_help)}, parser={action.help!r}"
                )
        non_null_defaults = [
            action.default for action in actions[destination] if action.default is not None
        ]
        if non_null_defaults:
            raise ValueError(
                f"{model} config/CLI default drift for {destination}: parser defaults "
                "must be None so lower-precedence config sources remain visible"
            )
        declared = [
            tuple(str(choice) for choice in action.choices)
            for action in actions[destination]
            if action.choices is not None
        ]
        if not declared:
            if field.choices:
                raise ValueError(
                    f"{model} config/CLI choice drift for {destination}: "
                    f"registered={sorted(field.choices)}, parser has no choices"
                )
            continue
        registered = (
            frozenset(config_registry().models())
            if destination == "model"
            else frozenset(field.choices)
        )
        for choices in declared:
            parsed = frozenset(choices)
            if registered != parsed:
                raise ValueError(
                    f"{model} config/CLI choice drift for {destination}: "
                    f"registered={sorted(registered)}, parser={sorted(parsed)}"
                )


def _resolve_callable(module_name: str, attribute: str) -> _ModelCommand:
    module = importlib.import_module(module_name)
    value: object = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return cast(_ModelCommand, value)


__all__ = [
    "CONFIG_CONTROL_DESTINATIONS",
    "ModelSelection",
    "model_summary",
    "resolve_model_command",
    "resolve_model_selection",
    "validate_model_parser",
]
