"""Contract between a model's complete CLI parser and its configuration contribution."""

import argparse

from kinomlx.invocation_controls import (
    INVOCATION_CONTROLS,
    MODEL_CONTROL,
    validate_invocation_controls,
)
from kinomlx.model_registry import config_registry

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


__all__ = ["CONFIG_CONTROL_DESTINATIONS", "validate_model_parser"]
