"""Generate parser groups from the migrated configuration declarations."""

import argparse

from kinomlx.config import ModelConfigSpec
from kinomlx.config_record_cli import ConfigArgumentParser


def add_config_arguments(parser: argparse.ArgumentParser, schema: ModelConfigSpec) -> None:
    """Consume every field in tables that own complete CLI bindings."""
    if any(field.cli_item is not None for _, field in schema.field_items()) and not isinstance(
        parser, ConfigArgumentParser
    ):
        raise TypeError(
            "grouped record arguments require ConfigArgumentParser for local item scope"
        )
    groups: dict[str, argparse._ArgumentGroup] = {}
    triggers = {implication.trigger for implication in schema.implications}
    for table in schema.tables:
        if not table.cli_generated:
            continue
        for field in table.fields:
            if not field.surfaces.supports("cli"):
                continue
            binding = field.cli
            if binding is None:
                raise TypeError(f"{schema.model}.{field.name}: missing CLI binding")
            if binding.group not in groups:
                groups[binding.group] = parser.add_argument_group(binding.group)
            path = (*table.path, field.name)
            table.add_cli_arguments(
                field,
                groups[binding.group],
                negative_help=field.help_text(negative=True),
                help=schema.implication_help(path) if path in triggers else field.help_text(),
            )
