"""Generate a declaration inventory without resolving invocation sources."""

import argparse

from kinomlx.config_records import tracks_presence
from kinomlx.config_rules import record_rules
from kinomlx.config_values import ConfigError
from kinomlx.invocation_controls import INVOCATION_CONTROLS

from .dump import dump_toml_value
from .registry import ConfigRecordSpec, ModelConfigSpec


def _record_inventory(schema: ConfigRecordSpec) -> dict[str, object]:
    members = []
    for field in schema.fields:
        default = schema.default_value(field)
        members.append(
            {
                "name": field.name,
                "surface_classification": field.surfaces.describe(),
                "cli_flags": list(field.cli.option_strings(field.cli_dest)) if field.cli else [],
                "type": str(schema.field_annotation(field)),
                "purpose": field.purpose,
                "constraints": field.valid_text,
                "local_constraints": [rule.describe() for rule in schema.local_constraints(field)],
                "default_kind": schema.default_kind(field),
                "default_toml": None if default is None else dump_toml_value(default),
                "item_schema": None
                if field.item_schema is None
                else _record_inventory(field.item_schema),
            }
        )
    return {
        "owner": f"{schema.record.__module__}.{schema.record.__qualname__}",
        "constructor_presence": "tracked",
        "members": members,
    }


def configuration_inventory(
    schema: ModelConfigSpec, *, field: str | None = None
) -> dict[str, object]:
    """Expose configuration and runtime ownership from the compiled model contribution."""
    tables = {table.path: table for table in schema.tables}
    environment = schema.environment_variables()
    entries: list[dict[str, object]] = []
    for path, declaration in schema.field_items():
        name = ".".join(path)
        if field is not None and name != field:
            continue
        table = tables.get(path[:-1])
        annotation = (
            declaration.input_annotation or type(declaration.default)
            if declaration.virtual
            else table.field_annotation(declaration)
            if table is not None
            else None
        )
        default = declaration.default if table is None else table.default_value(declaration)
        binding = declaration.cli
        cli: dict[str, object] | None = None
        if binding is not None:
            parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            actions = (
                table.add_cli_arguments(declaration, parser, help=None)
                if table is not None
                else binding.add_arguments(
                    parser,
                    destination=declaration.cli_dest,
                    annotation=annotation,
                    help=None,
                    choices=declaration.choices,
                )
            )
            cli = {
                "flags": [*action.option_strings for action in actions],
                "destination": declaration.cli_dest,
                "item_member": declaration.cli_item,
                "action": "record_items"
                if declaration.cli_item is not None
                else binding.action
                if isinstance(binding.action, str)
                else binding.action.__qualname__,
                "codec": getattr(binding.parse, "__name__", None)
                if binding.parse is not None
                else "inferred",
            }
        local_constraints = (
            [rule.describe() for rule in table.local_constraints(declaration)]
            if table is not None
            else []
        )
        entries.append(
            {
                "path": name,
                "owner": f"{table.record.__module__}.{table.record.__qualname__}.{declaration.name}"
                if table is not None and not declaration.virtual
                else f"{schema.model}:virtual:{name}",
                "type": str(annotation),
                "virtual": declaration.virtual,
                "item_schema": None
                if declaration.item_schema is None
                else _record_inventory(declaration.item_schema),
                "constructor_presence": "tracked"
                if table is not None and not declaration.virtual and tracks_presence(table.record)
                else "not_registered",
                "purpose": declaration.purpose,
                "constraints": declaration.valid_text,
                "local_constraints": local_constraints,
                "validation_rules": [
                    rule.describe()
                    for rule in record_rules(table.record)
                    if declaration.name in rule.reads
                ]
                if table is not None and not declaration.virtual
                else [],
                "default_kind": "value" if table is None else table.default_kind(declaration),
                "default_toml": None if default is None else dump_toml_value(default),
                "default_description": declaration.default_text,
                "cli": cli,
                "environment": list(environment.get(path, ())),
                "normalization": [
                    {
                        "rule": rule.name,
                        "phase": rule.phase,
                        "reads": path in rule.reads,
                        "writes": path in rule.writes,
                        "effects": sorted(rule.effects),
                    }
                    for rule in schema.layer_transforms
                    if path in rule.reads or path in rule.writes
                ],
                "implications": [
                    {
                        "trigger": ".".join(implication.trigger),
                        "targets": [".".join(target) for target, _ in implication.implied],
                    }
                    for implication in schema.implications
                    if path == implication.trigger or path in dict(implication.implied)
                ],
                "stage_policies": [
                    {
                        "stage": policy.stage,
                        "mode": policy.mode,
                        "reason": policy.reason,
                        "irrelevant_action": policy.irrelevant_action,
                        "projection": policy.attribute or declaration.name,
                    }
                    for policy in declaration.stage_policies
                ],
                "stage_coverage": "source_virtual"
                if declaration.virtual
                else "registered"
                if table is not None and table.stages
                else "no_restart"
                if schema.restart_stages == ()
                else "not_registered",
                "serialization": {
                    "strip": declaration.strip,
                    "drop_when_none": declaration.drop_when_none,
                    "stringify_large_int": declaration.stringify_large_int,
                    "multiline": declaration.multiline,
                },
                "persistence_policy": declaration.persistence,
                "surface_classification": declaration.surfaces.describe(),
                "output_policy": None
                if declaration.output_policy is None
                else declaration.output_policy.describe(),
            }
        )
    if field is not None and not entries:
        raise ConfigError(f"{schema.model}: unknown configuration field {field!r}")
    return {
        "model": schema.model,
        "restart": {
            "stages": list(schema.stages),
            "restart_stages": list(schema.restart_stages),
            "unavailable_reason": schema.no_restart_reason,
        },
        "api_boundaries": [boundary.describe() for boundary in schema.api_boundaries],
        "fields": entries,
        "invocation_controls": [control.describe() for control in INVOCATION_CONTROLS],
    }
