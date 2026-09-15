"""Readable starter TOML rendered from the global configuration registry."""

import shlex
import textwrap
from pathlib import Path

from kinomlx.config import (
    ConfigFieldSpec,
    ConfigTableSpec,
    as_sentence,
    dump_toml_value,
)
from kinomlx.model_registry import config_registry

_COMMENT_WIDTH = 96


def _comment(text: str) -> list[str]:
    return [
        f"# {line}" if line else "#"
        for line in textwrap.wrap(
            as_sentence(text),
            width=_COMMENT_WIDTH - 2,
            break_long_words=False,
            break_on_hyphens=False,
        )
    ]


def _labeled_comment(label: str, text: str) -> list[str]:
    initial = f"# {label}: "
    subsequent = "# " + " " * (len(label) + 2)
    return textwrap.wrap(
        as_sentence(text),
        width=_COMMENT_WIDTH,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _choice_comments(field: ConfigFieldSpec) -> list[str]:
    if not field.choice_help:
        return _labeled_comment("Valid", field.valid_text)
    rendered = [(f'"{choice}"', help_text) for choice, help_text in field.choice_help]
    value_width = max(len(value) for value, _help in rendered)
    lines = ["# Valid values:"]
    for value, help_text in rendered:
        initial = f"#   {value.ljust(value_width)}  "
        subsequent = "# " + " " * (len(initial) - 2)
        lines.extend(
            textwrap.wrap(
                as_sentence(help_text),
                width=_COMMENT_WIDTH,
                initial_indent=initial,
                subsequent_indent=subsequent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines


def _member_comments(field: ConfigFieldSpec) -> list[str]:
    """Describe each member of a record collection the way a table describes a field."""
    schema = field.item_schema
    if schema is None:
        return []
    rendered = []
    for member in schema.fields:
        if schema.default_kind(member) == "required":
            default = "Required."
        elif member.default_text is not None:
            default = f"Default: {as_sentence(member.default_text)}"
        else:
            value = schema.default_value(member)
            default = "Default: unset." if value is None else f"Default: {dump_toml_value(value)}."
        rendered.append(
            (
                member.name,
                f"{as_sentence(member.purpose)} Valid: {as_sentence(member.valid)} {default}",
            )
        )
    name_width = max(len(name) for name, _text in rendered)
    lines = ["# Members:"]
    for name, text in rendered:
        initial = f"#   {name.ljust(name_width)}  "
        lines.extend(
            textwrap.wrap(
                text,
                width=_COMMENT_WIDTH,
                initial_indent=initial,
                subsequent_indent="# " + " " * (len(initial) - 2),
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines


def _is_required(table: ConfigTableSpec | None, field: ConfigFieldSpec) -> bool:
    return table is not None and table.default_kind(field) == "required"


def _default_text(table: ConfigTableSpec | None, field: ConfigFieldSpec) -> str:
    if _is_required(table, field):
        # A field with no default has nothing to fall back to, so saying what it
        # would be if omitted would be false. Any nuance lives in a note.
        return "required."
    if field.default_text is not None:
        return as_sentence(field.default_text)
    value = field.default if table is None else table.default_value(field)
    if value is None:
        return "unset."
    return f"{dump_toml_value(value)}."


def _record_array(field: ConfigFieldSpec, value: object) -> list[str] | None:
    """Render a record collection one record per line, so no line runs long."""
    if field.item_schema is None or not isinstance(value, (list, tuple)) or not value:
        return None
    return [
        f"{field.name} = [",
        *(f"  {dump_toml_value(item)}," for item in value),
        "]",
    ]


def _assignment(field: ConfigFieldSpec, value: object, *, active: bool) -> list[str]:
    lines = _record_array(field, value)
    if lines is None:
        rendered = dump_toml_value(
            value,
            multiline=field.multiline,
            stringify_large_int=field.stringify_large_int,
        )
        lines = f"{field.name} = {rendered}".splitlines()
    if active:
        return lines
    return [f"# {line}" if line else "#" for line in lines]


def _render_field(
    field: ConfigFieldSpec,
    *,
    table: ConfigTableSpec | None,
    model: str,
    table_enabled: bool,
    include_assignment: bool = True,
) -> list[str]:
    lines = _comment(field.purpose)
    lines.append("#")
    lines.extend(_choice_comments(field))
    lines.extend(_member_comments(field))
    lines.extend(_labeled_comment("Default", _default_text(table, field)))
    variables = () if table is None else table.environment_variables(field, model=model)
    if variables:
        lines.extend(_labeled_comment("Environment", ", ".join(variables)))
    for note in field.notes:
        lines.extend(_labeled_comment("Note", note))
    if include_assignment:
        active = table_enabled and field.has_starter
        value = field.starter if active else field.example
        lines.extend(_assignment(field, value, active=active))
    return lines


def _render_table(table: ConfigTableSpec, *, model: str) -> list[str]:
    lines: list[str] = []
    if table.introduction is not None:
        lines.extend(_comment(table.introduction))
        lines.append("#")
    header = f"[{'.'.join(table.path)}]"
    if table.starter_enabled:
        lines.append(header)
        for field in table.fields:
            lines.extend(("", ""))
            lines.extend(_render_field(field, table=table, model=model, table_enabled=True))
        return lines

    for index, field in enumerate(table.fields):
        if index:
            lines.append("#")
        marker = " (required)" if _is_required(table, field) else ""
        lines.append(f"# {field.name}{marker}")
        lines.extend(
            _render_field(
                field,
                table=table,
                model=model,
                table_enabled=False,
                include_assignment=False,
            )
        )
    lines.append("#")
    lines.append(f"# {header}")
    for field in table.fields:
        lines.extend(_assignment(field, field.example, active=False))
    return lines


def config_template(
    model: str,
    *,
    destination: Path | str = Path("kino-config.toml"),
) -> str:
    """Return a complete starter configuration generated from one model schema."""
    schema = config_registry().model(model)
    lines = [
        f"# {schema.title}",
        "#",
        "# Run with:",
        "#",
        f"#   kinomlx --config {shlex.quote(str(destination))}",
        "#",
    ]
    for index, paragraph in enumerate(schema.introduction):
        if index:
            lines.append("#")
        lines.extend(_comment(paragraph))
    for field in schema.root_fields:
        lines.extend(("", ""))
        lines.extend(_render_field(field, table=None, model=model, table_enabled=True))
    for table in schema.tables:
        lines.extend(("", ""))
        lines.extend(_render_table(table, model=model))
    return "\n".join((*lines, "")) + "\n"


def template_models() -> tuple[str, ...]:
    """Return every model that contributes to the global config registry."""
    return config_registry().models()


__all__ = ["config_template", "template_models"]
