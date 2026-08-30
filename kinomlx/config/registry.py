"""Canonical configuration-field schemas contributed by each model."""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from enum import StrEnum
from typing import get_args, get_type_hints

from kinomlx.settings import EnvironmentSettings

ConfigPath = tuple[str, ...]


class ConfigGroup(StrEnum):
    """Semantic roles consumed by generic registry-backed behavior."""

    SAVE_ALL_CANDIDATE = "save_all_candidate"
    MODEL_SOURCE = "model_source"
    MODEL_MONOLITHIC_SOURCE = "model_monolithic_source"
    MODEL_SPLIT_SOURCE = "model_split_source"
    MODEL_GEMMA_SOURCE = "model_gemma_source"
    MODEL_TEXT_ENCODER_SOURCE = "model_text_encoder_source"
    MODEL_GENERATION_SELECTOR = "model_generation_selector"
    MODEL_VIDEO_VAE_SELECTOR = "model_video_vae_selector"
    MODEL_VIDEO_VAE_SOURCE = "model_video_vae_source"
    RESTART_DECODE_LOCKED = "restart_decode_locked"
    RESTART_STAGE2_LOCKED = "restart_stage2_locked"


_NO_VALUE = object()
_ENVIRONMENT_VARIABLE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_CONFIG_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_IMPERATIVE_VERBS = frozenset(
    {
        "Apply",
        "Build",
        "Choose",
        "Collect",
        "Compile",
        "Configure",
        "Describe",
        "Disable",
        "Emit",
        "Enable",
        "Generate",
        "Keep",
        "Let",
        "Load",
        "Mirror",
        "Override",
        "Pad",
        "Pretranspose",
        "Replace",
        "Reserve",
        "Reuse",
        "Save",
        "Select",
        "Set",
        "Substitute",
        "Use",
        "Write",
    }
)


def as_sentence(text: str) -> str:
    """Normalize one registry-owned prose fragment as a sentence."""
    value = text.strip()
    return value if value.endswith((".", "!", "?")) else value + "."


def _validate_imperative(text: str, *, location: str) -> None:
    value = text.strip()
    first = value.partition(" ")[0]
    if first not in _IMPERATIVE_VERBS:
        allowed = ", ".join(sorted(_IMPERATIVE_VERBS))
        raise ValueError(
            f"{location}: purpose must use the imperative convention; "
            f"first word {first!r} is not one of: {allowed}"
        )


def _record_fields(record: type[object]) -> dict[str, dataclasses.Field[object]]:
    if not isinstance(record, type) or not dataclasses.is_dataclass(record):
        name = getattr(record, "__name__", type(record).__name__)
        raise TypeError(f"configuration record {name} is not a dataclass type")
    reflected = dataclasses.fields(record)
    return {item.name: item for item in reflected}


def _field_default(item: dataclasses.Field[object]) -> object:
    if item.default is not MISSING:
        return copy.deepcopy(item.default)
    if item.default_factory is not MISSING:
        return item.default_factory()
    return _NO_VALUE


def _is_boolean_annotation(annotation: object) -> bool:
    """Return whether a record annotation accepts only bool, optionally with None."""
    if annotation is bool:
        return True
    arguments = frozenset(get_args(annotation))
    return bool in arguments and arguments <= {bool, type(None)}


def _set_path(target: dict[str, object], path: ConfigPath, value: object) -> None:
    if not path:
        raise ValueError("configuration paths cannot be empty")
    current = target
    for name in path[:-1]:
        existing = current.get(name)
        if existing is None:
            nested: dict[str, object] = {}
            current[name] = nested
            current = nested
            continue
        if not isinstance(existing, dict):
            raise ValueError(f"configuration path {'.'.join(path)} crosses scalar {name!r}")
        current = existing
    current[path[-1]] = copy.deepcopy(value)


def _path_value(source: Mapping[str, object], path: ConfigPath) -> tuple[bool, object]:
    current: object = source
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return False, _NO_VALUE
        current = current[name]
    return True, current


def _delete_path(target: dict[str, object], path: ConfigPath) -> None:
    if len(path) == 1:
        target.pop(path[0], None)
        return
    child = target.get(path[0])
    if not isinstance(child, dict):
        return
    _delete_path(child, path[1:])
    if not child:
        target.pop(path[0], None)


def _without_defaults(
    effective: Mapping[str, object],
    defaults: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in effective.items():
        default_present = name in defaults
        default = defaults.get(name)
        if isinstance(value, Mapping):
            nested_defaults = default if isinstance(default, Mapping) else {}
            nested = _without_defaults(value, nested_defaults)
            if nested:
                result[name] = nested
            continue
        if default_present and value == default:
            continue
        result[name] = copy.deepcopy(value)
    return result


@dataclass(frozen=True)
class ConfigFieldSpec:
    """Documentation, example, and CLI placement for one TOML field."""

    name: str
    purpose: str
    valid: str
    example: object
    cli_dest: str
    starter: object = _NO_VALUE
    virtual: bool = False
    default: object = _NO_VALUE
    default_text: str | None = None
    choices: tuple[str, ...] = ()
    choice_help: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    negative_purpose: str | None = None
    multiline: bool = False
    strip: bool = False
    stringify_large_int: bool = False
    groups: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if _CONFIG_NAME.fullmatch(self.name) is None:
            raise ValueError(f"invalid configuration field name {self.name!r}")
        if _CONFIG_NAME.fullmatch(self.cli_dest) is None:
            raise ValueError(f"{self.name}: invalid CLI destination {self.cli_dest!r}")
        if not self.purpose.strip():
            raise ValueError(f"{self.name}: purpose cannot be empty")
        _validate_imperative(self.purpose, location=self.name)
        if self.negative_purpose is not None:
            _validate_imperative(self.negative_purpose, location=f"{self.name} negative purpose")
        if not self.valid.strip() and not self.choices:
            raise ValueError(f"{self.name}: valid values cannot be empty")
        if self.virtual and self.default is _NO_VALUE:
            raise ValueError(f"{self.name}: virtual fields need an explicit default")
        if not self.virtual and self.default is not _NO_VALUE:
            raise ValueError(f"{self.name}: dataclass-owned defaults cannot be overridden")
        if self.choices and self.valid:
            raise ValueError(f"{self.name}: use choices or valid, not both")
        if len(self.choices) != len(set(self.choices)) or any(
            not isinstance(item, str) or not item for item in self.choices
        ):
            raise ValueError(f"{self.name}: choices must be unique non-empty strings")
        if self.choices and self.example not in self.choices:
            raise ValueError(f"{self.name}: example {self.example!r} is not a registered choice")
        if self.choices and self.has_starter and self.starter not in self.choices:
            raise ValueError(f"{self.name}: starter {self.starter!r} is not a registered choice")
        if self.virtual and self.choices and self.default not in self.choices:
            raise ValueError(f"{self.name}: default {self.default!r} is not a registered choice")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in self.choice_help
        ):
            raise ValueError(f"{self.name}: choice help must contain string pairs")
        choice_help = dict(self.choice_help)
        if len(choice_help) != len(self.choice_help):
            raise ValueError(f"{self.name}: choice help repeats a value")
        if choice_help and not self.choices:
            raise ValueError(f"{self.name}: choice help requires registered choices")
        if choice_help and tuple(choice_help) != self.choices:
            missing = sorted(set(self.choices) - set(choice_help))
            stale = sorted(set(choice_help) - set(self.choices))
            raise ValueError(
                f"{self.name}: choice-help drift; expected order={self.choices}, "
                f"actual order={tuple(choice_help)}, missing={missing}, stale={stale}"
            )
        if len(self.choices) > 1 and not choice_help:
            raise ValueError(f"{self.name}: multi-value choices require per-choice help")
        if any(not value.strip() for value in choice_help.values()):
            raise ValueError(f"{self.name}: choice help cannot be empty")
        if any(not isinstance(note, str) or not note.strip() for note in self.notes):
            raise ValueError(f"{self.name}: interaction notes cannot be empty")
        if self.multiline and not isinstance(self.example, str):
            raise ValueError(f"{self.name}: multiline presentation requires a string example")
        if self.multiline and self.has_starter and not isinstance(self.starter, str):
            raise ValueError(f"{self.name}: multiline presentation requires a string starter")
        if self.stringify_large_int and (
            isinstance(self.example, bool) or not isinstance(self.example, int)
        ):
            raise ValueError(f"{self.name}: large-integer stringification requires an int example")
        known_groups = frozenset(group.value for group in ConfigGroup)
        unknown_groups = sorted(set(self.groups) - known_groups)
        if unknown_groups:
            raise ValueError(f"{self.name}: unknown semantic groups: {', '.join(unknown_groups)}")

    @property
    def has_starter(self) -> bool:
        """Whether the starter template should activate this field."""
        return self.starter is not _NO_VALUE

    @property
    def valid_text(self) -> str:
        """Return the canonical human-readable constraint."""
        if self.choices:
            rendered = [f'"{choice}"' for choice in self.choices]
            if len(rendered) == 1:
                return rendered[0]
            if len(rendered) == 2:
                return f"{rendered[0]} or {rendered[1]}"
            return ", ".join(rendered[:-1]) + f", or {rendered[-1]}"
        return self.valid

    @property
    def choice_help_mapping(self) -> dict[str, str]:
        """Return per-choice explanations in the declared choice order."""
        return dict(self.choice_help)

    def help_text(self, *, negative: bool = False) -> str:
        """Return canonical argparse prose for this field."""
        if negative:
            purpose = self.negative_purpose or (f"Set {self.name.replace('_', ' ')} to false")
        else:
            purpose = self.purpose
        sentences = [as_sentence(purpose)]
        if self.choice_help:
            details = "; ".join(
                f"{choice}: {as_sentence(help_text).removesuffix('.')}"
                for choice, help_text in self.choice_help
            )
            sentences.append(f"Choices: {details}.")
        sentences.extend(as_sentence(note) for note in self.notes)
        return " ".join(sentences)


@dataclass(frozen=True)
class ConfigTableSpec:
    """One dataclass-owned TOML table in a contributed model schema."""

    path: ConfigPath
    record: type[object]
    fields: tuple[ConfigFieldSpec, ...]
    nested: tuple[str, ...] = ()
    starter_enabled: bool = True
    default_present: bool = True
    introduction: str | None = None

    def __post_init__(self) -> None:
        record_name = getattr(self.record, "__name__", type(self.record).__name__)
        location = f"{record_name} [{'.'.join(self.path)}]"
        if not self.path or any(_CONFIG_NAME.fullmatch(part) is None for part in self.path):
            raise ValueError(
                f"{record_name}: invalid configuration table path {'.'.join(self.path)!r}"
            )
        reflected = _record_fields(self.record)
        documented = [item.name for item in self.fields]
        if len(documented) != len(set(documented)):
            raise ValueError(f"{location} repeats a documented field")
        if len(self.nested) != len(set(self.nested)):
            raise ValueError(f"{location} repeats a nested table")
        overlap = set(documented) & set(self.nested)
        if overlap:
            raise ValueError(
                f"{location} fields collide with nested tables: {', '.join(sorted(overlap))}"
            )
        owned = {item.name for item in self.fields if not item.virtual} | set(self.nested)
        actual = set(reflected)
        if owned != actual:
            missing = sorted(actual - owned)
            stale = sorted(owned - actual)
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if stale:
                details.append("stale " + ", ".join(stale))
            raise ValueError(f"{location} schema drift: {'; '.join(details)}")
        if issubclass(self.record, EnvironmentSettings):
            for name in actual:
                self.record.environment_sources_for_field(name)
        for field in self.fields:
            default = self._declared_default(field)
            if default is _NO_VALUE:
                if self.default_present:
                    raise ValueError(
                        f"{location}.{field.name}: a default-present "
                        "config table cannot contain a required field"
                    )
                continue
            if field.choices and default is not None and default not in field.choices:
                raise ValueError(
                    f"{location}.{field.name}: default {default!r} is not a registered choice"
                )
            if field.default_text is not None:
                environment_backed = bool(self.environment_variables(field))
                if default is not None and not environment_backed:
                    raise ValueError(
                        f"{location}.{field.name}: default_text is permitted only for "
                        "an unset or environment-backed executable default"
                    )
            if field.multiline and default is not None and not isinstance(default, str):
                raise ValueError(
                    f"{location}.{field.name}: multiline presentation requires a string default"
                )
            if (
                field.stringify_large_int
                and default is not None
                and (isinstance(default, bool) or not isinstance(default, int))
            ):
                raise ValueError(
                    f"{location}.{field.name}: large-integer stringification requires an int default"
                )

    def _declared_default(self, field: ConfigFieldSpec) -> object:
        if field.virtual:
            return copy.deepcopy(field.default)
        reflected = _record_fields(self.record)
        return _field_default(reflected[field.name])

    def default_value(self, field: ConfigFieldSpec) -> object:
        """Return the executable built-in default for one documented field."""
        value = self._declared_default(field)
        return None if value is _NO_VALUE else value

    def default_mapping(self) -> dict[str, object]:
        """Return non-null direct defaults for this table."""
        result: dict[str, object] = {}
        for field in self.fields:
            value = self.default_value(field)
            if value is not None:
                result[field.name] = value
        return result

    def environment_variables(self, field: ConfigFieldSpec) -> tuple[str, ...]:
        """Return env-var names derived from the record's executable metadata."""
        if field.virtual or not issubclass(self.record, EnvironmentSettings):
            return ()
        entries = self.record.environment_sources_for_field(field.name)
        return tuple(
            match.group(1) for entry in entries for match in _ENVIRONMENT_VARIABLE.finditer(entry)
        )

    def environment_overrides(self) -> dict[str, object]:
        """Return values currently supplied by this table's declared environment."""
        if not issubclass(self.record, EnvironmentSettings):
            return {}
        names = tuple(field.name for field in self.fields if self.environment_variables(field))
        return self.record.overrides_from_env_fields(*names)


@dataclass(frozen=True)
class ConfigImplication:
    """Fields implied by one higher-level non-default selector."""

    trigger: ConfigPath
    trigger_value: object
    implied: tuple[tuple[ConfigPath, object], ...]
    expand_in_layers: bool = False
    excluded: tuple[ConfigPath, ...] = ()
    coverage_group: str | None = None


@dataclass(frozen=True)
class ModelConfigSpec:
    """One complete model contribution to the global configuration registry."""

    model: str
    title: str
    introduction: tuple[str, ...]
    root_fields: tuple[ConfigFieldSpec, ...]
    tables: tuple[ConfigTableSpec, ...]
    implications: tuple[ConfigImplication, ...] = ()

    def __post_init__(self) -> None:
        if _MODEL_NAME.fullmatch(self.model) is None:
            raise ValueError(f"invalid model config name {self.model!r}")
        if not self.title.strip() or not self.introduction:
            raise ValueError(f"{self.model}: config title and introduction are required")
        root_names = [item.name for item in self.root_fields]
        if len(root_names) != len(set(root_names)):
            raise ValueError(f"{self.model}: duplicate root configuration fields")
        if any(not item.virtual for item in self.root_fields):
            raise ValueError(f"{self.model}: root fields must declare explicit virtual defaults")
        model_fields = [field for field in self.root_fields if field.name == "model"]
        if len(model_fields) != 1:
            raise ValueError(f"{self.model}: exactly one root model field is required")
        model_field = model_fields[0]
        if (
            model_field.default != self.model
            or model_field.cli_dest != "model"
            or model_field.choices != (self.model,)
        ):
            raise ValueError(f"{self.model}: root model field does not match its contribution")

        by_path = {table.path: table for table in self.tables}
        if len(by_path) != len(self.tables):
            raise ValueError(f"{self.model}: duplicate configuration table paths")
        table_roots = {table.path[0] for table in self.tables}
        root_collisions = set(root_names) & table_roots
        if root_collisions:
            raise ValueError(
                f"{self.model}: root fields collide with tables: "
                f"{', '.join(sorted(root_collisions))}"
            )
        for table in self.tables:
            children = {
                candidate.path[-1] for candidate in self.tables if candidate.path[:-1] == table.path
            }
            if children != set(table.nested):
                raise ValueError(
                    f"[{'.'.join(table.path)}] nested schema drift: "
                    f"registered={sorted(children)}, owned={sorted(table.nested)}"
                )
            if len(table.path) > 1 and table.path[:-1] not in by_path:
                raise ValueError(f"[{'.'.join(table.path)}] has no parent table contribution")

        destinations = [
            field.cli_dest
            for field in (
                *self.root_fields,
                *(item for table in self.tables for item in table.fields),
            )
        ]
        if len(destinations) != len(set(destinations)):
            raise ValueError(f"{self.model}: duplicate CLI destinations in config schema")

        environment_paths: dict[str, ConfigPath] = {}
        for path, variables in self.environment_variables().items():
            for variable in variables:
                if variable in environment_paths:
                    raise ValueError(
                        f"{self.model}: environment variable {variable} is registered for "
                        f"both {'.'.join(environment_paths[variable])} and {'.'.join(path)}"
                    )
                environment_paths[variable] = path

        triggers: set[ConfigPath] = set()
        for implication in self.implications:
            self.field(implication.trigger)
            if implication.trigger in triggers:
                raise ValueError(
                    f"{self.model}: duplicate implication trigger {'.'.join(implication.trigger)}"
                )
            triggers.add(implication.trigger)
            implied_paths: set[ConfigPath] = set()
            for path, _value in implication.implied:
                self.field(path)
                if path == implication.trigger:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(path)} cannot imply itself"
                    )
                if path in implied_paths:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} repeats "
                        f"{'.'.join(path)}"
                    )
                implied_paths.add(path)
            excluded_paths: set[ConfigPath] = set()
            for path in implication.excluded:
                self.field(path)
                if path == implication.trigger:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(path)} cannot exclude itself"
                    )
                if path in excluded_paths:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} repeats "
                        f"excluded path {'.'.join(path)}"
                    )
                if path in implied_paths:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} both "
                        f"implies and excludes {'.'.join(path)}"
                    )
                excluded_paths.add(path)
            if implication.coverage_group is not None:
                covered = set(self.paths_in_group(implication.coverage_group))
                if not covered:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} uses "
                        f"empty coverage group {implication.coverage_group!r}"
                    )
                declared = implied_paths | excluded_paths
                if implication.coverage_group == ConfigGroup.SAVE_ALL_CANDIDATE:
                    table = self.table(implication.trigger[:-1])
                    annotations = get_type_hints(table.record)
                    convention_candidates = {
                        (*table.path, field.name)
                        for field in table.fields
                        if field.name != implication.trigger[-1]
                        and field.name.startswith("save_")
                        and _is_boolean_annotation(annotations.get(field.name))
                    }
                    unclassified = sorted(
                        ".".join(path) for path in convention_candidates - covered
                    )
                    if unclassified:
                        raise ValueError(
                            f"{self.model}: implication "
                            f"{'.'.join(implication.trigger)} has unclassified "
                            f"save-all boolean fields: {unclassified}"
                        )
                if covered != declared:
                    missing = sorted(".".join(path) for path in covered - declared)
                    stale = sorted(".".join(path) for path in declared - covered)
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} "
                        f"coverage drift for {implication.coverage_group!r}; "
                        f"missing={missing}, stale={stale}"
                    )

    def field_items(self) -> tuple[tuple[ConfigPath, ConfigFieldSpec], ...]:
        """Return every contributed field with its exact TOML path."""
        root = tuple(((field.name,), field) for field in self.root_fields)
        nested = tuple(
            ((*table.path, field.name), field) for table in self.tables for field in table.fields
        )
        return (*root, *nested)

    def paths_in_group(self, group: str) -> tuple[ConfigPath, ...]:
        """Return schema-ordered field paths carrying one semantic group."""
        return tuple(path for path, field in self.field_items() if group in field.groups)

    def cli_destinations_in_group(self, group: str) -> tuple[str, ...]:
        """Return schema-ordered argparse destinations carrying one semantic group."""
        return tuple(self.field(path).cli_dest for path in self.paths_in_group(group))

    def only_cli_destination_in_group(self, group: str) -> str:
        """Return the sole destination in a semantic group or fail construction."""
        destinations = self.cli_destinations_in_group(group)
        if len(destinations) != 1:
            raise ValueError(
                f"{self.model}: semantic group {group!r} must contain exactly one field, "
                f"got {destinations}"
            )
        return destinations[0]

    def table(self, path: ConfigPath) -> ConfigTableSpec:
        """Return one table contribution by exact path."""
        for table in self.tables:
            if table.path == path:
                return table
        raise KeyError(".".join(path))

    def field(self, path: ConfigPath) -> ConfigFieldSpec:
        """Return one field contribution by exact dotted path."""
        if len(path) == 1:
            for field in self.root_fields:
                if field.name == path[0]:
                    return field
            raise KeyError(path[0])
        table = self.table(path[:-1])
        for field in table.fields:
            if field.name == path[-1]:
                return field
        raise KeyError(".".join(path))

    def table_choices(self, path: ConfigPath) -> dict[str, tuple[str, ...]]:
        """Return constrained values for fields in one table."""
        return {field.name: field.choices for field in self.table(path).fields if field.choices}

    def table_help(
        self,
        path: ConfigPath,
        *,
        negative: bool = False,
    ) -> dict[str, str]:
        """Return canonical argparse help for fields in one table."""
        return {field.name: field.help_text(negative=negative) for field in self.table(path).fields}

    def cli_help(self, destination: str, *, negative: bool = False) -> str:
        """Return canonical argparse help for one registered destination."""
        try:
            field = self.cli_fields()[destination]
        except KeyError as exc:
            raise KeyError(f"unknown {self.model} CLI destination {destination!r}") from exc
        return field.help_text(negative=negative)

    def implication_help(self, trigger: ConfigPath) -> str:
        """Describe an implication directly from its registered field paths."""
        implication = next(
            (item for item in self.implications if item.trigger == trigger),
            None,
        )
        if implication is None:
            raise KeyError(".".join(trigger))
        enabled = "; ".join(
            as_sentence(self.field(path).purpose).removesuffix(".")
            for path, _value in implication.implied
        )
        text = f"{as_sentence(self.field(trigger).purpose)} Implies: {enabled}."
        if implication.excluded:
            excluded = "; ".join(
                as_sentence(self.field(path).purpose).removesuffix(".")
                for path in implication.excluded
            )
            text += f" Excludes large or specialized outputs: {excluded}."
        return text

    def cli_destinations(self) -> frozenset[str]:
        """Return every argparse destination contributed to invocation TOML."""
        return frozenset(self.cli_fields())

    def cli_fields(self) -> dict[str, ConfigFieldSpec]:
        """Return contributed fields keyed by their unique argparse destination."""
        fields = (*self.root_fields, *(item for table in self.tables for item in table.fields))
        return {field.cli_dest: field for field in fields}

    def cli_layer(self, options: object) -> dict[str, object]:
        """Build the model's TOML-shaped ordinary-CLI precedence layer."""
        layer: dict[str, object] = {}
        for field in self.root_fields:
            value = getattr(options, field.cli_dest, None)
            if value is not None:
                layer[field.name] = value
        for table in self.tables:
            values: dict[str, object] = {}
            for field in table.fields:
                value = getattr(options, field.cli_dest, None)
                if value is not None:
                    values[field.name] = value
            if values:
                present, current = _path_value(layer, table.path)
                if present:
                    if not isinstance(current, dict):
                        raise ValueError(
                            f"CLI schema path {'.'.join(table.path)} collides with a scalar"
                        )
                    current.update(values)
                else:
                    _set_path(layer, table.path, values)
        return layer

    def normalize_config(self, config: Mapping[str, object]) -> dict[str, object]:
        """Apply registry-declared semantic normalization before typed construction."""
        normalized = copy.deepcopy(dict(config))
        from .validate import ConfigError

        for path, field in self.field_items():
            present, value = _path_value(normalized, path)
            if not present:
                continue
            if field.strip:
                if not isinstance(value, str):
                    raise ConfigError(f"{'.'.join(path)}: expected string")
                value = value.strip()
                _set_path(normalized, path, value)
            if field.stringify_large_int and isinstance(value, str):
                if _DECIMAL_INTEGER.fullmatch(value) is None:
                    raise ConfigError(
                        f"{'.'.join(path)}: quoted integers must contain decimal digits only"
                    )
                _set_path(normalized, path, int(value))
        return normalized

    def dump_config(self, config: Mapping[str, object]) -> str:
        """Serialize config with this model's registry-owned presentation policy."""
        from .dump import dump_config

        multiline = frozenset(path for path, field in self.field_items() if field.multiline)
        stringify_large_int = frozenset(
            path for path, field in self.field_items() if field.stringify_large_int
        )
        return dump_config(
            config,
            multiline_paths=multiline,
            stringify_large_int_paths=stringify_large_int,
        )

    def top_level_tables(self) -> frozenset[str]:
        """Return accepted top-level table names."""
        return frozenset(table.path[0] for table in self.tables)

    def top_level_scalars(self) -> frozenset[str]:
        """Return accepted top-level scalar names."""
        return frozenset(field.name for field in self.root_fields)

    def default_config(self) -> dict[str, object]:
        """Return the pure built-in defaults without environment resolution."""
        result: dict[str, object] = {}
        for field in self.root_fields:
            value = copy.deepcopy(field.default)
            if value is not None:
                result[field.name] = value
        for table in self.tables:
            if not table.default_present:
                continue
            values = table.default_mapping()
            if values:
                _set_path(result, table.path, values)
        return result

    def expand_layer_implications(
        self,
        layer: Mapping[str, object],
    ) -> dict[str, object]:
        """Expand registered shorthands inside one precedence layer."""
        expanded = copy.deepcopy(dict(layer))
        for implication in self.implications:
            if not implication.expand_in_layers:
                continue
            present, trigger_value = _path_value(expanded, implication.trigger)
            if not present or trigger_value != implication.trigger_value:
                continue
            for path, implied_value in implication.implied:
                value_present, _value = _path_value(expanded, path)
                if not value_present:
                    _set_path(expanded, path, implied_value)
        return expanded

    def environment_variables(self) -> dict[ConfigPath, tuple[str, ...]]:
        """Return every env-backed field in this model contribution."""
        result: dict[ConfigPath, tuple[str, ...]] = {}
        for table in self.tables:
            for field in table.fields:
                variables = table.environment_variables(field)
                if variables:
                    result[(*table.path, field.name)] = variables
        return result

    def environment_config(self) -> dict[str, object]:
        """Return the currently resolved environment layer for this model."""
        result: dict[str, object] = {}
        for table in self.tables:
            values = table.environment_overrides()
            if values:
                _set_path(result, table.path, values)
        return result

    def non_default_config(self, effective: Mapping[str, object]) -> dict[str, object]:
        """Remove built-ins while preserving env masks and implication opt-outs."""
        sparse = _without_defaults(effective, self.default_config())
        model_present, model_value = _path_value(effective, ("model",))
        _set_path(sparse, ("model",), model_value if model_present else self.model)

        environment = self.environment_config()
        for path in self.environment_variables():
            environment_present, environment_value = _path_value(environment, path)
            effective_present, effective_value = _path_value(effective, path)
            if environment_present and effective_present and effective_value != environment_value:
                _set_path(sparse, path, effective_value)

        for implication in self.implications:
            present, trigger_value = _path_value(effective, implication.trigger)
            if not present or trigger_value != implication.trigger_value:
                continue
            _set_path(sparse, implication.trigger, trigger_value)
            for path, implied_value in implication.implied:
                value_present, effective_value = _path_value(effective, path)
                if not value_present:
                    continue
                if effective_value == implied_value:
                    _delete_path(sparse, path)
                else:
                    _set_path(sparse, path, effective_value)
        return sparse


class ConfigRegistry:
    """Global collection of validated model configuration contributions."""

    def __init__(self) -> None:
        self._models: dict[str, ModelConfigSpec] = {}
        self._frozen = False

    def register(self, contribution: ModelConfigSpec) -> None:
        """Register one model exactly once."""
        if self._frozen:
            raise RuntimeError("config registry is frozen")
        if contribution.model in self._models:
            raise ValueError(f"duplicate config schema for model {contribution.model!r}")
        self._models[contribution.model] = contribution

    def freeze(self) -> None:
        """Prevent mutation after every selectable model has contributed."""
        self._frozen = True

    def model(self, name: str) -> ModelConfigSpec:
        """Return one registered model schema."""
        return self._models[name]

    def models(self) -> tuple[str, ...]:
        """Return registered model names in stable order."""
        return tuple(sorted(self._models))

    def environment_variables(self) -> tuple[str, ...]:
        """Return the global union of environment variables from model records."""
        return tuple(
            sorted(
                {
                    variable
                    for model in self._models.values()
                    for variables in model.environment_variables().values()
                    for variable in variables
                }
            )
        )


__all__ = [
    "ConfigFieldSpec",
    "ConfigGroup",
    "ConfigImplication",
    "ConfigPath",
    "ConfigRegistry",
    "ConfigTableSpec",
    "ModelConfigSpec",
    "as_sentence",
]
