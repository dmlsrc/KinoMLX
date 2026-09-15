"""Canonical configuration-field schemas contributed by each model."""

import argparse
import copy
import dataclasses
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from enum import StrEnum
from functools import cached_property
from types import MappingProxyType
from typing import Literal, cast, get_args, get_origin, get_type_hints

from kinomlx.api_boundaries import APIBoundary, ConfigurationOwner
from kinomlx.config_bindings import CliBinding
from kinomlx.config_constraints import (
    LocalConstraint,
    canonicalize_field_value,
    field_constraints,
    validate_choice_binding,
    validate_field_constraints,
)
from kinomlx.config_record_cli import RecordArgument
from kinomlx.config_records import record_intent, tracks_presence
from kinomlx.config_rules import record_rules
from kinomlx.config_surfaces import SURFACES, SurfacePolicy
from kinomlx.config_values import (
    ConfigError,
    coerce_config_value,
    coerce_dataclass_fields,
    unknown_field_error,
)
from kinomlx.settings import (
    EnvironmentSettings,
    EnvironmentValue,
    environment_names,
    environment_value_is_unset,
    parse_environment_value,
    read_environment,
)

from .dump import dump_config as render_config
from .overrides import SetOverrides
from .paths import TargetPath, field_path, format_target_path
from .provenance import FieldOrigin, PreparedConfig, SourceLocation
from .stations import OutputPolicy, StationPolicy
from .transforms import DELETE_FIELD, TRANSFORM_PHASES, LayerPhase, LayerTransform

ConfigPath = tuple[str, ...]


class ConfigGroup(StrEnum):
    """Semantic roles consumed by generic registry-backed behavior."""

    SAVE_ALL_CANDIDATE = "save_all_candidate"
    SOURCE_SELECTOR = "source_selector"


_NO_VALUE = object()
_SHARED_ENVIRONMENT_FALLBACKS = {"output_dir": ("KINO_OUTPUT_DIR",)}
_ENVIRONMENT_VARIABLE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_CONFIG_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")


def as_sentence(text: str) -> str:
    """Normalize one registry-owned prose fragment as a sentence."""
    value = text.strip()
    return value if value.endswith((".", "!", "?")) else value + "."


def station_field_changed(current: object, parent: object, names: ConfigPath) -> bool:
    """Compare one owned field across two station records, through any nested records.

    Every station comparison reads a field this way, so nested records and
    collections are compared identically wherever a policy consults a value.
    An absent member on either side reads as one shared sentinel rather than as
    a difference against ``None``.
    """
    for name in names[:-1]:
        current = None if current is None else getattr(current, name)
        parent = None if parent is None else getattr(parent, name)
    return getattr(current, names[-1], _NO_VALUE) != getattr(parent, names[-1], _NO_VALUE)


def _record_fields(record: type[object]) -> dict[str, dataclasses.Field[object]]:
    if not isinstance(record, type) or not dataclasses.is_dataclass(record):
        name = getattr(record, "__name__", type(record).__name__)
        raise TypeError(f"configuration record {name} is not a dataclass type")
    reflected = dataclasses.fields(record)
    return {item.name: item for item in reflected}


_RECORD_ANNOTATIONS: dict[type[object], dict[str, object]] = {}


def _record_annotations(record: type[object]) -> dict[str, object]:
    if record not in _RECORD_ANNOTATIONS:
        _RECORD_ANNOTATIONS[record] = get_type_hints(record, include_extras=True)
    return _RECORD_ANNOTATIONS[record]


def _field_default(item: dataclasses.Field[object]) -> object:
    if item.default is not MISSING:
        return copy.deepcopy(item.default)
    if item.default_factory is not MISSING:
        return item.default_factory()
    return _NO_VALUE


def _field_default_kind(item: dataclasses.Field[object]) -> Literal["required", "value", "factory"]:
    if item.default is not MISSING:
        return "value"
    return "required" if item.default_factory is MISSING else "factory"


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


def _delete_path(target: dict[str, object], path: ConfigPath, *, prune_empty: bool = True) -> None:
    if len(path) == 1:
        target.pop(path[0], None)
        return
    child = target.get(path[0])
    if not isinstance(child, dict):
        return
    _delete_path(child, path[1:], prune_empty=prune_empty)
    if prune_empty and not child:
        target.pop(path[0], None)


@dataclass(frozen=True)
class ConfigFieldSpec:
    """Documentation, example, and CLI placement for one TOML field."""

    name: str
    purpose: str
    valid: str
    example: object
    cli_dest: str
    persistence: Literal["intent", "source"]
    surfaces: SurfacePolicy
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
    # A merged null means "the higher layer selected the alternative spelling",
    # so the key is removed instead of reaching typed construction as None.
    drop_when_none: bool = False
    groups: frozenset[str] = frozenset()
    cli: CliBinding | None = None
    layer_transforms: tuple[LayerTransform, ...] = ()
    input_annotation: object = None
    station_policies: tuple[StationPolicy, ...] = ()
    output_policy: OutputPolicy | None = None
    item_schema: ConfigRecordSpec | None = None
    cli_item: str | None = None
    environment_key: str | None = None

    def __post_init__(self) -> None:
        if self.persistence not in {"intent", "source"}:
            raise ValueError(
                f"{self.name}: persistence must be explicitly classified as intent or source"
            )
        if _CONFIG_NAME.fullmatch(self.name) is None:
            raise ValueError(f"invalid configuration field name {self.name!r}")
        if _CONFIG_NAME.fullmatch(self.cli_dest) is None:
            raise ValueError(f"{self.name}: invalid CLI destination {self.cli_dest!r}")
        if self.environment_key is None:
            object.__setattr__(self, "environment_key", self.name)
        elif _CONFIG_NAME.fullmatch(self.environment_key) is None:
            raise ValueError(f"{self.name}: invalid environment key {self.environment_key!r}")
        if not self.purpose.strip():
            raise ValueError(f"{self.name}: purpose cannot be empty")
        if self.negative_purpose is not None and not self.negative_purpose.strip():
            raise ValueError(f"{self.name}: negative purpose cannot be empty")
        if not self.valid.strip() and not self.choices:
            raise ValueError(f"{self.name}: valid values cannot be empty")
        if self.cli_item is not None and (
            self.item_schema is None
            or self.cli is None
            or _CONFIG_NAME.fullmatch(self.cli_item) is None
        ):
            raise ValueError(f"{self.name}: CLI items require a record schema, binding and member")
        if not isinstance(self.surfaces, SurfacePolicy):
            raise TypeError(f"{self.name}: explicit surface policy is required")
        if not self.surfaces.supports("toml") or not self.surfaces.supports("override"):
            raise ValueError(f"{self.name}: configuration fields require structured input surfaces")
        if not self.virtual and not self.surfaces.supports("api"):
            raise ValueError(f"{self.name}: typed record members expose their API surface")
        if self.virtual and self.surfaces.supports("api"):
            raise ValueError(f"{self.name}: virtual source inputs have no typed API member")
        if self.virtual and self.item_schema is not None:
            raise ValueError(f"{self.name}: record collections require a typed owner")
        if self.virtual and self.default is _NO_VALUE:
            raise ValueError(f"{self.name}: virtual fields need an explicit default")
        if self.input_annotation is not None and not self.virtual:
            raise ValueError(f"{self.name}: record-owned input annotations cannot be overridden")
        if self.virtual and self.default is None and self.input_annotation is None:
            raise ValueError(f"{self.name}: nullable virtual fields need an input annotation")
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
            any(type(item) is not int for item in self.example)
            if isinstance(self.example, (tuple, list))
            else type(self.example) is not int
        ):
            raise ValueError(f"{self.name}: large-integer stringification requires an int example")
        invalid_groups = sorted(
            str(group)
            for group in self.groups
            if not isinstance(group, str) or _CONFIG_NAME.fullmatch(group) is None
        )
        if invalid_groups:
            raise ValueError(f"{self.name}: invalid semantic groups: {', '.join(invalid_groups)}")

    def intent_contract(self) -> tuple[object, ...]:
        """Meaning shared by every contribution of the same typed field identity.

        Binding spellings, presentation and explicit surface/output/station policies
        may specialize per model. Typed values, constraints and input transforms
        retain one meaning for a reused record member.
        """
        return (
            self.persistence,
            self.valid,
            self.choices,
            self.strip,
            self.drop_when_none,
            self.stringify_large_int,
            self.layer_transforms,
            None if self.item_schema is None else self.item_schema.record,
            ()
            if self.item_schema is None
            else tuple(
                (member.name, member.intent_contract()) for member in self.item_schema.fields
            ),
        )

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


def _contains_record(annotation: object) -> bool:
    return bool(isinstance(annotation, type) and dataclasses.is_dataclass(annotation)) or any(
        _contains_record(argument) for argument in get_args(annotation)
    )


def _validate_item_schema(field: ConfigFieldSpec, annotation: object) -> None:
    if field.item_schema is not None:
        field.item_schema.validate_collection_annotation(annotation)
        for member in field.item_schema.fields:
            for surface in SURFACES:
                if field.surfaces.supports(surface) and not member.surfaces.supports(surface):
                    raise ValueError(
                        f"{field.name}.{member.name}: member excludes the collection's {surface} surface"
                    )
        if field.cli_item is None and any(
            member.cli is not None for member in field.item_schema.fields
        ):
            raise ValueError(f"{field.name}: member CLI bindings require an item token")
    elif _contains_record(annotation):
        raise ValueError(f"{field.name}: record collection is missing its item schema")


@dataclass(frozen=True)
class ConfigRecordSpec:
    """Owned members of a homogeneous record collection, independent of its surfaces."""

    record: type[object]
    fields: tuple[ConfigFieldSpec, ...]

    def __post_init__(self) -> None:
        reflected = _record_fields(self.record)
        record_rules(self.record)
        if not tracks_presence(self.record):
            raise ValueError("collection records require constructor presence tracking")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("record schema repeats a member")
        actual = {name for name, field in reflected.items() if field.init}
        if set(names) != actual:
            raise ValueError(f"{self.record.__name__}: record member coverage drift")
        for field in self.fields:
            if field.virtual or field.persistence != "intent":
                raise ValueError(f"{field.name}: record members must have typed intent owners")
            if (
                field.cli_item is not None
                or field.layer_transforms
                or field.station_policies
                or field.groups
                or field.strip
                or field.multiline
                or field.stringify_large_int
                or field.drop_when_none
            ):
                raise ValueError(f"{field.name}: record member policy is not implemented yet")
            annotation = self.field_annotation(field)
            validate_choice_binding(reflected[field.name], field.choices)
            _validate_item_schema(field, annotation)
            default = _field_default(reflected[field.name])
            if default is not _NO_VALUE:
                coerce_config_value(default, annotation, f"{self.record.__name__}.{field.name}")
                validate_field_constraints(
                    reflected[field.name],
                    annotation,
                    default,
                    f"{self.record.__name__}.{field.name}",
                    supplied=False,
                )
                if field.choices and default is not None and default not in field.choices:
                    raise ValueError(f"{field.name}: default is not a registered choice")

    def field_annotation(self, field: ConfigFieldSpec) -> object:
        return _record_annotations(self.record)[field.name]

    def local_constraints(self, field: ConfigFieldSpec) -> tuple[LocalConstraint, ...]:
        return () if field.virtual else field_constraints(_record_fields(self.record)[field.name])

    def default_kind(self, field: ConfigFieldSpec) -> Literal["required", "value", "factory"]:
        return _field_default_kind(_record_fields(self.record)[field.name])

    def default_value(self, field: ConfigFieldSpec) -> object:
        value = _field_default(_record_fields(self.record)[field.name])
        return None if value is _NO_VALUE else value

    def validate_collection_annotation(self, annotation: object) -> None:
        arguments = get_args(annotation)
        if not (
            (get_origin(annotation) is list and arguments == (self.record,))
            or (get_origin(annotation) is tuple and arguments == (self.record, Ellipsis))
        ):
            raise ValueError(f"collection annotation must contain {self.record.__name__} records")

    def coerce_members(self, value: object, location: str) -> dict[str, object]:
        """Admit supplied members without filling defaults or constructing a partial item."""
        members = coerce_dataclass_fields(self.record, value, table=location)
        for field in self.fields:
            if field.name not in members:
                continue
            value = members[field.name]
            member_location = f"{location}.{field.name}"
            if field.choices and value is not None and value not in field.choices:
                raise ConfigError(
                    f"{member_location}: expected one of {field.choices}, got {value!r}"
                )
            if field.item_schema is not None:
                field.item_schema.validate_members(value, location=member_location)
        return members

    def validate_members(self, values: object, *, location: str) -> None:
        """Apply member declarations after shared typed collection admission."""
        if not isinstance(values, (list, tuple)):
            raise ConfigError(f"{location}: expected record collection")
        for index, item in enumerate(values):
            self.coerce_members(item, f"{location}[{index}]")


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
    cli_generated: bool = False
    cli_group: str | None = None
    stations: tuple[str, ...] = ()
    station_record: type[object] | None = None

    def __post_init__(self) -> None:
        record_name = getattr(self.record, "__name__", type(self.record).__name__)
        location = f"{record_name} [{'.'.join(self.path)}]"
        if not self.path or any(_CONFIG_NAME.fullmatch(part) is None for part in self.path):
            raise ValueError(
                f"{record_name}: invalid configuration table path {'.'.join(self.path)!r}"
            )
        reflected = _record_fields(self.record)
        record_rules(self.record)
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
        if len(self.stations) != len(set(self.stations)) or any(not name for name in self.stations):
            raise ValueError(f"{location}: stations must be unique nonempty names")
        if bool(self.stations) != (self.station_record is not None):
            raise ValueError(f"{location}: station policies require a comparison record")
        if self.station_record is not None:
            projected = {
                name for name, item in _record_fields(self.station_record).items() if item.init
            }
            if projected != (
                {field.name for field in self.fields if not field.virtual} | set(self.nested)
            ):
                raise ValueError(f"{location}: station record field coverage drift")
        for field in self.fields:
            if field.output_policy is not None and (
                field.virtual or not _is_boolean_annotation(self.field_annotation(field))
            ):
                raise ValueError(f"{location}.{field.name}: output policy requires a boolean field")
            policies = {policy.station: policy for policy in field.station_policies}
            if len(policies) != len(field.station_policies):
                raise ValueError(f"{location}.{field.name}: duplicate station policy")
            expected = set() if field.virtual else set(self.stations)
            if set(policies) != expected:
                raise ValueError(f"{location}.{field.name}: station policy coverage drift")
            for policy in policies.values():
                attribute = policy.attribute or field.name
                if self.station_record is not None and not (
                    attribute in _record_fields(self.station_record)
                    or hasattr(self.station_record, attribute)
                ):
                    raise ValueError(
                        f"{location}.{field.name}: unknown station projection {attribute}"
                    )
        if issubclass(self.record, EnvironmentSettings):
            for field in self.fields:
                if not field.surfaces.supports("environment"):
                    raise ValueError(
                        f"{location}.{field.name}: Settings members retain their owned ENV surface"
                    )
            for name in actual:
                self.record.environment_sources_for_field(name)
        if self.cli_generated and issubclass(self.record, EnvironmentSettings):
            group = self.cli_group
            if not group:
                raise ValueError(f"{location}: Settings CLI tables require a group")
            object.__setattr__(
                self,
                "fields",
                tuple(
                    field
                    if field.cli is not None or not field.surfaces.supports("cli")
                    else dataclasses.replace(
                        field, cli=self.record.cli_binding_for_field(field.name, title=group)
                    )
                    for field in self.fields
                ),
            )
        for field in self.fields:
            _validate_item_schema(field, self.field_annotation(field))
            if not field.virtual:
                validate_choice_binding(reflected[field.name], field.choices)
            if field.surfaces.supports("cli") and field.cli is None:
                raise ValueError(
                    f"{location}.{field.name}: missing CLI binding or explicit exclusion"
                )
            if not field.surfaces.supports("cli") and field.cli is not None:
                raise ValueError(f"{location}.{field.name}: excluded CLI surface has a binding")
            if self.cli_generated and field.surfaces.supports("cli"):
                if field.cli is None:
                    raise ValueError(f"{location}.{field.name}: missing CLI binding")
                if field.cli_item is None:
                    field.cli.value_parser(self.field_annotation(field))
                else:
                    self.record_cli_bindings(field)
            default = self._declared_default(field)
            if default is _NO_VALUE:
                if self.default_present:
                    raise ValueError(
                        f"{location}.{field.name}: a default-present "
                        "config table cannot contain a required field"
                    )
                continue
            if not field.virtual:
                validate_field_constraints(
                    reflected[field.name],
                    self.field_annotation(field),
                    default,
                    f"{location}.{field.name}",
                    supplied=False,
                )
            if field.choices and default is not None and default not in field.choices:
                raise ValueError(
                    f"{location}.{field.name}: default {default!r} is not a registered choice"
                )
            if field.default_text is not None:
                environment_backed = issubclass(self.record, EnvironmentSettings) and bool(
                    self.record.environment_sources_for_field(field.name)
                )
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

    def record_cli_bindings(
        self, field: ConfigFieldSpec
    ) -> tuple[tuple[ConfigFieldSpec, CliBinding, bool], ...]:
        schema = field.item_schema
        if schema is None or field.cli is None or field.cli_item is None:
            raise ValueError(f"{field.name}: missing record CLI declaration")
        primary = next((member for member in schema.fields if member.name == field.cli_item), None)
        if primary is None:
            raise ValueError(f"{field.name}: unknown primary item member {field.cli_item!r}")
        if primary.cli is not None:
            raise ValueError(f"{field.name}: primary member is already assigned by the item token")
        bindings = (
            (primary, field.cli, True),
            *tuple(
                (member, member.cli, False) for member in schema.fields if member.cli is not None
            ),
        )
        flags: set[str] = set()
        for member, binding, starts in bindings:
            if binding.action != "store" or binding.nargs is not None:
                raise ValueError(f"{member.name}: record CLI bindings require one value token")
            if binding.group != field.cli.group:
                raise ValueError(f"{member.name}: record CLI bindings must share their item group")
            binding.value_parser(schema.field_annotation(member))
            options = binding.option_strings(field.cli_dest if starts else member.cli_dest)
            if flags.intersection(options):
                raise ValueError(f"{field.name}: duplicate record CLI flags")
            flags.update(options)
        return bindings

    def add_cli_arguments(
        self,
        field: ConfigFieldSpec,
        parser: argparse._ActionsContainer,
        *,
        help: str | None,
        negative_help: str | None = None,
    ) -> tuple[argparse.Action, ...]:
        binding = field.cli
        if binding is None:
            raise ValueError(f"{field.name}: missing CLI binding")
        if field.cli_item is None:
            return binding.add_arguments(
                parser,
                destination=field.cli_dest,
                annotation=self.field_annotation(field),
                choices=field.choices,
                help=help,
                negative_help=negative_help,
            )
        schema = field.item_schema
        assert schema is not None
        actions = []
        for member, member_binding, starts in self.record_cli_bindings(field):
            value_parser = member_binding.value_parser(schema.field_annotation(member))
            assert value_parser is not None
            actions.append(
                parser.add_argument(
                    *member_binding.option_strings(field.cli_dest if starts else member.cli_dest),
                    action=RecordArgument,
                    dest=field.cli_dest,
                    member=member.name,
                    starts_record=starts,
                    coerce=schema.coerce_members,
                    location=".".join((*self.path, field.name)),
                    item_flag=binding.option_strings(field.cli_dest)[0],
                    type=value_parser,
                    choices=None if starts else member.choices or None,
                    metavar=member_binding.metavar or member.name.upper(),
                    default=None,
                    help=help if starts else member.help_text(),
                )
            )
        return tuple(actions)

    def validate_cli_actions(self, field: ConfigFieldSpec, actions: list[argparse.Action]) -> None:
        binding = field.cli
        if binding is None:
            raise ValueError(f"{field.name}: missing CLI binding")
        if field.cli_item is None:
            binding.validate_actions(
                actions, annotation=self.field_annotation(field), destination=field.cli_dest
            )
            return
        expected = self.add_cli_arguments(
            field, argparse.ArgumentParser(add_help=False), help=field.help_text()
        )
        attributes = (
            "option_strings",
            "dest",
            "type",
            "nargs",
            "const",
            "default",
            "choices",
            "metavar",
            "help",
            "member",
            "starts_record",
            "coerce",
            "location",
            "item_flag",
        )
        if len(actions) != len(expected) or any(
            type(actual) is not type(wanted)
            or any(getattr(actual, name, _NO_VALUE) != getattr(wanted, name) for name in attributes)
            for actual, wanted in zip(actions, expected, strict=True)
        ):
            raise ValueError(f"CLI record binding drift for {field.cli_dest}")

    def locked_changes(
        self, current: object, parent: object, *, station: str
    ) -> tuple[ConfigPath, ...]:
        """Compare every registered locked semantic field at a supported station."""
        if station not in self.stations or self.station_record is None:
            raise ValueError(f"{'.'.join(self.path)}: unsupported station {station!r}")
        if (current is not None and not isinstance(current, self.station_record)) or (
            parent is not None and not isinstance(parent, self.station_record)
        ):
            raise TypeError("station comparison requires its registered typed records")
        changed = []
        for field in self.fields:
            for policy in field.station_policies:
                if policy.station == station and policy.mode == "locked":
                    attribute = policy.attribute or field.name
                    if station_field_changed(current, parent, (attribute,)):
                        changed.append((*self.path, field.name))
        return tuple(sorted(changed))

    def field_annotation(self, field: ConfigFieldSpec) -> object:
        """Read the executable type from the owned record, not a consumer table."""
        if field.virtual:
            return None
        return _record_annotations(self.record)[field.name]

    def _declared_default(self, field: ConfigFieldSpec) -> object:
        if field.virtual:
            return copy.deepcopy(field.default)
        reflected = _record_fields(self.record)
        record_rules(self.record)
        return _field_default(reflected[field.name])

    def local_constraints(self, field: ConfigFieldSpec) -> tuple[LocalConstraint, ...]:
        return () if field.virtual else field_constraints(_record_fields(self.record)[field.name])

    def default_kind(self, field: ConfigFieldSpec) -> Literal["required", "value", "factory"]:
        """Describe the owned default without evaluating a default factory."""
        if field.virtual:
            return "required" if field.default is _NO_VALUE else "value"
        return _field_default_kind(_record_fields(self.record)[field.name])

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

    def environment_variables(self, field: ConfigFieldSpec, *, model: str) -> tuple[str, ...]:
        """Use owned sources or the stable public key, independently of parser storage."""
        if not field.surfaces.supports("environment"):
            return ()
        if (
            self.path == ("settings",)
            and not field.virtual
            and issubclass(self.record, EnvironmentSettings)
        ):
            entries = self.record.environment_sources_for_field(field.name)
            variables = tuple(
                match.group(1)
                for entry in entries
                for match in _ENVIRONMENT_VARIABLE.finditer(entry)
            )
            if variables:
                return variables
        prefix = "KINO_" if self.path == ("settings",) else f"KINO_{model.upper()}_"
        assert field.environment_key is not None
        return (
            prefix + field.environment_key.upper(),
            *_SHARED_ENVIRONMENT_FALLBACKS.get(field.environment_key, ()),
        )

    def environment_overrides(self, *, model: str) -> dict[str, object]:
        """Resolve registered fields through the same typed values as TOML and --set."""
        return {name: entry.value for name, entry in self.environment_values(model=model).items()}

    def environment_values(self, *, model: str) -> dict[str, EnvironmentValue]:
        """Retain selected ENV bindings without resolving a second time for provenance."""
        declared = tuple(
            field.name
            for field in self.fields
            if self.path == ("settings",)
            and field.surfaces.supports("environment")
            and not field.virtual
            and issubclass(self.record, EnvironmentSettings)
            and self.record.environment_sources_for_field(field.name)
        )
        result = (
            self.record.values_from_env_fields(*declared)
            if issubclass(self.record, EnvironmentSettings)
            else {}
        )
        for field in self.fields:
            if field.name in declared:
                continue
            annotation = (
                type(field.example) if field.virtual else get_type_hints(self.record)[field.name]
            )
            for variable in self.environment_variables(field, model=model):
                raw = read_environment(variable)
                if not environment_value_is_unset(annotation, raw):
                    break
            else:
                continue
            assert raw is not None
            location = f"environment {variable} [{'.'.join(self.path)}]"
            try:
                value = parse_environment_value(annotation, raw)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{location}.{field.name}: {exc}") from exc
            if not field.virtual:
                try:
                    coerce_dataclass_fields(self.record, {field.name: value}, table=location)
                except ConfigError:
                    # Field coercion already reports this variable and field.
                    raise
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"{location}.{field.name}: {exc}") from exc
            result[field.name] = EnvironmentValue(value, variable, True)
        return result


@dataclass(frozen=True)
class ConfigImplication:
    """Fields implied by one higher-level non-default selector."""

    trigger: ConfigPath
    trigger_value: object
    implied: tuple[tuple[ConfigPath, object], ...]
    expand_in_layers: bool = False
    excluded: tuple[ConfigPath, ...] = ()
    coverage_group: str | None = None
    # Fields that follow the trigger at runtime while they stay unset; they are
    # not expanded into layers, so an explicit value anywhere keeps its meaning.
    inherited: tuple[ConfigPath, ...] = ()


@dataclass(frozen=True)
class ConfigTarget:
    """A concrete address resolved to its existing typed declaration owner."""

    path: TargetPath
    declaration_path: ConfigPath
    kind: Literal["group", "field", "item"]
    owner: ConfigTableSpec | ConfigRecordSpec | None
    field: ConfigFieldSpec | None = None


@dataclass(frozen=True)
class ModelConfigSpec:
    """One complete model contribution to the global configuration registry."""

    model: str
    title: str
    introduction: tuple[str, ...]
    root_fields: tuple[ConfigFieldSpec, ...]
    tables: tuple[ConfigTableSpec, ...]
    # Every contribution states its restart applicability. An empty station set
    # means the model has no saved restart operation and requires a reason.
    restart_stations: tuple[str, ...]
    implications: tuple[ConfigImplication, ...] = ()
    semantic_groups: frozenset[str] = frozenset()
    layer_transforms: tuple[LayerTransform, ...] = ()
    api_boundaries: tuple[APIBoundary, ...] = ()
    no_restart_reason: str | None = None

    def __post_init__(self) -> None:
        if _MODEL_NAME.fullmatch(self.model) is None:
            raise ValueError(f"invalid model config name {self.model!r}")
        if not self.title.strip() or not self.introduction:
            raise ValueError(f"{self.model}: config title and introduction are required")
        invalid_groups = sorted(
            str(group)
            for group in self.semantic_groups
            if not isinstance(group, str) or _CONFIG_NAME.fullmatch(group) is None
        )
        if invalid_groups:
            raise ValueError(
                f"{self.model}: invalid declared semantic groups: {', '.join(invalid_groups)}"
            )
        stations = {"generate", *(station for table in self.tables for station in table.stations)}
        for path, declaration in self.field_items():
            if declaration.output_policy is None:
                continue
            produced = declaration.output_policy.produced_at
            unknown_stations = produced - stations
            if unknown_stations:
                raise ValueError(
                    f"{'.'.join(path)}: unknown output stations {sorted(unknown_stations)}"
                )
            # Availability is the authority: a station that produces the group
            # may be asked for it, and one that does not cannot. A hand-written
            # mode beside the same field's OutputPolicy is a second answer to
            # the same question, and the two have drifted before.
            for policy in declaration.station_policies:
                expected = "mutable" if policy.station in produced else "irrelevant"
                if policy.mode != expected:
                    raise ValueError(
                        f"{'.'.join(path)}: station {policy.station} declares {policy.mode!r} "
                        f"but this artifact is "
                        f"{'produced' if policy.station in produced else 'not produced'} there"
                    )
        for declaration in self.root_fields:
            if declaration.surfaces.supports("cli") and declaration.cli is None:
                raise ValueError(f"{declaration.name}: missing root CLI binding")
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
        operations: set[object] = set()
        for boundary in self.api_boundaries:
            boundary.validate_tracked_projections()
            if boundary.model != self.model:
                raise ValueError(f"{self.model}: API boundary belongs to {boundary.model}")
            if boundary.operation in operations:
                raise ValueError(f"{self.model}: duplicate API operation ownership")
            operations.add(boundary.operation)
            for parameter in boundary.parameters:
                owner = parameter.owner
                if isinstance(owner, ConfigurationOwner):
                    table = by_path.get(owner.table)
                    if table is None or table.record is not owner.record:
                        raise ValueError(
                            f"{self.model}.{parameter.name}: API configuration owner "
                            f"does not match registered table {'.'.join(owner.table)}"
                        )
            # A runtime port is published as nonserializable. A registered table
            # record is the opposite of that, so naming one under a runtime
            # declaration is a misclassification, not a port. An unbounded type
            # parameter names no record and stays a port.
            table_records = {table.record: table.path for table in self.tables}
            for name, annotation in boundary.runtime_annotations().items():
                table_path = table_records.get(cast(type[object], annotation))
                if table_path is not None:
                    raise ValueError(
                        f"{self.model}.{name}: API runtime parameter names the registered "
                        f"table record for {'.'.join(table_path)}"
                    )
        if len(self.restart_stations) != len(set(self.restart_stations)) or any(
            not name for name in self.restart_stations
        ):
            raise ValueError(f"{self.model}: restart stations must be unique nonempty names")
        if self.restart_stations:
            if self.no_restart_reason is not None:
                raise ValueError(f"{self.model}: restart-enabled model cannot claim no restart")
        elif not isinstance(self.no_restart_reason, str) or not self.no_restart_reason.strip():
            raise ValueError(f"{self.model}: no-restart model requires a reason")
        expected_stations = set(self.restart_stations)
        for table in self.tables:
            if set(table.stations) != expected_stations:
                raise ValueError(f"{self.model}.{'.'.join(table.path)}: station coverage drift")
            for declaration in table.fields:
                if not declaration.virtual and any(
                    not policy.reason.strip() for policy in declaration.station_policies
                ):
                    raise ValueError(
                        f"{self.model}.{'.'.join((*table.path, declaration.name))}: "
                        "station policy needs a reason"
                    )
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

        # Environment names are public spellings owned per field, so a new field
        # that happens to reuse a key must be caught before any surface reads it.
        environment: dict[str, ConfigPath] = {}
        for owned_path, variables in self.environment_variables().items():
            for variable in variables:
                claimed = environment.setdefault(variable, owned_path)
                if claimed != owned_path:
                    raise ValueError(
                        f"{self.model}: environment name {variable} is claimed by "
                        f"{'.'.join(claimed)} and {'.'.join(owned_path)}"
                    )

        # Shared field contributions bring their behavior with them. Model
        # authors must not separately remember to install a seed/output rule.
        contributed: dict[str, LayerTransform] = {}
        for transform in (
            *(rule for _path, field in self.field_items() for rule in field.layer_transforms),
            *self.layer_transforms,
        ):
            existing = contributed.get(transform.name)
            if existing is not None and existing != transform:
                raise ValueError(f"{self.model}: conflicting transform {transform.name}")
            contributed[transform.name] = transform
        object.__setattr__(self, "layer_transforms", tuple(contributed.values()))

        declared_paths = {path for path, _field in self.field_items()}
        seen_transforms: dict[str, LayerTransform] = {}
        for transform in self.layer_transforms:
            if _CONFIG_NAME.fullmatch(transform.name) is None:
                raise ValueError(f"{self.model}: invalid transform name {transform.name!r}")
            if transform.name in seen_transforms:
                raise ValueError(f"{self.model}: duplicate transform {transform.name}")
            if transform.phase not in TRANSFORM_PHASES:
                raise ValueError(f"{transform.name}: invalid layer phase")
            if transform.effects - {"read_files"}:
                raise ValueError(f"{transform.name}: unknown effects")
            if transform.effects and transform.phase != "source":
                raise ValueError(f"{transform.name}: filesystem effects require the source phase")
            if not transform.reads or not transform.writes:
                raise ValueError(f"{transform.name}: reads and writes must be declared")
            for group, paths in transform.group_paths:
                if self.paths_in_group(group) != paths:
                    raise ValueError(f"{transform.name}: compiled group {group!r} has drifted")
            unknown = (transform.reads | transform.writes) - declared_paths
            if unknown:
                raise ValueError(f"{transform.name}: unknown field paths {sorted(unknown)}")
            for dependency in transform.after:
                if dependency not in seen_transforms:
                    raise ValueError(f"{transform.name}: dependency {dependency!r} must precede it")
                predecessor = seen_transforms[dependency]
                if TRANSFORM_PHASES.index(predecessor.phase) > TRANSFORM_PHASES.index(
                    transform.phase
                ):
                    raise ValueError(f"{transform.name}: dependency crosses phases backward")
            for predecessor in seen_transforms.values():
                if (
                    predecessor.phase == transform.phase
                    and (
                        predecessor.writes & (transform.reads | transform.writes)
                        or predecessor.reads & transform.writes
                    )
                    and predecessor.name not in transform.after
                ):
                    raise ValueError(
                        f"{transform.name}: overlapping transform {predecessor.name!r} "
                        "requires an explicit dependency"
                    )
            seen_transforms[transform.name] = transform

        common_groups = frozenset(group.value for group in ConfigGroup)
        used_groups = frozenset(
            group
            for field in (
                *self.root_fields,
                *(item for table in self.tables for item in table.fields),
            )
            for group in field.groups
        )
        unknown_groups = sorted(used_groups - common_groups - self.semantic_groups)
        if unknown_groups:
            raise ValueError(f"{self.model}: unknown semantic groups: {', '.join(unknown_groups)}")
        unused_groups = sorted(self.semantic_groups - used_groups)
        if unused_groups:
            raise ValueError(
                f"{self.model}: unused declared semantic groups: {', '.join(unused_groups)}"
            )

        for table in self.tables:
            if table.path == ("model_settings",) and issubclass(table.record, EnvironmentSettings):
                for field in table.fields:
                    expected = table.environment_variables(field, model=self.model)[0]
                    if table.record.environment_sources_for_field(field.name) != (
                        "{{" + expected + "}}",
                    ):
                        raise ValueError(
                            f"{self.model}.{field.name}: settings/registry environment drift"
                        )

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
            inherited_paths: set[ConfigPath] = set()
            for path in implication.inherited:
                self.field(path)
                if path == implication.trigger:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(path)} cannot inherit itself"
                    )
                if path in inherited_paths:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} repeats "
                        f"inherited path {'.'.join(path)}"
                    )
                if path in implied_paths or path in excluded_paths:
                    raise ValueError(
                        f"{self.model}: implication {'.'.join(implication.trigger)} both "
                        f"declares and inherits {'.'.join(path)}"
                    )
                inherited_paths.add(path)
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

    def locked_changes(
        self, table_path: ConfigPath, current: object, parent: object, *, station: str
    ) -> tuple[ConfigPath, ...]:
        """Compare a registered typed table and all its nested station projections."""
        tables = {table.path: table for table in self.tables}
        if table_path not in tables:
            raise ValueError(f"unknown station table {'.'.join(table_path)}")
        root = tables[table_path]
        changed = list(root.locked_changes(current, parent, station=station))
        for table in self.tables:
            if len(table.path) <= len(table_path) or table.path[: len(table_path)] != table_path:
                continue
            left, right = current, parent
            for name in table.path[len(table_path) :]:
                left = None if left is None else getattr(left, name)
                right = None if right is None else getattr(right, name)
            changed.extend(table.locked_changes(left, right, station=station))
        return tuple(sorted(changed))

    def station_changes(
        self,
        current: Mapping[str, object],
        parent: Mapping[str, object],
        *,
        station: str,
        active_explicit: frozenset[TargetPath],
    ) -> tuple[ConfigPath, ...]:
        """Admit one typed restart invocation through every registered table."""
        if station not in self.restart_stations:
            raise ValueError(f"{self.model}: unsupported restart station {station!r}")
        roots = {table.path[0] for table in self.tables if len(table.path) == 1}
        for label, records in (("current", current), ("parent", parent)):
            if set(records) != roots:
                missing = sorted(roots - set(records))
                stale = sorted(set(records) - roots)
                raise ValueError(
                    f"{self.model}: {label} station records drift; missing={missing}, stale={stale}"
                )
            for table in self.tables:
                if len(table.path) != 1:
                    continue
                record_type = table.station_record
                if record_type is None or not isinstance(records[table.path[0]], record_type):
                    raise TypeError(
                        f"{self.model}.{table.path[0]}: {label} station record must be "
                        f"{table.record.__name__}"
                    )
        changed: set[ConfigPath] = set()
        for table in self.tables:
            if len(table.path) == 1:
                changed.update(
                    self.locked_changes(
                        table.path, current[table.path[0]], parent[table.path[0]], station=station
                    )
                )
        for path, declaration in self.field_items():
            if declaration.virtual:
                continue
            policy = next(
                (policy for policy in declaration.station_policies if policy.station == station),
                None,
            )
            if policy is None:
                raise ValueError(f"{self.model}.{'.'.join(path)}: missing station policy")
            if (
                policy.mode != "irrelevant"
                or policy.irrelevant_action != "reject"
                or not any(supplied[: len(path)] == path for supplied in active_explicit)
            ):
                continue
            # Re-typing the value the parent already recorded is not a change.
            # A root both sides share records nothing to compare against, since
            # it describes this invocation rather than completed parent work, so
            # supplying such a field there remains a change on presence alone.
            root = path[0]
            if current[root] is parent[root] or station_field_changed(
                current[root], parent[root], path[1:]
            ):
                changed.add(path)
        return tuple(sorted(changed))

    @cached_property
    def _field_items(self) -> tuple[tuple[ConfigPath, ConfigFieldSpec], ...]:
        """Build the field index once per contribution.

        A contribution is immutable after construction, so every reader shares
        one index instead of rebuilding it. ``dataclasses.replace`` produces a
        new instance with an empty cache, which is the intended behavior.
        """
        root = tuple(((field.name,), field) for field in self.root_fields)
        nested = tuple(
            ((*table.path, field.name), field) for table in self.tables for field in table.fields
        )
        return (*root, *nested)

    def field_items(self) -> tuple[tuple[ConfigPath, ConfigFieldSpec], ...]:
        """Return every contributed field with its exact TOML path."""
        return self._field_items

    @cached_property
    def _fields_by_path(self) -> Mapping[ConfigPath, ConfigFieldSpec]:
        """Address every contributed field by its exact path."""
        return MappingProxyType(dict(self._field_items))

    @cached_property
    def _tables_by_path(self) -> Mapping[ConfigPath, ConfigTableSpec]:
        """Address every contributed table by its exact path."""
        return MappingProxyType({table.path: table for table in self.tables})

    def paths_in_group(self, group: str) -> tuple[ConfigPath, ...]:
        """Return schema-ordered field paths carrying one semantic group."""
        return tuple(path for path, field in self.field_items() if group in field.groups)

    def cli_destinations_in_group(self, group: str) -> tuple[str, ...]:
        """Return schema-ordered argparse destinations carrying one semantic group."""
        return tuple(self.field(path).cli_dest for path in self.paths_in_group(group))

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
        if implication.inherited:
            inherited = "; ".join(
                as_sentence(self.field(path).purpose).removesuffix(".")
                for path in implication.inherited
            )
            text += f" Also enables while unset: {inherited}."
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
        return {field.cli_dest: field for field in fields if field.surfaces.supports("cli")}

    def cli_layer(self, options: object) -> dict[str, object]:
        """Build the model's TOML-shaped ordinary-CLI precedence layer."""
        layer: dict[str, object] = {}
        for field in self.root_fields:
            if not field.surfaces.supports("cli"):
                continue
            value = getattr(options, field.cli_dest, None)
            if value is not None:
                layer[field.name] = value
        for table in self.tables:
            values: dict[str, object] = {}
            for field in table.fields:
                if not field.surfaces.supports("cli"):
                    continue
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

    def _transform_layer(
        self,
        layer: dict[str, object],
        phase: LayerPhase,
        explicit: set[TargetPath] | None = None,
        origins: dict[TargetPath, FieldOrigin] | None = None,
        source: str | None = None,
    ) -> dict[str, object]:
        for transform in self.layer_transforms:
            if transform.phase != phase:
                continue
            inputs = {}
            for path in transform.reads:
                present, value = _path_value(layer, path)
                if present:
                    inputs[path] = copy.deepcopy(value)
            source_explicit = explicit is not None and bool(transform.reads & explicit)
            parents = (
                tuple(origins[path] for path in sorted(inputs) if path in origins)
                if origins is not None
                else ()
            )
            try:
                patch = transform.run(MappingProxyType(inputs))
            except ConfigError as exc:
                # A layer transform states a relationship between keys it read,
                # so the remedy is neutral and the layer that holds them is what
                # the reader needs. Several layers can carry the same keys.
                if source is None or str(exc).startswith(f"{source}: "):
                    raise
                raise ConfigError(f"{source}: {exc}") from exc
            unexpected = patch.keys() - transform.writes
            if unexpected:
                raise ValueError(f"{transform.name}: undeclared writes {sorted(unexpected)}")
            for path, value in patch.items():
                was_explicit = explicit is not None and path in explicit
                if origins is not None:
                    for target in tuple(origins):
                        if target[: len(path)] == path:
                            del origins[target]
                    if value is not DELETE_FIELD:
                        origins[path] = FieldOrigin(
                            path,
                            "normalized",
                            was_explicit or (phase == "source" and source_explicit),
                            rule=transform.name,
                            inputs=parents,
                        )
                if value is DELETE_FIELD:
                    _delete_path(layer, path, prune_empty=False)
                    if explicit is not None:
                        explicit.difference_update(
                            target for target in tuple(explicit) if target[: len(path)] == path
                        )
                else:
                    _set_path(layer, path, value)
                    if phase == "source" and source_explicit and explicit is not None:
                        explicit.add(path)
        return layer

    def validate_layer(self, config: Mapping[str, object], *, source: str = "config") -> None:
        """Admit supplied shapes and primitive values before precedence can hide them."""
        fields = self._fields_by_path
        tables = self._tables_by_path

        def visit(values: Mapping[str, object], prefix: ConfigPath) -> None:
            for name, value in values.items():
                path = (*prefix, name)
                location = f"[{'.'.join(prefix)}].{name}" if prefix else name
                if path in tables:
                    if not isinstance(value, Mapping):
                        raise ConfigError(f"{source}: [{'.'.join(path)}] must be a table")
                    if not all(isinstance(key, str) for key in value):
                        raise ConfigError(f"{source}: {location}: table keys must be strings")
                    visit(value, path)
                    continue
                field = fields.get(path)
                if field is None:
                    allowed = {
                        candidate[-1]
                        for candidate in (*fields, *tables)
                        if candidate[:-1] == prefix
                    }
                    raise unknown_field_error(name, allowed, f"{source}: {location}")
                if value is None and field.drop_when_none:
                    continue
                annotation = field.input_annotation
                if annotation is None:
                    if field.virtual:
                        if field.default is None or field.default is _NO_VALUE:
                            raise ValueError(f"{location}: virtual input needs an input annotation")
                        annotation = type(field.default)
                    else:
                        annotation = tables[prefix].field_annotation(field)
                admitted = value
                if field.stringify_large_int:

                    def integer(item: object) -> object:
                        if isinstance(item, str) and _DECIMAL_INTEGER.fullmatch(item):
                            return int(item)
                        return item

                    admitted = (
                        [integer(item) for item in value]
                        if isinstance(value, (list, tuple))
                        else integer(value)
                    )
                if not field.virtual:
                    admitted = canonicalize_field_value(
                        _record_fields(tables[prefix].record)[name], admitted
                    )
                coerced = coerce_config_value(admitted, annotation, f"{source}: {location}")
                if not field.virtual:
                    try:
                        validate_field_constraints(
                            _record_fields(tables[prefix].record)[name],
                            annotation,
                            coerced,
                            f"{source}: {location}",
                        )
                    except ValueError as exc:
                        raise ConfigError(str(exc)) from exc
                if field.item_schema is not None:
                    field.item_schema.validate_members(admitted, location=f"{source}: {location}")
                if (
                    path != ("model",)
                    and field.choices
                    and admitted is not None
                    and admitted not in field.choices
                ):
                    raise ConfigError(
                        f"{source}: {location}: expected one of {field.choices}, got {admitted!r}"
                    )

        visit(config, ())

    def resolve_target(self, path: TargetPath) -> ConfigTarget:
        """Resolve registered groups and record members without needing collection values."""
        location = format_target_path(path)
        if not path or any(
            isinstance(part, bool) or (isinstance(part, int) and part < 0) for part in path
        ):
            raise ConfigError(f"{location}: invalid configuration target")
        tables = self._tables_by_path
        fields = self._fields_by_path
        prefix: ConfigPath = ()
        for position, part in enumerate(path):
            if not isinstance(part, str):
                raise ConfigError(f"{location}: index requires a registered record collection")
            prefix = (*prefix, part)
            table = tables.get(prefix)
            if table is not None:
                if position == len(path) - 1:
                    return ConfigTarget(path, prefix, "group", table)
                continue
            field = fields.get(prefix)
            if field is None:
                allowed = {
                    candidate[-1]
                    for candidate in (*fields, *tables)
                    if candidate[:-1] == prefix[:-1]
                }
                raise unknown_field_error(part, allowed, location)
            owner: ConfigTableSpec | ConfigRecordSpec | None = tables.get(prefix[:-1])
            remaining = path[position + 1 :]
            while remaining:
                index = remaining[0]
                if not isinstance(index, int) or field.item_schema is None:
                    raise ConfigError(
                        f"{location}: member traversal requires a registered record collection index"
                    )
                owner = field.item_schema
                remaining = remaining[1:]
                if not remaining:
                    return ConfigTarget(path, prefix, "item", owner)
                member = remaining[0]
                if not isinstance(member, str):
                    raise ConfigError(f"{location}: expected a record member after the index")
                members = {item.name: item for item in owner.fields}
                if member not in members:
                    raise unknown_field_error(member, set(members), location)
                field = members[member]
                prefix = (*prefix, member)
                remaining = remaining[1:]
            return ConfigTarget(path, prefix, "field", owner, field)
        raise ConfigError(f"{location}: unknown configuration target")

    def target_paths(self, config: Mapping[str, object]) -> frozenset[TargetPath]:
        """Project supplied member addresses and collection/item replacement identities."""
        paths: set[TargetPath] = set()

        def visit(path: TargetPath, value: object) -> None:
            target = self.resolve_target(path)
            if target.kind == "field":
                paths.add(path)
                if target.field is not None and target.field.item_schema is not None:
                    if not isinstance(value, (list, tuple)):
                        raise ConfigError(f"{format_target_path(path)}: expected record collection")
                    for index, item in enumerate(value):
                        visit((*path, index), item)
                return
            if target.kind == "item":
                paths.add(path)
                if dataclasses.is_dataclass(value) and not isinstance(value, type):
                    value = record_intent(value)
            if not isinstance(value, Mapping):
                raise ConfigError(f"{format_target_path(path)}: expected supplied members")
            if not value:
                paths.add(path)
            for name, member in cast(Mapping[str, object], value).items():
                visit((*path, name), member)

        for name, value in config.items():
            visit((name,), value)
        return frozenset(paths)

    def validate_assignment(
        self, path: TargetPath, value: object, *, source: str = "config"
    ) -> None:
        """Admit one addressed value independently of later precedence or item bounds."""
        if not any(isinstance(part, int) for part in path):
            layer: dict[str, object] = {}
            _set_path(layer, field_path(path), value)
            self.validate_layer(layer, source=source)
            return
        target = self.resolve_target(path)
        if not isinstance(target.owner, ConfigRecordSpec):
            raise ValueError("indexed target has no record owner")
        location = f"{source}: {format_target_path(path)}"
        if target.kind == "item":
            members = target.owner.coerce_members(value, location)
            missing = [
                field.name
                for field in target.owner.fields
                if target.owner.default_kind(field) == "required" and field.name not in members
            ]
            if missing:
                raise ConfigError(f"{location}: missing required members: {', '.join(missing)}")
        elif target.field is not None:
            target.owner.coerce_members(
                {target.field.name: value}, f"{source}: {format_target_path(path[:-1])}"
            )
        else:
            raise ValueError("indexed target has no member declaration")

    def parse_overrides(self, arguments: list[str] | None) -> SetOverrides:
        """Decode and admit every assignment, including later-replaced assignments."""
        overrides = SetOverrides.parse(arguments)
        for assignment in overrides.assignments:
            self.validate_assignment(
                assignment.path, assignment.value, source=f"--set #{assignment.ordinal}"
            )
        return overrides

    def prepare_sources_with_provenance(
        self,
        layers: tuple[tuple[str, Mapping[str, object]], ...],
        *,
        initial_origins: tuple[Mapping[TargetPath, FieldOrigin], ...] | None = None,
        layer_explicit: tuple[frozenset[TargetPath], ...] | None = None,
        admitted: bool = False,
    ) -> tuple[PreparedConfig, ...]:
        """Admit every source before performing each declared source read once.

        ``admitted`` is for a caller holding layers this schema already admitted,
        such as ``InvocationSources``, whose constructor validates them before
        any selector is read. Every other caller keeps admission here.
        """
        if initial_origins is not None and len(initial_origins) != len(layers):
            raise ValueError("source origins must match the supplied layers")
        if layer_explicit is not None and len(layer_explicit) != len(layers):
            raise ValueError("layer presence must match the supplied layers")
        if not admitted:
            for source, config in layers:
                self.validate_layer(config, source=source)
        prepared = []
        for index, (source, config) in enumerate(layers):
            supplied_origins = {} if initial_origins is None else initial_origins[index]
            origins: dict[TargetPath, FieldOrigin] = {}
            for path in self.target_paths(config):
                parent = next(
                    (
                        supplied_origins[path[:length]]
                        for length in range(len(path), 0, -1)
                        if path[:length] in supplied_origins
                    ),
                    None,
                )
                origins[path] = (
                    dataclasses.replace(parent, path=path)
                    if parent is not None
                    else FieldOrigin(path, "explicit", True, SourceLocation("config", source))
                )
            explicit: set[TargetPath] = (
                {path for path, origin in origins.items() if origin.explicit}
                if layer_explicit is None
                else set(layer_explicit[index])
            )
            values = self._transform_layer(
                copy.deepcopy(dict(config)), "source", explicit, origins, source=source
            )
            prepared.append(
                PreparedConfig(
                    values, MappingProxyType(origins), layer_explicit=frozenset(explicit)
                )
            )
        return tuple(prepared)

    def normalize_prepared_layer(self, state: PreparedConfig) -> PreparedConfig:
        """Run pure layer rules on decoded inputs without repeating source effects."""
        origins = dict(state.origins)
        explicit = set(state.active_explicit)
        values = self.normalize_layer(state.values, _explicit=explicit, _origins=origins)
        return PreparedConfig(
            values, MappingProxyType(origins), state.replacements, frozenset(explicit)
        )

    def prepare_layers_with_provenance(
        self,
        layers: tuple[tuple[str, Mapping[str, object]], ...],
        *,
        initial_origins: tuple[Mapping[TargetPath, FieldOrigin], ...] | None = None,
        admitted: bool = False,
    ) -> tuple[PreparedConfig, ...]:
        """Decode sources once, then normalize each layer with retained provenance."""
        return tuple(
            self.normalize_prepared_layer(state)
            for state in self.prepare_sources_with_provenance(
                layers, initial_origins=initial_origins, admitted=admitted
            )
        )

    def prepare_layers_with_presence(
        self, layers: tuple[tuple[str, Mapping[str, object]], ...]
    ) -> tuple[tuple[dict[str, object], frozenset[TargetPath]], ...]:
        """Prepare sources while retaining explicit canonical field presence."""
        return tuple(
            (state.values, state.explicit) for state in self.prepare_layers_with_provenance(layers)
        )

    def prepare_layers(
        self, layers: tuple[tuple[str, Mapping[str, object]], ...]
    ) -> tuple[dict[str, object], ...]:
        """Admit all sources before executing any source rule's filesystem reads."""
        return tuple(values for values, _explicit in self.prepare_layers_with_presence(layers))

    def prepare_layer(
        self, config: Mapping[str, object], *, source: str = "config"
    ) -> dict[str, object]:
        """Resolve declared source reads, then normalize one invocation layer."""
        return self.prepare_layers(((source, config),))[0]

    def normalize_layer(
        self,
        config: Mapping[str, object],
        *,
        _explicit: set[TargetPath] | None = None,
        _origins: dict[TargetPath, FieldOrigin] | None = None,
    ) -> dict[str, object]:
        """Run the declared pure layer sequence around shared implications."""
        layer = self._transform_layer(
            copy.deepcopy(dict(config)), "before_implications", _explicit, _origins
        )
        layer = self.expand_layer_implications(layer, _origins=_origins)
        return self._transform_layer(layer, "after_implications", _explicit, _origins)

    def normalize_effective(self, config: Mapping[str, object]) -> dict[str, object]:
        """Normalize policy after precedence merging or for resolved serialization."""
        return self._transform_layer(copy.deepcopy(dict(config)), "effective")

    def normalize_config(self, config: Mapping[str, object]) -> dict[str, object]:
        """Apply registry-declared semantic normalization before typed construction."""
        normalized = self.normalize_effective(config)
        for path, field in self.field_items():
            present, value = _path_value(normalized, path)
            if not present:
                continue
            if field.drop_when_none and value is None:
                _delete_path(normalized, path)
                continue
            if field.strip:
                if not isinstance(value, str):
                    raise ConfigError(f"{'.'.join(path)}: expected string")
                value = value.strip()
                _set_path(normalized, path, value)
            if field.stringify_large_int and isinstance(value, (list, tuple)):
                converted = []
                for item in value:
                    if isinstance(item, str):
                        if _DECIMAL_INTEGER.fullmatch(item) is None:
                            raise ConfigError(f"{'.'.join(path)}: expected decimal integers")
                        item = int(item)
                    converted.append(item)
                _set_path(normalized, path, converted)
            if field.stringify_large_int and isinstance(value, str):
                if _DECIMAL_INTEGER.fullmatch(value) is None:
                    raise ConfigError(
                        f"{'.'.join(path)}: quoted integers must contain decimal digits only"
                    )
                _set_path(normalized, path, int(value))
        return normalized

    def _require_station(self, station: str) -> None:
        known = {"generate", *(name for table in self.tables for name in table.stations)}
        if station not in known:
            raise ValueError(f"{self.model}: unsupported station {station!r}")

    def prepare_station_layer(self, state: PreparedConfig, *, station: str) -> PreparedConfig:
        """Limit implicit artifact selection to outputs produced by the selected station."""
        self._require_station(station)
        values = copy.deepcopy(state.values)
        origins = dict(state.origins)
        for path, declaration in self.field_items():
            policy = declaration.output_policy
            origin = origins.get(path)
            if (
                policy is not None
                and station not in policy.produced_at
                and origin is not None
                and origin.kind == "implied"
                and path not in state.active_explicit
            ):
                _delete_path(values, path)
                origins.pop(path)
        surviving = self.target_paths(values)
        return PreparedConfig(
            values,
            MappingProxyType(origins),
            state.replacements & surviving,
            state.active_explicit & surviving,
        )

    def validate_station_outputs(self, values: Mapping[str, object], *, station: str) -> None:
        """Reject explicit requests for artifacts unavailable at the selected station."""
        self._require_station(station)
        for path, declaration in self.field_items():
            if declaration.output_policy is None:
                continue
            present, value = _path_value(values, path)
            if present and value is True:
                try:
                    declaration.output_policy.selected(
                        True, save_all=False, station=station, field=".".join(path)
                    )
                except ValueError as exc:
                    raise ConfigError(str(exc)) from exc

    def validate_persisted_values(self, values: Mapping[str, object]) -> None:
        """Require canonical input fields, excluding unconsumed source-only spellings."""
        self.validate_layer(values, source="saved configuration")
        for path, declaration in self.field_items():
            present, _value = _path_value(values, path)
            if present and declaration.persistence == "source":
                raise ConfigError(
                    f"{'.'.join(path)}: source-only input must be resolved before persistence"
                )

    def dump_intent(self, intent: PreparedConfig) -> str:
        """Serialize retained inputs, rejecting clear markers TOML cannot replay."""
        self.validate_persisted_values(intent.values)
        retained = copy.deepcopy(intent.values)
        for path, origin in intent.origins.items():
            if (
                not origin.explicit
                and (
                    origin.kind == "implied"
                    or (
                        origin.kind == "default"
                        and origin.location is not None
                        and origin.location.kind == "default"
                    )
                )
                and all(isinstance(part, str) for part in path)
            ):
                _delete_path(retained, field_path(path))
        projected = self._transform_layer(retained, "serialization")
        for path, field in self.field_items():
            present, value = _path_value(projected, path)
            if present and field.strip:
                if not isinstance(value, str):
                    raise ConfigError(f"{'.'.join(path)}: expected string")
                _set_path(projected, path, value.strip())
        cleared: list[TargetPath] = []

        def omit_nulls(value: object, path: TargetPath) -> object:
            if isinstance(value, Mapping):
                result: dict[str, object] = {}
                for key, item in value.items():
                    target = (*path, str(key))
                    if item is None:
                        cleared.append(target)
                    else:
                        result[str(key)] = omit_nulls(item, target)
                return result
            if isinstance(value, (list, tuple)):
                return [omit_nulls(item, (*path, index)) for index, item in enumerate(value)]
            return value

        values = cast(dict[str, object], omit_nulls(projected, ()))
        values["model"] = self.model
        # Only pure rules run here: source files have already been decoded.
        replay = self.normalize_layer(values)
        for path in cleared:
            current: object = replay
            for part in path:
                if isinstance(part, str) and isinstance(current, Mapping):
                    current = current.get(part, _NO_VALUE)
                elif (
                    type(part) is int and isinstance(current, (list, tuple)) and part < len(current)
                ):
                    current = current[part]
                else:
                    current = _NO_VALUE
                    break
            if current is not None:
                raise ConfigError(
                    f"{format_target_path(path)}: TOML cannot preserve this explicit clearing; "
                    "select a declared alternative instead"
                )
        return self.dump_config(values)

    def dump_config(self, config: Mapping[str, object]) -> str:
        """Serialize config with this model's registry-owned presentation policy."""
        multiline = frozenset(path for path, field in self.field_items() if field.multiline)
        stringify_large_int = frozenset(
            path for path, field in self.field_items() if field.stringify_large_int
        )
        return render_config(
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
        *,
        _origins: dict[TargetPath, FieldOrigin] | None = None,
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
                    if _origins is not None:
                        trigger = _origins.get(implication.trigger)
                        _origins[path] = FieldOrigin(
                            path,
                            "implied",
                            False,
                            rule="implication:" + ".".join(implication.trigger),
                            inputs=() if trigger is None else (trigger,),
                        )
        return expanded

    def environment_variables(self) -> dict[ConfigPath, tuple[str, ...]]:
        """Return every env-backed field in this model contribution."""
        result: dict[ConfigPath, tuple[str, ...]] = {}
        for table in self.tables:
            for field in table.fields:
                variables = table.environment_variables(field, model=self.model)
                if variables:
                    result[(*table.path, field.name)] = variables
        return result

    def environment_config(self, *, exclude_tables: tuple[str, ...] = ()) -> dict[str, object]:
        """Return the currently resolved environment layer for this model."""
        return self.environment_state(exclude_tables=exclude_tables).values

    def environment_state(self, *, exclude_tables: tuple[str, ...] = ()) -> PreparedConfig:
        """Resolve values and exact selected ENV/default origins in a single pass."""
        known = {name for variables in self.environment_variables().values() for name in variables}
        unknown = environment_names(f"KINO_{self.model.upper()}_") - known
        if unknown:
            raise unknown_field_error(min(unknown), known, f"environment for {self.model}")
        result: dict[str, object] = {}
        origins: dict[TargetPath, FieldOrigin] = {}
        for table in self.tables:
            if table.path[0] in exclude_tables:
                continue
            try:
                values = table.environment_values(model=self.model)
            except ConfigError:
                raise
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"environment {table.path[0]}: {exc}") from exc
            if values:
                _set_path(result, table.path, {name: entry.value for name, entry in values.items()})
                for name, entry in values.items():
                    path = (*table.path, name)
                    origins[path] = FieldOrigin(
                        path,
                        "explicit" if entry.explicit else "default",
                        entry.explicit,
                        SourceLocation(
                            "environment" if entry.explicit else "default", entry.source
                        ),
                    )
        return PreparedConfig(result, MappingProxyType(origins))


class ConfigRegistry:
    """Global collection of validated model configuration contributions."""

    def __init__(self) -> None:
        self._models: dict[str, ModelConfigSpec] = {}
        self._shared_fields: dict[tuple[type[object], str], tuple[str, ConfigFieldSpec]] = {}
        self._frozen = False

    def register(self, contribution: ModelConfigSpec) -> None:
        """Register one model exactly once."""
        if self._frozen:
            raise RuntimeError("config registry is frozen")
        if contribution.model in self._models:
            raise ValueError(f"duplicate config schema for model {contribution.model!r}")
        owned = {
            (table.record, declaration.name): declaration
            for table in contribution.tables
            for declaration in table.fields
            if not declaration.virtual
        }
        for identity, declaration in owned.items():
            prior = self._shared_fields.get(identity)
            if prior is not None and prior[1].intent_contract() != declaration.intent_contract():
                record, name = identity
                raise ValueError(
                    f"{contribution.model}: shared field {record.__name__}.{name} "
                    f"conflicts with {prior[0]} semantics"
                )
        self._models[contribution.model] = contribution
        for identity, declaration in owned.items():
            self._shared_fields.setdefault(identity, (contribution.model, declaration))

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
