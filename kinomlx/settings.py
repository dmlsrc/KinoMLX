"""Infrastructure settings and the declarative environment/CLI bridge.

Only this module reads environment variables. Model packages declare their
own frozen setting records using :class:`EnvironmentSettings`; the host and
model records then travel separately through CLI/TOML assembly. This keeps
model internals out of the model-neutral configuration layer while preserving
one precedence path and one parser implementation.

Each setting field declares its environment source or fallback chain in
``metadata["env"]``. ``{{VAR}}`` references are resolved explicitly; bare
strings are literal fallback values. CLI values are parsed by their declared
types but never receive environment substitution.
"""

import argparse
import dataclasses
import os
import re
import tomllib
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar, Self, Union, cast, get_args, get_origin, get_type_hints

from kinomlx.config_bindings import CliAction, CliBinding
from kinomlx.config_constraints import NumberConstraint
from kinomlx.config_records import config_record, construct_record, replace_config
from kinomlx.config_rules import RecordRule, validate_record

_ENV_TEMPLATE_VAR = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_FALSE_WORDS = frozenset({"", "0", "false", "no", "off"})
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


def _parse_bool(raw: str) -> bool:
    """Parse conventional environment boolean spellings."""
    value = raw.strip().lower()
    if value in _FALSE_WORDS:
        return False
    if value in _TRUE_WORDS:
        return True
    raise ValueError("expected true/false, yes/no, on/off, or 1/0")


def environment_names(prefix: str) -> frozenset[str]:
    """Snapshot variable names in one owned configuration namespace."""
    return frozenset(name for name in os.environ if name.startswith(prefix))


def read_environment(name: str) -> str | None:
    """Read a registered variable, preserving an explicitly empty value."""
    return os.environ.get(name)


def _resolve_env_entry(entry: str) -> str | None:
    """Resolve one explicit environment template or literal fallback."""
    if "{{" not in entry:
        return entry
    parts: list[str] = []
    last = 0
    for match in _ENV_TEMPLATE_VAR.finditer(entry):
        parts.append(entry[last : match.start()])
        value = read_environment(match.group(1))
        if not value:
            return None
        parts.append(value)
        last = match.end()
    parts.append(entry[last:])
    return "".join(parts)


def parse_config_value(raw: str) -> object:
    """Parse a TOML scalar or collection, preserving ordinary unquoted text."""
    try:
        return cast(object, tomllib.loads(f"value = {raw}")["value"])
    except tomllib.TOMLDecodeError:
        return raw


def parse_environment_value(annotation: object, raw: str) -> object:
    """Use the shared CLI type parser for one registered environment value."""
    return _parser_for(annotation)(raw)


def environment_value_is_unset(annotation: object, raw: str | None) -> bool:
    """Empty paths are absent; empty text and sequences remain meaningful values."""
    return raw is None or (raw == "" and (annotation is Path or Path in get_args(annotation)))


def _parser_for(annotation: object) -> Callable[[str], object]:
    """Return a type-driven environment/CLI value parser."""
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (types.UnionType, Union):
        concrete = tuple(item for item in arguments if item is not type(None))
        if len(concrete) == 1:
            return _parser_for(concrete[0])
        if str in concrete:
            return lambda raw: parse_config_value(raw) if raw.lstrip().startswith("[") else raw
    if origin in (tuple, list):
        item_type = arguments[0] if arguments else str
        item_parser = _parser_for(item_type)

        def parse_tuple(raw: str) -> tuple[object, ...]:
            value = raw.strip()
            if not value or value.lower() in {"off", "none"}:
                return ()
            if value.startswith("["):
                parsed = parse_config_value(value)
                if not isinstance(parsed, list):
                    raise ValueError("expected a TOML array")
                return tuple(parsed)
            return tuple(item_parser(item.strip()) for item in value.split(",") if item.strip())

        return parse_tuple
    if annotation is bool:
        return _parse_bool
    if annotation is Path:
        return lambda raw: Path(raw).expanduser()
    if annotation is int:
        return int
    if annotation is float:
        return float
    return str


def _settings_fields(settings_type: type[object]) -> tuple[dataclasses.Field[object], ...]:
    """Reflect fields from a subclass that the mixin contract requires to be a dataclass."""
    # Typeshed cannot express that every concrete EnvironmentSettings subclass
    # is decorated with @dataclass, so keep that mismatch at this boundary.
    return cast(
        tuple[dataclasses.Field[object], ...],
        fields(settings_type),  # type: ignore[arg-type]
    )


def _environment_entries(item: dataclasses.Field[object]) -> tuple[str, ...]:
    env_spec: object = item.metadata.get("env")
    if env_spec is None:
        return ()
    entries: tuple[str, ...]
    if isinstance(env_spec, str):
        entries = (env_spec,)
    elif isinstance(env_spec, list | tuple) and all(isinstance(entry, str) for entry in env_spec):
        entries = tuple(entry for entry in env_spec if isinstance(entry, str))
    else:
        raise TypeError(f"{item.name}: env metadata must contain strings")
    for entry in entries:
        remainder = _ENV_TEMPLATE_VAR.sub("", entry)
        if "{{" in remainder or "}}" in remainder:
            raise TypeError(f"{item.name}: malformed environment template {entry!r}")
    return entries


def _cli_flags(item: dataclasses.Field[object]) -> tuple[str, ...]:
    """Return the declared public option spellings for one setting field."""
    raw: object = item.metadata.get("cli", "--" + item.name.replace("_", "-"))
    flags: tuple[str, ...]
    if isinstance(raw, str):
        flags = (raw,)
    elif isinstance(raw, list | tuple) and all(isinstance(flag, str) for flag in raw):
        flags = tuple(flag for flag in raw if isinstance(flag, str))
    else:
        raise TypeError(f"{item.name}: cli metadata must contain a string or sequence of strings")
    if (
        not flags
        or len(flags) != len(set(flags))
        or any(not flag.startswith("--") or len(flag) <= 2 for flag in flags)
    ):
        raise TypeError(f"{item.name}: cli metadata must contain unique long-option strings")
    return flags


def _settings_cli_binding(
    item: dataclasses.Field[object], annotation: object, title: str
) -> CliBinding:
    concrete = tuple(value for value in get_args(annotation) if value is not type(None))
    is_bool = annotation is bool or concrete == (bool,)
    return CliBinding(
        group=title,
        flags=_cli_flags(item),
        action=CliAction.PAIRED_BOOLEAN if is_bool else CliAction.STORE,
        parse=None if is_bool else _parser_for(annotation),
    )


@dataclass(frozen=True)
class EnvironmentValue:
    """A decoded value and the exact source expression selected by the resolver."""

    value: object
    source: str
    explicit: bool


class EnvironmentSettings:
    """Mixin for frozen dataclasses with declarative environment fields."""

    environment_prefix: ClassVar[str | None] = None
    # The TOML table this record is written at, so a rule violation names a
    # location the reader can edit rather than a Python class.
    config_table: ClassVar[str] = "[settings]"

    @classmethod
    def cli_binding_for_field(cls, name: str, *, title: str) -> CliBinding:
        """Compile the owned setting metadata for any CLI consumer."""
        hints: dict[str, object] = get_type_hints(cls)
        for item in _settings_fields(cls):
            if item.name == name:
                return _settings_cli_binding(item, hints[name], title)
        raise KeyError(f"unknown {cls.__name__} field {name!r}")

    @classmethod
    def from_env(cls) -> Self:
        """Build one setting record from all of its declared env sources."""
        return cls.from_env_fields(*(item.name for item in _settings_fields(cls)))

    @classmethod
    def from_env_fields(cls, *names: str) -> Self:
        """Build a record while reading only selected environment fields."""
        selected = cls.values_from_env_fields(*names)
        values = {name: entry.value for name, entry in selected.items()}
        presence = frozenset(name for name, entry in selected.items() if entry.explicit)
        return construct_record(cls, values, presence)

    @classmethod
    def values_from_env_fields(cls, *names: str) -> dict[str, EnvironmentValue]:
        """Decode once, retaining the chosen template or literal fallback."""
        requested = set(names)
        known = {item.name for item in _settings_fields(cls)}
        unknown = requested - known
        if unknown:
            raise KeyError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")

        kwargs: dict[str, EnvironmentValue] = {}
        hints: dict[str, object] = get_type_hints(cls)
        for item in _settings_fields(cls):
            if item.name not in requested:
                continue
            try:
                entries = cls.environment_sources_for_field(item.name)
            except TypeError as exc:
                raise TypeError(f"{cls.__name__}.{exc}") from exc
            for entry in entries:
                resolved = (
                    read_environment(entry[2:-2])
                    if cls.environment_prefix is not None and not _environment_entries(item)
                    else _resolve_env_entry(entry)
                )
                if environment_value_is_unset(hints[item.name], resolved):
                    continue
                assert resolved is not None
                try:
                    kwargs[item.name] = EnvironmentValue(
                        parse_environment_value(hints[item.name], resolved),
                        entry,
                        bool(_ENV_TEMPLATE_VAR.search(entry)),
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{item.name}: invalid value from {entry}") from exc
                break
        return kwargs

    @classmethod
    def environment_sources_for_field(cls, name: str) -> tuple[str, ...]:
        """Return the declared environment expressions for one record field."""
        reflected = {item.name: item for item in _settings_fields(cls)}
        if name not in reflected:
            raise KeyError(f"unknown {cls.__name__} field: {name}")
        try:
            entries = _environment_entries(reflected[name])
            if not entries and cls.environment_prefix is not None:
                return ("{{" + cls.environment_prefix + name.upper() + "}}",)
            return entries
        except TypeError as exc:
            raise TypeError(f"{cls.__name__}.{exc}") from exc

    def validate(self) -> None:
        """Run owned semantic checks on the fully composed Settings record."""
        validate_record(self, location=type(self).config_table)

    def with_overrides(self, **overrides: object) -> Self:
        """Return a record with non-None overrides applied."""
        clean = {name: value for name, value in overrides.items() if value is not None}
        return replace_config(self, **clean)


def _profile_log_requires_signposts(values: Mapping[str, object]) -> None:
    if values["profile_signpost_log"] is not None and not values["profile_signposts"]:
        raise ValueError("profile_signpost_log requires profile_signposts=true")


_PROFILE_LOG_RULE = RecordRule(
    "profile_log_requires_signposts",
    frozenset({"profile_signpost_log", "profile_signposts"}),
    _profile_log_requires_signposts,
)


@config_record
@dataclass(frozen=True)
class Settings(EnvironmentSettings):
    """Model-neutral cache, UX, profiling, and MLX allocator settings."""

    app_cache_dir: Path = field(
        default_factory=lambda: Path("~/Library/Caches/KinoMLX").expanduser(),
        metadata={
            "env": [
                "{{KINO_APP_CACHE_DIR}}",
                "{{XDG_CACHE_HOME}}/KinoMLX",
                "~/Library/Caches/KinoMLX",
            ]
        },
    )
    hf_home: Path = field(
        default_factory=lambda: Path("~/.cache/huggingface").expanduser(),
        metadata={"env": "{{HF_HOME}}"},
    )

    check_adapter_weights: bool = field(
        default=True, metadata={"env": "{{KINO_CHECK_ADAPTER_WEIGHTS}}"}
    )

    verbose: bool = field(default=False, metadata={"env": "{{KINO_VERBOSE}}"})
    quiet: bool = field(default=False, metadata={"env": "{{KINO_QUIET}}"})
    json_output: bool = field(
        default=False,
        metadata={"env": "{{KINO_JSON}}", "cli": "--json"},
    )

    profile_signposts: bool = field(
        default=False,
        metadata={"env": "{{KINO_PROFILE_SIGNPOSTS}}"},
    )
    profile_signpost_log: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_PROFILE_SIGNPOST_LOG}}", "config_rules": (_PROFILE_LOG_RULE,)},
    )

    mlx_cache_limit_gb: float | None = field(
        default=1.0,
        metadata={
            "env": "{{KINO_MLX_CACHE_LIMIT_GB}}",
            "config_constraints": (NumberConstraint(minimum=0),),
        },
    )
    # None wires up to the device's recommended working set; 0 leaves buffers unwired.
    mlx_wired_limit_gb: float | None = field(
        default=None,
        metadata={
            "env": "{{KINO_MLX_WIRED_LIMIT_GB}}",
            "config_constraints": (NumberConstraint(minimum=0),),
        },
    )


def add_settings_argparse_args(
    parser: argparse.ArgumentParser,
    settings_type: type[EnvironmentSettings],
    *,
    title: str,
    skip: set[str] | frozenset[str] = frozenset(),
    choices_by_field: Mapping[str, tuple[str, ...]] | None = None,
    help_by_field: Mapping[str, str] | None = None,
    negative_help_by_field: Mapping[str, str] | None = None,
) -> None:
    """Add type-driven flags for one infrastructure or model setting record."""
    group = parser.add_argument_group(title)
    declared_choices = {} if choices_by_field is None else choices_by_field
    declared_help = {} if help_by_field is None else help_by_field
    declared_negative_help = {} if negative_help_by_field is None else negative_help_by_field
    hints: dict[str, object] = get_type_hints(settings_type)
    for item in _settings_fields(settings_type):
        if item.name in skip:
            continue
        binding = _settings_cli_binding(item, hints[item.name], title)
        binding.add_arguments(
            group,
            destination=item.name,
            annotation=hints[item.name],
            choices=declared_choices.get(item.name, ()),
            help=declared_help.get(item.name),
            negative_help=declared_negative_help.get(item.name, declared_help.get(item.name)),
        )


def add_argparse_args(
    parser: argparse.ArgumentParser,
    *,
    skip: set[str] | frozenset[str] = frozenset(),
    choices_by_field: Mapping[str, tuple[str, ...]] | None = None,
    help_by_field: Mapping[str, str] | None = None,
    negative_help_by_field: Mapping[str, str] | None = None,
) -> None:
    """Add flags for model-neutral :class:`Settings` fields."""
    add_settings_argparse_args(
        parser,
        Settings,
        title="Infrastructure settings (override env vars)",
        skip=skip,
        choices_by_field=choices_by_field,
        help_by_field=help_by_field,
        negative_help_by_field=negative_help_by_field,
    )


def settings_from_args(args: argparse.Namespace, base: Settings) -> Settings:
    """Apply parsed infrastructure overrides to ``base``."""
    return base.with_overrides(
        **{item.name: getattr(args, item.name, None) for item in _settings_fields(Settings)}
    )


__all__ = [
    "EnvironmentSettings",
    "Settings",
    "add_argparse_args",
    "add_settings_argparse_args",
    "settings_from_args",
]
