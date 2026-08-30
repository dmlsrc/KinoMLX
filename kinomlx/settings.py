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

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol, Self, Union, cast, get_args, get_origin, get_type_hints

_ENV_TEMPLATE_VAR = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_FALSE_WORDS = frozenset({"", "0", "false", "no", "off"})
CACHE_MODE_CHOICES = ("auto", "rebuild")


class _KeywordConstructor[ResultT](Protocol):
    def __call__(self, **kwargs: object) -> ResultT: ...


def _parse_bool(raw: str) -> bool:
    """Parse conventional environment boolean spellings."""
    return raw.strip().lower() not in _FALSE_WORDS


def _resolve_env_entry(entry: str) -> str | None:
    """Resolve one explicit environment template or literal fallback."""
    if "{{" not in entry:
        return entry
    parts: list[str] = []
    last = 0
    for match in _ENV_TEMPLATE_VAR.finditer(entry):
        parts.append(entry[last : match.start()])
        value = os.environ.get(match.group(1))
        if not value:
            return None
        parts.append(value)
        last = match.end()
    parts.append(entry[last:])
    return "".join(parts)


def _parser_for(annotation: object) -> Callable[[str], object]:
    """Return a type-driven environment/CLI value parser."""
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (types.UnionType, Union):
        concrete = tuple(item for item in arguments if item is not type(None))
        if len(concrete) == 1:
            return _parser_for(concrete[0])
    if origin is tuple:
        item_type = arguments[0] if arguments else str
        item_parser = _parser_for(item_type)

        def parse_tuple(raw: str) -> tuple[object, ...]:
            value = raw.strip()
            if not value or value.lower() in {"off", "none"}:
                return ()
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


class EnvironmentSettings:
    """Mixin for frozen dataclasses with declarative environment fields."""

    @classmethod
    def from_env(cls) -> Self:
        """Build one setting record from all of its declared env sources."""
        return cls.from_env_fields(*(item.name for item in _settings_fields(cls)))

    @classmethod
    def from_env_fields(cls, *names: str) -> Self:
        """Build a record while reading only selected environment fields."""
        constructor = cast(_KeywordConstructor[Self], cls)
        return constructor(**cls.overrides_from_env_fields(*names))

    @classmethod
    def overrides_from_env_fields(cls, *names: str) -> dict[str, object]:
        """Return only values supplied by declared environment sources."""
        requested = set(names)
        known = {item.name for item in _settings_fields(cls)}
        unknown = requested - known
        if unknown:
            raise KeyError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")

        kwargs: dict[str, object] = {}
        hints: dict[str, object] = get_type_hints(cls)
        for item in _settings_fields(cls):
            if item.name not in requested:
                continue
            try:
                entries = _environment_entries(item)
            except TypeError as exc:
                raise TypeError(f"{cls.__name__}.{exc}") from exc
            for entry in entries:
                resolved = _resolve_env_entry(entry)
                if resolved is None:
                    continue
                try:
                    kwargs[item.name] = _parser_for(hints[item.name])(resolved)
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
            return _environment_entries(reflected[name])
        except TypeError as exc:
            raise TypeError(f"{cls.__name__}.{exc}") from exc

    def with_overrides(self, **overrides: object) -> Self:
        """Return a record with non-None overrides applied."""
        clean = {name: value for name, value in overrides.items() if value is not None}
        return dataclasses.replace(self, **clean)  # type: ignore[type-var]


@dataclass(frozen=True)
class Settings(EnvironmentSettings):
    """Model-neutral cache, UX, profiling, and MLX allocator settings."""

    cache_dir: Path = field(
        default_factory=lambda: Path("~/.cache/kinomlx").expanduser(),
        metadata={"env": "{{KINO_CACHE_DIR}}"},
    )
    cache_mode: str = field(default="auto", metadata={"env": "{{KINO_CACHE_MODE}}"})
    hf_home: Path = field(
        default_factory=lambda: Path("~/.cache/huggingface").expanduser(),
        metadata={"env": "{{HF_HOME}}"},
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
        metadata={"env": "{{KINO_PROFILE_SIGNPOST_LOG}}"},
    )

    mlx_cache_limit_gb: float | None = field(
        default=1.0,
        metadata={"env": "{{KINO_MLX_CACHE_LIMIT_GB}}"},
    )

    def validate(self) -> None:
        """Validate infrastructure settings."""
        if self.cache_mode not in CACHE_MODE_CHOICES:
            valid = ", ".join(CACHE_MODE_CHOICES)
            raise ValueError(f"cache_mode must be one of: {valid}")
        if self.profile_signpost_log is not None and not self.profile_signposts:
            raise ValueError("profile_signpost_log requires profile_signposts=true")
        if self.mlx_cache_limit_gb is not None and (
            not math.isfinite(self.mlx_cache_limit_gb) or self.mlx_cache_limit_gb < 0
        ):
            raise ValueError("mlx_cache_limit_gb must be finite and non-negative")


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
        try:
            flags = _cli_flags(item)
        except TypeError as exc:
            raise TypeError(f"{settings_type.__name__}.{exc}") from exc
        annotation = hints[item.name]
        concrete = tuple(value for value in get_args(annotation) if value is not type(None))
        is_bool = annotation is bool or concrete == (bool,)
        if is_bool:
            group.add_argument(
                *flags,
                dest=item.name,
                action="store_true",
                default=None,
                help=declared_help.get(item.name),
            )
            group.add_argument(
                *("--no-" + flag.removeprefix("--") for flag in flags),
                dest=item.name,
                action="store_false",
                default=None,
                help=declared_negative_help.get(
                    item.name,
                    declared_help.get(item.name),
                ),
            )
        else:
            group.add_argument(
                *flags,
                dest=item.name,
                type=_parser_for(annotation),
                choices=declared_choices.get(item.name),
                default=None,
                help=declared_help.get(item.name),
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
    "CACHE_MODE_CHOICES",
    "EnvironmentSettings",
    "Settings",
    "add_argparse_args",
    "add_settings_argparse_args",
    "settings_from_args",
]
