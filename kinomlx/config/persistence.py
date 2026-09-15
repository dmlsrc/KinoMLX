"""Current saved configuration format: canonical values and required presence metadata."""

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from kinomlx.config_values import ConfigError
from kinomlx.model_registry import config_registry

from .paths import TargetPath, format_target_path
from .provenance import FieldOrigin, PreparedConfig, SourceLocation

RUN_SCHEMA_VERSION = 2
CONFIGURATION_FORMAT = "kinomlx.configuration.v1"


def configuration_payload(model: str, intent: PreparedConfig) -> dict[str, object]:
    """Serialize canonical intent, without substituting execution observations."""
    config_registry().model(model).validate_persisted_values(intent.values)
    return {
        "format": CONFIGURATION_FORMAT,
        "model": model,
        "values": copy.deepcopy(intent.values),
        "explicit": [list(path) for path in sorted(intent.explicit, key=format_target_path)],
        "replacements": [
            list(path) for path in sorted(intent.replacements, key=format_target_path)
        ],
    }


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{location}: expected an object with string keys")
    return copy.deepcopy(dict(cast(Mapping[str, object], value)))


def _contains(values: Mapping[str, object], path: TargetPath) -> bool:
    current: object = values
    for part in path:
        if isinstance(part, str):
            if not isinstance(current, Mapping) or part not in current:
                return False
            current = current[part]
        else:
            if not isinstance(current, (tuple, list)) or part < 0 or part >= len(current):
                return False
            current = current[part]
    return True


def _paths(value: object, values: Mapping[str, object], location: str) -> frozenset[TargetPath]:
    if not isinstance(value, list):
        raise ConfigError(f"{location}: expected an array of field paths")
    paths: set[TargetPath] = set()
    for raw in value:
        if (
            not isinstance(raw, list)
            or not raw
            or any(
                not ((isinstance(part, str) and part) or (type(part) is int and part >= 0))
                for part in raw
            )
        ):
            raise ConfigError(f"{location}: invalid field path")
        path = tuple(cast(list[str | int], raw))
        if not _contains(values, path):
            raise ConfigError(f"{location}: {format_target_path(path)} is absent from values")
        if path in paths:
            raise ConfigError(f"{location}: duplicate path {format_target_path(path)}")
        paths.add(path)
    return frozenset(paths)


def read_configuration(payload: Mapping[str, object], *, model: str, source: str) -> PreparedConfig:
    """Require the current run and configuration formats; no snapshot fallback exists."""
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != RUN_SCHEMA_VERSION
    ):
        raise ConfigError(
            f"{source}: unsupported run schema; expected version {RUN_SCHEMA_VERSION}"
        )
    raw = _mapping(payload.get("configuration"), f"{source}.configuration")
    required = {"format", "model", "values", "explicit", "replacements"}
    if raw.keys() != required:
        raise ConfigError(f"{source}.configuration: expected fields {', '.join(sorted(required))}")
    if raw["format"] != CONFIGURATION_FORMAT or raw["model"] != model:
        raise ConfigError(f"{source}.configuration: unsupported format or model")
    values = _mapping(raw["values"], f"{source}.configuration.values")
    config_registry().model(model).validate_persisted_values(values)
    explicit = _paths(raw["explicit"], values, f"{source}.configuration.explicit")
    replacements = _paths(raw["replacements"], values, f"{source}.configuration.replacements")
    origins = {
        path: FieldOrigin(path, "inherited", True, SourceLocation("saved", source))
        for path in explicit
    }
    return PreparedConfig(values, MappingProxyType(origins), replacements)


def rebase_configuration(intent: PreparedConfig, values: dict[str, object]) -> PreparedConfig:
    """Retain metadata for surviving inputs after a station derives its replay context."""

    return PreparedConfig(
        copy.deepcopy(values),
        MappingProxyType(
            {path: origin for path, origin in intent.origins.items() if _contains(values, path)}
        ),
        frozenset(path for path in intent.replacements if _contains(values, path)),
    )
