"""TOML-typed ordered overrides with replacement and explicit-origin tracking."""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from kinomlx.config_records import record_intent
from kinomlx.config_values import ConfigError
from kinomlx.settings import parse_config_value

from .merge import merge_configs
from .paths import TargetPath, field_path, format_target_path, parse_target_path
from .provenance import FieldOrigin, PreparedConfig, SourceLocation


def parse_set_argument(argument: str) -> tuple[list[str | int], object]:
    """Parse an override into a concrete member/index path and a TOML value."""
    key_part, separator, value_part = argument.partition("=")
    if not separator:
        raise ConfigError(f"--set {argument!r}: expected <key>[.<key>...]=<value>")
    try:
        path = parse_target_path(key_part)
    except ConfigError as exc:
        raise ConfigError(f"--set {argument!r}: {exc}") from exc
    return list(path), parse_config_value(value_part)


@dataclass(frozen=True)
class OverrideAssignment:
    """One explicit assignment; ordinal is its one-based command-line position."""

    path: TargetPath
    value: object
    argument: str
    ordinal: int


def _assign(node: object, path: TargetPath, value: object, prefix: TargetPath = ()) -> object:
    key, *tail = path
    remainder = tuple(tail)
    location = format_target_path(prefix) or "configuration"
    if type(key) is int:
        if not isinstance(node, (list, tuple)):
            raise ConfigError(f"{location} is not a collection (found {type(node).__name__})")
        if key < 0 or key >= len(node):
            raise ConfigError(
                f"{location}[{key}]: index is outside collection of length {len(node)}"
            )
        result = list(node)
        result[key] = _assign(node[key], remainder, value, (*prefix, key)) if remainder else value
        return tuple(result) if isinstance(node, tuple) else result
    if not isinstance(key, str):
        raise ConfigError(f"{location}: invalid configuration member {key!r}")
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        node = record_intent(node)
    if not isinstance(node, Mapping):
        raise ConfigError(f"{location} is not a table (found {type(node).__name__})")
    mapping = dict(cast(Mapping[str, object], node))
    existing = mapping.get(key)
    if remainder:
        child = {} if existing is None and isinstance(remainder[0], str) else existing
        mapping[key] = _assign(child, remainder, value, (*prefix, key))
    else:
        mapping[key] = (
            merge_configs(existing, value)
            if isinstance(existing, Mapping) and isinstance(value, Mapping)
            else value
        )
    return mapping


def _apply(
    config: Mapping[str, object], assignments: tuple[OverrideAssignment, ...]
) -> dict[str, object]:
    result = dict(config)
    for assignment in assignments:
        try:
            result = cast(dict[str, object], _assign(result, assignment.path, assignment.value))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"--set {assignment.argument!r}: {exc}") from exc
    return result


def _leaf_values(path: TargetPath, value: object) -> tuple[tuple[TargetPath, object], ...]:
    if isinstance(value, dict) and value:
        return tuple(
            *_leaf_values((*path, key), child)
            for key, child in cast(dict[str, object], value).items()
        )
    return ((path, value),)


@dataclass(frozen=True)
class SetOverrides:
    """An ordered stream, normalized together as one precedence layer.

    Group tables overlay their members. Scalar and sequence assignments replace
    only their target; normalization sees surviving explicit values together.
    """

    assignments: tuple[OverrideAssignment, ...]

    @classmethod
    def parse(cls, arguments: list[str] | None) -> SetOverrides:
        assignments = []
        for ordinal, argument in enumerate(arguments or [], 1):
            path, value = parse_set_argument(argument)
            assignments.append(OverrideAssignment(tuple(path), value, argument, ordinal))
        return cls(tuple(assignments))

    def project(self, paths: frozenset[tuple[str, ...]]) -> SetOverrides:
        """Extract owned selector assignments without executing unrelated record edits."""
        selected = []
        for assignment in self.assignments:
            for path in sorted(paths):
                if path[: len(assignment.path)] != assignment.path:
                    continue
                value = assignment.value
                for member in path[len(assignment.path) :]:
                    if not isinstance(value, Mapping) or member not in value:
                        break
                    value = value[member]
                else:
                    selected.append(
                        OverrideAssignment(path, value, assignment.argument, assignment.ordinal)
                    )
        return SetOverrides(tuple(selected))

    @property
    def config(self) -> dict[str, object]:
        """Surviving explicit values for layer-local normalization."""
        return _apply({}, self.assignments)

    @property
    def origins(self) -> dict[tuple[str, ...], OverrideAssignment]:
        """Surviving explicit leaf origins, including explicit empty tables."""
        origins: dict[tuple[str, ...], OverrideAssignment] = {}
        for assignment in self.assignments:
            for target_path, value in _leaf_values(field_path(assignment.path), assignment.value):
                path = field_path(target_path)
                # An empty group does not discard earlier member assignments.
                if isinstance(value, dict) and any(
                    len(target) > len(path) and target[: len(path)] == path for target in origins
                ):
                    continue
                origins = {
                    target: origin
                    for target, origin in origins.items()
                    if target[: len(path)] != path and path[: len(target)] != target
                }
                origins[path] = assignment
        return origins

    def apply_state(
        self,
        base: PreparedConfig,
        *,
        targets: Callable[[Mapping[str, object]], frozenset[TargetPath]],
    ) -> PreparedConfig:
        """Execute against inherited values while updating only addressed source identities."""
        values = base.values
        origins = dict(base.origins)
        replacements = set(base.replacements)
        for assignment in self.assignments:
            values = _apply(values, (assignment,))
            surviving = targets(values)
            writes = (
                ((assignment.path, assignment.value),)
                if isinstance(assignment.path[-1], int)
                else _leaf_values(assignment.path, assignment.value)
            )
            for path, value in writes:
                # An empty group overlay changes no existing member's source.
                if (
                    isinstance(value, dict)
                    and not value
                    and not isinstance(path[-1], int)
                    and path not in surviving
                ):
                    continue
                origins = {
                    target: origin
                    for target, origin in origins.items()
                    if target[: len(path)] != path
                }
                for target in surviving:
                    if target[: len(path)] == path:
                        origins[target] = FieldOrigin(
                            target,
                            "explicit",
                            True,
                            SourceLocation("override", "--set", assignment.ordinal),
                        )
                if not isinstance(value, Mapping) or isinstance(path[-1], int):
                    replacements = {
                        target for target in replacements if target[: len(path)] != path
                    }
                    replacements.add(path)
            origins = {path: origin for path, origin in origins.items() if path in surviving}
            replacements.intersection_update(surviving)
        return PreparedConfig(values, frozendict(origins), frozenset(replacements))

    def merge(
        self,
        *layers: Mapping[str, object],
        normalized: Mapping[str, object],
    ) -> dict[str, object]:
        """Apply replacements to inherited state, then the normalized override layer.

        Replay against inherited state first to reject traversal through scalars.
        Normalization sees only surviving explicit assignments, so implications
        can replace inherited values while respecting explicit same-layer values.
        """
        base = merge_configs(*layers)
        _apply(base, self.assignments)
        # Clear assigned leaves before overlaying their normalized forms.
        # Group assignments preserve siblings, including in the same stream.
        replaced = {
            field_path(
                path[
                    : next(
                        (index for index, part in enumerate(path) if isinstance(part, int)),
                        len(path),
                    )
                ]
            )
            for item in self.assignments
            for path, value in _leaf_values(item.path, item.value)
            if not isinstance(value, dict) or any(isinstance(part, int) for part in item.path)
        }
        for path in sorted(replaced, key=len):
            node = base
            for key in path[:-1]:
                existing = node.get(key)
                if not isinstance(existing, dict):
                    break
                child = dict(cast(dict[str, object], existing))
                node[key] = child
                node = child
            else:
                node.pop(path[-1], None)
        return merge_configs(base, normalized)


def apply_set_overrides(
    config: Mapping[str, object], set_arguments: list[str] | None
) -> dict[str, object]:
    """Apply overrides in order to a copy of ``config``."""
    return _apply(config, SetOverrides.parse(set_arguments).assignments)
