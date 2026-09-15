"""Shared invocation source preparation and precedence around a saved base."""

import argparse
import copy
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import cast

from kinomlx.config_records import supplied_fields
from kinomlx.config_values import ConfigError, dataclass_from_mapping, dataclass_to_mapping
from kinomlx.settings import EnvironmentSettings

from .load import load_config
from .merge import field_paths, merge_configs
from .overrides import SetOverrides
from .paths import TargetPath, field_path
from .provenance import FieldOrigin, PreparedConfig, SourceLocation
from .registry import ConfigGroup, ConfigPath, ModelConfigSpec

_LOG = logging.getLogger(__name__)

# One admitted layer of source-selector intent and the surface that wrote it.
SelectionLayer = tuple[str, Mapping[str, object]]

# Every restart failure is a problem with the table the user wrote.
RESTART_LOCATION = "[restart]"


@contextmanager
def restart_source(run: Path) -> Iterator[None]:
    """Report every failure of a restart source as one located configuration error.

    Opening a saved run reads a file, decodes JSON, admits a persisted
    configuration and resolves the artifacts the selected station consumes.
    Each of those can fail for a reason the user controls through ``[restart]``,
    so the failure is named after that table and rendered as a configuration
    error rather than reaching the console as a traceback. A failure that
    already names the table keeps its own text; one that names nothing gains
    the run it was reading.
    """
    try:
        yield
    except ConfigError as exc:
        message = str(exc)
        if message.startswith(f"{RESTART_LOCATION}:"):
            raise
        raise ConfigError(f"{RESTART_LOCATION}: {message}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigError(f"{RESTART_LOCATION}: restart run {run}: {exc}") from exc


def config_table(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Read one declared table from a raw configuration layer."""
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a table")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"[{name}] keys must be strings")
    return cast(Mapping[str, object], value)


def refuse_restart_changes(
    schema: ModelConfigSpec, changed: Sequence[ConfigPath], *, station: str
) -> None:
    """Refuse one restart in the declared words of every field it turned away.

    Every station policy is required to carry a nonempty reason, so the refusal
    quotes it rather than asserting one generic cause for inputs that a station
    may have locked or may simply never read.
    """
    if not changed:
        return
    details = []
    for path in changed:
        policy = next(
            item for item in schema.field(path).station_policies if item.station == station
        )
        name = f"[{path[0]}]." + ".".join(path[1:])
        relation = (
            "is not used on replay"
            if policy.mode == "irrelevant"
            else "belongs to an earlier station"
        )
        details.append(f"{name} {relation}: {policy.reason}")
    raise ConfigError(f"[restart]: {station} restart: " + "; ".join(details))


def _restart_key_surfaces(layers: Sequence[SelectionLayer]) -> dict[str, str]:
    """Name the highest-precedence layer that wrote each present restart key."""
    written: dict[str, str] = {}
    for label, values in layers:
        written.update(dict.fromkeys(config_table(values, "restart"), label))
    return written


def _restart_key_surface_text(
    key: str, label: str, variables: Mapping[ConfigPath, tuple[str, ...]]
) -> str:
    """Describe one restart key's surface, naming the variable behind an env read."""
    if label != "environment":
        return label
    declared = variables.get(("restart", key), ())
    return f"environment {declared[0]}" if declared else label


def restart_selection[RestartT](
    record: type[RestartT],
    *layers: SelectionLayer,
    schema: ModelConfigSpec,
    source: str,
) -> RestartT | None:
    """Resolve restart selection before applying its prior invocation as a base.

    A ``[restart]`` table with no ``run`` is a restart request only when a layer
    above the environment wrote one of its keys. An exported restart variable
    left in a shell would otherwise turn every later ordinary run into a failed
    restart, with no way to clear it inside the invocation; this is the one
    place an environment value does not simply yield to a higher layer.
    """
    ordered = tuple(layers)
    merged = merge_configs(*(config_table(values, "restart") for _label, values in ordered))
    if not merged:
        return None
    if not merged.get("run"):
        variables = schema.environment_variables()
        written = _restart_key_surfaces(ordered)
        surfaces = {
            key: _restart_key_surface_text(key, label, variables)
            for key, label in sorted(written.items())
        }
        if all(label == "environment" for label in written.values()):
            _LOG.debug(
                "Ignoring a [restart] table with no run supplied only by %s",
                ", ".join(f"[restart].{key} from {text}" for key, text in surfaces.items()),
            )
            return None
        raise ConfigError(
            f"{source}: [restart].run is required; "
            + "; ".join(f"[restart].{key} was set by {text}" for key, text in surfaces.items())
        )
    try:
        return dataclass_from_mapping(record, merged, table="[restart]")
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[restart]: {exc}") from exc


def _project_values(
    base: Mapping[str, object], paths: frozenset[tuple[str, ...]]
) -> dict[str, object]:
    selected: dict[str, object] = {}
    for path in paths:
        value: object = base
        for member in path:
            if not isinstance(value, Mapping) or member not in value:
                break
            value = value[member]
        else:
            node = selected
            for member in path[:-1]:
                child = node.setdefault(member, {})
                if not isinstance(child, dict):
                    raise ValueError("projected field paths overlap")
                node = child
            node[path[-1]] = value
    return selected


def _source_selection(
    schema: ModelConfigSpec,
    layers: tuple[Mapping[str, object], ...],
    overrides: SetOverrides,
) -> dict[str, object]:
    paths = frozenset(schema.paths_in_group(ConfigGroup.SOURCE_SELECTOR))
    selected = _project_values(merge_configs(*layers), paths)
    return merge_configs(selected, overrides.project(paths).config)


def _compose_states(schema: ModelConfigSpec, states: tuple[PreparedConfig, ...]) -> PreparedConfig:
    values: dict[str, object] = {}
    origins: dict[TargetPath, FieldOrigin] = {}
    replacements: set[TargetPath] = set()
    for state in states:
        for path in field_paths(state.values):
            replacements = {target for target in replacements if target[: len(path)] != path}
        replacements.update(state.replacements)
        values = merge_configs(values, state.values)
        origins.update(state.origins)
        surviving = schema.target_paths(values)
        origins = {path: origin for path, origin in origins.items() if path in surviving}
    return PreparedConfig(values, MappingProxyType(origins), frozenset(replacements & surviving))


@dataclass(frozen=True)
class ConfigLayer:
    """One named sparse source before and after its declared preparation."""

    source: str
    raw: dict[str, object]
    values: dict[str, object]
    origins: Mapping[TargetPath, FieldOrigin]
    replacements: frozenset[TargetPath] = frozenset()
    layer_explicit: frozenset[TargetPath] | None = None

    @property
    def explicit(self) -> frozenset[TargetPath]:
        return (
            frozenset(path for path, origin in self.origins.items() if origin.explicit)
            if self.layer_explicit is None
            else self.layer_explicit
        )


@dataclass(frozen=True)
class InvocationLayers:
    """Prepared inputs with one precedence protocol shared by every model."""

    environment: ConfigLayer
    file: ConfigLayer
    cli: ConfigLayer
    override: ConfigLayer
    overrides: SetOverrides
    schema: ModelConfigSpec
    saved_base: PreparedConfig | None = None

    def selection(self) -> dict[str, object]:
        """Project source selectors through the same declarations as raw collection."""
        return _source_selection(
            self.schema,
            (
                self.environment.values,
                {} if self.saved_base is None else self.saved_base.values,
                self.file.values,
                self.cli.values,
            ),
            self.overrides,
        )

    def supplied_values(self) -> dict[str, object]:
        """Project prepared sources without saved-only fields or replaying their edits."""
        return merge_configs(
            self.environment.values, self.file.values, self.cli.values, self.override.values
        )

    def merge(self) -> dict[str, object]:
        """Compose ENV < saved base < file < CLI < ordered overrides."""
        saved = self.saved_base
        return self.overrides.merge(
            self.environment.values,
            {} if saved is None else saved.values,
            self.file.values,
            self.cli.values,
            normalized=self.override.values,
        )

    def state(self) -> PreparedConfig:
        """Compose sparse intent and its origins together before effective resolution."""
        return self._state

    @cached_property
    def _state(self) -> PreparedConfig:
        """Compose once; the prepared layers never change after construction.

        Every caller of ``state``, ``provenance``, ``explicit_paths`` and
        ``active_explicit_paths`` reads the same composition, and one model
        reaches it three or four times per invocation.
        """
        saved = self.saved_base
        origins: dict[TargetPath, FieldOrigin] = {}
        layers = [(self.environment.values, self.environment.origins)]
        if saved is not None:
            layers.append((saved.values, saved.origins))
        layers.extend(
            (
                (self.file.values, self.file.origins),
                (self.cli.values, self.cli.origins),
                (self.override.values, self.override.origins),
            )
        )
        for layer_values, layer_origins in layers:
            replaced = field_paths(layer_values)
            origins = {
                path: origin
                for path, origin in origins.items()
                if not any(path[: len(leaf)] == leaf for leaf in replaced)
            }
            supplied = self.schema.target_paths(layer_values)
            origins.update(
                {path: origin for path, origin in layer_origins.items() if path in supplied}
            )
        values = self.merge()
        surviving = self.schema.target_paths(values)
        return PreparedConfig(
            values,
            MappingProxyType(
                {path: origin for path, origin in origins.items() if path in surviving}
            ),
            self.override.replacements & surviving,
        )

    def provenance(self) -> Mapping[TargetPath, FieldOrigin]:
        """Return the surviving source and rule explanations."""
        return self.state().origins

    def explicit_paths(self) -> frozenset[TargetPath]:
        """Derive supplied presence from surviving provenance, including normalized sources."""
        return self.state().explicit

    def active_explicit_paths(self) -> frozenset[TargetPath]:
        """Current-run assignments that survive saved bases and higher layers."""
        return frozenset(
            path
            for path, origin in self.state().origins.items()
            if origin.explicit and origin.kind != "inherited"
        )


@dataclass(frozen=True)
class InvocationSources:
    """Admitted sparse sources whose collection operations have not executed."""

    schema: ModelConfigSpec
    layers: tuple[tuple[str, dict[str, object]], ...]
    environment_origins: Mapping[TargetPath, FieldOrigin]
    overrides: SetOverrides

    def __post_init__(self) -> None:
        # Admission is this record's invariant, not a step its factory happens to
        # run: it must precede reading any source selector and loading a restart
        # manifest from disk. Holding it here lets prepare() skip a second pass
        # over the same layers.
        for source, values in self.layers:
            self.schema.validate_layer(values, source=source)

    def selection(self) -> dict[str, object]:
        """Choose the source before loading saved values or running source effects."""
        return _source_selection(
            self.schema, tuple(values for _source, values in self.layers), self.overrides
        )

    def selection_layers(self) -> tuple[SelectionLayer, ...]:
        """Project the same selectors per layer, keeping the surface that wrote each."""
        paths = frozenset(self.schema.paths_in_group(ConfigGroup.SOURCE_SELECTOR))
        return (
            *((source, _project_values(values, paths)) for source, values in self.layers),
            ("--set", self.overrides.project(paths).config),
        )

    def restart_selection[RestartT](
        self, record: type[RestartT], *, source: str
    ) -> tuple[RestartT | None, InvocationSources]:
        """Resolve the restart selection and return the sources it leaves behind.

        An environment-only ``[restart]`` table with no ``run`` is not a restart
        request, and it is not retained intent either. The sources returned
        beside such a selection no longer carry those keys, so
        ``--print-config`` and ``--save-config`` describe the run that actually
        happens and the saved file re-runs as one layer. Keys written by any
        other layer are untouched; those already raised if they lacked a ``run``.
        """
        layers = self.selection_layers()
        selection = restart_selection(record, *layers, schema=self.schema, source=source)
        if selection is not None:
            return selection, self
        ignored = frozenset(
            key for key, label in _restart_key_surfaces(layers).items() if label == "environment"
        )
        return None, self._without_environment_restart(ignored)

    def _without_environment_restart(self, ignored: frozenset[str]) -> InvocationSources:
        """Drop the named restart keys, and their origins, from the environment layer."""
        if not ignored:
            return self
        layers = []
        for source, values in self.layers:
            restart = values.get("restart")
            if source != "environment" or not isinstance(restart, Mapping):
                layers.append((source, values))
                continue
            remaining = {name: value for name, value in values.items() if name != "restart"}
            kept = {
                name: value
                for name, value in cast(Mapping[str, object], restart).items()
                if name not in ignored
            }
            if kept:
                remaining["restart"] = kept
            layers.append((source, remaining))
        origins = {
            path: origin
            for path, origin in self.environment_origins.items()
            if not (len(path) >= 2 and path[0] == "restart" and path[1] in ignored)
        }
        return replace(self, layers=tuple(layers), environment_origins=origins)

    def prepare(
        self, saved: PreparedConfig | None = None, *, station: str = "generate"
    ) -> InvocationLayers:
        """Prepare lower layers, then execute one override stream against their saved context."""
        origins: tuple[dict[TargetPath, FieldOrigin], ...] = tuple(
            {
                path: FieldOrigin(path, "explicit", True, SourceLocation(kind, source))
                for path in field_paths(values)
            }
            for kind, (source, values) in zip(
                ("environment", "file", "cli"), self.layers, strict=True
            )
        )
        origins[0].update(self.environment_origins)
        saved_values = {} if saved is None else copy.deepcopy(saved.values)
        self.schema.validate_layer(saved_values, source="saved configuration")
        saved_paths = self.schema.target_paths(saved_values)
        if saved is not None and not (saved.explicit | saved.replacements) <= saved_paths:
            raise ConfigError("saved configuration presence does not match its declared fields")
        prepared = tuple(
            self.schema.prepare_station_layer(state, station=station)
            for state in self.schema.prepare_layers_with_provenance(
                self.layers, initial_origins=origins, admitted=True
            )
        )
        inherited = PreparedConfig(
            saved_values,
            {
                path: FieldOrigin(
                    path,
                    "inherited",
                    saved is not None and path in saved.explicit,
                    None
                    if saved is None or path not in saved.origins
                    else saved.origins[path].location,
                )
                for path in saved_paths
            },
            frozenset() if saved is None else saved.replacements,
        )
        base = _compose_states(self.schema, (prepared[0], inherited, prepared[1], prepared[2]))
        applied = self.overrides.apply_state(base, targets=self.schema.target_paths)
        own = frozenset(
            path
            for path, origin in applied.origins.items()
            if origin.location is not None and origin.location.kind == "override"
        )
        fields = frozenset(
            field_path(
                path[
                    : next(
                        (index for index, part in enumerate(path) if isinstance(part, int)),
                        len(path),
                    )
                ]
            )
            for path in own
        )
        raw = _project_values(applied.values, fields)
        contextual_origins = {
            path: origin
            for path, origin in applied.origins.items()
            if any(path[: len(field)] == field for field in fields)
        }
        decoded = self.schema.prepare_sources_with_provenance(
            (("--set", raw),), initial_origins=(contextual_origins,), layer_explicit=(own,)
        )[0]
        normalized = self.schema.normalize_prepared_layer(
            PreparedConfig(
                decoded.values, decoded.origins, applied.replacements, decoded.active_explicit
            )
        )
        normalized = self.schema.prepare_station_layer(normalized, station=station)
        environment, file, cli = (
            ConfigLayer(
                source,
                values,
                state.values,
                state.origins,
                state.replacements,
                state.active_explicit,
            )
            for (source, values), state in zip(self.layers, prepared, strict=True)
        )
        override = ConfigLayer(
            "--set",
            raw,
            normalized.values,
            normalized.origins,
            normalized.replacements,
            normalized.active_explicit,
        )
        result = InvocationLayers(
            environment, file, cli, override, self.overrides, self.schema, inherited
        )
        selected = _source_selection(
            self.schema,
            (self.layers[0][1], saved_values, *(values for _source, values in self.layers[1:])),
            self.overrides,
        )
        if _project_values(result.merge(), field_paths(selected)) != selected:
            raise ConfigError("source selectors changed during configuration preparation")
        return result


def collect_invocation_sources(
    schema: ModelConfigSpec,
    options: argparse.Namespace,
    *,
    base_settings: EnvironmentSettings | None = None,
    base_model_settings: EnvironmentSettings | None = None,
) -> InvocationSources:
    """Decode and admit sources without evaluating ordered collection operations."""
    overrides = schema.parse_overrides(options.set_overrides)
    excluded = tuple(
        name
        for name, base in (("settings", base_settings), ("model_settings", base_model_settings))
        if base is not None
    )

    environment_state = schema.environment_state(exclude_tables=excluded)
    initial_origins = dict(environment_state.origins)
    for name, base in (("settings", base_settings), ("model_settings", base_model_settings)):
        if base is None:
            continue
        presence = supplied_fields(base)
        if presence is None:
            raise ConfigError(f"API {name}: a presence-tracked configuration record is required")
        values = dataclass_to_mapping(base, omit_none=False, preserve_nested_intent=True)
        environment_state.values[name] = values
        for path in schema.target_paths({name: values}):
            explicit = len(path) > 1 and path[1] in presence
            initial_origins[path] = FieldOrigin(
                path,
                "explicit" if explicit else "default",
                explicit,
                SourceLocation("api", type(base).__name__),
            )
    raw = (
        ("environment", environment_state.values),
        (
            str(options.config) if options.config is not None else "config file",
            load_config(options.config) if options.config is not None else {},
        ),
        ("command line", schema.cli_layer(options)),
    )
    return InvocationSources(schema, raw, initial_origins, overrides)


def collect_invocation_layers(
    schema: ModelConfigSpec,
    options: argparse.Namespace,
    *,
    base_settings: EnvironmentSettings | None = None,
    base_model_settings: EnvironmentSettings | None = None,
) -> InvocationLayers:
    """Collect and prepare a fresh invocation without a saved-source selection step."""
    return collect_invocation_sources(
        schema,
        options,
        base_settings=base_settings,
        base_model_settings=base_model_settings,
    ).prepare()
