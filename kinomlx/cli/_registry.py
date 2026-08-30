"""Lightweight model CLI, command, and runner registry."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

from kinomlx.config import (
    ConfigError,
    ConfigRegistry,
    ModelConfigSpec,
    load_config,
    parse_set_argument,
)

if TYPE_CHECKING:
    import mlx.core as mx

    from kinomlx.media.frames import CloseableVideoFrameStream
    from kinomlx.media.signals import VideoSignalSpec


class _RuntimeGeneration(Protocol):
    """Generation products used by the installed orchestration boundary."""

    @property
    def frames(self) -> CloseableVideoFrameStream: ...

    @property
    def audio_waveform(self) -> mx.array | None: ...

    @property
    def audio_sample_rate(self) -> int | None: ...

    @property
    def signal(self) -> VideoSignalSpec: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def metadata(self) -> dict[str, object]: ...

    def runtime_diagnostics(self) -> dict[str, object]: ...

    def close(self) -> None: ...


class _ObjectCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _Recipe(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _RuntimeGeneration: ...


class _Runner(Protocol):
    """Uniform host operations required from a generation model runner."""

    def run(self, recipe: _Recipe, request: object) -> _RuntimeGeneration: ...

    def restart(self, request: object, restart: object) -> _RuntimeGeneration: ...


class _ArtifactOptions(Protocol):
    @property
    def save_latents(self) -> bool | None: ...

    @property
    def save_media_conditioning(self) -> bool | None: ...

    @property
    def save_text_conditioning(self) -> bool | None: ...


class _ArtifactContribution(Protocol):
    """Model-contributed artifact vocabulary used by the generic CLI host."""

    def sidecar_paths(self, video_path: Path) -> dict[str, Path]: ...

    def requested_artifacts(
        self,
        options: _ArtifactOptions,
        *,
        save_all: bool = False,
        has_media_conditioning: bool = False,
    ) -> frozenset[str]: ...

    def restart_artifacts(
        self,
        requested: frozenset[str],
        *,
        phase: str,
    ) -> frozenset[str]: ...


class _CLIContribution(Protocol):
    def add_arguments(
        self,
        parser: argparse.ArgumentParser,
        settings_type: type[object],
        config_schema: ModelConfigSpec,
    ) -> None: ...


@dataclass(frozen=True)
class ModalModelSpec:
    """One model whose modality does not fit the flat generation grammar.

    ``--model <name>`` still selects it, but the invocation is routed to the
    model's own complete parser instead of the generation assembly - video
    generation keeps the default grammar, other modalities (still expansion,
    future audio or restoration tools) own theirs. ``module``/``function``
    name a lazily imported ``Callable[[list[str]], int]`` receiving the full
    argument list, so model discovery stays runtime-light.
    """

    name: str
    help: str
    module: str
    function: str
    config_schema_module: str


MODAL_MODEL_SPECS = {
    "gmnet": ModalModelSpec(
        name="gmnet",
        help="expand one SDR still to HDR (EXR / PQ HEIC) with GMNet",
        module="kinomlx.models.gmnet.cli",
        function="run_gmnet_command",
        config_schema_module="kinomlx.models.gmnet.config_schema",
    ),
}


@dataclass(frozen=True)
class ModelSelection:
    """The model selected by config, ordinary CLI flags, and ``--set``."""

    name: str
    explicit: bool
    source: str


def model_choices() -> tuple[str, ...]:
    """Every selectable ``--model`` value, generation and modal alike."""
    return tuple(sorted({*MODEL_SPECS, *MODAL_MODEL_SPECS}))


def _bootstrap_options(arguments: list[str]) -> argparse.Namespace:
    """Parse only model-selection inputs while leaving model flags untouched."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="set_overrides", action="append", default=None)
    try:
        options, _unknown = parser.parse_known_args(arguments)
    except argparse.ArgumentError as exc:
        raise ConfigError(f"model selection: {exc}") from exc
    return options


def resolve_model_selection(
    arguments: list[str],
    *,
    tolerate_errors: bool = False,
) -> ModelSelection:
    """Resolve model precedence without importing a model or native runtime.

    Selection follows the same precedence as a complete invocation: config,
    then ``--model``, then ordered ``--set model=...`` overrides. Absence of a
    selector preserves the compatible LTX-2 default.
    """
    try:
        options = _bootstrap_options(arguments)
    except ConfigError:
        if tolerate_errors:
            return ModelSelection(name="ltx2", explicit=False, source="default")
        raise
    selected: object = None
    explicit = False
    source = "default"
    if options.config is not None:
        try:
            file_config = load_config(options.config)
        except ConfigError:
            if not tolerate_errors:
                raise
        else:
            if "model" in file_config:
                selected = file_config["model"]
                explicit = True
                source = f"config {options.config}"
    if options.model is not None:
        selected = options.model
        explicit = True
        source = "--model"
    for argument in options.set_overrides or []:
        try:
            path, value = parse_set_argument(argument)
        except ConfigError:
            if tolerate_errors:
                continue
            raise
        if path == ["model"]:
            selected = value
            explicit = True
            source = f"--set {argument!r}"
    if selected is None:
        selected = "ltx2"
    if not isinstance(selected, str):
        if tolerate_errors:
            return ModelSelection(name="ltx2", explicit=False, source="default")
        raise ConfigError(f"{source}: model must be a string, got {type(selected).__name__}")
    if selected not in model_choices():
        if tolerate_errors:
            return ModelSelection(name="ltx2", explicit=False, source="default")
        supported = ", ".join(model_choices())
        raise ConfigError(f"{source}: unknown model {selected!r}; expected one of {supported}")
    return ModelSelection(name=selected, explicit=explicit, source=source)


def resolve_modal_model(
    selection: ModelSelection | str,
) -> Callable[[list[str]], int] | None:
    """Return a modal model's command, or ``None`` for the generation family."""
    name = selection.name if isinstance(selection, ModelSelection) else selection
    spec = MODAL_MODEL_SPECS.get(name)
    if spec is None:
        return None
    return cast(Callable[[list[str]], int], _resolve_callable(spec.module, spec.function))


def modal_model_summary() -> str:
    """One help line enumerating the modal models behind ``--model``."""
    rendered = "; ".join(
        f"'--model {spec.name}' - {spec.help}" for spec in MODAL_MODEL_SPECS.values()
    )
    return (
        f"Modal models with their own argument set: {rendered}. "
        "See 'kinomlx --model <name> --help'. Generate a model-specific TOML "
        "starter with 'kinomlx config init --model <name>'."
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    help: str
    cli_module: str
    settings_module: str
    settings_class: str
    runner_module: str
    runner_class: str
    recipe_module: str
    recipe_function: str
    restart_recipe_module: str
    restart_request_class: str
    artifact_module: str
    config_schema_module: str


MODEL_SPECS = {
    "ltx2": ModelSpec(
        name="ltx2",
        help="Distilled LTX-2.3 text/image-to-video",
        cli_module="kinomlx.models.ltx2.cli",
        settings_module="kinomlx.models.ltx2.settings",
        settings_class="LTX2Settings",
        runner_module="kinomlx.models.ltx2.runner",
        runner_class="LTX2Runner",
        recipe_module="kinomlx.models.ltx2.pipelines.distilled",
        recipe_function="generate_distilled",
        restart_recipe_module="kinomlx.models.ltx2.pipelines.restart",
        restart_request_class="DistilledRestart",
        artifact_module="kinomlx.models.ltx2.artifacts",
        config_schema_module="kinomlx.models.ltx2.config_schema",
    )
}


def _load_cli_contribution(spec: ModelSpec) -> _CLIContribution:
    """Import one lightweight CLI contribution through Python's module cache."""
    return cast(_CLIContribution, importlib.import_module(spec.cli_module))


def _load_settings_contribution(spec: ModelSpec) -> ModuleType:
    """Import one model settings record through its lazy parent package."""
    return importlib.import_module(spec.settings_module)


@lru_cache(maxsize=1)
def config_registry() -> ConfigRegistry:
    """Build the global registry from every model's lightweight contribution."""
    registry = ConfigRegistry()
    specs: tuple[ModelSpec | ModalModelSpec, ...] = (
        *MODEL_SPECS.values(),
        *MODAL_MODEL_SPECS.values(),
    )
    for spec in specs:
        try:
            module = importlib.import_module(spec.config_schema_module)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"model {spec.name!r} config contribution: {exc}") from exc
        contribution: object = getattr(module, "CONFIG_SCHEMA", None)
        if not isinstance(contribution, ModelConfigSpec):
            raise TypeError(
                f"model {spec.name!r} config contribution must expose ModelConfigSpec "
                "as CONFIG_SCHEMA"
            )
        if contribution.model != spec.name:
            raise ValueError(f"model {spec.name!r} contributed schema for {contribution.model!r}")
        registry.register(contribution)
    if registry.models() != model_choices():
        raise ValueError(
            "global config registry does not match selectable models: "
            f"registered={registry.models()}, selectable={model_choices()}"
        )
    registry.freeze()
    return registry


CONFIG_CONTROL_DESTINATIONS = frozenset(
    {
        "config",
        "help",
        "only_non_defaults",
        "print_config",
        "save_config",
        "set_overrides",
    }
)


def validate_model_parser(parser: argparse.ArgumentParser, model: str) -> None:
    """Fail when a complete public parser drifts from its model contribution."""
    schema = config_registry().model(model)
    actions: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if action.dest in CONFIG_CONTROL_DESTINATIONS:
            continue
        actions.setdefault(action.dest, []).append(action)

    expected = schema.cli_destinations()
    actual = frozenset(actions)
    if expected != actual:
        missing = sorted(expected - actual)
        unregistered = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing parser destinations: " + ", ".join(missing))
        if unregistered:
            details.append("unregistered parser destinations: " + ", ".join(unregistered))
        raise ValueError(f"{model} config/CLI drift: {'; '.join(details)}")

    for destination, field in schema.cli_fields().items():
        for action in actions[destination]:
            negative_only = isinstance(action, argparse._StoreFalseAction) or (
                bool(action.option_strings)
                and all(option.startswith("--no-") for option in action.option_strings)
            )
            accepted_help = {field.help_text(negative=negative_only)}
            if not negative_only:
                for implication in schema.implications:
                    if schema.field(implication.trigger).cli_dest == destination:
                        accepted_help.add(schema.implication_help(implication.trigger))
            if action.help not in accepted_help:
                raise ValueError(
                    f"{model} config/CLI help drift for {destination}: "
                    f"registered={sorted(accepted_help)}, parser={action.help!r}"
                )
        non_null_defaults = [
            action.default for action in actions[destination] if action.default is not None
        ]
        if non_null_defaults:
            raise ValueError(
                f"{model} config/CLI default drift for {destination}: parser defaults "
                "must be None so lower-precedence config sources remain visible"
            )
        declared = [
            tuple(str(choice) for choice in action.choices)
            for action in actions[destination]
            if action.choices is not None
        ]
        if not declared:
            if field.choices:
                raise ValueError(
                    f"{model} config/CLI choice drift for {destination}: "
                    f"registered={sorted(field.choices)}, parser has no choices"
                )
            continue
        registered = (
            frozenset(config_registry().models())
            if destination == "model"
            else frozenset(field.choices)
        )
        for choices in declared:
            parsed = frozenset(choices)
            if registered != parsed:
                raise ValueError(
                    f"{model} config/CLI choice drift for {destination}: "
                    f"registered={sorted(registered)}, parser={sorted(parsed)}"
                )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Load lightweight model contributions and add their argument groups."""
    for spec in MODEL_SPECS.values():
        module = _load_cli_contribution(spec)
        settings_module = _load_settings_contribution(spec)
        settings_type: object = getattr(settings_module, spec.settings_class)
        if not isinstance(settings_type, type):
            raise TypeError(f"model settings contribution {spec.settings_class!r} is not a class")
        module.add_arguments(parser, settings_type, config_registry().model(spec.name))


def _resolve_callable(module_name: str, attribute: str) -> _ObjectCallable:
    module = importlib.import_module(module_name)
    value: object = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"{module_name}.{attribute} is not callable")
    return cast(_ObjectCallable, value)


def create_runner(model: str, **kwargs: object) -> _Runner:
    """Construct the selected model runner after argument resolution."""
    spec = MODEL_SPECS[model]
    constructor = _resolve_callable(spec.runner_module, spec.runner_class)
    return cast(_Runner, constructor(**kwargs))


def resolve_recipe(model: str) -> _Recipe:
    """Resolve the selected public recipe after invocation assembly."""
    spec = MODEL_SPECS[model]
    return cast(_Recipe, _resolve_callable(spec.recipe_module, spec.recipe_function))


def create_restart_request(model: str, **kwargs: object) -> object:
    """Construct one model-owned typed restart selection."""
    spec = MODEL_SPECS[model]
    constructor = _resolve_callable(spec.restart_recipe_module, spec.restart_request_class)
    return constructor(**kwargs)


def resolve_artifact_contribution(model: str) -> _ArtifactContribution:
    """Resolve the selected model's names, paths, and output selections."""
    module = importlib.import_module(MODEL_SPECS[model].artifact_module)
    return cast(_ArtifactContribution, module)


__all__ = [
    "CONFIG_CONTROL_DESTINATIONS",
    "MODAL_MODEL_SPECS",
    "MODEL_SPECS",
    "ModalModelSpec",
    "ModelSelection",
    "ModelSpec",
    "add_model_arguments",
    "config_registry",
    "create_restart_request",
    "create_runner",
    "modal_model_summary",
    "model_choices",
    "resolve_artifact_contribution",
    "resolve_modal_model",
    "resolve_model_selection",
    "resolve_recipe",
    "validate_model_parser",
]
