"""Resolve TOML, environment, and CLI input into typed GMNet invocations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from kinomlx.cli._registry import config_registry
from kinomlx.config import (
    ConfigError,
    apply_set_overrides,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    load_config,
    merge_configs,
    normalize_output_selection,
    validate_top_level,
)
from kinomlx.settings import Settings

from .settings import GMNetSettings
from .types import GMNetExpandConfig, GMNetOutputConfig, GMNetRequest


@dataclass(frozen=True)
class GMNetInvocation:
    """One fully resolved GMNet invocation before output reservation."""

    settings: Settings
    model_settings: GMNetSettings
    output: GMNetOutputConfig
    request: GMNetRequest | None
    resolved_config: dict[str, object]


def _cli_layer(options: argparse.Namespace) -> dict[str, object]:
    return config_registry().model("gmnet").cli_layer(options)


def _validate_settings(settings: Settings) -> None:
    try:
        settings.validate()
    except ValueError as exc:
        raise ConfigError(f"[settings].{exc}") from exc


def _validate_model_settings(settings: GMNetSettings) -> None:
    try:
        settings.validate()
    except ValueError as exc:
        raise ConfigError(f"[model_settings].{exc}") from exc


def assemble(
    options: argparse.Namespace,
    *,
    base_settings: Settings | None = None,
    base_model_settings: GMNetSettings | None = None,
) -> GMNetInvocation:
    """Resolve defaults < env < TOML < CLI < ``--set`` exactly once."""
    source = str(options.config) if options.config is not None else "invocation"
    schema = config_registry().model("gmnet")
    file_layer = normalize_output_selection(
        schema.expand_layer_implications(
            load_config(options.config) if options.config is not None else {}
        )
    )
    cli_layer = normalize_output_selection(schema.expand_layer_implications(_cli_layer(options)))
    set_layer = normalize_output_selection(
        schema.expand_layer_implications(apply_set_overrides({}, options.set_overrides))
    )
    explicit = merge_configs(file_layer, cli_layer, set_layer)
    output_overrides = GMNetOutputConfig.overrides_from_env_fields("directory")
    output_environment = {"output": output_overrides} if output_overrides else {}
    merged = merge_configs(output_environment, file_layer, cli_layer, set_layer)
    merged = schema.normalize_config(merged)
    validate_top_level(
        merged,
        tables=schema.top_level_tables(),
        scalars=schema.top_level_scalars(),
        source=source,
    )
    model = merged.get("model", "gmnet")
    if model != "gmnet":
        raise ConfigError(f"{source}: model {model!r} does not select GMNet")

    settings_overrides = coerce_dataclass_fields(
        Settings,
        merged.get("settings", {}),
        table="[settings]",
    )
    if base_settings is None:
        try:
            base_settings = Settings.from_env()
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environment settings: {exc}") from exc
    settings = base_settings.with_overrides(**settings_overrides)
    _validate_settings(settings)

    model_overrides = coerce_dataclass_fields(
        GMNetSettings,
        merged.get("model_settings", {}),
        table="[model_settings]",
    )
    if base_model_settings is None:
        try:
            base_model_settings = GMNetSettings.from_env()
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environment model settings: {exc}") from exc
    model_settings = base_model_settings.with_overrides(**model_overrides)
    _validate_model_settings(model_settings)

    expand = dataclass_from_mapping(
        GMNetExpandConfig,
        merged.get("expand", {}),
        table="[expand]",
    )
    output = dataclass_from_mapping(
        GMNetOutputConfig,
        merged.get("output", {}),
        table="[output]",
    )
    request = None if expand.image is None else GMNetRequest(expand.image)

    resolved: dict[str, object] = {
        "model": "gmnet",
        "settings": dataclass_to_mapping(settings),
        "model_settings": dataclass_to_mapping(model_settings),
        "output": dataclass_to_mapping(output),
    }
    if request is not None or "expand" in explicit:
        resolved["expand"] = dataclass_to_mapping(expand)
    return GMNetInvocation(
        settings=settings,
        model_settings=model_settings,
        output=output,
        request=request,
        resolved_config=resolved,
    )


def validate_for_execution(invocation: GMNetInvocation) -> None:
    """Validate fields allowed to remain absent for ``--print-config``."""
    if invocation.request is None:
        raise ConfigError("GMNet needs --image <sdr still>")
    source = invocation.request.image
    if not source.is_file():
        raise ConfigError(f"[expand].image does not exist: {source}")
    if source.suffix.lower() == ".exr":
        raise ConfigError("[expand].image must be a display-referred SDR still, not an EXR")
    from .output import GMNetOutputError, plan_gmnet_output

    try:
        plan_gmnet_output(invocation.request, invocation.output)
    except GMNetOutputError as exc:
        raise ConfigError(f"[output]: {exc}") from exc


__all__ = [
    "GMNetExpandConfig",
    "GMNetInvocation",
    "assemble",
    "validate_for_execution",
]
