"""Resolve TOML, environment, and CLI input into typed GMNet invocations."""

import argparse
from dataclasses import dataclass

from kinomlx.cli.resource_operation import Operation, refuse_resource_only_work
from kinomlx.config import (
    ConfigError,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    validate_top_level,
)
from kinomlx.config.assembly import collect_invocation_layers
from kinomlx.config.paths import TargetPath
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config_rules import validate_record
from kinomlx.model_registry import config_registry
from kinomlx.settings import Settings

from .output import GMNetOutputError, plan_gmnet_output
from .settings import GMNetSettings
from .types import GMNetExpandConfig, GMNetOutputConfig, GMNetRequest


@dataclass(frozen=True)
class GMNetInvocation:
    """One fully resolved GMNet invocation before output reservation."""

    settings: Settings
    model_settings: GMNetSettings
    operation: Operation
    output: GMNetOutputConfig
    request: GMNetRequest | None
    resolved_config: dict[str, object]
    config_intent: PreparedConfig

    @property
    def explicit_config_paths(self) -> frozenset[TargetPath]:
        """Supplied presence comes from retained intent, never resolved-value equality."""
        return self.config_intent.explicit


def assemble(
    options: argparse.Namespace,
    *,
    base_settings: Settings | None = None,
    base_model_settings: GMNetSettings | None = None,
) -> GMNetInvocation:
    """Resolve defaults < env < TOML < CLI < ``--set`` exactly once."""
    source = str(options.config) if options.config is not None else "invocation"
    schema = config_registry().model("gmnet")
    inputs = collect_invocation_layers(
        schema, options, base_settings=base_settings, base_model_settings=base_model_settings
    )
    intent = inputs.state()
    explicit = intent.values
    merged = schema.normalize_config(intent.values)
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
        base_settings = Settings()
    settings = base_settings.with_overrides(**settings_overrides)
    settings.validate()

    model_overrides = coerce_dataclass_fields(
        GMNetSettings,
        merged.get("model_settings", {}),
        table="[model_settings]",
    )
    if base_model_settings is None:
        base_model_settings = GMNetSettings()
    model_settings = base_model_settings.with_overrides(**model_overrides)
    model_settings.validate()

    expand = dataclass_from_mapping(
        GMNetExpandConfig,
        merged.get("expand", {}),
        table="[expand]",
    )
    operation = dataclass_from_mapping(
        Operation,
        merged.get("operation", {}),
        table="[operation]",
    )
    output = dataclass_from_mapping(
        GMNetOutputConfig,
        merged.get("output", {}),
        table="[output]",
    )
    validate_record(output, location="[output]")
    refuse_resource_only_work(
        dry_run=operation.dry_run,
        requested=expand.image is not None,
        work="an expansion",
    )
    request = None if expand.image is None else GMNetRequest(expand.image)

    resolved: dict[str, object] = {
        "model": "gmnet",
        "settings": dataclass_to_mapping(settings),
        "model_settings": dataclass_to_mapping(model_settings),
        "operation": dataclass_to_mapping(operation),
        "output": dataclass_to_mapping(output),
    }
    if request is not None or "expand" in explicit:
        resolved["expand"] = dataclass_to_mapping(expand)
    return GMNetInvocation(
        settings=settings,
        model_settings=model_settings,
        operation=operation,
        output=output,
        request=request,
        resolved_config=resolved,
        config_intent=intent,
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

    try:
        plan_gmnet_output(invocation.request, invocation.output)
    except GMNetOutputError as exc:
        raise ConfigError(f"[output]: {exc}") from exc


__all__ = [
    "GMNetInvocation",
    "assemble",
    "validate_for_execution",
]
