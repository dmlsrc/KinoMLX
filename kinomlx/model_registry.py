"""Model catalog and configuration contributions shared by every host surface."""

import importlib
from dataclasses import dataclass
from functools import lru_cache

from kinomlx.config import ConfigRegistry, ModelConfigSpec


@dataclass(frozen=True)
class ModelSpec:
    """One model-owned inference command and config schema."""

    name: str
    display_name: str
    help: str
    command_module: str
    command_function: str
    config_schema_module: str
    default: bool = False


MODEL_SPECS = {
    "flux2": ModelSpec(
        name="flux2",
        display_name="FLUX.2 Klein",
        help="generate Klein base, distilled, and KV still images",
        command_module="kinomlx.models.flux2.command",
        command_function="run_flux2_command",
        config_schema_module="kinomlx.models.flux2.config_schema",
    ),
    "gmnet": ModelSpec(
        name="gmnet",
        display_name="GMNet",
        help="expand one SDR still to HDR (EXR / PQ HEIC) with GMNet",
        command_module="kinomlx.models.gmnet.command",
        command_function="run_gmnet_command",
        config_schema_module="kinomlx.models.gmnet.config_schema",
    ),
    "krea2": ModelSpec(
        name="krea2",
        display_name="Krea 2",
        help="generate Krea 2 Raw or Turbo still images",
        command_module="kinomlx.models.krea2.command",
        command_function="run_krea2_command",
        config_schema_module="kinomlx.models.krea2.config_schema",
    ),
    "ltx2": ModelSpec(
        name="ltx2",
        display_name="LTX-2",
        help="generate distilled LTX-2.3/LTX-2.5 video",
        command_module="kinomlx.models.ltx2.command",
        command_function="run_ltx2_command",
        config_schema_module="kinomlx.models.ltx2.config_schema",
        default=True,
    ),
}

_DEFAULT_MODELS = tuple(name for name, spec in MODEL_SPECS.items() if spec.default)
if len(_DEFAULT_MODELS) != 1:
    raise RuntimeError("the model registry must declare exactly one default model")
DEFAULT_MODEL = _DEFAULT_MODELS[0]


def model_choices() -> tuple[str, ...]:
    """Every selectable model-owned inference command."""
    return tuple(sorted(MODEL_SPECS))


@lru_cache(maxsize=1)
def config_registry() -> ConfigRegistry:
    """Build the global registry from every model contribution."""
    registry = ConfigRegistry()
    for spec in MODEL_SPECS.values():
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
        missing_tables = [
            ".".join(table.path) for table in contribution.tables if not table.cli_generated
        ]
        if missing_tables:
            raise ValueError(
                f"{spec.name}: config tables lack generated CLI bindings: {missing_tables}"
            )
        registry.register(contribution)
    if registry.models() != model_choices():
        raise ValueError(
            "global config registry does not match selectable models: "
            f"registered={registry.models()}, selectable={model_choices()}"
        )
    registry.freeze()
    return registry
