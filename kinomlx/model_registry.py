"""Global configuration registry built from every model's contribution."""

from functools import lru_cache

from kinomlx.config import ConfigRegistry, ModelConfigSpec
from kinomlx.model_catalog import MODEL_SPECS, model_choices
from kinomlx.models.flux2 import config_schema as flux2_config_schema
from kinomlx.models.gmnet import config_schema as gmnet_config_schema
from kinomlx.models.krea2 import config_schema as krea2_config_schema
from kinomlx.models.ltx2 import config_schema as ltx2_config_schema

# Each catalog model's contribution module; the registry reads CONFIG_SCHEMA when it is built.
_CONTRIBUTIONS = {
    "flux2": flux2_config_schema,
    "gmnet": gmnet_config_schema,
    "krea2": krea2_config_schema,
    "ltx2": ltx2_config_schema,
}
if _CONTRIBUTIONS.keys() != MODEL_SPECS.keys():
    raise RuntimeError("every catalog model must contribute exactly one config schema")


@lru_cache(maxsize=1)
def config_registry() -> ConfigRegistry:
    """Build the global registry from every model contribution."""
    registry = ConfigRegistry()
    for spec in MODEL_SPECS.values():
        contribution: object = getattr(_CONTRIBUTIONS[spec.name], "CONFIG_SCHEMA", None)
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
