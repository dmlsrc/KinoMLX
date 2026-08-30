"""Generic TOML composition, typed overrides, validation, and dumping."""

from .dump import dump_config, dump_toml_value
from .load import compose_config, load_config
from .merge import merge_configs, normalize_output_selection
from .overrides import apply_set_overrides, parse_set_argument
from .registry import (
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigPath,
    ConfigRegistry,
    ConfigTableSpec,
    ModelConfigSpec,
    as_sentence,
)
from .validate import (
    ConfigError,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    validate_top_level,
)

__all__ = [
    "ConfigError",
    "ConfigFieldSpec",
    "ConfigGroup",
    "ConfigImplication",
    "ConfigPath",
    "ConfigRegistry",
    "ConfigTableSpec",
    "ModelConfigSpec",
    "as_sentence",
    "apply_set_overrides",
    "coerce_dataclass_fields",
    "compose_config",
    "dataclass_from_mapping",
    "dataclass_to_mapping",
    "dump_config",
    "dump_toml_value",
    "load_config",
    "merge_configs",
    "normalize_output_selection",
    "parse_set_argument",
    "validate_top_level",
]
