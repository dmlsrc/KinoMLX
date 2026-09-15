"""Generic TOML composition, typed overrides, validation, and dumping."""

from kinomlx.config_bindings import CliBinding
from kinomlx.config_values import (
    ConfigError,
    coerce_dataclass_fields,
    dataclass_from_mapping,
    dataclass_to_mapping,
    validate_top_level,
)

from .dump import dump_config, dump_toml_value
from .load import compose_config, load_config
from .merge import merge_configs
from .overrides import SetOverrides, apply_set_overrides, parse_set_argument
from .registry import (
    ConfigFieldSpec,
    ConfigGroup,
    ConfigImplication,
    ConfigPath,
    ConfigRecordSpec,
    ConfigRegistry,
    ConfigTableSpec,
    ModelConfigSpec,
    as_sentence,
)

__all__ = [
    "CliBinding",
    "ConfigError",
    "ConfigFieldSpec",
    "ConfigGroup",
    "ConfigImplication",
    "ConfigPath",
    "ConfigRecordSpec",
    "ConfigRegistry",
    "ConfigTableSpec",
    "ModelConfigSpec",
    "SetOverrides",
    "apply_set_overrides",
    "as_sentence",
    "coerce_dataclass_fields",
    "compose_config",
    "dataclass_from_mapping",
    "dataclass_to_mapping",
    "dump_config",
    "dump_toml_value",
    "load_config",
    "merge_configs",
    "parse_set_argument",
    "validate_top_level",
]
