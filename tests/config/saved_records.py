"""Current-format synthetic run configuration for station tests."""

from kinomlx.config.persistence import RUN_SCHEMA_VERSION, configuration_payload
from kinomlx.config.provenance import PreparedConfig


def saved_run_configuration(model: str, values: dict[str, object]) -> dict[str, object]:
    """Build a replay context with explicit test values and no user-source assignments."""
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "configuration": configuration_payload(model, PreparedConfig(values, {})),
        "effective_config": values,
    }
