"""GMNet's no-restart, direct recipe, and request-presence contracts."""

import argparse
from pathlib import Path

import pytest

from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.config import dataclass_from_mapping
from kinomlx.config_records import record_intent, supplied_fields, tracks_presence
from kinomlx.models.gmnet.config_schema import CONFIG_SCHEMA
from kinomlx.models.gmnet.pipelines import expand_gmnet
from kinomlx.models.gmnet.resources import prepare_resources
from kinomlx.models.gmnet.types import GMNetExpandConfig, GMNetRequest


def test_gmnet_explicitly_has_no_saved_restart_stage():
    assert CONFIG_SCHEMA.restart_stages == ()
    assert CONFIG_SCHEMA.no_restart_reason
    assert all(not table.stages for table in CONFIG_SCHEMA.tables)
    assert all(not field.stage_policies for table in CONFIG_SCHEMA.tables for field in table.fields)
    variables = CONFIG_SCHEMA.environment_variables()
    assert variables[("output", "path")][0] == "KINO_GMNET_OUTPUT_PATH"
    assert variables[("output", "directory")][0] == "KINO_GMNET_OUTPUT_DIR"
    assert variables[("output", "prefix")][0] == "KINO_GMNET_OUTPUT_PREFIX"


def test_public_gmnet_request_tracks_input_presence_and_matches_config_constraints():
    assert tracks_presence(GMNetRequest)
    direct = GMNetRequest(Path("still.png"))
    assert supplied_fields(direct) == frozenset({"image"})
    assert record_intent(direct) == {"image": Path("still.png")}
    with pytest.raises(ValueError, match="must end in"):
        GMNetRequest(Path("linear.exr"))
    with pytest.raises(ValueError, match="must end in"):
        GMNetExpandConfig(image=Path("linear.exr"))


def test_cli_and_direct_gmnet_inputs_use_the_same_typed_owner():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, CONFIG_SCHEMA)
    layer = CONFIG_SCHEMA.cli_layer(parser.parse_args(["--image", "still.png"]))
    config = dataclass_from_mapping(GMNetExpandConfig, layer["expand"], table="[expand]")
    assert config.image == GMNetRequest(Path("still.png")).image
    with pytest.raises(ValueError, match="must end in"):
        dataclass_from_mapping(
            GMNetExpandConfig,
            {"image": Path("linear.exr")},
            table="[expand]",
        )


def test_gmnet_direct_public_operations_have_declared_owners():
    operations = {boundary.operation for boundary in CONFIG_SCHEMA.api_boundaries}
    assert {prepare_resources, expand_gmnet} <= operations
    for boundary in CONFIG_SCHEMA.api_boundaries:
        boundary.validate_tracked_projections()
