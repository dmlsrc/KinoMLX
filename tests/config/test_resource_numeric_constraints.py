"""Resource bounds are enforced by their typed owner on every supplied layer."""

import importlib

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_values import dataclass_from_mapping
from kinomlx.model_registry import config_registry
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import Settings


@pytest.mark.parametrize("name", ["mlx_cache_limit_gb", "mlx_wired_limit_gb"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_memory_limits_reject_invalid_api_and_structured_values(name, value):
    with pytest.raises(ValueError, match=name):
        Settings(**{name: value})
    with pytest.raises(ValueError, match=name):
        dataclass_from_mapping(Settings, {name: value}, table="settings")


@pytest.mark.parametrize("value", [None, 0, 0.5, 1])
def test_memory_limit_absence_zero_and_fractional_gib_remain_distinct(value):
    settings = Settings(mlx_cache_limit_gb=value, mlx_wired_limit_gb=value)
    assert settings.mlx_cache_limit_gb == value
    assert settings.mlx_wired_limit_gb == value


@pytest.mark.parametrize("name", ["transformer_resident_blocks", "compile_block_group_size"])
@pytest.mark.parametrize("value", [0, 49, 1.5, True])
def test_bounded_block_counts_reject_invalid_api_values(name, value):
    with pytest.raises(ValueError, match=name):
        LTX2Settings(**{name: value})


@pytest.mark.parametrize(
    "name",
    ["video_ff_quantize_group_size", "video_ff_quantize_bits", "transformer_compile_group_size"],
)
@pytest.mark.parametrize("value", [0, 1.5, True])
def test_positive_integer_controls_reject_invalid_api_values(name, value):
    with pytest.raises(ValueError, match=name):
        LTX2Settings(**{name: value})


@pytest.mark.parametrize("value", [(0, 0), (-1,), (48,), (0.5,), (True,)])
def test_layer_indices_share_range_type_and_uniqueness_admission(value):
    with pytest.raises(ValueError, match="video_ff_layout_layers"):
        LTX2Settings(video_ff_layout_layers=value)
    with pytest.raises(ValueError, match="video_ff_layout_layers"):
        dataclass_from_mapping(
            LTX2Settings, {"video_ff_layout_layers": value}, table="model_settings"
        )


def test_layer_order_and_boundary_indices_survive():
    assert LTX2Settings(video_ff_layout_layers=(47, 0, 23)).video_ff_layout_layers == (47, 0, 23)


@pytest.mark.parametrize("model", ["ltx2", "krea2", "flux2", "gmnet"])
def test_invalid_memory_layer_cannot_be_hidden_by_a_later_valid_override(tmp_path, model):
    path = tmp_path / "config.toml"
    path.write_text("[settings]\nmlx_cache_limit_gb = -1\n")
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    assemble = importlib.import_module(f"kinomlx.models.{model}.config").assemble
    with pytest.raises(ConfigError, match="mlx_cache_limit_gb"):
        assemble(
            parser.parse_args(
                ["--config", str(path), "--mlx-cache-limit-gb", "1", "--print-config"]
            )
        )


def test_resource_constraints_are_visible_through_schema_inspection():
    schema = config_registry().model("ltx2")
    table = next(table for table in schema.tables if table.path == ("model_settings",))
    field = next(field for field in table.fields if field.name == "transformer_resident_blocks")
    assert table.local_constraints(field)[0].describe() == {
        "kind": "integer",
        "finite": True,
        "minimum": 1,
        "maximum": 48,
    }
