"""LTX scalar geometry is checked before source selection or derivation."""

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_constraints import NumberConstraint
from kinomlx.config_records import supplied_fields
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble
from kinomlx.models.ltx2.types import LTX2Request


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("width", 0),
        ("width", 1000),
        ("height", 63),
        ("frames", 0),
        ("frames", 8),
        ("fps", 0),
        ("fps", float("inf")),
        ("duration", 0),
        ("duration", float("nan")),
        ("seed", -1),
        ("seed", 2**64),
        ("seeds", ()),
        ("seeds", (-1,)),
        ("generated_keyframes", -1),
    ],
)
def test_invalid_scalar_geometry_fails_at_constructor_admission(name, value):
    with pytest.raises(ValueError, match=name):
        LTX2Request(**{name: value})


def test_valid_duration_derivation_keeps_intent_and_frame_grid():
    request = LTX2Request(duration=6, fps=24, width=64, height=64, seed=2**64 - 1)
    assert request.frames == 145
    assert supplied_fields(request) == frozenset({"duration", "fps", "width", "height", "seed"})


def test_malformed_lower_geometry_cannot_be_hidden_by_cli(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[generate]\nwidth = 1000\n")
    options = build_parser().parse_args(
        ["--config", str(config), "--width", "1024", "--print-config"]
    )
    with pytest.raises(ConfigError, match="width"):
        assemble(options)


def test_common_constructor_keeps_ltx_nested_adapter_presence():
    request = LTX2Request(adapters=[{"path": "adapter.safetensors", "strength": 1.0}])
    assert request.adapters[0].path.name == "adapter.safetensors"
    assert supplied_fields(request.adapters[0]) == frozenset({"path", "strength"})


@pytest.mark.parametrize("value", [-1, 0, float("inf"), float("nan"), True])
def test_strict_lower_numeric_bound_rejects_zero_and_nonfinite_values(value):
    with pytest.raises(ValueError, match="must"):
        NumberConstraint(minimum=0, exclusive_minimum=True).validate(value)


def test_strict_lower_bound_admits_positive_values_and_describes_its_contract():
    constraint = NumberConstraint(minimum=0, maximum=1, exclusive_minimum=True)
    constraint.validate(1e-12)
    constraint.validate(1)
    assert constraint.describe()["exclusive_minimum"] is True
    with pytest.raises(ValueError, match="between"):
        constraint.validate(2)


@pytest.mark.parametrize("bounds", [{}, {"minimum": 0, "maximum": 0}])
def test_strict_lower_bound_rejects_missing_or_empty_ranges(bounds):
    with pytest.raises(ValueError, match="exclusive minimum"):
        NumberConstraint(**bounds, exclusive_minimum=True)
