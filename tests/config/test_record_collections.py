"""Typed and structured record collections share coercion and serialization."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kinomlx.config import ConfigError, dataclass_from_mapping, dataclass_to_mapping, dump_config


@dataclass(frozen=True)
class Layer:
    source: Path
    opacity: float = 1.0
    label: str | None = None


@dataclass(frozen=True)
class Layers:
    layers: tuple[Layer, ...] = ()
    groups: tuple[tuple[Layer, ...], ...] = ()
    counts: tuple[int, ...] = (1, 2)
    alternatives: list[Layer] = field(default_factory=list)


def test_record_tuple_round_trips_toml_preserving_order_duplicates_and_absence():
    value = Layers(layers=(Layer(Path("same.png")), Layer(Path("same.png"), 0.0)))
    mapping = dataclass_to_mapping(value)
    assert mapping["layers"] == (
        {"source": Path("same.png"), "opacity": 1.0},
        {"source": Path("same.png"), "opacity": 0.0},
    )
    assert mapping["counts"] == (1, 2)
    restored = dataclass_from_mapping(Layers, tomllib.loads(dump_config(mapping)), table="layers")
    assert restored == value


def test_nested_record_sequences_and_optional_fields_project_recursively():
    item = Layer(Path("source.png"))
    value = Layers(groups=((item,), ()), alternatives=[item])
    mapping = dataclass_to_mapping(value, omit_none=False)
    assert mapping["groups"] == (({"source": item.source, "opacity": 1.0, "label": None},), ())
    assert mapping["alternatives"] == [{"source": item.source, "opacity": 1.0, "label": None}]
    assert dataclass_to_mapping(Layers())["layers"] == ()


def test_typed_and_mapping_items_reach_the_same_coercion_boundary():
    typed = dataclass_from_mapping(
        Layers, {"layers": (Layer(Path("source.png"), 0.5),)}, table="generate"
    )
    structured = dataclass_from_mapping(
        Layers, {"layers": [{"source": "source.png", "opacity": 0.5}]}, table="generate"
    )
    assert typed == structured
    assert dataclass_from_mapping(Layers, typed, table="generate") == structured


@pytest.mark.parametrize("typed", [True, False])
def test_bad_typed_member_cannot_bypass_primitive_admission(typed):
    item = Layer(Path("source.png"), True) if typed else {"source": "source.png", "opacity": True}
    with pytest.raises(ConfigError, match=r"generate.layers\[0\].opacity: expected float"):
        dataclass_from_mapping(Layers, {"layers": [item]}, table="generate")


def test_typed_input_ignores_computed_members_when_reconstructing():
    @dataclass(frozen=True)
    class Count:
        count: int = 1
        doubled: int = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "doubled", self.count * 2)

    assert dataclass_from_mapping(Count, Count(3), table="count") == Count(3)
