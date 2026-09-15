"""Ordered concrete item mutations preserve record replacement and input ownership."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from kinomlx.config import ConfigError, apply_set_overrides, parse_set_argument
from kinomlx.config.paths import format_target_path, parse_target_path
from kinomlx.config_records import config_record, record_intent


@config_record
@dataclass(frozen=True)
class Layer:
    source: Path
    opacity: float = 1.0


def test_concrete_paths_keep_indices_separate_from_member_names():
    path = ("generate", "layers", 1, "opacity")
    assert parse_target_path(" generate . layers[1] . opacity ") == path
    assert format_target_path(path) == "generate.layers[1].opacity"
    assert parse_set_argument("generate.layers[1].opacity=0.0") == (list(path), 0.0)


@pytest.mark.parametrize(
    "path",
    [
        "generate.layers[-1]",
        "generate.layers[*]",
        "generate.layers[1:2]",
        "generate.layers[]",
        "generate.layers[1",
        "generate.layers[1]opacity",
        "generate..layers",
        "[1].source",
        "generate.layers[+1]",
    ],
)
def test_invalid_index_and_path_syntax_is_rejected(path):
    with pytest.raises(ConfigError, match="invalid configuration path"):
        parse_set_argument(path + "=0")


def test_leaf_patch_preserves_siblings_and_duplicate_path_items():
    base = {
        "generate": {
            "layers": [
                {"source": "same.png", "opacity": 0.25},
                {"source": "same.png", "opacity": 0.75},
            ],
            "steps": 12,
        }
    }
    original = deepcopy(base)
    result = apply_set_overrides(base, ["generate.layers[1].opacity=0"])
    assert result == {
        "generate": {
            "layers": [
                {"source": "same.png", "opacity": 0.25},
                {"source": "same.png", "opacity": 0},
            ],
            "steps": 12,
        }
    }
    assert base == original


def test_whole_item_replacement_discards_its_old_controls():
    base = {"generate": {"layers": [{"source": "old.png", "opacity": 0.25}]}}
    result = apply_set_overrides(base, ['generate.layers[0]={source="new.png"}'])
    assert result == {"generate": {"layers": [{"source": "new.png"}]}}
    assert base["generate"]["layers"][0]["opacity"] == 0.25


def test_collection_replacement_and_index_patch_obey_order():
    base = {"generate": {"layers": [{"source": "old.png", "opacity": 0.25}]}}
    replacement = 'generate.layers=[{source="new.png"}]'
    patch = "generate.layers[0].opacity=0.5"
    assert apply_set_overrides(base, [replacement, patch]) == {
        "generate": {"layers": [{"source": "new.png", "opacity": 0.5}]}
    }
    assert apply_set_overrides(base, [patch, replacement]) == {
        "generate": {"layers": [{"source": "new.png"}]}
    }
    assert apply_set_overrides(base, ["generate={layers=[],steps=7}"]) == {
        "generate": {"layers": [], "steps": 7}
    }


@pytest.mark.parametrize(
    ("base", "target", "error"),
    [
        ({}, "generate.layers[0].opacity", "not a collection"),
        ({"generate": {"layers": []}}, "generate.layers[0].opacity", "length 0"),
        ({"generate": {"layers": [{}]}}, "generate.layers[1].opacity", "length 1"),
        ({"generate": {"layers": False}}, "generate.layers[0].opacity", "not a collection"),
        (
            {"generate": {"layers": [{"source": "image.png"}]}},
            "generate.layers[0].source.name",
            "not a table",
        ),
    ],
)
def test_invalid_traversal_is_not_rescued_by_a_later_replacement(base, target, error):
    original = deepcopy(base)
    with pytest.raises(ConfigError, match=error):
        apply_set_overrides(base, [target + "=1", "generate.layers=[]"])
    assert base == original


def test_typed_record_patch_preserves_sparse_intent_and_tuple_transport():
    original = Layer(Path("image.png"))
    base = {"generate": {"layers": (original,)}}
    result = apply_set_overrides(base, ["generate.layers[0].opacity=1.0"])
    assert result == {"generate": {"layers": ({"source": Path("image.png"), "opacity": 1.0},)}}
    assert record_intent(original) == {"source": Path("image.png")}


def test_nested_collection_addresses_preserve_unmodified_groups():
    base = {"generate": {"groups": [{"layers": [{"source": "first.png"}]}]}}
    result = apply_set_overrides(base, ["generate.groups[0].layers[0].opacity=0.5"])
    assert result == {
        "generate": {"groups": [{"layers": [{"source": "first.png", "opacity": 0.5}]}]}
    }
    assert base == {"generate": {"groups": [{"layers": [{"source": "first.png"}]}]}}
