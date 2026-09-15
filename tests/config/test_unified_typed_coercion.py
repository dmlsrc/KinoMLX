"""Direct typed inputs and structured values share one coercion path."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from kinomlx.config import ConfigError, dataclass_from_mapping
from kinomlx.config_records import config_record, record_intent, replace_config, supplied_fields
from kinomlx.models.flux2.settings import Flux2Settings
from kinomlx.models.gmnet.settings import GMNetSettings
from kinomlx.models.krea2.settings import Krea2Settings
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.settings import Settings

type Names = str | Sequence[str]


@config_record
@dataclass(frozen=True)
class Item:
    path: Path
    enabled: bool = False


@config_record
@dataclass(frozen=True)
class Request:
    names: Names
    items: tuple[Item, ...] = ()
    count: int = 0


def test_direct_and_mapping_inputs_produce_equal_typed_records_and_presence():
    raw = {
        "names": ["first", "second"],
        "items": [{"path": "image.png", "enabled": False}],
        "count": 0,
    }
    direct = Request(**raw)
    mapped = dataclass_from_mapping(Request, raw, table="request")
    assert direct == mapped
    assert direct.names == ("first", "second")
    assert direct.items == (Item(Path("image.png"), enabled=False),)
    assert record_intent(direct) == record_intent(mapped)
    assert supplied_fields(direct.items[0]) == frozenset({"path", "enabled"})


@pytest.mark.parametrize(
    "values",
    [
        {"count": True},
        {"count": 1.5},
        {"items": [{"path": "x", "enabled": "false"}]},
        {"names": ["first", 2]},
        {"names": b"bytes"},
    ],
)
def test_direct_and_mapping_inputs_reject_the_same_bad_shapes(values):
    raw = {"names": "first", **values}
    with pytest.raises(ConfigError):
        Request(**raw)
    with pytest.raises(ConfigError):
        dataclass_from_mapping(Request, raw, table="request")


def test_controlled_copy_coerces_new_values_without_marking_omitted_fields():
    original = Request(names="first")
    updated = replace_config(original, items=[{"path": "image.png"}])
    assert updated.items == (Item(Path("image.png")),)
    assert supplied_fields(updated) == frozenset({"names", "items"})
    assert supplied_fields(updated.items[0]) == frozenset({"path"})


@pytest.mark.parametrize(
    ("record", "name"),
    [
        (Settings, "hf_home"),
        (Flux2Settings, "source_path"),
        (Krea2Settings, "vae_path"),
        (LTX2Settings, "weights_path"),
        (GMNetSettings, "weights_path"),
    ],
)
def test_all_model_api_path_fields_use_shared_coercion(record, name):
    assert getattr(record(**{name: "local/source"}), name) == Path("local/source")
    with pytest.raises(ConfigError, match=name):
        record(**{name: 42})


def test_invalid_primitive_is_rejected_before_post_init():
    events = []

    @config_record
    @dataclass(frozen=True)
    class Observed:
        enabled: bool

        def __post_init__(self):
            events.append(self.enabled)

    with pytest.raises(ConfigError, match="enabled"):
        Observed(enabled="false")
    assert events == []
    assert Observed(enabled=False).enabled is False
    assert events == [False]
