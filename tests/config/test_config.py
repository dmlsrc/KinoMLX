"""Generic config composition, override, schema, and dump contracts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from kinomlx.config import (
    ConfigError,
    apply_set_overrides,
    dataclass_from_mapping,
    dump_config,
    dump_toml_value,
    load_config,
    merge_configs,
    parse_set_argument,
    validate_top_level,
)


def test_recursive_merge_preserves_tables_and_replaces_arrays() -> None:
    base = {
        "generate": {"width": 1024, "image": {"path": "a.png", "strength": 0.5}},
        "items": [1, 2],
    }
    overlay = {
        "generate": {"image": {"strength": 0.9}},
        "items": [3],
    }
    merged = merge_configs(base, overlay)
    assert merged == {
        "generate": {"width": 1024, "image": {"path": "a.png", "strength": 0.9}},
        "items": [3],
    }
    assert base["generate"]["image"]["strength"] == 0.5


def test_load_config_wraps_file_and_toml_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(missing)

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[broken\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(invalid)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("model=gmnet", (["model"], "gmnet")),
        ("generate.seed=7", (["generate", "seed"], 7)),
        ("generate.audio=true", (["generate", "audio"], True)),
        ("generate.scale=0.5", (["generate", "scale"], 0.5)),
        ('generate.tags=["a", "b"]', (["generate", "tags"], ["a", "b"])),
        ("generate.prompt=plain text", (["generate", "prompt"], "plain text")),
        ('generate.pin="42"', (["generate", "pin"], "42")),
    ],
)
def test_set_values_use_toml_types(raw: str, expected: tuple[list[str], object]) -> None:
    assert parse_set_argument(raw) == expected


def test_set_applies_in_order_and_copies_nested_tables() -> None:
    base = {"generate": {"image": {"path": "a.png", "strength": 0.5}}}
    final = apply_set_overrides(
        base,
        ["generate.image.strength=0.7", "generate.image.strength=0.9"],
    )
    assert final["generate"]["image"]["strength"] == 0.9
    assert base["generate"]["image"]["strength"] == 0.5


def test_set_can_replace_a_root_scalar() -> None:
    assert apply_set_overrides({"model": "ltx2"}, ["model=gmnet"]) == {"model": "gmnet"}


def test_set_rejects_malformed_or_scalar_paths() -> None:
    with pytest.raises(ConfigError, match="expected"):
        parse_set_argument("generate.seed")
    with pytest.raises(ConfigError, match="is not a table"):
        apply_set_overrides({"generate": 1}, ["generate.seed=4"])


def test_dump_round_trips_nested_values_and_paths() -> None:
    config = {
        "model": "ltx2",
        "generate": {
            "prompt": "line one\nline two",
            "image": {"path": Path("frame.png"), "strength": 0.9},
        },
        "output": {"path": Path("out.mp4"), "quality": 0.65},
    }
    rendered = dump_config(config)
    assert "[generate.image]" in rendered
    assert "image = {" not in rendered
    parsed = tomllib.loads(rendered)
    assert parsed == {
        "model": "ltx2",
        "generate": {
            "prompt": "line one\nline two",
            "image": {"path": "frame.png", "strength": 0.9},
        },
        "output": {"path": "out.mp4", "quality": 0.65},
    }


def test_dump_omits_empty_tables_instead_of_looking_truncated() -> None:
    rendered = dump_config(
        {
            "model": "ltx2",
            "generate": {"prompt": "test", "image": {}},
            "model_artifacts": {},
        }
    )

    assert "[generate]" in rendered
    assert "[generate.image]" not in rendered
    assert "[model_artifacts]" not in rendered


def test_generic_toml_dump_refuses_out_of_range_integers() -> None:
    assert dump_toml_value(-(2**63)) == str(-(2**63))
    assert dump_toml_value(2**63 - 1) == str(2**63 - 1)
    with pytest.raises(ConfigError, match="signed 64-bit"):
        dump_toml_value(2**63)


@dataclass(frozen=True)
class _Schema:
    count: int = 1
    path: Path | None = None


@dataclass(frozen=True)
class _TupleSchema:
    fixed: tuple[int, str] = (1, "one")
    repeated: tuple[int, ...] = ()


def test_dataclass_schema_coerces_paths_and_rejects_bool_as_int() -> None:
    value = dataclass_from_mapping(_Schema, {"path": "~/x"}, table="[sample]")
    assert value.path == Path("~/x").expanduser()
    with pytest.raises(ConfigError, match="expected int"):
        dataclass_from_mapping(_Schema, {"count": True}, table="[sample]")


def test_dataclass_schema_rejects_non_tables_and_non_string_keys() -> None:
    with pytest.raises(ConfigError, match="expected table, got int"):
        dataclass_from_mapping(_Schema, 1, table="[sample]")
    with pytest.raises(ConfigError, match="table keys must be strings"):
        dataclass_from_mapping(_Schema, {1: "one"}, table="[sample]")


def test_schema_unknown_field_suggests_close_name() -> None:
    with pytest.raises(ConfigError, match=r"did you mean 'count'.*known: count, path"):
        dataclass_from_mapping(_Schema, {"cont": 2}, table="[sample]")


def test_tuple_schema_validates_fixed_and_repeated_item_types() -> None:
    value = dataclass_from_mapping(
        _TupleSchema,
        {"fixed": [2, "two"], "repeated": [3, 4]},
        table="[sample]",
    )
    assert value == _TupleSchema(fixed=(2, "two"), repeated=(3, 4))
    with pytest.raises(ConfigError, match="expected 2 items"):
        dataclass_from_mapping(_TupleSchema, {"fixed": [1]}, table="[sample]")
    with pytest.raises(ConfigError, match=r"fixed\[1\]: expected string"):
        dataclass_from_mapping(_TupleSchema, {"fixed": [1, 2]}, table="[sample]")


def test_top_level_schema_rejects_unknown_and_non_table_values() -> None:
    with pytest.raises(ConfigError, match="unknown field 'generat'"):
        validate_top_level({"generat": {}}, tables={"generate"})
    with pytest.raises(ConfigError, match=r"\[generate\] must be a table"):
        validate_top_level({"generate": 3}, tables={"generate"})
