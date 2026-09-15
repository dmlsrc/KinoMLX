"""Real adapter declarations exercise the shared grouped CLI and indexed source engine."""

import argparse
import tomllib
from dataclasses import make_dataclass
from pathlib import Path

import pytest

from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.config import (
    ConfigFieldSpec,
    ConfigTableSpec,
    ModelConfigSpec,
    dataclass_from_mapping,
    dataclass_to_mapping,
)
from kinomlx.config.assembly import collect_invocation_layers
from kinomlx.config_record_cli import ConfigArgumentParser
from kinomlx.config_records import config_record, record_intent
from kinomlx.config_surfaces import MODEL_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.models.flux2.adapters.config_schema import ADAPTERS as FLUX_ADAPTERS
from kinomlx.models.krea2.adapters.config_schema import ADAPTERS as KREA_ADAPTERS
from kinomlx.models.ltx2.adapters.config_schema import ADAPTERS as LTX_ADAPTERS

DECLARATIONS = (KREA_ADAPTERS, FLUX_ADAPTERS, LTX_ADAPTERS)


def configuration(field):
    owner = config_record(
        make_dataclass(
            "Generate", [("adapters", tuple[field.item_schema.record, ...], ())], frozen=True
        )
    )
    schema = ModelConfigSpec(
        model="fixture",
        title="Adapter contribution test",
        introduction=("Test only.",),
        restart_stations=(),
        no_restart_reason="The adapter contribution fixture has no saved station.",
        root_fields=(
            ConfigFieldSpec(
                "model",
                "Select model",
                "",
                "fixture",
                "model",
                persistence="intent",
                virtual=True,
                default="fixture",
                choices=("fixture",),
                surfaces=MODEL_SURFACES,
                cli=MODEL_CONTROL.binding,
            ),
        ),
        tables=(ConfigTableSpec(("generate",), owner, (field,), cli_generated=True),),
    )
    parser = ConfigArgumentParser()
    add_config_arguments(parser, schema)
    parser.add_argument("--unrelated")
    parser.add_argument("--set", dest="set_overrides", action="append")
    parser.add_argument("--config", type=Path)
    return schema, parser


def parse(field, arguments):
    schema, parser = configuration(field)
    layer = schema.cli_layer(parser.parse_args(arguments))
    return dataclass_from_mapping(schema.tables[0].record, layer["generate"], table="generate")


@pytest.mark.parametrize("field", DECLARATIONS)
def test_grouped_shorthand_is_sparse_and_local_even_for_repeated_paths(field):
    config = parse(
        field,
        [
            "--adapter",
            "same.safetensors:0.5",
            "--unrelated",
            "value",
            "--adapter-exclude",
            "attn",
            "--adapter",
            "same.safetensors",
        ],
    )
    assert [record_intent(item) for item in config.adapters] == [
        {"path": Path("same.safetensors"), "strength": 0.5, "exclude": ("attn",)},
        {"path": Path("same.safetensors")},
    ]
    expanded = parse(field, ["--adapter", "same.safetensors", "--adapter-strength", "0.5"])
    assert record_intent(expanded.adapters[0]) == {
        "path": Path("same.safetensors"),
        "strength": 0.5,
    }


@pytest.mark.parametrize("field", DECLARATIONS)
@pytest.mark.parametrize("exists", [False, True])
def test_shorthand_does_not_depend_on_file_existence(field, exists, tmp_path):
    literal = tmp_path / "style.safetensors:0.5"
    if exists:
        literal.touch()
    config = parse(field, ["--adapter", str(literal)])
    assert config.adapters[0].path == tmp_path / "style.safetensors"
    assert config.adapters[0].strength == 0.5


@pytest.mark.parametrize("field", DECLARATIONS)
def test_nonnumeric_suffix_remains_literal(field):
    config = parse(field, ["--adapter", "style.safetensors:latest"])
    assert record_intent(config.adapters[0]) == {"path": Path("style.safetensors:latest")}


@pytest.mark.parametrize("field", DECLARATIONS)
@pytest.mark.parametrize(
    "text", ["a.safetensors:1e999", "a.safetensors:nan", "a.safetensors:1:2:3"]
)
def test_invalid_numeric_shorthand_fails_without_any_files(field, text, capsys):
    with pytest.raises(SystemExit):
        parse(field, ["--adapter", text])
    error = capsys.readouterr().err
    assert "finite" in error or "numeric suffix" in error


@pytest.mark.parametrize("field", DECLARATIONS)
@pytest.mark.parametrize(
    "arguments",
    [
        ["--adapter-strength", "0.5"],
        ["--adapter", "a.safetensors:0.5", "--adapter-strength", "0.5"],
        ["--adapter", "a", "--adapter-strength", "0.5", "--lora-strength", "0.5"],
        ["--adapter", "a", "--adapter-exclude", "attn", "--adapter-exclude", "attn"],
    ],
)
def test_orphan_and_duplicate_modifiers_fail_in_shared_record_action(field, arguments, capsys):
    with pytest.raises(SystemExit):
        parse(field, arguments)
    error = capsys.readouterr().err
    assert "requires a preceding" in error or "duplicate member" in error


def test_ltx_two_number_shorthand_sets_only_stages_on_its_own_item():
    config = parse(
        LTX_ADAPTERS,
        [
            "--adapter",
            "a.safetensors:0:0.5",
            "--adapter",
            "b.safetensors:0.8",
            "--adapter",
            "c.safetensors",
        ],
    )
    assert [record_intent(item) for item in config.adapters] == [
        {"path": Path("a.safetensors"), "stage1_strength": 0.0, "stage2_strength": 0.5},
        {"path": Path("b.safetensors"), "strength": 0.8},
        {"path": Path("c.safetensors")},
    ]


@pytest.mark.parametrize("field", [KREA_ADAPTERS, FLUX_ADAPTERS])
def test_scheduled_and_constant_siblings_have_independent_controls(field):
    config = parse(
        field,
        [
            "--adapter",
            "a",
            "--adapter-strength",
            "0.8",
            "--adapter",
            "b",
            "--adapter-step-strengths",
            "1,0.8,0",
        ],
    )
    first, second = [item.selection() for item in config.adapters]
    assert first.strength == 0.8
    assert first.schedule is None
    assert second.schedule.per_step == (1.0, 0.8, 0.0)
    assert "strength" not in record_intent(config.adapters[1])


@pytest.mark.parametrize("field", [KREA_ADAPTERS, FLUX_ADAPTERS])
def test_two_number_shorthand_is_not_an_image_model_capability(field, capsys):
    with pytest.raises(SystemExit):
        parse(field, ["--adapter", "a.safetensors:0.5:0.8"])
    assert "one numeric suffix" in capsys.readouterr().err


def test_inherited_namespace_items_do_not_authorize_orphan_modifiers(capsys):
    _, parser = configuration(KREA_ADAPTERS)
    namespace = argparse.Namespace(adapters=[{"path": Path("saved")}])
    with pytest.raises(SystemExit):
        parser.parse_args(["--adapter-strength", "0.5"], namespace=namespace)
    assert "requires a preceding" in capsys.readouterr().err


@pytest.mark.parametrize("field", DECLARATIONS)
def test_indexed_patch_uses_real_declared_owners_and_preserves_other_item_members(field):
    schema, parser = configuration(field)
    args = parser.parse_args(
        [
            "--adapter",
            "a.safetensors:0.5",
            "--adapter",
            "b.safetensors",
            "--set",
            "generate.adapters[1].strength=0.25",
        ]
    )
    inputs = collect_invocation_layers(schema, args)
    merged = inputs.merge()
    config = dataclass_from_mapping(schema.tables[0].record, merged["generate"], table="generate")
    assert [record_intent(item) for item in config.adapters] == [
        {"path": Path("a.safetensors"), "strength": 0.5},
        {"path": Path("b.safetensors"), "strength": 0.25},
    ]


@pytest.mark.parametrize("field", DECLARATIONS)
def test_reloadable_projection_preserves_nested_intent_and_root_values(field):
    config = parse(
        field,
        ["--adapter", "a.safetensors", "--adapter-strength", "0.5", "--adapter", "b.safetensors"],
    )
    schema, _ = configuration(field)
    projection = dataclass_to_mapping(config, preserve_nested_intent=True)
    assert projection == {
        "adapters": (
            {"path": Path("a.safetensors"), "strength": 0.5},
            {"path": Path("b.safetensors")},
        )
    }
    document = schema.dump_config({"model": "fixture", "generate": projection})
    reloaded = dataclass_from_mapping(
        type(config), tomllib.loads(document)["generate"], table="generate"
    )
    assert [record_intent(item) for item in reloaded.adapters] == [
        record_intent(item) for item in config.adapters
    ]
    # Full observation projection is still available and does not claim input presence.
    assert dataclass_to_mapping(config)["adapters"][1]["strength"] == 1.0
    assert dataclass_to_mapping(type(config)(), preserve_nested_intent=True) == {"adapters": ()}


@pytest.mark.parametrize("field", [KREA_ADAPTERS, FLUX_ADAPTERS])
def test_scheduled_record_export_does_not_manufacture_a_strength_conflict(field):
    config = parse(field, ["--adapter", "a", "--adapter-step-strengths", "1,0"])
    schema, _ = configuration(field)
    document = schema.dump_config(
        {"generate": dataclass_to_mapping(config, preserve_nested_intent=True)}
    )
    reloaded = dataclass_from_mapping(
        type(config), tomllib.loads(document)["generate"], table="generate"
    )
    assert reloaded.adapters[0].selection().schedule.per_step == (1.0, 0.0)
    assert record_intent(reloaded.adapters[0]) == {"path": Path("a"), "step_strengths": (1.0, 0.0)}
