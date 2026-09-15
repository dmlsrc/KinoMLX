"""A non-adapter record contribution uses the real configuration surfaces."""

import argparse
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.config import (
    CliBinding,
    ConfigError,
    ConfigFieldSpec,
    ConfigRecordSpec,
    ConfigTableSpec,
    ModelConfigSpec,
    dataclass_from_mapping,
)
from kinomlx.config.assembly import collect_invocation_layers, collect_invocation_sources
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config.provenance import FieldOrigin, PreparedConfig, SourceLocation
from kinomlx.config_constraints import NumberConstraint
from kinomlx.config_record_cli import ConfigArgumentParser
from kinomlx.config_records import config_record, record_intent
from kinomlx.config_surfaces import CONFIG_SURFACES, MODEL_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.settings import parse_config_value


@config_record
@dataclass(frozen=True)
class Layer:
    source: Path
    opacity: float = field(default=1.0, metadata={"config_constraints": (NumberConstraint(),)})


@config_record
@dataclass(frozen=True)
class Generate:
    layers: tuple[Layer, ...] = ()


def layer_schema():
    return ConfigRecordSpec(
        Layer,
        (
            ConfigFieldSpec(
                "source",
                "Select source",
                "a path",
                Path("image.png"),
                "source",
                persistence="intent",
                surfaces=CONFIG_SURFACES,
            ),
            ConfigFieldSpec(
                "opacity",
                "Set opacity",
                "a finite number",
                1.0,
                "opacity",
                persistence="intent",
                surfaces=CONFIG_SURFACES,
            ),
        ),
    )


def fixture_schema():
    return ModelConfigSpec(
        model="fixture",
        title="Fixture configuration",
        introduction=("Synthetic record contribution.",),
        restart_stations=(),
        no_restart_reason="The synthetic record contribution has no saved station.",
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
        tables=(
            ConfigTableSpec(
                ("generate",),
                Generate,
                (
                    ConfigFieldSpec(
                        "layers",
                        "Select layers",
                        "an ordered record collection",
                        (),
                        "layers",
                        persistence="intent",
                        item_schema=layer_schema(),
                        cli=CliBinding("Layers", parse=parse_config_value),
                        surfaces=CONFIG_SURFACES,
                    ),
                ),
                cli_generated=True,
            ),
        ),
    )


def test_record_declaration_reaches_existing_cli_env_toml_api_and_whole_set(monkeypatch):
    schema = fixture_schema()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, schema)
    text = '[{source="same.png"},{source="same.png",opacity=0.0}]'
    monkeypatch.setenv("KINO_FIXTURE_LAYERS", text)
    cli = schema.cli_layer(parser.parse_args(["--layers", text]))
    env = schema.environment_config()
    toml = tomllib.loads("[generate]\nlayers=" + text)
    overrides = schema.parse_overrides(["generate.layers=" + text]).config
    api = {"generate": {"layers": (Layer(Path("same.png")), Layer(Path("same.png"), 0.0))}}
    expected = {
        "layers": ({"source": Path("same.png")}, {"source": Path("same.png"), "opacity": 0.0})
    }
    for raw in (cli, env, toml, overrides, api):
        admitted = schema.prepare_layer(raw)
        typed = dataclass_from_mapping(Generate, admitted["generate"], table="generate")
        assert record_intent(typed) == expected
        restored = tomllib.loads(schema.dump_config({"generate": record_intent(typed)}))
        assert (
            record_intent(dataclass_from_mapping(Generate, restored["generate"], table="generate"))
            == expected
        )
    assert schema.cli_layer(parser.parse_args([])) == {}
    assert "--layers" in parser.format_help()


def test_record_inventory_reflects_member_ownership_and_required_path():
    item = configuration_inventory(fixture_schema(), field="generate.layers")["fields"][0]
    record = item["item_schema"]
    assert record["owner"].endswith(".Layer")
    assert record["constructor_presence"] == "tracked"
    assert [(field["name"], field["default_kind"]) for field in record["members"]] == [
        ("source", "required"),
        ("opacity", "value"),
    ]
    assert record["members"][1]["default_toml"] == "1.0"


def test_member_coverage_and_collection_annotation_drift_fail_at_registration():
    record = layer_schema()
    with pytest.raises(ValueError, match="coverage drift"):
        replace(record, fields=record.fields[:1])
    with pytest.raises(ValueError, match="repeats a member"):
        replace(record, fields=record.fields + record.fields[:1])
    with pytest.raises(ValueError, match="annotation must contain Layer"):
        record.validate_collection_annotation(tuple[str, ...])
    with pytest.raises(ValueError, match="annotation must contain Layer"):
        record.validate_collection_annotation(tuple[Layer, Layer])
    record.validate_collection_annotation(list[Layer])
    record.validate_collection_annotation(tuple[Layer, ...])


def test_untracked_records_and_unimplemented_member_policies_are_rejected():
    @dataclass(frozen=True)
    class Untracked:
        source: Path
        opacity: float = 1.0

    record = layer_schema()
    with pytest.raises(ValueError, match="presence tracking"):
        replace(record, record=Untracked)
    with pytest.raises(ValueError, match="policy is not implemented"):
        replace(record, fields=(replace(record.fields[0], strip=True), record.fields[1]))


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ('[{source="image.png",opcaity=1.0}]', "unknown field"),
        ('[{source="image.png",opacity=true}]', "expected float"),
        ('[{source="image.png",opacity=inf}]', "must be finite"),
        ('["image.png"]', "expected table"),
    ],
)
def test_invalid_member_source_is_not_hidden_by_collection_replacement(value, error):
    with pytest.raises(ConfigError, match=error):
        fixture_schema().parse_overrides(["generate.layers=" + value, "generate.layers=[]"])


def test_record_member_choices_are_admitted_from_the_owned_declaration():
    @config_record
    @dataclass(frozen=True)
    class Blend:
        mode: str = "normal"

    record = ConfigRecordSpec(
        Blend,
        (
            ConfigFieldSpec(
                "mode",
                "Choose blend",
                "",
                "normal",
                "mode",
                persistence="intent",
                choices=("normal", "screen"),
                choice_help=(("normal", "Use normal blending"), ("screen", "Use screen blending")),
                surfaces=CONFIG_SURFACES,
            ),
        ),
    )
    record.validate_members([{"mode": "screen"}], location="generate.layers")
    with pytest.raises(ConfigError, match=r"layers\[0\].mode.*expected one of"):
        record.validate_members([{"mode": "typo"}], location="generate.layers")


def test_collection_members_cannot_skip_their_owned_schema():
    schema = fixture_schema()
    table = schema.tables[0]
    with pytest.raises(ValueError, match="missing its item schema"):
        replace(table, fields=(replace(table.fields[0], item_schema=None),))


def test_invalid_record_defaults_fail_at_registration():
    @config_record
    @dataclass(frozen=True)
    class Invalid:
        opacity: float = True

    with pytest.raises(ConfigError, match="expected float"):
        ConfigRecordSpec(Invalid, (layer_schema().fields[1],))


def test_typed_record_defaults_reach_inspection_and_toml_without_manual_projection():
    @config_record
    @dataclass(frozen=True)
    class Defaults:
        layers: tuple[Layer, ...] = (Layer(Path("default.png")),)

    schema = fixture_schema()
    schema = replace(schema, tables=(replace(schema.tables[0], record=Defaults),))
    defaults = tomllib.loads(schema.dump_config(schema.default_config()))
    assert defaults["generate"]["layers"] == [{"source": "default.png", "opacity": 1.0}]
    item = configuration_inventory(schema, field="generate.layers")["fields"][0]
    assert (
        tomllib.loads("layers=" + item["default_toml"])["layers"] == defaults["generate"]["layers"]
    )
    direct = schema.dump_config({"generate": {"layers": (Layer(Path("input.png"), 0.0),)}})
    assert tomllib.loads(direct)["generate"]["layers"] == [{"source": "input.png", "opacity": 0.0}]


def grouped_schema():
    schema = fixture_schema()
    table = schema.tables[0]
    record = layer_schema()
    record = replace(
        record,
        fields=(
            record.fields[0],
            replace(
                record.fields[1],
                cli=CliBinding("Layers", flags=("--layer-opacity", "--opacity")),
            ),
        ),
    )
    collection = replace(
        table.fields[0],
        item_schema=record,
        cli_item="source",
        cli=CliBinding("Layers", flags=("--layer",)),
    )
    return replace(schema, tables=(replace(table, fields=(collection,)),))


def grouped_parser(schema=None):
    schema = grouped_schema() if schema is None else schema
    parser = ConfigArgumentParser()
    add_config_arguments(parser, schema)
    return parser


def test_grouped_item_and_member_flags_produce_sparse_ordered_records():
    schema = grouped_schema()
    parser = grouped_parser(schema)
    options = parser.parse_args(
        [
            "--layer",
            "same.png",
            "--layer-opacity",
            "0",
            "--layer",
            "same.png",
            "--layer",
            "last.png",
            "--opacity",
            "1",
        ]
    )
    raw = schema.cli_layer(options)
    typed = dataclass_from_mapping(
        Generate, schema.prepare_layer(raw)["generate"], table="generate"
    )
    assert record_intent(typed) == {
        "layers": (
            {"source": Path("same.png"), "opacity": 0.0},
            {"source": Path("same.png")},
            {"source": Path("last.png"), "opacity": 1.0},
        )
    }
    assert "--layer SOURCE" in parser.format_help()
    assert "--layer-opacity OPACITY" in parser.format_help()
    assert schema.cli_layer(parser.parse_args([])) == {}
    fresh = schema.cli_layer(parser.parse_args(["--layer", "fresh.png"]))
    assert fresh == {"generate": {"layers": [{"source": Path("fresh.png")}]}}


@pytest.mark.parametrize(
    "flags",
    [
        ["--layer-opacity", "1"],
        ["--layer", "first.png", "--layer-opacity", "1", "--opacity", "1"],
    ],
)
def test_orphan_and_duplicate_modifiers_fail_even_for_equal_values(flags, capsys):
    with pytest.raises(SystemExit):
        grouped_parser().parse_args(flags)
    error = capsys.readouterr().err
    assert any(message in error for message in ("preceding --layer", "duplicate member"))


def test_inherited_namespace_items_do_not_satisfy_a_local_modifier(capsys):
    namespace = argparse.Namespace(layers=[{"source": Path("inherited.png")}])
    with pytest.raises(SystemExit):
        grouped_parser().parse_args(["--layer-opacity", "0.5"], namespace=namespace)
    assert "preceding --layer" in capsys.readouterr().err
    assert namespace.layers == [{"source": Path("inherited.png")}]
    result = grouped_parser().parse_args(["--layer", "new.png"], namespace=namespace)
    assert result.layers == [{"source": Path("new.png")}]


def test_item_codec_assignments_share_duplicate_detection(capsys):
    def decode(text):
        source, separator, opacity = text.partition("@")
        return {"source": source, "opacity": float(opacity)} if separator else {"source": source}

    schema = grouped_schema()
    table = schema.tables[0]
    collection = replace(
        table.fields[0], cli=CliBinding("Layers", flags=("--layer",), parse=decode)
    )
    schema = replace(schema, tables=(replace(table, fields=(collection,)),))
    parser = grouped_parser(schema)
    assert parser.parse_args(["--layer", "first.png@0.5"]).layers == [
        {"source": Path("first.png"), "opacity": 0.5}
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["--layer", "first.png@0.5", "--opacity", "0.5"])
    assert "duplicate member assignment: opacity" in capsys.readouterr().err


def test_grouped_inspection_and_drift_checks_consume_the_same_declarations():
    schema = grouped_schema()
    parser = grouped_parser(schema)
    table = schema.tables[0]
    actions = [action for action in parser._actions if action.dest == "layers"]
    table.validate_cli_actions(table.fields[0], actions)
    entry = configuration_inventory(schema, field="generate.layers")["fields"][0]
    assert entry["cli"]["flags"] == ["--layer", "--layer-opacity", "--opacity"]
    assert entry["cli"]["action"] == "record_items"
    assert entry["cli"]["item_member"] == "source"
    assert entry["item_schema"]["members"][1]["cli_flags"] == ["--layer-opacity", "--opacity"]
    actions[1].help = "unowned help"
    with pytest.raises(ValueError, match="record binding drift"):
        table.validate_cli_actions(table.fields[0], actions)


def test_grouped_binding_registration_rejects_unowned_targets_and_flags():
    schema = grouped_schema()
    table = schema.tables[0]
    collection = table.fields[0]
    with pytest.raises(ValueError, match="unknown primary"):
        replace(table, fields=(replace(collection, cli_item="typo"),))
    record = collection.item_schema
    duplicate = replace(
        record,
        fields=(
            record.fields[0],
            replace(
                record.fields[1],
                cli=CliBinding("Layers", flags=("--layer",)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="duplicate record CLI flags"):
        replace(table, fields=(replace(collection, item_schema=duplicate),))
    with pytest.raises(ValueError, match="require an item token"):
        replace(table, fields=(replace(collection, cli_item=None),))


def test_reused_namespace_cannot_supply_a_current_item_to_a_new_parse(capsys):
    parser = grouped_parser()
    namespace = parser.parse_args(["--layer", "first.png"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--layer-opacity", "1"], namespace=namespace)
    assert "preceding --layer" in capsys.readouterr().err
    assert namespace.layers == [{"source": Path("first.png")}]
    parser.parse_args(["--layer", "second.png"], namespace=namespace)
    assert namespace.layers == [{"source": Path("second.png")}]


def test_grouped_arguments_require_the_owned_parser_lifecycle():
    with pytest.raises(TypeError, match="require ConfigArgumentParser"):
        add_config_arguments(argparse.ArgumentParser(), grouped_schema())


def test_concrete_targets_resolve_to_the_same_registered_member_owner():
    schema = fixture_schema()
    collection = schema.resolve_target(("generate", "layers"))
    item = schema.resolve_target(("generate", "layers", 1))
    member = schema.resolve_target(("generate", "layers", 1, "opacity"))
    assert collection.kind == "field"
    assert item.kind == "item"
    assert member.kind == "field"
    assert member.owner is item.owner is collection.field.item_schema
    assert member.declaration_path == ("generate", "layers", "opacity")
    assert member.path == ("generate", "layers", 1, "opacity")
    assert member.field is item.owner.fields[1]
    assert schema.resolve_target(("generate",)).kind == "group"
    assert schema.resolve_target(("model",)).owner is None


@pytest.mark.parametrize(
    "path",
    [
        ("generate", "layers", "opacity"),
        ("generate", "layers", 1, "missing"),
        ("generate", "layers", 1, 0),
        ("generate", "layers", 1, "opacity", 0),
        ("generate", "layers", -1),
        ("generate", "layers", True),
        ("generate", 0),
        (),
    ],
)
def test_target_resolution_rejects_unowned_or_nonrecord_traversal(path):
    with pytest.raises(ConfigError):
        fixture_schema().resolve_target(path)


def test_indexed_source_admission_uses_owned_constraints_without_materializing_an_item():
    schema = fixture_schema()
    # Bounds belong to ordered execution; admission must not invent preceding items.
    schema.validate_assignment(("generate", "layers", 10, "opacity"), 0.0)
    schema.validate_assignment(("generate", "layers", 10), {"source": "image.png"})
    with pytest.raises(ConfigError, match="finite"):
        schema.parse_overrides(["generate.layers[1].opacity=nan", "generate.layers[1].opacity=0.5"])
    with pytest.raises(ConfigError, match="opacity"):
        schema.validate_assignment(("generate", "layers", 1, "opacity"), True)
    with pytest.raises(ConfigError, match="unknown"):
        schema.validate_assignment(("generate", "layers", 1), {"unknown": 0.0})


def test_atomic_item_replacement_requires_its_own_required_members():
    with pytest.raises(ConfigError, match="missing required members: source"):
        fixture_schema().parse_overrides(
            ["generate.layers[1]={opacity=0.5}", 'generate.layers[1]={source="image.png"}']
        )


def inherited_layers(schema):
    values = {
        "generate": {
            "layers": [
                {"source": "same.png", "opacity": 0.2},
                {"source": "same.png", "opacity": 0.4},
            ]
        }
    }
    origins = {
        path: FieldOrigin(path, "inherited", False, SourceLocation("saved", "parent"))
        for path in schema.target_paths(values)
    }
    return PreparedConfig(values, origins)


def test_indexed_patch_preserves_inherited_sibling_origins_and_sparse_explicitness():
    schema = fixture_schema()
    base = inherited_layers(schema)
    path = ("generate", "layers", 1, "opacity")
    result = schema.parse_overrides(["generate.layers[1].opacity=0.0"]).apply_state(
        base, targets=schema.target_paths
    )
    assert result.explicit == {path}
    assert result.origins[path].location.position == 1
    for sibling in base.origins.keys() - {path}:
        assert result.origins[sibling] is base.origins[sibling]
    assert base.values["generate"]["layers"][1]["opacity"] == 0.4
    assert result.values["generate"]["layers"][1]["opacity"] == 0.0


def test_atomic_item_replacement_erases_old_members_and_retains_replacement_identity():
    schema = fixture_schema()
    base = inherited_layers(schema)
    result = schema.parse_overrides(['generate.layers[1]={source="new.png"}']).apply_state(
        base, targets=schema.target_paths
    )
    item = ("generate", "layers", 1)
    assert result.explicit == {item, (*item, "source")}
    assert (*item, "opacity") not in result.origins
    assert item in result.replacements
    assert result.values["generate"]["layers"][1] == {"source": "new.png"}
    assert (
        result.origins[("generate", "layers", 0, "opacity")]
        is base.origins[("generate", "layers", 0, "opacity")]
    )


def test_collection_replacement_then_patch_keeps_distinct_surviving_assignment_origins():
    schema = fixture_schema()
    base = inherited_layers(schema)
    result = schema.parse_overrides(
        [
            'generate={layers=[{source="new.png"}]}',
            "generate.layers[0].opacity=0.0",
        ]
    ).apply_state(base, targets=schema.target_paths)
    collection = ("generate", "layers")
    assert result.origins[(*collection, 0, "source")].location.position == 1
    assert result.origins[(*collection, 0, "opacity")].location.position == 2
    assert collection in result.replacements
    assert not any(path[:3] == (*collection, 1) for path in result.origins)
    reversed_result = schema.parse_overrides(
        [
            "generate.layers[0].opacity=0.0",
            'generate={layers=[{source="new.png"}]}',
        ]
    ).apply_state(base, targets=schema.target_paths)
    assert (*collection, 0, "opacity") not in reversed_result.origins
    assert (*collection, 0, "opacity") not in reversed_result.replacements


def test_typed_record_target_projection_uses_supplied_members_only():
    schema = fixture_schema()
    values = {"generate": {"layers": (Layer(Path("one.png")), Layer(Path("two.png"), 1.0))}}
    paths = schema.target_paths(values)
    assert ("generate", "layers", 0, "opacity") not in paths
    assert ("generate", "layers", 1, "opacity") in paths


def test_invocation_sources_keep_member_origins_when_collections_replace(tmp_path, monkeypatch):
    schema = fixture_schema()
    monkeypatch.setenv("KINO_FIXTURE_LAYERS", '[{source="env.png",opacity=0.1}]')
    config = tmp_path / "run.toml"
    config.write_text('[generate]\nlayers=[{source="file.png"}]\n')
    options = argparse.Namespace(config=config, set_overrides=None)
    parent = inherited_layers(schema)
    layers = collect_invocation_sources(schema, options).prepare(PreparedConfig(parent.values, {}))
    result = layers.state()
    assert result.values["generate"]["layers"] == [{"source": "file.png"}]
    source = ("generate", "layers", 0, "source")
    assert result.origins[source].location.kind == "file"
    assert result.origins[source].path == source
    assert ("generate", "layers", 0, "opacity") not in result.origins
    assert ("generate", "layers", 1) not in result.origins
    assert layers.environment.origins[source].location.kind == "environment"
    assert layers.environment.origins[source].location.identifier == "KINO_FIXTURE_LAYERS"


def test_source_collection_does_not_execute_an_index_before_saved_state_is_loaded():
    options = argparse.Namespace(config=None, set_overrides=["generate.layers[99].opacity=0.5"])
    sources = collect_invocation_sources(fixture_schema(), options)
    assert sources.selection() == {}
    assert sources.overrides.assignments[0].path == ("generate", "layers", 99, "opacity")
    with pytest.raises(ConfigError, match="finite"):
        collect_invocation_sources(
            fixture_schema(),
            argparse.Namespace(config=None, set_overrides=["generate.layers[99].opacity=nan"]),
        )


@pytest.mark.parametrize("source", ["environment", "file", "cli", "saved"])
def test_contextual_indexed_invocation_uses_the_selected_lower_collection(
    source, tmp_path, monkeypatch
):
    schema = fixture_schema()
    config = tmp_path / "run.toml"
    values = {"generate": {"layers": [{"source": "one.png", "opacity": 0.2}]}}
    config.write_text('[generate]\nlayers=[{source="one.png",opacity=0.2}]\n')
    if source == "environment":
        monkeypatch.setenv("KINO_FIXTURE_LAYERS", '[{source="one.png",opacity=0.2}]')
    options = argparse.Namespace(
        config=config if source == "file" else None,
        layers=values["generate"]["layers"] if source == "cli" else None,
        set_overrides=["generate.layers[0].opacity=0.5"],
    )
    saved = values if source == "saved" else {}
    sources = collect_invocation_sources(schema, options)
    inputs = sources.prepare(PreparedConfig(saved, {}))
    result = inputs.state()
    member = ("generate", "layers", 0, "opacity")
    sibling = ("generate", "layers", 0, "source")
    assert list(result.values["generate"]["layers"]) == [{"source": "one.png", "opacity": 0.5}]
    assert inputs.override.explicit == {member}
    assert result.origins[member].location.kind == "override"
    if source == "saved":
        assert not result.origins[sibling].explicit
        assert result.origins[sibling].kind == "inherited"
    else:
        assert result.origins[sibling].explicit
        assert result.origins[sibling].location.kind == source
    assert inputs.override.origins[sibling] == result.origins[sibling]


def test_contextual_invocation_replacement_order_and_bounds_use_saved_state():
    schema = fixture_schema()
    saved = inherited_layers(schema).values
    arguments = [
        "generate.layers[1].opacity=0.0",
        'generate.layers[0]={source="replacement.png"}',
    ]
    inputs = collect_invocation_sources(
        schema, argparse.Namespace(config=None, set_overrides=arguments)
    ).prepare(PreparedConfig(saved, {}))
    result = inputs.state()
    assert result.values["generate"]["layers"] == [
        {"source": "replacement.png"},
        {"source": "same.png", "opacity": 0.0},
    ]
    assert ("generate", "layers", 0, "opacity") not in result.origins
    assert ("generate", "layers", 0) in result.replacements
    with pytest.raises(ConfigError, match="outside collection"):
        collect_invocation_sources(
            schema,
            argparse.Namespace(
                config=None,
                set_overrides=[
                    "generate.layers=[]",
                    "generate.layers[0].opacity=0.5",
                    'generate.layers=[{source="later.png"}]',
                ],
            ),
        ).prepare(PreparedConfig(saved, {}))


def test_grouped_cli_and_indexed_override_share_one_context_and_roundtrip():
    schema = grouped_schema()
    options = grouped_parser(schema).parse_args(
        [
            "--layer",
            "same.png",
            "--layer-opacity",
            "0.2",
            "--layer",
            "same.png",
        ]
    )
    options.config = None
    options.set_overrides = ["generate.layers[1].opacity=0.0"]
    inputs = collect_invocation_layers(schema, options)
    result = inputs.state()
    assert result.values["generate"]["layers"] == [
        {"source": Path("same.png"), "opacity": 0.2},
        {"source": Path("same.png"), "opacity": 0.0},
    ]
    assert inputs.override.explicit == {("generate", "layers", 1, "opacity")}
    restored = dataclass_from_mapping(
        Generate, tomllib.loads(schema.dump_config(result.values))["generate"], table="generate"
    )
    assert [record_intent(item) for item in restored.layers] == result.values["generate"]["layers"]


def test_collection_replacement_removes_earlier_indexed_override_presence():
    schema = fixture_schema()
    saved = inherited_layers(schema).values
    sources = collect_invocation_sources(
        schema,
        argparse.Namespace(
            config=None,
            set_overrides=[
                "generate.layers[1].opacity=0.0",
                'generate={layers=[{source="new.png"}]}',
            ],
        ),
    )
    inputs = sources.prepare(PreparedConfig(saved, {}))
    result = inputs.state()
    assert result.values["generate"]["layers"] == [{"source": "new.png"}]
    assert not any(path[:3] == ("generate", "layers", 1) for path in result.origins)
    assert ("generate", "layers", 0, "opacity") not in inputs.override.explicit
    assert ("generate", "layers") in result.replacements


def test_prepared_invocation_has_no_late_saved_context_interface():
    schema = fixture_schema()
    saved = inherited_layers(schema).values
    inputs = collect_invocation_sources(
        schema, argparse.Namespace(config=None, set_overrides=["generate.layers[0].opacity=0.5"])
    ).prepare(PreparedConfig(saved, {}))
    with pytest.raises(TypeError):
        inputs.state({"generate": {"layers": [{"source": "different.png"}]}})
    saved["generate"]["layers"][0]["source"] = "mutated.png"
    assert inputs.state().values["generate"]["layers"][0]["source"] == "same.png"


def test_collection_parser_storage_rename_preserves_canonical_paths_and_environment(monkeypatch):
    original = grouped_schema()
    table = original.tables[0]
    collection = replace(table.fields[0], cli_dest="parser_private_collection")
    schema = replace(original, tables=(replace(table, fields=(collection,)),))
    monkeypatch.setenv("KINO_FIXTURE_LAYERS", '[{source="input.png",opacity=0.25}]')
    parser = grouped_parser(schema)
    options = parser.parse_args(["--layer", "input.png", "--opacity", "0.25"])
    expected = {"generate": {"layers": [{"source": Path("input.png"), "opacity": 0.25}]}}
    assert options.parser_private_collection == expected["generate"]["layers"]
    assert not hasattr(options, "layers")
    assert schema.cli_layer(options) == expected
    assert schema.environment_variables() == {("generate", "layers"): ("KINO_FIXTURE_LAYERS",)}
    environment = schema.prepare_layer(schema.environment_config())
    typed = dataclass_from_mapping(Generate, environment["generate"], table="generate")
    assert record_intent(typed) == {"layers": ({"source": Path("input.png"), "opacity": 0.25},)}
    assert schema.resolve_target(("generate", "layers", 0, "opacity")).field.name == "opacity"
    assert original.environment_variables() == schema.environment_variables()
