"""Schema-owned CLI construction, extension, and independently expected behavior."""

import argparse
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from kinomlx.cli._registry import validate_model_parser
from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.config import (
    CliBinding,
    ConfigError,
    ConfigFieldSpec,
    ConfigTableSpec,
    ModelConfigSpec,
    dataclass_from_mapping,
)
from kinomlx.config_surfaces import CONFIG_SURFACES, MODEL_SURFACES
from kinomlx.invocation_controls import MODEL_CONTROL
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.arguments import build_parser as flux_parser
from kinomlx.models.gmnet.arguments import build_parser as gmnet_parser
from kinomlx.models.krea2.arguments import build_parser as krea_parser
from kinomlx.models.ltx2.arguments import build_parser as ltx_parser


@dataclass(frozen=True)
class SensorConfig:
    count: int = 1
    enabled: bool = True
    sources: tuple[Path, ...] = ()


def sensor_schema() -> ModelConfigSpec:
    return ModelConfigSpec(
        model="sensor",
        title="Sensor configuration",
        introduction=("Synthetic non-model inputs.",),
        restart_stations=(),
        no_restart_reason="The synthetic sensor contribution has no saved station.",
        root_fields=(
            ConfigFieldSpec(
                "model",
                "Select model",
                "",
                "sensor",
                "model",
                persistence="intent",
                virtual=True,
                default="sensor",
                choices=("sensor",),
                surfaces=MODEL_SURFACES,
                cli=MODEL_CONTROL.binding,
            ),
        ),
        tables=(
            ConfigTableSpec(
                path=("capture",),
                record=SensorConfig,
                cli_generated=True,
                fields=(
                    ConfigFieldSpec(
                        "count",
                        "Set sample count",
                        "an integer",
                        7,
                        "count",
                        persistence="intent",
                        cli=CliBinding("Capture"),
                        surfaces=CONFIG_SURFACES,
                    ),
                    ConfigFieldSpec(
                        "enabled",
                        "Enable capture",
                        "a boolean",
                        False,
                        "enabled",
                        persistence="intent",
                        cli=CliBinding("Capture", action="boolean"),
                        surfaces=CONFIG_SURFACES,
                    ),
                    ConfigFieldSpec(
                        "sources",
                        "Select sources",
                        "path array",
                        (Path("source"),),
                        "sources",
                        persistence="intent",
                        cli=CliBinding("Capture", flags=("--source",), action="append"),
                        surfaces=CONFIG_SURFACES,
                    ),
                ),
            ),
        ),
    )


def test_new_declaration_reaches_cli_env_toml_and_typed_api(monkeypatch):
    schema = sensor_schema()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, schema)
    assert schema.cli_layer(parser.parse_args([])) == {}
    expected = {"capture": {"count": 7, "enabled": False, "sources": [Path("a"), Path("b")]}}
    parsed = parser.parse_args(["--count", "7", "--no-enabled", "--source", "a", "--source", "b"])
    assert schema.cli_layer(parsed) == expected
    monkeypatch.setenv("KINO_SENSOR_COUNT", "7")
    monkeypatch.setenv("KINO_SENSOR_ENABLED", "false")
    monkeypatch.setenv("KINO_SENSOR_SOURCES", '["a", "b"]')
    env = schema.environment_config()
    cli = schema.cli_layer(parsed)
    toml = tomllib.loads(schema.dump_config(cli))
    api = {"count": 7, "enabled": False, "sources": ["a", "b"]}
    expected_record = SensorConfig(count=7, enabled=False, sources=(Path("a"), Path("b")))
    for values in (cli["capture"], env["capture"], toml["capture"], api):
        assert dataclass_from_mapping(SensorConfig, values, table="capture") == expected_record
    action = next(a for a in parser._actions if a.dest == "count")
    assert action.type is int
    assert action.help == "Set sample count."
    assert action.default is None


def test_new_field_requires_no_consumer_edit(monkeypatch):
    @dataclass(frozen=True)
    class ExtendedSensor(SensorConfig):
        exposure: float = 0.5

    base = sensor_schema()
    field = ConfigFieldSpec(
        "exposure",
        "Set exposure",
        "a number",
        0.75,
        "exposure",
        persistence="intent",
        cli=CliBinding("Capture"),
        surfaces=CONFIG_SURFACES,
    )
    table = replace(base.tables[0], record=ExtendedSensor, fields=(*base.tables[0].fields, field))
    schema = replace(base, tables=(table,))
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, schema)
    options = parser.parse_args(["--exposure", "0.75"])
    assert schema.cli_layer(options) == {"capture": {"exposure": 0.75}}
    monkeypatch.setenv("KINO_SENSOR_EXPOSURE", "0.75")
    assert schema.environment_config() == {"capture": {"exposure": 0.75}}
    assert tomllib.loads(schema.dump_config(schema.cli_layer(options))) == {
        "capture": {"exposure": 0.75}
    }
    assert (
        dataclass_from_mapping(ExtendedSensor, {"exposure": 0.75}, table="capture").exposure == 0.75
    )


def test_migrated_table_rejects_field_without_binding():
    table = sensor_schema().tables[0]
    with pytest.raises(ValueError, match="missing CLI binding"):
        replace(table, fields=(replace(table.fields[0], cli=None), *table.fields[1:]))


@pytest.mark.parametrize(
    "annotation", [tuple[int, ...], dict[str, int], int | str, int | dict[str, int]]
)
def test_complex_shapes_require_explicit_transport_codec(annotation):
    with pytest.raises(TypeError, match="needs a codec"):
        CliBinding("Capture").value_parser(annotation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flags": ("--same", "--same")},
        {"flags": ("bare",)},
        {"flags": ("--bad=value",)},
        {"group": ""},
        {"action": "boolean", "parse": str},
        {"action": "boolean", "nargs": "?"},
        {"const": "auto"},
        {"nargs": 0},
    ],
)
def test_invalid_binding_fails_at_declaration(kwargs):
    with pytest.raises(ValueError, match="CLI|boolean"):
        CliBinding(**({"group": "Capture"} | kwargs))


@pytest.mark.parametrize("builder", [krea_parser, flux_parser])
def test_real_aliases_repetition_and_optional_values(builder):
    options = builder().parse_args(
        [
            "--prompt",
            "first",
            "--prompt",
            "second",
            "--lora",
            "a.safetensors",
            "--lora-strength",
            "0.5",
            "--adapter",
            "b.safetensors",
            "--adapter-strength",
            "0.8",
            "--lora-step-strengths",
            "1,0",
            "--adapter-sigma-window",
            "0.3:0.8",
            "--match-source-aspect",
            "--save-all-sidecars",
            "--no-save-text-embeddings",
            "--restart",
            "run.json",
            "--restart-from",
            "denoise",
            "--restart-index",
            "2",
            "--seed",
            "7",
            "7",
            "9",
        ]
    )
    assert options.prompt == ["first", "second"]
    assert options.adapters == [
        {"path": Path("a.safetensors"), "strength": 0.5},
        {
            "path": Path("b.safetensors"),
            "strength": 0.8,
            "step_strengths": (1.0, 0.0),
            "sigma_window": "0.3:0.8",
        },
    ]
    assert options.match_source_aspect == "long"
    assert options.save_all_sidecars is True
    assert options.save_text_conditioning is False
    assert options.restart_run == Path("run.json")
    assert options.restart_phase == "denoise"
    assert options.restart_index == 2
    assert options.seeds == (7, 7, 9)
    assert options.seed is None


@pytest.mark.parametrize(("model", "builder"), [("krea2", krea_parser), ("flux2", flux_parser)])
@pytest.mark.parametrize(
    ("attribute", "value"),
    [("type", str), ("nargs", "+"), ("option_strings", ["--wrong"]), ("const", "wrong")],
)
def test_downstream_parser_mutation_fails(model, builder, attribute, value):
    parser = builder()
    action = next(a for a in parser._actions if a.dest == "width")
    setattr(action, attribute, value)
    with pytest.raises(ValueError, match="CLI binding drift for width"):
        validate_model_parser(parser, model)


def test_false_and_invalid_cli_values_keep_existing_contract(capsys):
    parser = krea_parser()
    assert parser.parse_args(["--no-turbo-sda"]).turbo_sda is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--width", "not-an-integer"])
    assert "invalid int value" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser.parse_args(["--adapter-step-strengths", "1,nan"])
    assert "finite" in capsys.readouterr().err


def test_direct_typed_admission_rejects_boolean_integer():
    with pytest.raises(ConfigError, match="expected int"):
        dataclass_from_mapping(SensorConfig, {"count": True}, table="capture")


def test_all_model_tables_bind_exactly_the_fields_that_support_the_cli():
    for model in config_registry().models():
        schema = config_registry().model(model)
        for table in schema.tables:
            assert table.cli_generated
            for field in table.fields:
                assert (field.cli is not None) is field.surfaces.supports("cli"), (
                    f"{model} {'.'.join(table.path)}.{field.name}"
                )


def test_adapter_weight_check_binds_only_where_adapter_weights_are_checked():
    for model in config_registry().models():
        field = config_registry().model(model).field(("settings", "check_adapter_weights"))
        checks_adapters = model != "gmnet"
        assert field.surfaces.supports("cli") is checks_adapters
        assert (field.cli is not None) is checks_adapters
        assert field.surfaces.supports("environment")


@pytest.mark.parametrize("builder", [krea_parser, flux_parser, ltx_parser, gmnet_parser])
def test_settings_bindings_have_independent_positive_and_negative_help(builder):
    parser = builder()
    positive, negative = [action for action in parser._actions if action.dest == "verbose"]
    assert positive.option_strings == ["--verbose"]
    assert negative.option_strings == ["--no-verbose"]
    assert positive.help == "Enable verbose logs."
    assert negative.help == "Set verbose to false."
    assert parser.parse_args([]).verbose is None
    assert parser.parse_args(["--verbose", "--no-verbose"]).verbose is False
    assert parser.parse_args(["--no-verbose", "--verbose"]).verbose is True


def test_ltx_paired_convenience_controls_and_stage_aliases():
    parser = ltx_parser()
    values = parser.parse_args(
        [
            "--generate-audio",
            "--no-generate-audio",
            "--reference-prompt-padding",
            "--compact-prompt",
            "--sigmas-stage-1",
            "1,0.5,0",
            "--stage2-sigmas",
            "0.8,0",
            "--adapter",
            "style.safetensors",
            "--lora-stage1-strength",
            "0.3",
            "--adapter-stage2-strength",
            "0.7",
            "--save-debug-sidecars",
            "--no-save-metadata",
        ]
    )
    assert values.generate_audio is False
    assert values.pad_prompt_to_max is False
    assert values.stage_1_sigmas == (1.0, 0.5, 0.0)
    assert values.stage_2_sigmas == (0.8, 0.0)
    assert values.adapters == [
        {"path": Path("style.safetensors"), "stage1_strength": 0.3, "stage2_strength": 0.7}
    ]
    assert values.save_all_sidecars is True
    assert values.save_run_log is False
    reverse = parser.parse_args(
        [
            "--compact-prompt",
            "--reference-prompt-padding",
            "--no-generate-audio",
            "--generate-audio",
        ]
    )
    assert reverse.pad_prompt_to_max is True
    assert reverse.generate_audio is True


def test_gmnet_image_and_output_binding_semantics():
    values = gmnet_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output",
            "out.exr",
            "--exr",
            "--no-heic",
            "--save-all-sidecars",
            "--no-save-gain-map",
        ]
    )
    assert values.image == Path("input.png")
    assert values.output_path == Path("out.exr")
    assert values.exr is True
    assert values.heic is False
    assert values.save_all_sidecars is True
    assert values.save_gain_map is False


def test_negative_alias_is_part_of_the_checked_binding():
    parser = ltx_parser()
    action = next(
        action for action in parser._actions if action.option_strings == ["--compact-prompt"]
    )
    action.option_strings = ["--no-padding"]
    with pytest.raises(ValueError, match="CLI binding drift for pad_prompt_to_max"):
        validate_model_parser(parser, "ltx2")


def test_registry_rejects_new_table_without_cli_participation(monkeypatch):
    from kinomlx.models.gmnet import config_schema

    original = config_schema.CONFIG_SCHEMA
    incomplete = replace(
        original, tables=(replace(original.tables[0], cli_generated=False), *original.tables[1:])
    )
    config_registry.cache_clear()
    try:
        with monkeypatch.context() as context:
            context.setattr(config_schema, "CONFIG_SCHEMA", incomplete)
            with pytest.raises(ValueError, match="config tables lack generated CLI bindings"):
                config_registry()
    finally:
        config_registry.cache_clear()


def test_settings_new_field_compiles_existing_metadata():
    from dataclasses import field

    from kinomlx.settings import EnvironmentSettings

    @dataclass(frozen=True)
    class SettingsExample(EnvironmentSettings):
        directory: Path = field(default=Path("default"), metadata={"cli": ("--directory", "--dir")})

    table = ConfigTableSpec(
        path=("settings",),
        record=SettingsExample,
        cli_generated=True,
        cli_group="Example settings",
        fields=(
            ConfigFieldSpec(
                "directory",
                "Set directory",
                "a path",
                Path("sample"),
                "directory",
                persistence="intent",
                surfaces=CONFIG_SURFACES,
            ),
        ),
    )
    binding = table.fields[0].cli
    assert binding is not None
    parser = argparse.ArgumentParser()
    binding.add_arguments(parser, destination="directory", annotation=Path, help="Set directory.")
    assert parser.parse_args(["--dir", "~/input"]).directory == Path.home() / "input"
    assert parser.parse_args([]).directory is None
