"""Independent extension, capability removal and failure-atomicity acceptance."""

import argparse
import copy
import tomllib
from dataclasses import dataclass, field, replace

import pytest

import kinomlx.cli.config_templates as templates
from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.config import CliBinding, ConfigError, ConfigFieldSpec, ConfigRegistry, ConfigTableSpec
from kinomlx.config.assembly import collect_invocation_sources
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config.provenance import PreparedConfig
from kinomlx.config.stations import StationPolicy
from kinomlx.config.transforms import LayerTransform
from kinomlx.config_constraints import NumberConstraint
from kinomlx.config_records import config_record, record_intent
from kinomlx.config_rules import RecordRule, validate_record
from kinomlx.config_surfaces import CONFIG_SURFACES
from kinomlx.config_values import dataclass_from_mapping
from tests.config.test_record_schema import fixture_schema, grouped_parser, grouped_schema


@config_record
@dataclass(frozen=True)
class Presentation:
    enabled: bool = True


@config_record
@dataclass(frozen=True)
class ExposurePresentation(Presentation):
    exposure: float = field(
        default=0.5, metadata={"config_constraints": (NumberConstraint(minimum=0, maximum=1),)}
    )


def _member(name, example, *, action="store"):
    return ConfigFieldSpec(
        name=name,
        purpose=f"Set {name}",
        valid="a declared typed value",
        example=example,
        cli_dest=name,
        persistence="intent",
        surfaces=CONFIG_SURFACES,
        cli=CliBinding("Presentation", action=action),
        station_policies=(StationPolicy("render", "mutable", reason="Rendering runs again."),),
    )


def _schema(*, extended=False, model="fixture"):
    base = fixture_schema()
    fields = (_member("enabled", True, action="boolean"),)
    if extended:
        fields += (_member("exposure", 0.75),)
    table = ConfigTableSpec(
        ("output",),
        ExposurePresentation if extended else Presentation,
        fields,
        cli_generated=True,
        stations=("render",),
        station_record=ExposurePresentation if extended else Presentation,
    )
    return replace(
        base,
        model=model,
        root_fields=(replace(base.root_fields[0], default=model, example=model, choices=(model,)),),
        tables=(table,),
        restart_stations=("render",),
        no_restart_reason=None,
    )


def _parser(schema):
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, schema)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="set_overrides", action="append", default=[])
    return parser


def _admit(schema, values):
    prepared = schema.prepare_layer(values)
    table = schema.tables[0]
    result = dataclass_from_mapping(table.record, prepared.get("output", {}), table="output")
    validate_record(result, location="[output]")
    return result


def test_scalar_extension_reaches_every_surface_reference_and_restart(monkeypatch):
    schema = _schema(extended=True)
    registry = ConfigRegistry()
    registry.register(schema)
    registry.freeze()
    monkeypatch.setattr(templates, "config_registry", lambda: registry)
    monkeypatch.setenv("KINO_FIXTURE_EXPOSURE", "0.75")
    parser = _parser(schema)
    cli = schema.cli_layer(parser.parse_args(["--exposure", "0.75"]))
    env = schema.environment_config()
    structured = tomllib.loads("[output]\nexposure=0.75")
    override = schema.parse_overrides(["output.exposure=0.75"]).config
    api = {"output": record_intent(ExposurePresentation(exposure=0.75))}
    for raw in (cli, env, structured, override, api):
        admitted = _admit(schema, raw)
        assert record_intent(admitted) == {"exposure": 0.75}
        dumped = schema.dump_config({"output": record_intent(admitted)})
        assert tomllib.loads(dumped) == {"output": {"exposure": 0.75}}
        assert (
            schema.station_changes(
                {"output": admitted},
                {"output": ExposurePresentation()},
                station="render",
                active_explicit=frozenset({("output", "exposure")}),
            )
            == ()
        )
    assert "--exposure" in parser.format_help()
    template = templates.config_template("fixture")
    assert "KINO_FIXTURE_EXPOSURE" in template
    assert "exposure = 0.75" in template
    entry = configuration_inventory(schema, field="output.exposure")["fields"][0]
    assert entry["type"] == "<class 'float'>"
    assert entry["local_constraints"][0]["minimum"] == 0
    assert entry["local_constraints"][0]["maximum"] == 1
    assert entry["station_policies"][0]["reason"] == "Rendering runs again."


@pytest.mark.parametrize("saved", [False, True])
def test_new_model_uses_shared_source_precedence_and_presence(tmp_path, monkeypatch, saved):
    schema = _schema(extended=True, model="newmodel")
    monkeypatch.setenv("KINO_NEWMODEL_EXPOSURE", "0.1")
    config = tmp_path / "new-model.toml"
    config.write_text("[output]\nexposure=0.3\nenabled=false\n")
    parser = _parser(schema)
    base = PreparedConfig({"output": {"exposure": 0.2}}, {}) if saved else None
    cases = (
        ([], 0.2 if saved else 0.1),
        (["--config", str(config)], 0.3),
        (["--config", str(config), "--exposure", "0.4"], 0.4),
        (["--config", str(config), "--set", "output.exposure=0.5", "--exposure", "0.4"], 0.5),
    )
    for args, expected in cases:
        layers = collect_invocation_sources(schema, parser.parse_args(args)).prepare(
            base, station="render" if saved else "generate"
        )
        typed = _admit(schema, layers.merge())
        assert typed.exposure == expected
        assert record_intent(typed)["exposure"] == expected
        if args:
            assert typed.enabled is False
            assert record_intent(typed)["enabled"] is False
    with pytest.raises(ConfigError, match="did you mean 'exposure'"):
        schema.prepare_layer({"output": {"exposur": 0.2}})


def _requires_enabled(values):
    if values["exposure"] > 0 and not values["enabled"]:
        raise ValueError("enable output.enabled before selecting output.exposure")


def test_removing_capability_removes_its_binding_rule_and_remedy():
    @config_record
    @dataclass(frozen=True)
    class ExposureCapability(Presentation):
        exposure: float = field(
            default=0.5,
            metadata={
                "config_rules": (
                    RecordRule(
                        "exposure_requires_output",
                        frozenset({"enabled", "exposure"}),
                        _requires_enabled,
                    ),
                )
            },
        )

    extended = _schema(extended=True)
    capability_table = replace(
        extended.tables[0], record=ExposureCapability, station_record=ExposureCapability
    )
    capable = replace(extended, tables=(capability_table,))
    removed = _schema()
    assert "--exposure" in _parser(capable).format_help()
    assert "--exposure" not in _parser(removed).format_help()
    assert ("output", "exposure") not in removed.environment_variables()
    capable_fields = configuration_inventory(capable)["fields"]
    assert any(
        rule["rule"] == "exposure_requires_output"
        for entry in capable_fields
        for rule in entry["validation_rules"]
    )
    assert all(
        not entry["validation_rules"] for entry in configuration_inventory(removed)["fields"]
    )
    with pytest.raises(ValueError, match="enable output.enabled"):
        _admit(capable, {"output": {"enabled": False, "exposure": 0.5}})
    assert _admit(removed, {"output": {"enabled": False}}).enabled is False
    with pytest.raises(ConfigError, match="did you mean 'exposure'"):
        capable.prepare_layer({"output": {"exposur": 0.2}})
    with pytest.raises(ConfigError) as error:
        removed.prepare_layer({"output": {"exposur": 0.2}})
    assert "did you mean 'exposure'" not in str(error.value)
    assert "known: enabled" in str(error.value)


def test_shared_typed_identity_rejects_reinterpretation_and_accepts_explicit_station_policy():
    registry = ConfigRegistry()
    first = _schema()
    registry.register(first)
    second = _schema(model="second")
    original = second.tables[0]
    incompatible = replace(original.fields[0], valid="an inverted boolean")
    with pytest.raises(ValueError, match="shared field Presentation.enabled conflicts"):
        registry.register(replace(second, tables=(replace(original, fields=(incompatible,)),)))
    assert registry.models() == ("fixture",)
    locked = replace(
        original.fields[0],
        station_policies=(
            StationPolicy("render", "locked", reason="This model caches presentation."),
        ),
    )
    specialized = replace(second, tables=(replace(original, fields=(locked,)),))
    registry.register(specialized)
    assert specialized.station_changes(
        {"output": Presentation(False)},
        {"output": Presentation()},
        station="render",
        active_explicit=frozenset({("output", "enabled")}),
    ) == (("output", "enabled"),)


def test_transform_cycle_fails_deterministically_at_contribution_construction():
    schema = _schema(extended=True)
    target = ("output", "exposure")
    first = LayerTransform(
        "first", frozenset({target}), frozenset({target}), lambda values: {}, after=("second",)
    )
    second = replace(first, name="second", after=("first",))
    for ordered in ((first, second), (second, first)):
        with pytest.raises(ValueError, match="must precede"):
            replace(schema, layer_transforms=ordered)


@pytest.mark.parametrize("failure", ["operation", "later_rule"])
def test_failed_override_or_later_rule_preserves_caller_and_saved_state(failure):
    schema = grouped_schema()
    saved_values = {"generate": {"layers": [{"source": "first.png", "opacity": 0.5}]}}
    saved = PreparedConfig(saved_values, {})
    if failure == "operation":
        assignments = ["generate.layers[0].opacity=0.25", "generate.layers[3].opacity=0"]
        message = "outside collection"
    else:
        target = ("generate", "layers")

        def refuse(values):
            raise ValueError("later rule refused the complete collection")

        rule = LayerTransform("refuse_layers", frozenset({target}), frozenset({target}), refuse)
        schema = replace(schema, layer_transforms=(rule,))
        assignments = ["generate.layers[0].opacity=0.25"]
        message = "later rule refused"
    parser = grouped_parser(schema)
    options = parser.parse_args([])
    options.config = None
    options.set_overrides = assignments
    options_before = copy.deepcopy(vars(options))
    saved_before = copy.deepcopy(saved.values)
    results = []
    with pytest.raises((ConfigError, ValueError), match=message):
        results.append(collect_invocation_sources(schema, options).prepare(saved))
    assert results == []
    assert saved.values == saved_before == saved_values
    assert vars(options) == options_before
