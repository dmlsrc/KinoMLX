"""Invocation utility declarations participate in generated parser drift checks."""

import argparse
from dataclasses import replace
from pathlib import Path

import pytest

from kinomlx.cli._registry import validate_model_parser
from kinomlx.config.inspection import configuration_inventory
from kinomlx.config_bindings import CliBinding
from kinomlx.invocation_controls import INVOCATION_CONTROLS, InvocationControl
from kinomlx.model_registry import config_registry, model_choices
from kinomlx.models.flux2.arguments import build_parser


@pytest.mark.parametrize(
    "destination", ["config", "set_overrides", "print_config", "save_config", "only_non_defaults"]
)
@pytest.mark.parametrize("attribute", ["help", "default", "required"])
def test_utility_mutation_fails_model_parser_validation(destination, attribute):
    parser = build_parser()
    action = next(action for action in parser._actions if action.dest == destination)
    setattr(action, attribute, True if attribute == "required" else "drift")
    with pytest.raises(ValueError, match="invocation control drift"):
        validate_model_parser(parser, "flux2")


@pytest.mark.parametrize("model", model_choices())
def test_model_binding_and_utility_exclusions_are_visible(model):
    inventory = configuration_inventory(config_registry().model(model))
    model_field = next(field for field in inventory["fields"] if field["path"] == "model")
    assert model_field["cli"]["flags"] == ["--model"]
    controls = {control["name"]: control for control in inventory["invocation_controls"]}
    assert controls["model"]["configuration_target"] == "model"
    for name in ("config", "print_config", "save_config", "set_overrides", "only_non_defaults"):
        assert controls[name]["configuration_target"] is None
        surfaces = controls[name]["surface_classification"]
        assert surfaces["cli"]["status"] == "supported"
        assert surfaces["toml"]["status"] == "unavailable"
        assert surfaces["toml"]["reason"]


def test_neutral_utility_contribution_uses_existing_typed_binding():
    control = InvocationControl(
        "manifest",
        Path,
        CliBinding("Export", metavar="PATH"),
        "Write an export manifest.",
        "utility",
    )
    parser = argparse.ArgumentParser(add_help=False)
    control.add(parser)
    assert parser.parse_args(["--manifest", "export.json"]).manifest == Path("export.json")
    assert parser.parse_args([]).manifest is None
    assert control.describe()["surface_classification"]["toml"]["reason"]


def test_control_requires_reason_for_unavailable_surfaces():
    from kinomlx.config_surfaces import SurfacePolicy

    with pytest.raises(ValueError, match="nonempty reasons"):
        SurfacePolicy((("toml", ""),))


@pytest.mark.parametrize("attribute", ["option_strings", "type", "nargs"])
def test_model_selector_binding_mutation_is_detected(attribute):
    parser = build_parser()
    action = next(action for action in parser._actions if action.dest == "model")
    changes = {"option_strings": ["--wrong-model"], "type": int, "nargs": "+"}
    setattr(action, attribute, changes[attribute])
    with pytest.raises(ValueError, match="CLI binding drift for model"):
        validate_model_parser(parser, "flux2")


def test_utility_dependency_uses_declared_flags_and_is_inspectable():
    from kinomlx.invocation_controls import validate_control_inputs

    parser = build_parser()
    with pytest.raises(
        ValueError, match="--only-non-defaults requires --print-config or --save-config"
    ):
        validate_control_inputs(parser.parse_args(["--only-non-defaults"]))
    validate_control_inputs(parser.parse_args(["--only-non-defaults", "--print-config"]))
    control = next(
        control for control in INVOCATION_CONTROLS if control.name == "only_non_defaults"
    )
    assert control.describe()["requires_any"] == ["print_config", "save_config"]


def test_unknown_utility_dependency_fails_declaration_validation():
    from kinomlx.invocation_controls import validate_control_declarations

    invalid = replace(INVOCATION_CONTROLS[1], requires_any=("missing",))
    with pytest.raises(ValueError, match="unknown invocation dependencies"):
        validate_control_declarations((invalid,))


def test_duplicate_utility_flag_fails_declaration_validation():
    from kinomlx.invocation_controls import validate_control_declarations

    first = INVOCATION_CONTROLS[1]
    duplicate = replace(
        first, name="another_config", binding=CliBinding("Invocation", flags=("--config",))
    )
    with pytest.raises(ValueError, match="duplicate invocation control flags"):
        validate_control_declarations((first, duplicate))
