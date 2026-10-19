"""Bootstrap selectors must agree with complete model parsing and admission."""

import argparse
import importlib
import tomllib
from pathlib import Path

import pytest

from kinomlx.cli._registry import _bootstrap_options, resolve_model_selection
from kinomlx.cli.invocation import add_selection_arguments
from kinomlx.cli.main import main
from kinomlx.model_catalog import model_choices


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_shared_selector_parser_matches_complete_model_parser(model):
    arguments = [
        "--config=config.toml",
        "--model",
        "ltx2",
        "--model",
        model,
        "--set",
        "settings.verbose=false",
        "--set",
        f'model="{model}"',
    ]
    bootstrap = _bootstrap_options(arguments)
    parser = importlib.import_module(f"kinomlx.models.{model}.arguments").build_parser()
    complete = parser.parse_args(arguments)
    for name in ("model", "config", "set_overrides"):
        assert getattr(bootstrap, name) == getattr(complete, name)
    assert bootstrap.config == Path("config.toml")


@pytest.mark.parametrize("model", ["gmnet", "flux2", "krea2", "ltx2"])
def test_bootstrap_and_final_output_agree_across_source_precedence(tmp_path, capsys, model):
    config = tmp_path / "run.toml"
    config.write_text('model="ltx2"\n')
    arguments = [
        "--config",
        str(config),
        "--model",
        "flux2",
        "--set",
        'model="krea2"',
        "--set",
        f'model="{model}"',
        "--print-config",
    ]
    assert resolve_model_selection(arguments).name == model
    assert main(arguments) == 0
    assert tomllib.loads(capsys.readouterr().out)["model"] == model


def test_selection_projection_does_not_hide_malformed_discarded_input(capsys):
    arguments = ["--set", "model=7", "--set", 'model="gmnet"', "--print-config"]
    assert resolve_model_selection(arguments).name == "gmnet"
    assert main(arguments) != 0
    assert "--set #1" in capsys.readouterr().err


def test_help_can_select_valid_model_after_malformed_override():
    selected = resolve_model_selection(
        ["--set", "malformed", "--set", 'model="gmnet"'], tolerate_errors=True
    )
    assert selected.name == "gmnet"


def test_shared_selector_bindings_preserve_terminator_and_sparse_defaults():
    parser = argparse.ArgumentParser(add_help=False)
    add_selection_arguments(parser, choices=model_choices())
    options = parser.parse_args([])
    assert vars(options) == {"model": None, "config": None, "set_overrides": None}
    assert resolve_model_selection(["--", "--model", "gmnet"]).name == "ltx2"
