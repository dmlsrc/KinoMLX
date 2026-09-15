"""Independent public-spelling oracle and grouped-option cursor regression."""

import json
import tomllib
from pathlib import Path

import pytest

from kinomlx.cli.main import main
from kinomlx.config.inspection import configuration_inventory
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.arguments import build_parser as flux_parser
from kinomlx.models.gmnet.arguments import build_parser as gmnet_parser
from kinomlx.models.krea2.arguments import build_parser as krea_parser
from kinomlx.models.ltx2.arguments import build_parser as ltx_parser
from tests.config.test_record_schema import grouped_parser, grouped_schema

_BUILDERS = {
    "flux2": flux_parser,
    "gmnet": gmnet_parser,
    "krea2": krea_parser,
    "ltx2": ltx_parser,
}
_SNAPSHOT = Path(__file__).with_name("fixtures") / "public_spellings.json"


@pytest.mark.parametrize("model", tuple(_BUILDERS))
def test_public_flags_and_environment_names_match_reviewed_snapshot(model):
    expected = json.loads(_SNAPSHOT.read_text())[model]
    parser = _BUILDERS[model]()
    flags = sorted({flag for action in parser._actions for flag in action.option_strings})
    fields = configuration_inventory(config_registry().model(model))["fields"]
    environment = sorted({name for field in fields for name in field["environment"]})
    assert {"flags": flags, "environment": environment} == expected


def test_set_and_global_option_between_grouped_item_and_modifier_keep_cursor():
    schema = grouped_schema()
    parser = grouped_parser(schema)
    parser.add_argument("--set", dest="set_overrides", action="append")
    parser.add_argument("--verbose", action="store_true")
    options = parser.parse_args(
        [
            "--layer",
            "first.png",
            "--set",
            "generate.layers[0].opacity=0.75",
            "--layer-opacity",
            "0.25",
            "--layer",
            "second.png",
            "--verbose",
            "--set",
            "generate.layers[1].opacity=0.25",
            "--opacity",
            "0.5",
        ]
    )
    assert options.set_overrides == [
        "generate.layers[0].opacity=0.75",
        "generate.layers[1].opacity=0.25",
    ]
    assert options.verbose is True
    assert schema.cli_layer(options) == {
        "generate": {
            "layers": [
                {"source": Path("first.png"), "opacity": 0.25},
                {"source": Path("second.png"), "opacity": 0.5},
            ]
        }
    }


@pytest.mark.parametrize("model", ["flux2", "krea2", "ltx2"])
def test_set_layer_wins_even_when_first_override_precedes_model_flag(model, capsys):
    assert (
        main(
            [
                "--model",
                model,
                "--print-config",
                "--set",
                "generate.seed=7",
                "--seed",
                "9",
                "--set",
                "generate.seed=11",
            ]
        )
        == 0
    )
    assert tomllib.loads(capsys.readouterr().out)["generate"]["seed"] == 11
