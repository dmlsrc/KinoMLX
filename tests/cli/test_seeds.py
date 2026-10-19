"""Seed lists have consistent CLI, configuration, and prompt expansion semantics."""

import tomllib

import pytest

from kinomlx.config import ConfigError
from kinomlx.models.flux2.arguments import build_parser as flux_parser
from kinomlx.models.flux2.config import assemble as flux_assemble
from kinomlx.models.flux2.config_schema import CONFIG_SCHEMA as flux_schema
from kinomlx.models.krea2.arguments import build_parser as krea_parser
from kinomlx.models.krea2.config import assemble as krea_assemble
from kinomlx.models.krea2.config_schema import CONFIG_SCHEMA as krea_schema
from kinomlx.models.ltx2.arguments import build_parser as ltx_parser
from kinomlx.models.ltx2.config import assemble as ltx_assemble
from kinomlx.models.ltx2.config_schema import CONFIG_SCHEMA as ltx_schema

MODELS = [
    (flux_parser, flux_assemble, flux_schema),
    (krea_parser, krea_assemble, krea_schema),
    (ltx_parser, ltx_assemble, ltx_schema),
]


def request(invocation):
    return (
        invocation.generate
        if hasattr(invocation, "generate") and not hasattr(invocation, "request")
        else invocation.request
    )


@pytest.mark.parametrize(("parser", "assemble", "schema"), MODELS)
@pytest.mark.parametrize(
    "arguments", [["--seed", "9", "2", "9"], ["--seed", "9,2,9"], ["--seeds", "9", "2,9"]]
)
def test_cli_list_and_effective_config_roundtrip(parser, assemble, schema, arguments, tmp_path):
    invocation = assemble(parser().parse_args(["--prompt", "test", *arguments]))
    assert request(invocation).selected_seeds() == (9, 2, 9)
    text = schema.dump_config(invocation.resolved_config)
    assert "seed" not in tomllib.loads(text)["generate"]
    path = tmp_path / "settings.toml"
    path.write_text(text)
    restored = assemble(parser().parse_args(["--config", str(path)]))
    assert request(restored).selected_seeds() == (9, 2, 9)


@pytest.mark.parametrize(("parser", "assemble", "schema"), MODELS)
def test_full_u64_list_and_selection_precedence(parser, assemble, schema, tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text('[generate]\nprompt="test"\nseeds=[9, 2]\n')
    invocation = assemble(parser().parse_args(["--config", str(config), "--seed", "7"]))
    assert request(invocation).selected_seeds() == (7,)
    invocation = assemble(parser().parse_args(["--prompt", "test", "--seed", str(2**64 - 1), "0"]))
    text = schema.dump_config(invocation.resolved_config)
    config.write_text(text)
    assert request(assemble(parser().parse_args(["--config", str(config)]))).selected_seeds() == (
        2**64 - 1,
        0,
    )
    config.write_text('[generate]\nprompt="test"\nseed=1\nseeds=[9, 2]\n')
    with pytest.raises(ConfigError, match="mutually exclusive"):
        assemble(parser().parse_args(["--config", str(config)]))


@pytest.mark.parametrize(("parser", "assemble", "schema"), MODELS[:2])
def test_image_modes_and_variation_conflict(parser, assemble, schema):
    for mode in ("sequential", "stage", "resident"):
        invocation = assemble(
            parser().parse_args(["--prompt", "test", "--seed", "9", "2", "--seed-mode", mode])
        )
        assert request(invocation).seed_mode == mode
        assert request(invocation).image_count == 2
    with pytest.raises(ValueError, match="images_per_prompt"):
        assemble(
            parser().parse_args(
                ["--prompt", "test", "--seed", "9", "2", "--images-per-prompt", "2"]
            )
        )


def test_ltx_rejects_image_lifetime_modes():
    with pytest.raises(SystemExit):
        ltx_parser().parse_args(["--seed-mode", "resident"])
