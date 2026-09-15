"""Krea configuration and execution share typed field declarations."""

import argparse
from collections.abc import Sequence
from dataclasses import fields

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_bindings import CliBinding
from kinomlx.config_constraints import field_choices
from kinomlx.config_records import supplied_fields
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.config_schema import CONFIG_SCHEMA
from kinomlx.models.krea2.types import Krea2GenerateConfig, Krea2Request

type SampleNumbers = Sequence[int]
type AmbiguousNumbers = Sequence[int | str]


def test_configuration_and_request_share_each_input_field_object():
    config = {item.name: item for item in fields(Krea2GenerateConfig)}
    request = {item.name: item for item in fields(Krea2Request) if item.init}
    assert config.keys() == request.keys()
    assert all(config[name] is request[name] for name in config)


@pytest.mark.parametrize(
    "name", ["seed_mode", "noise_backend", "vae_tiling", "reference_fit", "match_source_aspect"]
)
def test_owned_choices_reject_before_request_source_reads(monkeypatch, name):
    from kinomlx.models.krea2 import request_sources

    def unexpected(*args, **kwargs):
        pytest.fail("source read before local admission")

    monkeypatch.setattr(request_sources, "read_text_conditioning_prompts", unexpected)
    assert CONFIG_SCHEMA.field(("generate", name)).choices == field_choices(Krea2Request, name)
    for record in (Krea2GenerateConfig, Krea2Request):
        with pytest.raises(ValueError, match=name):
            record(text_conditioning="saved.safetensors", **{name: "unknown"})


def test_common_constructor_preserves_nested_presence_through_request_projection():
    config = Krea2GenerateConfig(
        prompt="hello", adapters=[{"path": "adapter.safetensors", "strength": 1.0}]
    )
    request = config.request()
    assert request is not None
    assert supplied_fields(request) == supplied_fields(config) == frozenset({"prompt", "adapters"})
    assert supplied_fields(request.adapters[0]) == frozenset({"path", "strength"})


def test_alias_owned_prompt_keeps_repeated_cli_order():
    options = build_parser().parse_args(
        ["--prompt", "first", "--prompt", "second", "--print-config"]
    )
    invocation = assemble(options)
    assert invocation.generate.prompt == ("first", "second")
    assert invocation.request.prompts == ("first", "second")


def test_repeated_cli_infers_scalar_from_sequence_alias():
    binding = CliBinding("Samples", action="append")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="append", type=binding.value_parser(SampleNumbers))
    assert parser.parse_args(["--sample", "9", "--sample", "2"]).sample == [9, 2]
    with pytest.raises(TypeError, match="needs a codec"):
        binding.value_parser(AmbiguousNumbers)


@pytest.mark.parametrize("record", [Krea2GenerateConfig, Krea2Request])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("width", 0),
        ("height", 0),
        ("seed", -1),
        ("seed", 2**64),
        ("seeds", ()),
        ("seeds", (-1,)),
        ("seeds", (2**64,)),
        ("steps", 0),
        ("images_per_prompt", 0),
        ("vae_tile_size", 0),
        ("vae_tile_stride", 0),
        ("grounding_px", -1),
        ("guidance", -1),
        ("guidance", float("inf")),
        ("mu", float("nan")),
        ("reference_strength", -1),
        ("scene_reference_strength", float("inf")),
    ],
)
def test_numeric_admission_precedes_request_source_reads(monkeypatch, record, name, value):
    from kinomlx.models.krea2 import request_sources

    def unexpected(*args, **kwargs):
        pytest.fail("source read before numeric admission")

    monkeypatch.setattr(request_sources, "read_text_conditioning_prompts", unexpected)
    with pytest.raises(ValueError, match=name):
        record(text_conditioning="saved.safetensors", **{name: value})


def test_malformed_lower_numeric_input_cannot_be_hidden_by_cli(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[generate]\nsteps = 0\n")
    options = build_parser().parse_args(["--config", str(config), "--steps", "4", "--print-config"])
    with pytest.raises(ConfigError, match="steps"):
        assemble(options)


def test_shared_numeric_owner_admits_valid_edge_values():
    config = Krea2GenerateConfig(
        seed=2**64 - 1,
        seeds=(0, 2**64 - 1),
        grounding_px=0,
        guidance=0,
        reference_strength=0,
        scene_reference_strength=0,
        mu=-1,
    )
    assert config.seeds == (0, 2**64 - 1)
    assert config.grounding_px == config.guidance == config.reference_strength == 0
