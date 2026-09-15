"""Krea request admission, final policies and explicit source observations."""

import re
from dataclasses import fields

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_constraints import LengthConstraint
from kinomlx.config_records import record_intent, replace_config, supplied_fields
from kinomlx.config_rules import validate_record
from kinomlx.models.krea2 import request_sources
from kinomlx.models.krea2.arguments import build_parser
from kinomlx.models.krea2.config import assemble
from kinomlx.models.krea2.request_sources import resolve_request_sources
from kinomlx.models.krea2.types import Krea2GenerateConfig, Krea2Request


@pytest.mark.parametrize("record", [Krea2GenerateConfig, Krea2Request])
@pytest.mark.parametrize(
    ("values", "override", "message"),
    [
        (
            {"seeds": (9, 2), "images_per_prompt": 2},
            {"images_per_prompt": 1},
            "Explicit seeds cannot be combined",
        ),
        (
            {"vae_tile_size": 128},
            {"vae_tile_stride": 64},
            "vae_tile_stride must be smaller",
        ),
        (
            {"prompt_unconditional": True},
            {"prompt_unconditional": False},
            "prompt_unconditional cannot be combined",
        ),
        (
            {"match_source_aspect": "long"},
            {"source_images": ("source.png",)},
            "match_source_aspect requires source images",
        ),
        (
            {"grounding_system_prompt": "identity"},
            {"source_images": ("source.png",)},
            "grounding system prompt and reference boost",
        ),
        (
            {"edit_adapter_strength": 2.0},
            {"source_images": ("source.png",)},
            "edit adapter options require source images",
        ),
        (
            {"turbo_4step_strength": 0.75},
            {"turbo_4step": True},
            "turbo_4step_strength and turbo_4step_exclusions require",
        ),
        (
            {"turbo_sda_sigma_window": "0.93"},
            {"turbo_sda": True},
            "turbo_sda_sigma_window and turbo_sda_step_strengths require",
        ),
        (
            {
                "turbo_sda": True,
                "turbo_sda_sigma_window": "0.93",
                "turbo_sda_step_strengths": (1.0,),
            },
            {"turbo_sda_step_strengths": ()},
            "use turbo_sda_sigma_window or turbo_sda_step_strengths, not both",
        ),
    ],
)
def test_final_policy_accepts_an_explicit_override(record, values, override, message):
    request = record(prompt="scene", **values)
    with pytest.raises(ValueError, match=rf"\[generate\]: {re.escape(message)}"):
        validate_record(request, location="[generate]")
    validate_record(replace_config(request, **override), location="[generate]")


@pytest.mark.parametrize("record", [Krea2GenerateConfig, Krea2Request])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"turbo_sda_strength": 1.5},
            "turbo_sda_strength and turbo_sda_exclusions require turbo_sda",
        ),
        (
            {"turbo_sda_exclusions": "output"},
            "turbo_sda_strength and turbo_sda_exclusions require turbo_sda",
        ),
        (
            {
                "turbo_sda": True,
                "turbo_sda_strength": 1.5,
                "turbo_sda_step_strengths": (1.0, 1.0, 0.0, 0.0),
            },
            "turbo_sda_step_strengths replaces turbo_sda_strength; put the magnitude in the array",
        ),
        (
            {
                "turbo_4step": True,
                "turbo_4step_strength": 1.5,
                "turbo_4step_step_strengths": (1.0, 1.0, 0.0, 0.0),
            },
            "turbo_4step_step_strengths replaces turbo_4step_strength; "
            "put the magnitude in the array",
        ),
    ],
)
def test_a_profile_magnitude_needs_its_switch_and_yields_to_its_step_array(record, values, message):
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        validate_record(record(prompt="scene", **values), location="[generate]")
    assert str(error.value).endswith(message)


@pytest.mark.parametrize("record", [Krea2GenerateConfig, Krea2Request])
@pytest.mark.parametrize("profile", ["turbo_4step", "turbo_sda"])
def test_a_profile_step_array_alone_carries_the_magnitude(record, profile):
    validate_record(
        record(
            prompt="scene",
            **{profile: True, f"{profile}_step_strengths": (1.5, 1.5, 0.0, 0.0)},
        ),
        location="[generate]",
    )


def test_an_ignored_profile_magnitude_is_refused_on_the_command_line():
    options = build_parser().parse_args(
        ["--model", "krea2", "--prompt", "scene", "--turbo-sda-strength", "1.5", "--print-config"]
    )
    with pytest.raises(ConfigError, match="turbo_sda_strength and turbo_sda_exclusions"):
        assemble(options)


@pytest.mark.parametrize("record", [Krea2GenerateConfig, Krea2Request])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("prompt", ()),
        ("negative_prompt", ()),
        ("system_prompt", "x" * 65_537),
        ("grounding_system_prompt", "x" * 65_537),
        ("edit_adapter_strength", 0),
        ("turbo_sda_strength", float("nan")),
        ("turbo_4step_exclusions", "unknown"),
        ("edit_adapter_exclusions", "unknown"),
        ("turbo_sda_exclusions", "unknown"),
        ("turbo_sda_sigma_window", "0.97:0.93"),
        ("turbo_4step_step_strengths", (float("nan"),)),
        ("sigmas", (0.5, 1.0, 0.0)),
        ("reference_boost_regions", ("unknown",)),
    ],
)
def test_malformed_local_values_fail_without_source_reads(record, name, value, monkeypatch):
    def unexpected(*args):
        pytest.fail("admission attempted source access")

    monkeypatch.setattr(request_sources, "read_text_conditioning_prompts", unexpected)
    with pytest.raises(ValueError, match=name):
        record(text_conditioning="saved.safetensors", **{name: value})


def test_construction_and_copies_do_not_observe_saved_files(monkeypatch, tmp_path):
    observations = []

    def read(path):
        observations.append(path)
        return ("inherited scene",), ("inherited negative",)

    monkeypatch.setattr(request_sources, "read_text_conditioning_prompts", read)
    raw = Krea2Request(text_conditioning=tmp_path / "saved.safetensors")
    copied = replace_config(raw, seed=42)
    assert not observations
    resolved = resolve_request_sources(copied)
    assert observations == [raw.text_conditioning]
    assert resolved.prompts == ("inherited scene",)
    assert resolved.negative_prompts == ("inherited negative",)
    assert resolved.prompt is None
    assert resolved.negative_prompt is None
    assert supplied_fields(resolved) == supplied_fields(copied)
    assert record_intent(resolved) == record_intent(copied)
    assert raw.prompts == ()
    updated = replace_config(resolved, text_conditioning=tmp_path / "other.safetensors")
    assert len(observations) == 1
    resolve_request_sources(updated)
    assert observations[-1] == tmp_path / "other.safetensors"


def test_surviving_conflict_fails_before_conditioning_read(monkeypatch):
    def unexpected(*args):
        pytest.fail("source observation preceded final policy")

    monkeypatch.setattr(request_sources, "read_text_conditioning_prompts", unexpected)
    request = Krea2Request(text_conditioning="absent.safetensors", source_images=("source.png",))
    with pytest.raises(ValueError, match=r"\[generate\]: source images cannot be combined"):
        resolve_request_sources(request)


def test_layer_override_resolves_tile_conflict_and_preserves_seed_order(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[generate]\nprompt = "scene"\nvae_tile_size = 128\nseeds = [9, 2, 9]\nimages_per_prompt = 2\n'
    )
    options = build_parser().parse_args(
        [
            "--config",
            str(path),
            "--images-per-prompt",
            "1",
            "--set",
            "generate.vae_tile_stride=64",
            "--print-config",
        ]
    )
    request = assemble(options).request
    assert request.selected_seeds() == (9, 2, 9)
    assert request.vae_tile_stride == 64


def test_bad_lower_profile_grammar_cannot_be_hidden(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[generate]\nturbo_sda_sigma_window = "0.97:0.93"\n')
    options = build_parser().parse_args(
        [
            "--config",
            str(path),
            "--set",
            'generate.turbo_sda_sigma_window=""',
            "--print-config",
        ]
    )
    with pytest.raises(ConfigError, match="turbo_sda_sigma_window"):
        assemble(options)


def test_request_and_config_use_the_same_semantic_field_owners():
    request_fields = {item.name: item for item in fields(Krea2Request)}
    for item in fields(Krea2GenerateConfig):
        assert request_fields[item.name] is item
    assert {
        rule.name
        for item in fields(Krea2GenerateConfig)
        for rule in item.metadata.get("config_rules", ())
    } == {
        "image_seeds",
        "vae_tiling",
        "profiles",
        "references",
        "prompt",
        "recipe_mode",
        "negative_cfg",
    }


@pytest.mark.parametrize("maximum", [-1, True, 1.5])
def test_invalid_length_maximum_is_rejected(maximum):
    with pytest.raises(ValueError, match="maximum length"):
        LengthConstraint(maximum=maximum)


def test_owned_maximum_length_reports_and_enforces_endpoint():
    constraint = LengthConstraint(minimum=1, maximum=3)
    constraint.validate("abc")
    assert constraint.describe()["maximum"] == 3
    with pytest.raises(ValueError, match="at most 3"):
        constraint.validate("abcd")
