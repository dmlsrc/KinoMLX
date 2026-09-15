"""Klein request choices have one owner across input surfaces."""

import re

import pytest

from kinomlx.config import ConfigError
from kinomlx.config_constraints import field_choices
from kinomlx.config_records import replace_config, supplied_fields
from kinomlx.models.flux2.arguments import build_parser
from kinomlx.models.flux2.config import assemble
from kinomlx.models.flux2.config_schema import CONFIG_SCHEMA
from kinomlx.models.flux2.types import Flux2Request, Flux2RestartConfig


@pytest.mark.parametrize(
    "name", ["seed_mode", "noise_backend", "reference_fit", "match_source_aspect", "cfg_mode"]
)
def test_request_choice_owner_agrees_with_schema_and_rejects_api_unknowns(name):
    assert CONFIG_SCHEMA.field(("generate", name)).choices == field_choices(Flux2Request, name)
    with pytest.raises(ValueError, match=name):
        Flux2Request(**{name: "unknown"})


def test_invalid_lower_reference_fit_cannot_be_hidden_by_cli(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[generate]\nreference_fit = "unknown"\n')
    options = build_parser().parse_args(
        ["--config", str(config), "--reference-fit", "fit", "--print-config"]
    )
    with pytest.raises(ConfigError, match="reference_fit"):
        assemble(options)


def test_common_constructor_retains_nested_adapter_presence_without_model_coercer():
    request = Flux2Request(adapters=[{"path": "adapter.safetensors", "strength": 1.0}])
    assert isinstance(request.adapters, tuple)
    assert request.adapters[0].path.name == "adapter.safetensors"
    assert supplied_fields(request.adapters[0]) == frozenset({"path", "strength"})
    assert supplied_fields(request) == frozenset({"adapters"})


@pytest.mark.parametrize("index", [0, -1, True, 1.5])
def test_restart_index_rejected_at_local_admission(index):
    with pytest.raises(ValueError, match="index"):
        Flux2RestartConfig(run="run.json", index=index)


@pytest.mark.parametrize(
    ("values", "repair", "message"),
    [
        (
            {"prompt": "hello", "prompt_unconditional": True},
            {"prompt_unconditional": False},
            "prompt_unconditional cannot be combined",
        ),
        (
            {"seeds": (9, 2), "images_per_prompt": 2},
            {"images_per_prompt": 1},
            "Explicit seeds cannot be combined",
        ),
        (
            {"seed": 2**64 - 1, "images_per_prompt": 2},
            {"images_per_prompt": 1},
            "seeds must be unsigned 64-bit integers",
        ),
        (
            {"text_conditioning": "saved.safetensors", "reference_fit": "fit"},
            {"reference_fit": None},
            "Saved conditioning replaces source images",
        ),
        (
            {"text_conditioning": "saved.safetensors", "prompt": ("a", "b")},
            {"prompt": "a"},
            "Saved conditioning replay requires one prompt",
        ),
    ],
)
def test_final_request_rules_admit_local_values_then_validate_surviving_combination(
    values, repair, message
):
    request = Flux2Request(**values)
    with pytest.raises(ValueError, match=rf"\[generate\]: {re.escape(message)}"):
        request.validate("distilled")
    replace_config(request, **repair).validate("distilled")


def test_higher_layer_repairs_seed_combination(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[generate]\nseeds = [9, 2, 9]\nimages_per_prompt = 2\n")
    options = build_parser().parse_args(
        ["--config", str(config), "--images-per-prompt", "1", "--print-config"]
    )
    assert assemble(options).generate.selected_seeds() == (9, 2, 9)


@pytest.mark.parametrize(
    "values",
    [{"seed": -1}, {"seed": 2**64}, {"seeds": ()}, {"seeds": (-1,)}, {"images_per_prompt": 0}],
)
def test_invalid_seed_values_fail_without_final_validation(values):
    with pytest.raises(ValueError, match=next(iter(values))):
        Flux2Request(**values)


def test_conflicting_conditioning_fails_before_source_access(monkeypatch):
    from kinomlx.models.flux2 import config

    def unexpected(*args, **kwargs):
        pytest.fail("opened a source before validating the surviving request")

    monkeypatch.setattr(config, "load_conditioning", unexpected)
    monkeypatch.setattr(config, "image_size", unexpected)
    options = build_parser().parse_args(
        [
            "--text-conditioning",
            "saved.safetensors",
            "--source-image",
            "source.png",
            "--print-config",
        ]
    )
    with pytest.raises(ValueError, match=r"\[generate\]: Saved conditioning replaces"):
        assemble(options)
