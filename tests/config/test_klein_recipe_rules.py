"""Direct sampling and layered Klein requests share recipe and capability policy."""

import pytest

from kinomlx.config_rules import validate_capabilities, validate_record
from kinomlx.models.flux2.adapters.inputs import Flux2AdapterConfig
from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.sampling import SamplingRecipe
from kinomlx.models.flux2.types import Flux2Request


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": "bad"}, "steps"),
        ({"guidance": 0.5}, "guidance"),
        ({"guidance": float("nan")}, "guidance"),
        ({"cfg_mode": "unknown"}, "cfg_mode"),
        ({"sigmas": (1.0, -0.5)}, "sigmas"),
        ({"sigmas": (1.0, 0.6, 0.7, 0.0)}, "sigmas"),
    ],
)
def test_direct_recipe_rejects_malformed_input_at_construction(kwargs, error):
    with pytest.raises(ValueError, match=error):
        SamplingRecipe(**kwargs)


@pytest.mark.parametrize("variant", ["distilled", "kv"])
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"guidance": 4.0}, "External CFG"), ({"cfg_mode": "batched"}, "base model")],
)
def test_request_and_direct_recipe_share_profile_restrictions(variant, kwargs, error):
    request = Flux2Request(prompt="scene", **kwargs)
    validate_record(request, location="[generate]")
    with pytest.raises(ValueError, match=error):
        validate_capabilities(request, {"variant": variant}, location="[generate]")
    recipe = SamplingRecipe(variant=Flux2Variant(variant), **kwargs)
    with pytest.raises(ValueError, match=error):
        recipe.resolved()


@pytest.mark.parametrize("variant", list(Flux2Variant))
def test_profile_and_explicit_schedule_defaults_share_one_resolution(variant):
    request = Flux2Request(prompt="scene", steps=12, sigmas=(1.0, 0.5, 0.0))
    request.validate(variant)
    direct = SamplingRecipe(variant, steps=12, sigmas=request.sigmas)
    assert request.recipe(variant).resolved() == direct.resolved()
    assert direct.resolved()[0] == 2
    assert request.steps == 12


@pytest.mark.parametrize("variant", ["base", "distilled", "kv"])
def test_negative_prompt_requires_active_guidance(variant):
    request = Flux2Request(prompt="scene", negative_prompt="noise", guidance=1.0)
    validate_record(request, location="[generate]")
    with pytest.raises(ValueError, match="Negative prompts require active base CFG"):
        request.validate(variant)


def test_negative_row_cardinality_is_final_input_policy():
    request = Flux2Request(prompt=("a", "b"), negative_prompt=("x", "y", "z"))
    with pytest.raises(ValueError, match="negative_prompt requires one value or one per prompt"):
        validate_record(request, location="[generate]")


def test_capability_context_is_required_for_klein():
    with pytest.raises(ValueError, match="missing required observations: variant"):
        validate_capabilities(Flux2Request(prompt="scene"), {}, location="[generate]")


def test_adapter_schedule_is_checked_against_resolved_explicit_schedule():
    request = Flux2Request(
        prompt="scene",
        steps=12,
        sigmas=(1.0, 0.5, 0.0),
        adapters=(Flux2AdapterConfig(path="adapter", step_strengths=(0.2, 0.8)),),
    )
    request.validate("distilled")
    mismatch = Flux2Request(
        prompt="scene",
        steps=2,
        adapters=(Flux2AdapterConfig(path="adapter", step_strengths=(0.2, 0.8, 1.0)),),
    )
    with pytest.raises(ValueError, match="step"):
        mismatch.validate("distilled")


def test_recompute_requires_kv_profile_at_final_recipe_validation():
    recipe = SamplingRecipe(recompute_reference_kv=True)
    with pytest.raises(ValueError, match="requires the KV recipe"):
        recipe.resolved()
    assert SamplingRecipe(Flux2Variant.KV, recompute_reference_kv=True).resolved() == (4, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"width": 65}, "multiples of 16"),
        ({"reference_fit": "fit"}, "requires source_images"),
        ({"match_source_aspect": "long"}, "requires a reference"),
        (
            {
                "source_images": ("missing.png",),
                "width": 512,
                "height": 512,
                "match_source_aspect": "width",
            },
            "pins one dimension",
        ),
        (
            {"source_images": ("missing.png",), "height": 512, "match_source_aspect": "width"},
            "explicit aspect axis",
        ),
    ],
)
def test_geometry_relationships_fail_before_source_reads(kwargs, error):
    request = Flux2Request(prompt="scene", **kwargs)
    with pytest.raises(ValueError, match=error):
        validate_record(request, location="[generate]")


def test_single_reference_can_pin_an_unaligned_size_before_geometry_resolution():
    request = Flux2Request(prompt="scene", source_images=("missing.png",), width=513)
    validate_record(request, location="[generate]")
