"""Klein schedules: the resolution-aware shift, and the explicit override of it."""

import mlx.core as mx
import pytest

from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.sampling import SamplingRecipe, schedule
from kinomlx.models.flux2.types import Flux2Request

TOKENS = 64 * 64


def test_the_shift_produces_one_more_value_than_its_steps() -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, steps=4)
    values = recipe.sigma_array(TOKENS).tolist()
    assert len(values) == 5
    assert (values[0], values[-1]) == (1.0, 0.0)
    assert not recipe.explicit_sigmas


def test_the_shift_depends_on_the_token_grid() -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, steps=4)
    small = recipe.sigma_array(32 * 32).tolist()
    large = recipe.sigma_array(96 * 96).tolist()
    assert small != large


def test_explicit_values_replace_the_shift_and_ignore_the_grid() -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, sigmas=(1.0, 0.6, 0.0))
    assert recipe.explicit_sigmas
    assert recipe.sigma_array(TOKENS).tolist() == pytest.approx([1.0, 0.6, 0.0])
    assert recipe.sigma_array(TOKENS).dtype == mx.float32
    assert recipe.sigma_array(32 * 32).tolist() == recipe.sigma_array(96 * 96).tolist()


def test_the_array_fixes_the_resolved_step_count() -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, steps=12, sigmas=(1.0, 0.5, 0.0))
    assert recipe.resolved()[0] == 2


def test_an_absent_array_leaves_the_profile_step_count() -> None:
    profile_steps = SamplingRecipe(Flux2Variant.DISTILLED).resolved()[0]
    assert profile_steps >= 1
    assert SamplingRecipe(Flux2Variant.DISTILLED).sigma_array(TOKENS).shape == (profile_steps + 1,)


def test_the_built_in_shift_is_unchanged_by_the_override_seam() -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, steps=6)
    direct = schedule(TOKENS, 6)
    assert mx.array_equal(recipe.sigma_array(TOKENS), direct).item()


def test_a_request_carries_its_array_into_the_recipe() -> None:
    request = Flux2Request(prompt="p", sigmas=(1.0, 0.75, 0.25, 0.0))
    recipe = request.recipe("distilled")
    assert recipe.sigmas == (1.0, 0.75, 0.25, 0.0)
    assert recipe.resolved()[0] == 3


def test_a_malformed_array_is_rejected_when_the_recipe_resolves() -> None:
    with pytest.raises(ValueError, match="decrease strictly"):
        Flux2Request(prompt="p", sigmas=(1.0, 0.2, 0.6, 0.0)).recipe("distilled").sigma_array(
            TOKENS
        )
