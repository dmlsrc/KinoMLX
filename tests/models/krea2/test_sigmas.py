"""Explicit Krea schedules supersede the resolved step count and shift."""

import itertools

import mlx.core as mx
import pytest

from kinomlx.models.krea2.catalog import Krea2Variant
from kinomlx.models.krea2.metadata import default_scheduler_config
from kinomlx.models.krea2.sampling import (
    Krea2Recipe,
    resolve_geometry,
    resolve_recipe,
    timestep_schedule,
)
from kinomlx.models.krea2.types import Krea2Request


def _recipe(**changes) -> Krea2Recipe:
    base: dict[str, object] = {
        "variant": Krea2Variant.TURBO,
        "steps": 8,
        "guidance": 0.0,
        "mu": 1.15,
        "effective_mu": 1.15,
        "cfg_enabled": False,
        "matches_variant_defaults": True,
    }
    return Krea2Recipe(**{**base, **changes})


def _resolved(request: Krea2Request) -> Krea2Recipe:
    geometry = resolve_geometry(request, spatial_compression=8, patch_size=2)
    return resolve_recipe(Krea2Variant.TURBO, request, geometry, default_scheduler_config())


def test_a_generated_schedule_starts_at_one_and_ends_at_zero() -> None:
    values = timestep_schedule(_recipe()).tolist()
    assert len(values) == 9
    assert (values[0], values[-1]) == (1.0, 0.0)
    assert all(later < earlier for earlier, later in itertools.pairwise(values))
    assert not _recipe().explicit_sigmas


def test_the_trained_turbo_node_is_the_second_step() -> None:
    # The community SDA adapter trains at this level; its gate covers steps 0 and 1.
    assert round(timestep_schedule(_recipe()).tolist()[1], 6) == 0.956724


def test_explicit_values_replace_the_generated_schedule() -> None:
    recipe = _recipe(sigmas=(1.0, 0.8, 0.3, 0.0), steps=3)
    assert recipe.explicit_sigmas
    assert timestep_schedule(recipe).tolist() == pytest.approx([1.0, 0.8, 0.3, 0.0])
    assert timestep_schedule(recipe).dtype == mx.float32


def test_metadata_names_which_source_produced_the_schedule() -> None:
    assert _recipe().to_metadata()["sigma_source"] == "shifted-uniform"
    assert _recipe(sigmas=(1.0, 0.0), steps=1).to_metadata()["sigma_source"] == "explicit"


def test_the_array_fixes_the_step_count_the_recipe_resolves() -> None:
    recipe = _resolved(Krea2Request(prompt="p", sigmas=(1.0, 0.7, 0.4, 0.1, 0.0), steps=8))
    # Four steps, not the eight the configuration layer resolved from the variant.
    assert recipe.steps == 4
    assert recipe.sigmas == (1.0, 0.7, 0.4, 0.1, 0.0)
    assert not recipe.matches_variant_defaults
    assert timestep_schedule(recipe).tolist() == pytest.approx([1.0, 0.7, 0.4, 0.1, 0.0])


def test_an_absent_array_leaves_the_generated_schedule_alone() -> None:
    recipe = _resolved(Krea2Request(prompt="p"))
    assert recipe.sigmas == ()
    assert recipe.steps == 8
    assert recipe.matches_variant_defaults


def test_four_step_profile_preserves_direct_explicit_sampler_values() -> None:
    defaults = _resolved(Krea2Request(prompt="p", turbo_4step=True))
    assert (defaults.steps, defaults.guidance, defaults.mu) == (4, 0.0, 1.15)
    request = Krea2Request(prompt="p", turbo_4step=True, steps=6, guidance=1.25, mu=0.8)
    recipe = _resolved(request)
    assert (recipe.steps, recipe.guidance, recipe.mu) == (6, 1.25, 0.8)
    assert len(timestep_schedule(recipe).tolist()) == 7


def test_a_malformed_array_is_rejected_when_the_request_is_built() -> None:
    with pytest.raises(ValueError, match="decrease strictly"):
        Krea2Request(prompt="p", sigmas=(1.0, 0.5, 0.9, 0.0))


def test_an_explicit_array_supersedes_the_four_step_profile_schedule() -> None:
    request = Krea2Request(prompt="p", sigmas=(1.0, 0.5, 0.0), turbo_4step=True)
    assert request.sigmas == (1.0, 0.5, 0.0)
    assert "an explicit sigma schedule" in " ".join(request.profile_deviations())
