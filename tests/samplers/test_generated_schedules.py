"""Every model's own spacing must survive the validation the helper applies.

Validating the generated branch is only safe if no legitimate configuration emits
a level that fails the contract. A float32 tie at a high step count would turn a
working run into an error, so the range each model actually accepts is swept here
rather than argued about.
"""

import math

import mlx.core as mx
import pytest

from kinomlx.models.flux2.catalog import Flux2Variant
from kinomlx.models.flux2.sampling import SamplingRecipe, schedule
from kinomlx.models.krea2.catalog import Krea2Variant
from kinomlx.models.krea2.sampling import Krea2Recipe, timestep_schedule
from kinomlx.models.ltx2.sigmas import stage_sigmas
from kinomlx.samplers.sigmas import SigmaSchedule

STEPS = (1, 2, 3, 4, 8, 16, 28, 52, 120)
MU = (0.0, 0.5, 0.8, 1.15, 1.9, 3.0, 5.0)
TOKENS = (256, 1024, 4096, 4300, 9216, 16384, 65536)


def _krea(steps: int, mu: float) -> Krea2Recipe:
    return Krea2Recipe(Krea2Variant.TURBO, steps, 0.0, mu, mu, False, False)


def _shifted_uniform(steps: int, mu: float) -> mx.array:
    """The pre-helper expression, kept here as the numeric baseline."""
    timesteps = mx.linspace(1.0, 0.0, steps + 1, dtype=mx.float32)
    exp_mu = math.exp(mu)
    shifted = exp_mu / (exp_mu + (1.0 / timesteps - 1.0))
    return mx.concatenate([shifted[:-1], mx.zeros((1,), dtype=mx.float32)])


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("mu", MU)
def test_the_krea_shift_satisfies_the_schedule_contract(steps, mu) -> None:
    values = timestep_schedule(_krea(steps, mu)).tolist()
    assert SigmaSchedule(tuple(values)).steps == steps


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("mu", MU)
def test_the_helper_does_not_change_the_krea_shift(steps, mu) -> None:
    assert mx.array_equal(timestep_schedule(_krea(steps, mu)), _shifted_uniform(steps, mu)).item()


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_the_klein_shift_satisfies_the_schedule_contract(steps, tokens) -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, steps=steps)
    assert SigmaSchedule(tuple(recipe.sigma_array(tokens).tolist())).steps == steps


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_the_helper_does_not_change_the_klein_shift(steps, tokens) -> None:
    recipe = SamplingRecipe(Flux2Variant.DISTILLED, steps=steps)
    assert mx.array_equal(recipe.sigma_array(tokens), schedule(tokens, steps)).item()


def test_the_ltx2_pinned_schedules_satisfy_the_contract() -> None:
    first, second = stage_sigmas()
    assert SigmaSchedule(first).steps == 8
    assert SigmaSchedule(second).steps == 3
