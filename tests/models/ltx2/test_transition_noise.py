"""LTX-2 ancestral-transition noise tests."""

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.models.ltx2.transition_noise import SeededGaussianNoise


def test_seeded_transition_noise_is_reproducible_and_ordered() -> None:
    """A fresh stream repeats, while video then audio consume distinct draws."""
    first = SeededGaussianNoise(seed=10_042, stage=1, transitions=7)
    second = SeededGaussianNoise(seed=10_042, stage=1, transitions=7)
    arguments = {
        "stage": 1,
        "transition": 0,
        "shape": (1, 2, 3),
        "dtype": mx.float32,
    }
    first_video = first(modality="video", **arguments)
    first_audio = first(modality="audio", **arguments)
    second_video = second(modality="video", **arguments)
    second_audio = second(modality="audio", **arguments)

    np.testing.assert_array_equal(np.asarray(first_video), np.asarray(second_video))
    np.testing.assert_array_equal(np.asarray(first_audio), np.asarray(second_audio))
    assert not np.array_equal(np.asarray(first_video), np.asarray(first_audio))


def test_transition_bound_follows_the_schedule_rather_than_the_pinned_length() -> None:
    """A longer stage-1 schedule draws past the distilled schedule's last transition.

    The default pin drives seven transitions that draw noise, so a fixed ceiling of
    six fit it exactly and refused every longer override partway through stage 1.
    """
    arguments = {"stage": 1, "modality": "video", "shape": (1, 2), "dtype": mx.float32}
    longer = SeededGaussianNoise(seed=1, stage=1, transitions=11)
    for transition in range(11):
        assert longer(transition=transition, **arguments).shape == (1, 2)

    with pytest.raises(ValueError, match="drives 11 transition"):
        longer(transition=11, **arguments)

    default = SeededGaussianNoise(seed=1, stage=1, transitions=8)
    with pytest.raises(ValueError, match="drives 8 transition"):
        default(transition=8, **arguments)
    with pytest.raises(ValueError, match="drives 8 transition"):
        default(transition=-1, **arguments)


def test_a_schedule_with_no_transitions_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="at least one transition"):
        SeededGaussianNoise(seed=1, stage=1, transitions=0)


def test_each_stream_draws_only_for_its_own_stage() -> None:
    """Stage 2 opens its own stream; neither stream answers for the other stage."""
    arguments = {"transition": 0, "modality": "video", "shape": (1, 2), "dtype": mx.float32}
    stage_2 = SeededGaussianNoise(seed=20_042, stage=2, transitions=3)
    assert stage_2(stage=2, **arguments).shape == (1, 2)
    with pytest.raises(ValueError, match="stage-2 ancestral noise stream was asked for stage 1"):
        stage_2(stage=1, **arguments)
    with pytest.raises(ValueError, match="stage-2 schedule, which drives 3 transition"):
        stage_2(stage=2, **{**arguments, "transition": 3})


@pytest.mark.parametrize("stage", [0, 3])
def test_a_stage_outside_the_two_distilled_stages_is_refused(stage: int) -> None:
    with pytest.raises(ValueError, match="stage 1 or 2"):
        SeededGaussianNoise(seed=1, stage=stage, transitions=3)
