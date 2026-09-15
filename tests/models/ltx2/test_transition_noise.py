"""LTX-2 ancestral-transition noise tests."""

import mlx.core as mx
import numpy as np

from kinomlx.models.ltx2.transition_noise import SeededGaussianNoise


def test_seeded_transition_noise_is_reproducible_and_ordered() -> None:
    """A fresh stream repeats, while video then audio consume distinct draws."""
    first = SeededGaussianNoise(seed=10_042)
    second = SeededGaussianNoise(seed=10_042)
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
