"""Behavioral tests for ``kinomlx.models.ltx2.sigmas``."""

from __future__ import annotations

from itertools import pairwise

import mlx.core as mx
import numpy as np

from kinomlx.models.ltx2.sigmas import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
    DISTILLED_STAGE_2_START_INDEX,
    distilled_stage_1_sigmas,
    distilled_stage_2_sigmas,
)

# ---------------------------------------------------------------------------
# Stage 1 - 9 values, 8 steps
# ---------------------------------------------------------------------------


def test_stage_1_has_nine_values() -> None:
    assert len(DISTILLED_STAGE_1_SIGMAS) == 9


def test_stage_1_starts_at_one_ends_at_zero() -> None:
    assert DISTILLED_STAGE_1_SIGMAS[0] == 1.0
    assert DISTILLED_STAGE_1_SIGMAS[-1] == 0.0


def test_stage_1_host_values_match_official_float32_tensor() -> None:
    official_values = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
    expected = tuple(float(value) for value in np.asarray(official_values, dtype=np.float32))
    assert expected == DISTILLED_STAGE_1_SIGMAS


def test_stage_1_is_strictly_monotonic_decreasing() -> None:
    for prev, cur in pairwise(DISTILLED_STAGE_1_SIGMAS):
        assert cur < prev, f"non-decreasing pair: {prev} -> {cur}"


def test_stage_1_as_mx_array_is_float32() -> None:
    arr = distilled_stage_1_sigmas()
    assert arr.dtype == mx.float32
    assert arr.shape == (9,)
    np.testing.assert_array_equal(
        np.asarray(arr), np.asarray(DISTILLED_STAGE_1_SIGMAS, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Stage 2 - 4 values, 3 steps; mirrors the tail of stage 1
# ---------------------------------------------------------------------------


def test_stage_2_has_four_values() -> None:
    assert len(DISTILLED_STAGE_2_SIGMAS) == 4


def test_stage_2_ends_at_zero() -> None:
    assert DISTILLED_STAGE_2_SIGMAS[-1] == 0.0


def test_stage_2_is_strictly_monotonic_decreasing() -> None:
    for prev, cur in pairwise(DISTILLED_STAGE_2_SIGMAS):
        assert cur < prev


def test_stage_2_host_values_match_independent_official_float32_tensor() -> None:
    official_values = (0.909375, 0.725, 0.421875, 0.0)
    expected = tuple(float(value) for value in np.asarray(official_values, dtype=np.float32))
    assert expected == DISTILLED_STAGE_2_SIGMAS


def test_stage_2_mirrors_tail_of_stage_1() -> None:
    """Stage 2's complete schedule equals stage 1's named tail.

    Stage 1 ends ..., 0.909375, 0.725, 0.421875, 0.0.
    Stage 2 begins 0.909375, 0.725, 0.421875, 0.0.
    The non-zero values and terminal zero overlap exactly.
    """
    assert DISTILLED_STAGE_1_SIGMAS[DISTILLED_STAGE_2_START_INDEX:] == DISTILLED_STAGE_2_SIGMAS


def test_stage_2_as_mx_array_is_float32() -> None:
    arr = distilled_stage_2_sigmas()
    assert arr.dtype == mx.float32
    assert arr.shape == (4,)
