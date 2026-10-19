"""LTX-2 denoises in two stages, so each schedule is pinned and overridden alone."""

import pytest

from kinomlx.models.ltx2.pipelines.distilled import sampling_facts
from kinomlx.models.ltx2.sigmas import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
    stage_sigmas,
)
from kinomlx.models.ltx2.types import LTX2Request
from kinomlx.models.ltx2.vocabularies import LTX2StageSampler

_PUBLISHED = (LTX2StageSampler.DETERMINISTIC, LTX2StageSampler.DETERMINISTIC)


def test_no_override_returns_both_pinned_schedules() -> None:
    assert stage_sigmas() == (DISTILLED_STAGE_1_SIGMAS, DISTILLED_STAGE_2_SIGMAS)


def test_the_pinned_schedules_drive_eight_and_three_steps() -> None:
    first, second = stage_sigmas()
    assert (len(first) - 1, len(second) - 1) == (8, 3)


def test_stage_two_legitimately_starts_below_one() -> None:
    # It continues from stage 1's tail after the spatial upscaler.
    _, second = stage_sigmas()
    assert second[0] < 1.0
    assert second[-1] == 0.0


@pytest.mark.parametrize("stage", [1, 2])
def test_overriding_one_stage_leaves_the_other_pinned(stage) -> None:
    supplied = (0.9, 0.5, 0.0)
    first, second = stage_sigmas(*((supplied, ()) if stage == 1 else ((), supplied)))
    changed, kept = (first, second) if stage == 1 else (second, first)
    pinned = DISTILLED_STAGE_2_SIGMAS if stage == 1 else DISTILLED_STAGE_1_SIGMAS
    assert changed == supplied
    assert kept == pinned


def test_both_stages_can_be_overridden_together() -> None:
    first, second = stage_sigmas((1.0, 0.5, 0.0), (0.4, 0.0))
    assert (first, second) == ((1.0, 0.5, 0.0), (0.4, 0.0))


@pytest.mark.parametrize("values", [(0.5, 0.9, 0.0), (1.2, 0.0), (1.0,)])
@pytest.mark.parametrize("stage", [1, 2])
def test_a_malformed_stage_array_is_rejected(values, stage) -> None:
    with pytest.raises(ValueError, match="decrease strictly|between 0 and 1|at least two values"):
        stage_sigmas(*((values, ()) if stage == 1 else ((), values)))


def test_announced_facts_report_both_schedules_and_their_step_counts() -> None:
    facts = sampling_facts(LTX2Request(prompt="p"), _PUBLISHED)
    assert facts["stage_1_sigmas"] == list(DISTILLED_STAGE_1_SIGMAS)
    assert facts["stage_2_sigmas"] == list(DISTILLED_STAGE_2_SIGMAS)
    assert (facts["stage_1_steps"], facts["stage_2_steps"]) == (8, 3)
    assert facts["sigma_source"] == "distilled-literal"


def test_announced_facts_mark_an_override_and_its_new_step_count() -> None:
    request = LTX2Request(prompt="p", stage_2_sigmas=(0.9, 0.5, 0.0))
    facts = sampling_facts(request, _PUBLISHED)
    assert facts["sigma_source"] == "explicit"
    assert facts["stage_2_sigmas"] == [0.9, 0.5, 0.0]
    assert (facts["stage_1_steps"], facts["stage_2_steps"]) == (8, 2)
