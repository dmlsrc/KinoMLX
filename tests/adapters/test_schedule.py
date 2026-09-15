"""Per-step strength arrays, their text spelling, and their step-count contract."""

import pytest

from kinomlx.adapters.inputs import ScheduledAdapterInput
from kinomlx.adapters.schedule import (
    AdapterSchedule,
    parse_step_strengths,
    require_schedule_length,
)
from kinomlx.adapters.selection import (
    AdapterSelection,
    require_adapter_schedule_steps,
)


def test_schedule_reads_one_strength_per_step() -> None:
    schedule = AdapterSchedule((1.0, 1.0, 0.0, 0.25))
    assert len(schedule) == 4
    assert [schedule.at(index) for index in range(4)] == [1.0, 1.0, 0.0, 0.25]


def test_schedule_rejects_a_step_outside_its_length() -> None:
    schedule = AdapterSchedule((1.0, 0.0))
    with pytest.raises(IndexError, match="outside a 2-step"):
        schedule.at(2)


@pytest.mark.parametrize("values", [(), (1.0, float("nan")), (float("inf"),)])
def test_schedule_requires_a_finite_nonempty_array(values) -> None:
    with pytest.raises(ValueError, match="at least one strength|must be finite"):
        AdapterSchedule(values)


def test_schedule_reports_effectiveness_and_constancy() -> None:
    assert not AdapterSchedule((0.0, 0.0)).effective
    assert AdapterSchedule((0.0, 0.5)).effective
    assert AdapterSchedule((0.7, 0.7)).constant
    assert not AdapterSchedule((0.7, 0.0)).constant


def test_schedule_metadata_records_the_array_and_active_steps() -> None:
    assert AdapterSchedule((1.0, 1.0, 0.0)).to_metadata() == {
        "per_step": [1.0, 1.0, 0.0],
        "active_steps": [0, 1],
    }


def test_text_spelling_parses_a_comma_separated_array() -> None:
    assert parse_step_strengths("1, 1,0,0.5") == (1.0, 1.0, 0.0, 0.5)


@pytest.mark.parametrize("text", ["", "1,,0", "1,high", "1,inf"])
def test_text_spelling_rejects_malformed_arrays(text) -> None:
    with pytest.raises(ValueError, match="comma-separated|all be numbers|must be finite"):
        parse_step_strengths(text)


def test_length_check_names_both_counts() -> None:
    with pytest.raises(ValueError, match="has 8 step strengths.*runs 4 steps"):
        require_schedule_length(AdapterSchedule((1.0,) * 8), 4, label="adapter a.safetensors")


def test_record_selections_schedule_one_item_and_leave_others_fused(tmp_path) -> None:
    selections = (
        ScheduledAdapterInput(tmp_path / "a.safetensors", step_strengths=(1.0, 0.0)).selection(),
        ScheduledAdapterInput(tmp_path / "b.safetensors").selection(),
    )
    assert selections[0].schedule is not None
    assert selections[1].schedule is None
    assert selections[1].strength == 1.0


def test_record_selection_rejects_a_constant_strength_beside_an_array(tmp_path) -> None:
    with pytest.raises(ValueError, match="not both"):
        ScheduledAdapterInput(
            tmp_path / "a.safetensors", strength=0.8, step_strengths=(1.0, 0.0)
        ).selection()


def test_selection_reports_schedule_effectiveness_and_first_strength(tmp_path) -> None:
    path = tmp_path / "a.safetensors"
    off = AdapterSelection(path=path, strength=1.0, schedule=AdapterSchedule((0.0, 0.0)))
    gated = AdapterSelection(path=path, strength=1.0, schedule=AdapterSchedule((0.5, 0.0)))
    assert not off.effective
    assert gated.effective
    assert gated.effective_strength == 0.5
    assert AdapterSelection(path=path, strength=0.0).effective is False


def test_step_count_check_covers_every_scheduled_selection(tmp_path) -> None:
    selections = (
        AdapterSelection(path=tmp_path / "a.safetensors", strength=1.0),
        AdapterSelection(
            path=tmp_path / "b.safetensors", strength=1.0, schedule=AdapterSchedule((1.0, 0.0))
        ),
    )
    require_adapter_schedule_steps(selections, 2)
    with pytest.raises(ValueError, match="b.safetensors"):
        require_adapter_schedule_steps(selections, 8)
