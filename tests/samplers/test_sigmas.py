"""Sigma schedules: presentation, validated user schedules, and the scheduler seam."""

import mlx.core as mx
import pytest

from kinomlx.samplers.sigmas import (
    ExplicitScheduler,
    ScheduleRequest,
    SigmaSchedule,
    parse_sigmas,
    render_sigmas,
    resolve_schedule,
    resolve_sigma_schedule,
    resolved_sigmas,
    sigma_metadata,
)


def test_schedule_reports_the_steps_its_values_drive() -> None:
    assert SigmaSchedule((1.0, 0.5, 0.0)).steps == 2


def test_schedule_keeps_a_tail_above_zero_and_a_head_below_one() -> None:
    # LTX-2 stage 2 continues from stage 1's tail, and stopping early is a choice.
    assert SigmaSchedule((0.9, 0.4)).steps == 1


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((1.0,), "at least two values"),
        ((1.0, float("nan")), "must be finite"),
        ((1.2, 0.0), "between 0 and 1"),
        ((1.0, -0.1), "between 0 and 1"),
        ((0.5, 0.9, 0.0), "decrease strictly"),
        ((0.5, 0.5, 0.0), "decrease strictly"),
    ],
)
def test_schedule_rejects_a_malformed_array(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        SigmaSchedule(values)


def test_schedule_converts_to_an_fp32_array_and_plain_metadata() -> None:
    schedule = SigmaSchedule((1.0, 0.25, 0.0))
    assert schedule.to_array().dtype == mx.float32
    assert schedule.to_metadata() == [1.0, 0.25, 0.0]


def test_short_schedules_render_in_full() -> None:
    assert render_sigmas((1.0, 0.9567238, 0.0)) == "1, 0.956724, 0"


def test_long_schedules_keep_their_head_and_report_their_length() -> None:
    rendered = render_sigmas([1.0 - index / 40 for index in range(40)], limit=8)
    assert rendered.startswith("1, 0.975, 0.95, 0.925, 0.9,")
    assert rendered.endswith("(40 values)")


def test_rendering_accepts_an_mlx_array_and_an_empty_schedule() -> None:
    assert render_sigmas(mx.array([1.0, 0.0], dtype=mx.float32)) == "1, 0"
    assert render_sigmas(()) == "none"


def test_metadata_passes_through_none_and_flattens_an_array() -> None:
    assert sigma_metadata(None) is None
    assert sigma_metadata(mx.array([1.0, 0.5, 0.0], dtype=mx.float32)) == [1.0, 0.5, 0.0]


def test_text_spelling_parses_and_validates_in_one_step() -> None:
    assert parse_sigmas("1, 0.75,0.5, 0") == (1.0, 0.75, 0.5, 0.0)
    with pytest.raises(ValueError, match="decrease strictly"):
        parse_sigmas("1,0.5,0.75,0")
    with pytest.raises(ValueError, match="all be numbers"):
        parse_sigmas("1,low,0")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_sigmas("1,,0")


def test_a_declared_step_count_must_agree_with_the_array() -> None:
    assert resolve_sigma_schedule((1.0, 0.5, 0.0), steps=2, label="sigmas").steps == 2
    assert resolve_sigma_schedule((1.0, 0.5, 0.0), steps=None, label="sigmas").steps == 2
    with pytest.raises(ValueError, match="drive 2 steps, but steps is set to 8"):
        resolve_sigma_schedule((1.0, 0.5, 0.0), steps=8, label="sigmas")


def test_explicit_scheduler_returns_its_values_for_any_request() -> None:
    scheduler = ExplicitScheduler((1.0, 0.5, 0.0))
    resolved = resolve_schedule(scheduler, ScheduleRequest(steps=2, image_tokens=4096))
    assert resolved.values == (1.0, 0.5, 0.0)
    assert scheduler.name == "explicit"


def test_a_request_needs_no_token_grid_when_the_scheduler_ignores_it() -> None:
    assert resolve_schedule(ExplicitScheduler((1.0, 0.0)), ScheduleRequest(steps=1)).steps == 1


def test_resolving_names_the_scheduler_that_returned_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="scheduler 'explicit' returned 2 values"):
        resolve_schedule(ExplicitScheduler((1.0, 0.0)), ScheduleRequest(steps=5))


def test_resolving_validates_whatever_a_scheduler_produced() -> None:
    with pytest.raises(ValueError, match="decrease strictly"):
        resolve_schedule(ExplicitScheduler((0.5, 0.9, 0.0)), ScheduleRequest(steps=2))


def test_a_custom_scheduler_satisfies_the_seam_without_inheriting_anything() -> None:
    class Linear:
        """A spacing written against the protocol alone."""

        name = "linear"

        def prepare(self, request: ScheduleRequest) -> tuple[float, ...]:
            return tuple(1.0 - index / request.steps for index in range(request.steps + 1))

    resolved = resolve_schedule(Linear(), ScheduleRequest(steps=4))
    assert resolved.values == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert resolved.steps == 4


def test_a_supplied_schedule_wins_over_the_generated_one() -> None:
    assert resolved_sigmas((1.0, 0.5, 0.0), lambda: (1.0, 0.9, 0.8, 0.0), label="test").values == (
        1.0,
        0.5,
        0.0,
    )


def test_an_absent_schedule_falls_through_to_the_generated_one() -> None:
    assert resolved_sigmas((), lambda: (1.0, 0.9, 0.0), label="test").values == (1.0, 0.9, 0.0)


def test_the_generated_branch_is_validated_too() -> None:
    # A spacing that emits a bad level is refused here, not inside a denoise loop.
    with pytest.raises(ValueError, match="test: sigma values must decrease strictly"):
        resolved_sigmas((), lambda: (0.5, 0.9, 0.0), label="test")


def test_a_supplied_schedule_is_validated_and_named() -> None:
    with pytest.raises(ValueError, match="Krea 2 schedule: sigma values must be finite"):
        resolved_sigmas((1.0, float("nan")), lambda: (1.0, 0.0), label="Krea 2 schedule")


def test_the_generator_is_not_called_when_a_schedule_is_supplied() -> None:
    calls = []

    def generate():
        calls.append(1)
        return (1.0, 0.0)

    resolved_sigmas((1.0, 0.5, 0.0), generate, label="test")
    assert calls == []


def test_an_mlx_array_from_a_generator_is_accepted() -> None:
    values = resolved_sigmas(
        (), lambda: mx.array([1.0, 0.5, 0.0], dtype=mx.float32), label="test"
    ).values
    assert values == (1.0, 0.5, 0.0)
