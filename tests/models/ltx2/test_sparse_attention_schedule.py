"""Sparse attention schedules: spelling, step resolution, restart projections, and the loop."""

import mlx.core as mx
import pytest

from kinomlx.config_rules import validate_record
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.models.ltx2.config import assemble
from kinomlx.models.ltx2.denoise import denoise_loop
from kinomlx.models.ltx2.kernels.sol_attention import morton_order
from kinomlx.models.ltx2.sparse_attention import (
    AUTO_STAGE_2_TAUS,
    SparseAttentionSchedule,
    canonical_layer_ranges,
    canonical_step_schedule,
    is_dense_schedule,
    parse_layer_ranges,
    parse_step_schedule,
    resolve_schedule,
    stage_steps,
)
from kinomlx.models.ltx2.types import LatentState, LTX2Request


@pytest.mark.parametrize(
    ("text", "steps", "expected"),
    [
        ("dense", 3, (None, None, None)),
        ("1.5", 3, (1.5, 1.5, 1.5)),
        ("1,1.25,1.5", 3, (1.0, 1.25, 1.5)),
        ("dense,1,dense", 3, (None, 1.0, None)),
        ("5-8=1.5", 8, (None, None, None, None, 1.5, 1.5, 1.5, 1.5)),
        ("2=1, 4-5=0.75,7-7=dense", 8, (None, 1.0, None, 0.75, 0.75, None, None, None)),
        ("1-3=-0.5", 3, (-0.5, -0.5, -0.5)),
    ],
)
def test_schedules_resolve_each_step(text: str, steps: int, expected: tuple) -> None:
    assert parse_step_schedule(text).resolve(steps, label="schedule", stage=1) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "must not be empty"),
        ("  ", "must not be empty"),
        ("1,,2", "empty entry"),
        ("1,x", 'tau number or "dense"'),
        ("inf", "must be finite"),
        ("nan", "must be finite"),
        ("1,2=1.5", "not both"),
        ("0=1", "numbered from 1"),
        ("3-2=1", "numbered from 1"),
        ("a-2=1", "numbered from 1"),
        ("1-3=1,3=2", "sets step 3 twice"),
    ],
)
def test_malformed_schedules_fail_with_the_reason(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_step_schedule(text)


def test_a_schedule_must_match_the_stage_step_count() -> None:
    with pytest.raises(ValueError, match="lists 2 steps, but the stage runs 3"):
        parse_step_schedule("1,2").resolve(3, label="stage_2_sparse_attention", stage=2)
    with pytest.raises(ValueError, match="names step 5, but the stage runs 3"):
        parse_step_schedule("2-5=1").resolve(3, label="stage_2_sparse_attention", stage=2)


def test_auto_takes_sol_engines_stage_2_schedule() -> None:
    assert AUTO_STAGE_2_TAUS == (1.0, 1.25, 1.5)
    auto = parse_step_schedule(" auto ")
    assert auto.resolve(3, label="stage_2_sparse_attention", stage=2) == AUTO_STAGE_2_TAUS
    assert canonical_step_schedule(" auto ") == "auto"
    assert not is_dense_schedule("auto")
    with pytest.raises(ValueError, match="stage-2 schedule; stage 1 takes taus"):
        auto.resolve(8, label="stage_1_sparse_attention", stage=1)
    with pytest.raises(ValueError, match="but the stage runs 4 steps; write the taus out"):
        auto.resolve(4, label="stage_2_sparse_attention", stage=2)
    with pytest.raises(ValueError, match="tau number"):
        parse_step_schedule("auto,1")


def test_requests_take_auto_on_stage_2_only() -> None:
    request = LTX2Request(prompt="p", stage_2_sparse_attention="auto")
    validate_record(request, location="[generate]")
    first, second = request.sparse_attention_schedules()
    assert not first.active
    assert second == SparseAttentionSchedule((1.0, 1.25, 1.5), (0,))
    # The spelled-out schedule is the same stage 2, so a restart between them changes nothing.
    spelled = LTX2Request(prompt="p", stage_2_sparse_attention="1,1.25,1.5")
    assert request.sparse_attention_stage_profiles == spelled.sparse_attention_stage_profiles
    with pytest.raises(ValueError, match="stage-2 schedule"):
        validate_record(
            LTX2Request(prompt="p", stage_1_sparse_attention="auto"), location="[generate]"
        )
    with pytest.raises(ValueError, match="runs 2 steps"):
        validate_record(
            LTX2Request(
                prompt="p", stage_2_sparse_attention="auto", stage_2_sigmas=(0.9, 0.5, 0.0)
            ),
            location="[generate]",
        )


def test_cli_switches_stage_2_to_auto() -> None:
    options = build_parser().parse_args(
        ["--prompt", "p", "--stage2-sparse-attention", "auto", "--print-config"]
    )
    request = assemble(options).request
    assert request.stage_2_sparse_attention == "auto"
    assert request.sparse_attention_schedules()[1].taus == AUTO_STAGE_2_TAUS


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        (" 1, 1.25 ,1.50 ", "1,1.25,1.5"),
        ("dense, 2.0", "dense,2"),
        ("dense,dense", "dense,dense"),
        ("7-8=1.5,1=dense, 3-3=1", "1=dense,3=1,7-8=1.5"),
        ("1e-1", "0.1"),
    ],
)
def test_schedules_have_one_canonical_spelling(text: str, canonical: str) -> None:
    assert canonical_step_schedule(text) == canonical
    assert canonical_step_schedule(canonical) == canonical


def test_dense_schedules_are_recognized_in_either_form() -> None:
    assert is_dense_schedule("dense")
    assert is_dense_schedule("dense,dense,dense")
    assert is_dense_schedule("1-8=dense")
    assert not is_dense_schedule("dense,1")
    assert not is_dense_schedule("8=1.5")


@pytest.mark.parametrize(
    ("text", "layers", "canonical"),
    [
        ("0", (0,), "0"),
        ("none", (), "none"),
        ("47, 0,46", (0, 46, 47), "0,46-47"),
        ("0-3,5", (0, 1, 2, 3, 5), "0-3,5"),
        ("0,1,2", (0, 1, 2), "0-2"),
        ("0-47", tuple(range(48)), "0-47"),
    ],
)
def test_dense_layers_take_numbers_and_ranges(text: str, layers: tuple, canonical: str) -> None:
    assert parse_layer_ranges(text) == layers
    assert canonical_layer_ranges(text) == canonical


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "must not be empty"),
        ("48", "from 0 to 47"),
        ("-1", "numbers and ranges"),
        ("5-3", "from 0 to 47"),
        ("1,0-2", "repeat layer 1"),
        ("x", "numbers and ranges"),
    ],
)
def test_malformed_dense_layers_fail_with_the_reason(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_layer_ranges(text)


def test_request_rejects_a_schedule_that_does_not_fit_its_stage() -> None:
    with pytest.raises(ValueError, match="stage_2_sparse_attention lists 2 steps"):
        validate_record(
            LTX2Request(prompt="p", stage_2_sparse_attention="1,2"), location="[generate]"
        )
    # Two explicit stage-2 sigmas intervals make the same schedule fit.
    request = LTX2Request(
        prompt="p", stage_2_sparse_attention="1,2", stage_2_sigmas=(0.9, 0.5, 0.0)
    )
    validate_record(request, location="[generate]")
    assert request.sparse_attention_schedules()[1].taus == (1.0, 2.0)


def test_requests_default_to_dense_attention_on_both_stages() -> None:
    request = LTX2Request(prompt="p")
    first, second = request.sparse_attention_schedules()
    assert not first.active
    assert not second.active
    assert len(first.taus) == 8
    assert len(second.taus) == 3
    assert first.dense_layers == second.dense_layers == (0,)
    assert request.sparse_attention_stage1_profile is None
    assert request.sparse_attention_stage_profiles == (None, None)


def test_restart_projections_compare_effective_schedules() -> None:
    sparse = LTX2Request(prompt="p", stage_1_sparse_attention="1.5", stage_1_dense_layers="0,47")
    respelled = LTX2Request(
        prompt="p",
        stage_1_sparse_attention="1-8=1.5",
        stage_1_dense_layers="47,0",
    )
    assert sparse.sparse_attention_stage1_profile == respelled.sparse_attention_stage1_profile
    moved = LTX2Request(prompt="p", stage_1_sparse_attention="1.5", stage_1_dense_layers="0")
    assert sparse.sparse_attention_stage1_profile != moved.sparse_attention_stage1_profile
    # Dense layers of a dense stage change nothing it ran.
    dense = LTX2Request(prompt="p", stage_1_dense_layers="0-3")
    assert dense.sparse_attention_stage1_profile is None
    stage_2 = LTX2Request(prompt="p", stage_2_sparse_attention="1,1.25,1.5")
    assert stage_2.sparse_attention_stage1_profile is None
    assert stage_2.sparse_attention_stage_profiles == (
        None,
        SparseAttentionSchedule((1.0, 1.25, 1.5), (0,)),
    )


def test_schedule_renders_steps_and_layers_and_records_a_receipt() -> None:
    schedule = resolve_schedule("dense,1,1.5", "0,46-47", steps=3, label="stage", stage=2)
    assert schedule.render() == "dense, 1, 1.5; dense layers 0,46-47"
    assert schedule.to_metadata() == {
        "method": "sol-attn",
        "token_order": "morton",
        "taus": [None, 1.0, 1.5],
        "dense_layers": [0, 46, 47],
    }
    assert resolve_schedule("dense", "0", steps=3, label="stage", stage=2).to_metadata() is None


def test_stage_steps_share_one_token_order_and_skip_dense_steps() -> None:
    schedule = SparseAttentionSchedule((None, 1.0, 1.5), (0, 47))
    steps = stage_steps(schedule, grid=(3, 4, 5), tokens=70)
    assert steps[0] is None
    first, second = steps[1], steps[2]
    assert first is not None
    assert second is not None
    assert (first.tau, second.tau) == (1.0, 1.5)
    assert first.order is second.order
    assert mx.array_equal(first.order, morton_order(3, 4, 5, tokens=70)).item()
    assert not first.applies_to(0)
    assert first.applies_to(1)
    assert not first.applies_to(47)
    assert stage_steps(SparseAttentionSchedule((None, None), (0,)), grid=(1, 1, 1), tokens=1) == (
        None,
        None,
    )


def _state() -> LatentState:
    return LatentState(
        latent=mx.ones((1, 2, 3), dtype=mx.float32),
        denoise_mask=mx.ones((1, 2, 1)),
        positions=mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        clean_latent=mx.zeros((1, 2, 3)),
    )


class _RecordingX0:
    def __init__(self) -> None:
        self.sparse = []

    def __call__(self, video, audio=None):
        self.sparse.append(video.sparse_attention)
        return mx.zeros_like(video.latent), None


def test_denoise_loop_hands_each_step_its_sparse_record() -> None:
    schedule = SparseAttentionSchedule((1.0, None, 1.5), (0,))
    steps = stage_steps(schedule, grid=(1, 1, 2), tokens=2)
    model = _RecordingX0()
    denoise_loop(
        _state(),
        None,
        (0.9, 0.7, 0.4, 0.0),
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
        video_sparse_attention=steps,
    )
    assert model.sparse == list(steps)
    model = _RecordingX0()
    denoise_loop(
        _state(),
        None,
        (0.9, 0.7, 0.4, 0.0),
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
    )
    assert model.sparse == [None, None, None]


def test_denoise_loop_rejects_a_sparse_plan_for_another_schedule() -> None:
    steps = stage_steps(SparseAttentionSchedule((1.0, 1.0), (0,)), grid=(1, 1, 2), tokens=2)
    with pytest.raises(ValueError, match="lists 2 steps for a 3-step schedule"):
        denoise_loop(
            _state(),
            None,
            (0.9, 0.7, 0.4, 0.0),
            transformer=_RecordingX0(),
            video_context=mx.zeros((1, 1, 4)),
            video_sparse_attention=steps,
        )
