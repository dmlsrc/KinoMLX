import json
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.generated_keyframes import (
    append_generated_keyframe_slots,
    extract_generated_keyframes,
)
from kinomlx.models.ltx2.keyframe_positions import generated_keyframe_indices
from kinomlx.models.ltx2.state import create_video_latent_tools, init_video_latent_state
from kinomlx.types import VideoLatentShape

_FIXTURE = Path(__file__).parent / "fixtures/ltx25_generated_keyframe_contract.json"


def test_generated_keyframe_selection_matches_neutral_contract() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for case in fixture["selection_cases"]:
        assert generated_keyframe_indices(case["frames"], case["count"]) == tuple(case["indices"])


def test_generated_slots_append_after_existing_tokens_and_round_trip() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))["slot_case"]
    shape = VideoLatentShape(*fixture["latent_shape"])
    tools = create_video_latent_tools(shape, fps=fixture["fps"])
    state = init_video_latent_state(tools, dtype=mx.float32)
    prepared, layout, mask = append_generated_keyframe_slots(
        state,
        tools,
        pixel_frames=fixture["pixel_frames"],
        count=fixture["count"],
    )
    mx.eval(prepared.latent, prepared.positions, mask)

    assert layout.frame_indices == (5, 11)
    assert layout.tokens_per_slot == fixture["tokens_per_slot"]
    assert layout.first_token == fixture["first_slot_token"]
    assert prepared.latent.shape[1] == fixture["total_tokens"]
    assert mask.shape == (1, fixture["total_tokens"])
    assert int(mx.sum(mask).item()) == fixture["tokens_per_slot"] * 3
    for slot, expected in enumerate(fixture["slot_time_bounds"]):
        start = layout.first_token + slot * layout.tokens_per_slot
        actual = prepared.positions[0, 0, start, :].tolist()
        assert actual == pytest.approx(expected)

    tokens = mx.arange(prepared.latent.size, dtype=mx.float32).reshape(prepared.latent.shape)
    extracted = extract_generated_keyframes(tokens, tools, layout)
    assert tuple(extracted.shape) == (1, 128, 2, 2, 2)
    expected_tokens = tokens[:, layout.first_token :]
    assert mx.array_equal(tools.patchifier.patchify(extracted), expected_tokens).item()


def test_generated_slots_reject_impossible_or_duplicate_interior_requests() -> None:
    with pytest.raises(ValueError, match="interior"):
        generated_keyframe_indices(1, 1)
    with pytest.raises(ValueError, match="interior"):
        generated_keyframe_indices(9, 8)


def _slot_case() -> tuple[dict, VideoLatentShape]:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))["slot_case"]
    return fixture, VideoLatentShape(*fixture["latent_shape"])


def test_initial_slots_seed_the_generative_latent_and_leave_clean_zero() -> None:
    fixture, shape = _slot_case()
    tools = create_video_latent_tools(shape, fps=fixture["fps"])
    state = init_video_latent_state(tools, dtype=mx.float32)
    count = fixture["count"]
    slot_shape = (shape.batch, shape.channels, count, shape.height, shape.width)
    initial = mx.arange(mx.prod(mx.array(slot_shape)).item(), dtype=mx.float32).reshape(slot_shape)

    prepared, layout, _mask = append_generated_keyframe_slots(
        state,
        tools,
        pixel_frames=fixture["pixel_frames"],
        count=count,
        initial=initial,
    )

    slots = slice(layout.first_token, layout.first_token + layout.token_count)
    assert mx.array_equal(prepared.latent[:, slots], tools.patchifier.patchify(initial)).item()
    assert mx.array_equal(
        extract_generated_keyframes(prepared.latent, tools, layout), initial
    ).item()
    assert not mx.any(prepared.clean_latent[:, slots]).item()
    assert mx.all(prepared.denoise_mask[:, slots] == 1).item()


def test_initial_slots_must_match_the_stage_latent_size() -> None:
    fixture, shape = _slot_case()
    tools = create_video_latent_tools(shape, fps=fixture["fps"])
    state = init_video_latent_state(tools, dtype=mx.float32)
    wrong = mx.zeros((shape.batch, shape.channels, fixture["count"], shape.height * 2, shape.width))
    with pytest.raises(ValueError, match="initial generated keyframes have shape"):
        append_generated_keyframe_slots(
            state,
            tools,
            pixel_frames=fixture["pixel_frames"],
            count=fixture["count"],
            initial=wrong,
        )
