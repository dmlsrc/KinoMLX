from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.generated_keyframes import (
    append_generated_keyframe_slots,
    extract_generated_keyframes,
    generated_keyframe_indices,
)
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
