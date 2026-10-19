"""Selecting one image of a multi-image source run for a saved-stage restart."""

import pytest

from kinomlx.restart_selection import (
    fingerprint_key,
    resolve_restart_index,
    selected_artifact,
    source_image_count,
)


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ({}, 1),
        ({"images": "one.png"}, 1),
        ({"images": ["one.png"]}, 1),
        ({"images": ["one.png", "two.png", "three.png"]}, 3),
        # A record that kept only tensors still reports its image count.
        ({"final_latents": ["a.safetensors", "b.safetensors"]}, 2),
        # Older records kept one combined tensor product; that is one image.
        ({"images": ["one.png", "two.png"], "final_latents": "combined.safetensors"}, 2),
    ],
)
def test_source_image_count_reads_whichever_artifacts_a_record_kept(outputs, expected) -> None:
    assert source_image_count(outputs) == expected


def test_a_recorded_planned_count_outranks_what_a_run_managed_to_publish() -> None:
    # A run canceled after its first image keyed every digest by the planned
    # count, so the record's own count is what resolves those digests.
    canceled = {"images": ["one.png"], "final_latents": ["one.safetensors"]}
    assert source_image_count(canceled, planned=3) == 3
    assert fingerprint_key("final_latents", index=0, count=3) == "final_latents.0"


@pytest.mark.parametrize("planned", [None, 0, -1, True, 2.0, "2", [2]])
def test_a_record_without_a_usable_planned_count_falls_back_to_its_artifacts(planned) -> None:
    outputs = {"images": ["one.png", "two.png"]}
    assert source_image_count(outputs, planned=planned) == 2


def test_single_image_runs_need_no_index_and_reject_a_second_one() -> None:
    assert resolve_restart_index(None, 1) == 0
    assert resolve_restart_index(1, 1) == 0
    with pytest.raises(ValueError, match="between 1 and 1"):
        resolve_restart_index(2, 1)


def test_multi_image_runs_require_an_index_and_report_the_range() -> None:
    with pytest.raises(ValueError, match="recorded 3 images; select one"):
        resolve_restart_index(None, 3)
    assert [resolve_restart_index(index, 3) for index in (1, 2, 3)] == [0, 1, 2]
    for rejected in (0, 4):
        with pytest.raises(ValueError, match="between 1 and 3"):
            resolve_restart_index(rejected, 3)


def test_index_rejects_non_integers_including_booleans() -> None:
    for rejected in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="must be an integer"):
            resolve_restart_index(rejected, 3)


def test_selected_artifact_passes_single_paths_through_and_indexes_lists() -> None:
    assert selected_artifact(None, index=0, label="outputs.x") is None
    assert selected_artifact("only.safetensors", index=0, label="outputs.x") == "only.safetensors"
    entries = ["first.safetensors", "second.safetensors"]
    assert selected_artifact(entries, index=1, label="outputs.x") == "second.safetensors"


def test_selected_artifact_reports_a_missing_or_malformed_entry() -> None:
    with pytest.raises(ValueError, match="output 3 is missing"):
        selected_artifact(["a", "b"], index=2, label="outputs.x")
    with pytest.raises(ValueError, match="must hold path strings"):
        selected_artifact(["a", 2], index=0, label="outputs.x")
    with pytest.raises(ValueError, match="path string or an array"):
        selected_artifact(7, index=0, label="outputs.x")


def test_fingerprint_key_matches_how_runs_record_their_digests() -> None:
    assert fingerprint_key("final_latents", index=0, count=1) == "final_latents"
    assert fingerprint_key("final_latents", index=2, count=4) == "final_latents.2"
