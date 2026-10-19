"""Compaction preserves valid text, including template suffixes after padding."""

import mlx.core as mx

from kinomlx.models.krea2.transformer.inputs import compact_text_groups


def test_compaction_gathers_valid_rows_and_keeps_five_suffix_tokens_and_positions() -> None:
    hidden = mx.arange(20, dtype=mx.float32).reshape(2, 10, 1, 1)
    positions = mx.arange(72, dtype=mx.float32).reshape(2, 12, 3)
    mask = mx.array(
        [
            [True, False, False, False, False, True, True, True, True, True, True, True],
            [False, True, False, False, False, True, True, True, True, True, True, True],
        ]
    )
    (group,) = compact_text_groups(hidden, positions, mask)
    assert group.rows == (0, 1)
    assert group.hidden_states[..., 0, 0].tolist() == [
        [0, 5, 6, 7, 8, 9],
        [11, 15, 16, 17, 18, 19],
    ]
    for row, prefix in ((0, 0), (1, 1)):
        indices = mx.array([prefix, 5, 6, 7, 8, 9, 10, 11])
        assert mx.array_equal(group.position_ids[row], positions[row, indices]).item()


def test_already_compact_text_reuses_arrays() -> None:
    hidden = mx.zeros((2, 3, 1, 2))
    positions = mx.zeros((2, 5, 3))
    (group,) = compact_text_groups(hidden, positions, mx.ones((2, 5), dtype=mx.bool_))
    assert group.hidden_states is hidden
    assert group.position_ids is positions
