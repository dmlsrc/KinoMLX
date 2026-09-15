"""Compact Krea text rows without losing suffix tokens or changing positions."""

from dataclasses import dataclass
from typing import cast

import mlx.core as mx


@dataclass(frozen=True)
class Krea2TextGroup:
    """Equal-length text rows and their original batch indices."""

    rows: tuple[int, ...]
    hidden_states: mx.array
    position_ids: mx.array


def compact_text_groups(
    hidden_states: mx.array,
    position_ids: mx.array,
    attention_mask: mx.array,
) -> tuple[Krea2TextGroup, ...]:
    """Gather valid rows, grouping unequal lengths instead of padding them."""
    batch, text_length = hidden_states.shape[:2]
    masks = cast(list[list[bool]], attention_mask[:, :text_length].tolist())
    indices = [[index for index, valid in enumerate(row) if valid] for row in masks]
    by_length: dict[int, list[int]] = {}
    for row, valid in enumerate(indices):
        by_length.setdefault(len(valid), []).append(row)
    if len(by_length) == 1 and text_length in by_length:
        return (Krea2TextGroup(tuple(range(batch)), hidden_states, position_ids),)
    groups: list[Krea2TextGroup] = []
    for rows in by_length.values():
        gathers = [mx.array(indices[row], dtype=mx.int32) for row in rows]
        hidden = mx.stack(
            [hidden_states[row, take] for row, take in zip(rows, gathers, strict=True)]
        )
        text_positions = mx.stack(
            [position_ids[row, take] for row, take in zip(rows, gathers, strict=True)]
        )
        image_positions = position_ids[mx.array(rows), text_length:]
        positions = mx.concatenate([text_positions, image_positions], axis=1)
        groups.append(Krea2TextGroup(tuple(rows), hidden, positions))
    return tuple(groups)
