"""Image spans and three-axis positions independent of prompt or adapter policy."""

from dataclasses import dataclass

import mlx.core as mx

from .vision import Qwen3VLVisualFeatures


@dataclass(frozen=True)
class Qwen3VLImageSpan:
    """One contiguous image-token span in an unpadded single prompt."""

    start: int
    features: Qwen3VLVisualFeatures

    @property
    def length(self) -> int:
        return self.features.grid[0] * self.features.grid[1]


def multimodal_positions(length: int, spans: tuple[Qwen3VLImageSpan, ...]) -> mx.array:
    """Text advances on all axes; image tokens advance on their spatial axes."""
    cursor = 0
    offset = 0
    pieces: list[mx.array] = []
    for span in spans:
        if isinstance(span.start, bool) or not isinstance(span.start, int):
            raise ValueError("Qwen image span start must be an integer")
        height, width = span.features.grid
        if min(height, width) <= 0 or span.start < cursor or span.start + span.length > length:
            raise ValueError("Qwen image spans must be ordered, disjoint, and inside the prompt")
        text_length = span.start - cursor
        text = mx.arange(text_length, dtype=mx.int32) + offset
        pieces.append(mx.broadcast_to(text[None], (3, text_length)))
        offset += text_length
        row = mx.repeat(mx.arange(height, dtype=mx.int32), width)
        column = mx.tile(mx.arange(width, dtype=mx.int32), height)
        pieces.append(mx.stack([mx.zeros_like(row), row, column]) + offset)
        offset += max(height, width)
        cursor = span.start + span.length
    tail = mx.arange(length - cursor, dtype=mx.int32) + offset
    pieces.append(mx.broadcast_to(tail[None], (3, length - cursor)))
    return mx.concatenate(pieces, axis=1)[:, None, :]
