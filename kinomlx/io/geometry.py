"""Shared source-aspect output sizing for image models."""

import math
from enum import StrEnum


class MatchSourceAspect(StrEnum):
    """Which output axis a source aspect ratio is pinned to."""

    LONG = "long"
    SHORT = "short"
    WIDTH = "width"
    HEIGHT = "height"


MATCH_SOURCE_ASPECT_CHOICES = tuple(item.value for item in MatchSourceAspect)


class ReferenceFit(StrEnum):
    """How a reference image is placed on the output grid."""

    FIT = "fit"
    CROP = "crop"


REFERENCE_FIT_CHOICES = tuple(item.value for item in ReferenceFit)
_AXES = ("width", "height")


def pinned_axis(mode: str, source_size: tuple[int, int]) -> str:
    """Map long/short onto the source's actual axes; width/height are literal."""
    if mode in _AXES:
        return mode
    if mode not in MATCH_SOURCE_ASPECT_CHOICES:
        raise ValueError(
            f"match_source_aspect must be one of: {', '.join(MATCH_SOURCE_ASPECT_CHOICES)}"
        )
    width, height = source_size
    longer = "height" if height >= width else "width"
    return longer if mode == "long" else ("width" if longer == "height" else "height")


def derive_output_size(
    source_size: tuple[int, int],
    *,
    axis: str,
    pixels: int,
    alignment: int = 16,
) -> tuple[int, int]:
    """Round the pinned axis up to the alignment and the other to its nearest multiple."""
    source_width, source_height = source_size
    if min(source_width, source_height) <= 0:
        raise ValueError("source image dimensions must be positive")
    if pixels <= 0 or alignment <= 0:
        raise ValueError("output size and alignment must be positive")
    pinned = math.ceil(pixels / alignment) * alignment
    if axis == "width":
        derived = pinned * source_height / source_width
    elif axis == "height":
        derived = pinned * source_width / source_height
    else:
        raise ValueError("axis must be width or height")
    other = max(alignment, round(derived / alignment) * alignment)
    return (pinned, other) if axis == "width" else (other, pinned)
