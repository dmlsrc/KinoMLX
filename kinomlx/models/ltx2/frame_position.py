"""Pure named frame-position grammar shared by admission, logging, and execution."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from kinomlx.config_constraints import LengthConstraint

FIRST = "first"
LAST = "last"

NAMED_FRAME_POSITIONS = (FIRST, LAST)
"""Positions expressible without knowing how many frames a run holds."""

# A numeric position is decimal digits alone, so no sign, exponent or spacing
# reaches the integer branch and every admitted number is non-negative.
_DECIMAL_POSITION = re.compile(r"[0-9]+")


def canonical_frame_position(value: object) -> object:
    """Spell a bare frame number the way the field stores it.

    TOML, ``--set`` and the API deliver ``120`` as a number; the field is a
    string so that ``first`` and ``last`` share it. Both spellings mean the same
    thing, and the saved configuration keeps the string. A negative or
    non-integral number is passed through as text so ``validate`` names the
    actual problem.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return value


def parse_frame_position(spec: str) -> str | int:
    """Map ``first``, ``last``, or a non-negative integer to its position value."""
    text = spec.strip().lower()
    if text in NAMED_FRAME_POSITIONS:
        return text
    if _DECIMAL_POSITION.fullmatch(text) is not None:
        return int(text)
    raise ValueError(f"frame position {spec!r} must be 'first', 'last', or a non-negative integer")


def is_first_position(spec: str) -> bool:
    """Report whether a position names frame 0 without knowing the frame count."""
    position = parse_frame_position(spec)
    return position == FIRST or position == 0


def resolve_frame_position(spec: str, frames: int, *, predicted: bool = False) -> int:
    """Resolve one position against a known frame count.

    ``predicted`` says the count came from the duration head rather than from
    the request, which the out-of-range refusal then reports.
    """
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("a frame position resolves against at least one frame")
    position = parse_frame_position(spec)
    if isinstance(position, int):
        if position >= frames:
            extent = f"predicted {frames}" if predicted else str(frames)
            raise ValueError(f"image frame_index {position} is outside {extent} frames")
        return position
    return 0 if position == FIRST else frames - 1


def repeated_position_message(frame: object, earlier: str, later: str) -> str:
    """Name the frame two conditions share, and both spellings when they differ."""
    label = frame if isinstance(frame, int) else repr(frame)
    message = f"image frame_index {label} is conditioned more than once"
    if earlier.strip().lower() == later.strip().lower():
        return message
    return f"{message}: {earlier!r} and {later!r} both name frame {label}"


def resolve_frame_positions(
    specs: Sequence[str], frames: int, *, predicted: bool = False
) -> tuple[int, ...]:
    """Resolve ordered positions, refusing two conditions on one absolute frame."""
    spellings: dict[int, str] = {}
    resolved: list[int] = []
    for spec in specs:
        index = resolve_frame_position(spec, frames, predicted=predicted)
        earlier = spellings.get(index)
        if earlier is not None:
            raise ValueError(repeated_position_message(index, earlier, spec))
        spellings[index] = spec
        resolved.append(index)
    return tuple(resolved)


@dataclass(frozen=True)
class FramePositionConstraint(LengthConstraint):
    """Own the named-position or non-negative-integer grammar as a local field constraint."""

    minimum: int = 1

    def canonicalize(self, value: object) -> object:
        return canonical_frame_position(value)

    def validate(self, value: object) -> None:
        if not isinstance(value, str):
            raise ValueError("must be a frame position name or an integer string")
        super().validate(value)
        parse_frame_position(value)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "kind": "frame_position",
            "named_positions": list(NAMED_FRAME_POSITIONS),
            "numeric_minimum": 0,
        }


__all__ = [
    "FIRST",
    "LAST",
    "NAMED_FRAME_POSITIONS",
    "FramePositionConstraint",
    "canonical_frame_position",
    "is_first_position",
    "parse_frame_position",
    "repeated_position_message",
    "resolve_frame_position",
    "resolve_frame_positions",
]
