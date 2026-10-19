"""LTX-2 image-condition members, CLI shorthand, and ordered collection declaration."""

import re
from dataclasses import dataclass
from pathlib import Path

from kinomlx.config import CliBinding, ConfigFieldSpec, ConfigRecordSpec
from kinomlx.config_bindings import argument_value
from kinomlx.config_surfaces import CONFIG_SURFACES

from .frame_position import LAST, NAMED_FRAME_POSITIONS
from .types import LTX2ImageConditioningConfig

_GROUP = "LTX-2 generation"
_FRAME_SEPARATOR = "@"
_STRENGTH_SEPARATOR = ":"
# A frame position is a named position or decimal digits alone, so an ordinary
# path such as photo@2x.png keeps its suffix instead of being read as a position.
_FRAME_TOKEN = re.compile(rf"[0-9]+|{'|'.join(NAMED_FRAME_POSITIONS)}", re.IGNORECASE)

_PATH = ConfigFieldSpec(
    "path",
    "Set the image-conditioning source",
    "a readable image path",
    Path("input.png"),
    "image_path",
    persistence="intent",
    surfaces=CONFIG_SURFACES,
)
_FRAME_INDEX = ConfigFieldSpec(
    "frame_index",
    "Choose the conditioned output frame",
    '"first", "last", or a non-negative integer, bare or quoted '
    '(in TOML, frame_index = 120 or "120"; saved as the string)',
    LAST,
    "image_frame_index",
    persistence="intent",
    cli=CliBinding(_GROUP, flags=("--image-frame-index",)),
    surfaces=CONFIG_SURFACES,
)
_STRENGTH = ConfigFieldSpec(
    "strength",
    "Set the image-conditioning strength",
    "a number from 0 through 1",
    1.0,
    "image_strength",
    persistence="intent",
    cli=CliBinding(_GROUP, flags=("--image-strength",)),
    surfaces=CONFIG_SURFACES,
)
_STAGE1_STRENGTH = ConfigFieldSpec(
    "stage1_strength",
    "Override this condition's stage-1 strength",
    "a number from 0 through 1",
    0.9,
    "image_stage1_strength",
    persistence="intent",
    cli=CliBinding(_GROUP, flags=("--image-stage1-strength",)),
    surfaces=CONFIG_SURFACES,
)
_STAGE2_STRENGTH = ConfigFieldSpec(
    "stage2_strength",
    "Override this condition's stage-2 strength",
    "a number from 0 through 1",
    0.5,
    "image_stage2_strength",
    persistence="intent",
    cli=CliBinding(_GROUP, flags=("--image-stage2-strength",)),
    surfaces=CONFIG_SURFACES,
)

IMAGE_RECORD = ConfigRecordSpec(
    LTX2ImageConditioningConfig,
    (_PATH, _FRAME_INDEX, _STRENGTH, _STAGE1_STRENGTH, _STAGE2_STRENGTH),
)


def _peel_numbers(text: str) -> tuple[str, tuple[float, ...]]:
    """Take trailing ``:number`` suffixes from the right, leaving the path side.

    A suffix is numeric when ``float`` accepts it, so an ordinary path such as
    ``a:2.png`` or ``style:latest.png`` keeps every character it was given.
    """
    remainder = text
    numbers: list[float] = []
    while True:
        prefix, separator, suffix = remainder.rpartition(_STRENGTH_SEPARATOR)
        if not separator:
            break
        try:
            numbers.append(float(suffix))
        except ValueError:
            break
        remainder = prefix
    numbers.reverse()
    return remainder, tuple(numbers)


@dataclass(frozen=True)
class ImageShorthand:
    """CLI-only suffix decoding into owned fields, without probing the filesystem."""

    schema: ConfigRecordSpec

    def __post_init__(self) -> None:
        if any(
            not any(member is owned for owned in self.schema.fields)
            for member in (_PATH, _FRAME_INDEX, _STRENGTH, _STAGE1_STRENGTH, _STAGE2_STRENGTH)
        ):
            raise ValueError("image shorthand fields must belong to its record schema")

    def __call__(self, text: str) -> dict[str, object]:
        remainder, numbers = _peel_numbers(text)
        if len(numbers) > 2:
            raise ValueError("image shorthand requires one or two numeric suffix values")
        prefix, separator, suffix = remainder.rpartition(_FRAME_SEPARATOR)
        indexed = bool(separator) and _FRAME_TOKEN.fullmatch(suffix) is not None
        path = prefix if indexed else remainder
        if not path:
            raise ValueError("image shorthand requires a path before an @frame or :strength suffix")
        members: dict[str, object] = {_PATH.name: path}
        if indexed:
            # The record stores the spelling the user typed; its field owns the
            # grammar, so a name and an integer arrive here the same way.
            members[_FRAME_INDEX.name] = suffix
        if numbers:
            # One number is the strength both stages hold, two are the stage
            # strengths alone, exactly as two numbers mean on --adapter.
            targets = (_STRENGTH,) if len(numbers) == 1 else (_STAGE1_STRENGTH, _STAGE2_STRENGTH)
            members.update(
                (member.name, number) for member, number in zip(targets, numbers, strict=True)
            )
        # The ordinary record action performs field-owned local admission.
        return members


def _collection_example() -> tuple[dict[str, object], ...]:
    """Show the headline case as records a user could write: a first and a last frame.

    The second record also carries a stage override, so the example states the
    inheritance rule it depends on: stage 1 holds the common strength and stage
    2 holds its own.
    """
    return (
        {_PATH.name: Path("first.png")},
        {
            _PATH.name: Path("last.png"),
            _FRAME_INDEX.name: LAST,
            _STRENGTH.name: 0.8,
            _STAGE2_STRENGTH.name: 0.5,
        },
    )


IMAGES = ConfigFieldSpec(
    "images",
    "Condition frames on ordered images",
    "an ordered array of image-condition records",
    _collection_example(),
    "images",
    persistence="intent",
    item_schema=IMAGE_RECORD,
    cli_item=_PATH.name,
    cli=CliBinding(
        _GROUP,
        flags=("--image",),
        parse=argument_value(ImageShorthand(IMAGE_RECORD), "image"),
    ),
    surfaces=CONFIG_SURFACES,
)

__all__ = ["IMAGES", "IMAGE_RECORD", "ImageShorthand"]
