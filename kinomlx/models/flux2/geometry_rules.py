"""Klein geometry choices that require input values but no source-image reads."""

from pathlib import Path
from typing import cast

from kinomlx.config_rules import RecordRule, RuleInputs


def output_aspect_mode(
    reference_count: int, width: int | None, height: int | None, selected: str | None
) -> str | None:
    """Resolve implicit single-reference aspect matching and validate pinned dimensions."""
    mode = selected
    if mode is None and reference_count == 1 and (width is None or height is None):
        mode = "width" if width is not None else "height" if height is not None else "long"
    if mode is not None:
        if not reference_count:
            raise ValueError("match_source_aspect requires a reference image")
        if width is not None and height is not None:
            raise ValueError(
                "match_source_aspect pins one dimension; pass width or height, not both"
            )
        if (mode == "width" and height is not None) or (mode == "height" and width is not None):
            raise ValueError("explicit aspect axis must agree with the supplied dimension")
    elif any(value is not None and value % 16 for value in (width, height)):
        raise ValueError("explicit Klein output dimensions must be multiples of 16")
    return mode


def _reference_geometry(values: RuleInputs) -> None:
    sources = cast(tuple[Path, ...], values["source_images"])
    if values["reference_fit"] is not None and not sources:
        raise ValueError("reference_fit requires source_images")
    output_aspect_mode(
        len(sources),
        cast(int | None, values["width"]),
        cast(int | None, values["height"]),
        cast(str | None, values["match_source_aspect"]),
    )


REFERENCE_GEOMETRY = RecordRule(
    "reference_geometry",
    frozenset({"source_images", "reference_fit", "match_source_aspect", "width", "height"}),
    _reference_geometry,
)
