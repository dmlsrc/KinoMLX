"""Pure output-format policy owned by GMNet configuration fields."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kinomlx.config_rules import RecordRule
from kinomlx.debug.metadata import sidecar_selected


def selected_output_formats(
    path: Path | None,
    exr: bool | None,
    heic: bool | None,
    save_gain_map: bool | None,
    save_all_sidecars: bool,
) -> tuple[bool, bool, bool]:
    """Resolve enabled artifacts and enforce exact-path and nonempty-output policy."""
    suffix = "" if path is None else path.suffix.lower()
    write_exr = (path is None or suffix == ".exr") if exr is None else exr
    write_heic = (path is None or suffix == ".heic") if heic is None else heic
    if suffix == ".exr" and not write_exr:
        raise ValueError("an exact .exr path requires EXR output")
    if suffix == ".heic" and not write_heic:
        raise ValueError("an exact .heic path requires HEIC output")
    gain_map = sidecar_selected(save_gain_map, save_all=save_all_sidecars)
    if not (write_exr or write_heic or gain_map):
        raise ValueError("nothing to write: EXR, HEIC, and gain-map outputs are disabled")
    return write_exr, write_heic, gain_map


def _output_formats(values: Mapping[str, object]) -> None:
    selected_output_formats(
        cast(Path | None, values["path"]),
        cast(bool | None, values["exr"]),
        cast(bool | None, values["heic"]),
        cast(bool | None, values["save_gain_map"]),
        cast(bool, values["save_all_sidecars"]),
    )


OUTPUT_FORMATS = RecordRule(
    "output_formats",
    frozenset({"path", "exr", "heic", "save_gain_map", "save_all_sidecars"}),
    _output_formats,
)
