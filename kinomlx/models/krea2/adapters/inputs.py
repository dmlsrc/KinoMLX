"""Krea2 adapter input capabilities and category ownership."""

from dataclasses import dataclass
from pathlib import Path

from kinomlx.adapters.inputs import ScheduledAdapterInput, exclusion_field
from kinomlx.adapters.selection import AdapterSelection
from kinomlx.config_records import config_record
from kinomlx.config_values import dataclass_from_mapping

from .categories import ADAPTER_EXCLUDE_CATEGORIES


@config_record
@dataclass(frozen=True)
class Krea2AdapterConfig(ScheduledAdapterInput):
    """One ordered user adapter with this model's admitted controls."""

    exclude: tuple[str, ...] = exclusion_field(ADAPTER_EXCLUDE_CATEGORIES)


def profile_selection(
    path: Path,
    *,
    strength: float,
    exclusions: str,
    location: str,
    step_strengths: tuple[float, ...] = (),
    sigma_window: str = "",
) -> AdapterSelection:
    """Adapt resolved profile controls to the same typed record as user selections.

    Profiles retain their existing optional-control and category-string interface.
    An active step array owns magnitude; unused resolved defaults are not input intent.
    """
    exclude = (
        ()
        if exclusions.strip().lower() in {"", "none", "off"}
        else tuple(item.strip() for item in exclusions.split(",") if item.strip())
    )
    values: dict[str, object] = {"path": path, "exclude": exclude}
    if step_strengths:
        values["step_strengths"] = step_strengths
    else:
        values["strength"] = strength
    if sigma_window:
        values["sigma_window"] = sigma_window
    return dataclass_from_mapping(Krea2AdapterConfig, values, table=location).selection(
        location=location
    )
