"""Flux2 adapter input capabilities and category ownership."""

from dataclasses import dataclass

from kinomlx.adapters.inputs import ScheduledAdapterInput, exclusion_field
from kinomlx.config_records import config_record

from .categories import ADAPTER_EXCLUDE_CATEGORIES


@config_record
@dataclass(frozen=True)
class Flux2AdapterConfig(ScheduledAdapterInput):
    """One ordered user adapter with this model's admitted controls."""

    exclude: tuple[str, ...] = exclusion_field(ADAPTER_EXCLUDE_CATEGORIES)
