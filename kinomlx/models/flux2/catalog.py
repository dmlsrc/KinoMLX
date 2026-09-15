"""Explicit official Klein 9B checkpoint and recipe profiles."""

from dataclasses import dataclass
from enum import StrEnum


class Flux2Variant(StrEnum):
    BASE = "base"
    DISTILLED = "distilled"
    KV = "kv"


@dataclass(frozen=True)
class Flux2Profile:
    variant: Flux2Variant
    repo_id: str
    steps: int
    guidance: float
    kv: bool = False


PROFILES = (
    Flux2Profile(Flux2Variant.BASE, "black-forest-labs/FLUX.2-klein-base-9B", 50, 4.0),
    Flux2Profile(Flux2Variant.DISTILLED, "black-forest-labs/FLUX.2-klein-9B", 4, 1.0),
    Flux2Profile(Flux2Variant.KV, "black-forest-labs/FLUX.2-klein-9b-kv", 4, 1.0, True),
)


def profile(variant: str) -> Flux2Profile:
    selected = Flux2Variant(variant)
    return next(item for item in PROFILES if item.variant is selected)
