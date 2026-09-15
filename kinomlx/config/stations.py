"""Field-owned station comparison policy for typed request projections."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class StationPolicy:
    """One field's admission decision at a saved restart station."""

    station: str
    mode: Literal["mutable", "locked", "irrelevant"]
    attribute: str | None = None
    reason: str = field(default="", kw_only=True)
    irrelevant_action: Literal["ignore", "reject"] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not self.station or self.mode not in ("mutable", "locked", "irrelevant"):
            raise ValueError("station policy requires a station and valid mode")
        if self.attribute is not None and not self.attribute.isidentifier():
            raise ValueError("station projection must name a typed record attribute")
        if self.attribute is not None and self.mode != "locked":
            raise ValueError("only locked station policies may select a projection")
        if not isinstance(self.reason, str) or (self.reason and not self.reason.strip()):
            raise ValueError("station reason must be nonempty prose when supplied")
        if (self.mode == "irrelevant") != (self.irrelevant_action is not None):
            raise ValueError("irrelevant station policy requires an ignore or reject action")
        if self.irrelevant_action is not None and self.irrelevant_action not in (
            "ignore",
            "reject",
        ):
            raise ValueError("irrelevant station action must be ignore or reject")


@dataclass(frozen=True)
class OutputPolicy:
    """Stations that can produce one selectable output artifact."""

    produced_at: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.produced_at, frozenset) or any(
            not isinstance(station, str) or not station.replace("-", "_").isidentifier()
            for station in self.produced_at
        ):
            raise ValueError("output policy requires frozen station names")

    def describe(self) -> dict[str, object]:
        """Expose the same availability used by artifact selection and admission."""
        return {"produced_at": sorted(self.produced_at)}

    def selected(self, value: bool | None, *, save_all: bool, station: str, field: str) -> bool:
        """Explicit unavailable output requests fail; save-all selects available artifacts."""
        if station not in self.produced_at:
            if value is True:
                raise ValueError(f"{field} is not produced by a {station} restart")
            return False
        return save_all if value is None else value


def station_output_defaults(
    values: Mapping[str, object], policies: Mapping[str, OutputPolicy], *, station: str
) -> dict[str, object]:
    """A new station does not inherit requests to reproduce unavailable parent artifacts."""
    return {
        **values,
        **{name: False for name, policy in policies.items() if station not in policy.produced_at},
    }
