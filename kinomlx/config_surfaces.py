"""Explicit transport support and reasons for deliberately unavailable surfaces."""

from dataclasses import dataclass
from typing import Literal

Surface = Literal["cli", "environment", "toml", "override", "api"]
SURFACES: tuple[Surface, ...] = ("cli", "environment", "toml", "override", "api")


@dataclass(frozen=True)
class SurfacePolicy:
    """All configuration transports are supported except explicitly reasoned exclusions."""

    unavailable: tuple[tuple[Surface, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.unavailable, tuple) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or entry[0] not in SURFACES
            or not isinstance(entry[1], str)
            or not entry[1].strip()
            for entry in self.unavailable
        ):
            raise ValueError("surface exclusions require known surfaces and nonempty reasons")
        names = tuple(surface for surface, _reason in self.unavailable)
        if len(names) != len(set(names)):
            raise ValueError("duplicate surface exclusions")

    def supports(self, surface: Surface) -> bool:
        return all(name != surface for name, _reason in self.unavailable)

    def describe(self) -> dict[str, object]:
        excluded = dict(self.unavailable)
        return {
            surface: {"status": "unavailable", "reason": excluded[surface]}
            if surface in excluded
            else {"status": "supported"}
            for surface in SURFACES
        }


CONFIG_SURFACES = SurfacePolicy()
SOURCE_SURFACES = SurfacePolicy(
    (("api", "Source spelling is resolved into canonical typed inputs before construction."),)
)
MODEL_SURFACES = SurfacePolicy(
    (
        ("api", "Select the model-specific API; model is not a request-record argument."),
        ("environment", "Model selection uses explicit CLI, TOML or ordered overrides."),
    )
)
STRUCTURED_SURFACES = SurfacePolicy(
    (("cli", "This contribution has no ordinary CLI binding; use a structured input surface."),)
)
UTILITY_SURFACES = SurfacePolicy(
    tuple(
        (surface, "Command utility; not a configuration record field.")
        for surface in SURFACES
        if surface != "cli"
    )
)
