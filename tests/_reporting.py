"""Progress event recorder for tests and standalone validation gates."""

from dataclasses import dataclass, field


@dataclass
class RecordingReporter:
    """Capture ordered progress events for assertions."""

    events: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def phase_start(
        self,
        phase: str,
        *,
        total: float | None = None,
        unit: str = "it",
        aggregate_total: float | None = None,
    ) -> None:
        self.events.append(("start", phase, {"total": total, "unit": unit}))

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        self.events.append(("advance", phase, {"advance": advance}))

    def phase_end(self, phase: str) -> None:
        self.events.append(("end", phase, {}))
