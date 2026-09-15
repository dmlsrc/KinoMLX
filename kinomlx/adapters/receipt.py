"""Tensor-complete, data-only adapter placement records."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LOW_COVERAGE_WARNING_THRESHOLD = 0.5


def log_adapter_receipt(
    logger: logging.Logger,
    *,
    model: str,
    path: Path,
    adapter_types: Iterable[str],
    strength: float,
    placed_targets: int,
    complete_targets: int,
    structural_coverage: float,
    exclusions: Iterable[str] = (),
    details: Iterable[tuple[str, object]] = (),
    position: tuple[int, int] | None = None,
) -> None:
    """Log the common factual adapter-placement summary used by model loaders."""
    fields: list[tuple[str, object]] = [
        ("Model", model),
        ("Types", ", ".join(adapter_types) or "unknown"),
        ("Strength", f"{strength:g}"),
        ("Targets", f"{placed_targets}/{complete_targets}"),
        ("Coverage", f"{100 * structural_coverage:.1f}%"),
        ("Excluded", ", ".join(exclusions) or "none"),
    ]
    fields.extend((name.replace("_", " ").title(), value) for name, value in details)
    heading = (
        "Adapter result" if position is None else f"Adapter result {position[0]}/{position[1]}"
    )
    logger.info("%s: %s", heading, path.name)
    logger.info("  %s", " | ".join(f"{label}: {value}" for label, value in fields))
    logger.info("  Path: %s", path.expanduser().absolute())


@dataclass(frozen=True)
class AdapterPlacement:
    source_keys: tuple[str, ...]
    target: str | None
    kind: str
    status: str
    derived_from: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, object]:
        result: dict[str, object] = {
            "source_keys": list(self.source_keys),
            "target": self.target,
            "kind": self.kind,
            "status": self.status,
        }
        if self.derived_from:
            result["derived_from"] = list(self.derived_from)
        return result


@dataclass(frozen=True)
class AdapterReceipt:
    path: Path
    source_identity: str
    strength: float
    exclusions: tuple[str, ...]
    base_variant: str
    declared_model: str | None
    declared_variant: str | None
    placements: tuple[AdapterPlacement, ...]
    adapter_types: tuple[str, ...]
    variant_pairing: str = "undeclared"
    effective: bool = True
    warnings: tuple[str, ...] = ()
    step_strengths: tuple[float, ...] = ()
    sigma_window: dict[str, object] | None = None
    # Whether the loader fused this adapter into the weight or installed a live
    # term; the loader states it from the routing decision it actually made.
    application: str = "fused"

    @property
    def placed_targets(self) -> int:
        return sum(item.status == "applied" for item in self.placements)

    @property
    def complete_targets(self) -> int:
        return sum(
            item.kind not in ("unknown", "shared_basis", "training_only")
            for item in self.placements
        )

    @property
    def coverage(self) -> float:
        return self.placed_targets / self.complete_targets if self.complete_targets else 0.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source_identity": self.source_identity,
            "strength": self.strength,
            "exclusions": list(self.exclusions),
            "base_variant": self.base_variant,
            "declared_model": self.declared_model,
            "declared_variant": self.declared_variant,
            "adapter_types": list(self.adapter_types),
            "variant_pairing": self.variant_pairing,
            "effective": self.effective,
            "complete_targets": self.complete_targets,
            "placed_targets": self.placed_targets,
            "structural_coverage": self.coverage,
            "placements": [item.to_metadata() for item in self.placements],
            "warnings": list(self.warnings),
            "step_strengths": list(self.step_strengths),
            "sigma_window": self.sigma_window,
            "application": self.application,
        }
