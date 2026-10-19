"""Data-only LTX-2 adapter placement receipt."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LTX2AdapterReceipt:
    """Factual placement receipt for one adapter against one transformer graph."""

    path: Path
    fingerprint: str
    base_model_generation: str | None
    declared_model_generation: str | None
    generation_mismatch: bool | None
    strength: float
    knockouts: tuple[str, ...]
    complete_targets: int
    placed_targets: int
    structural_coverage: float
    bare_parameter_targets: int
    target_categories: tuple[tuple[str, int], ...]
    skipped_reasons: tuple[tuple[str, int], ...]
    effective: bool
    warnings: tuple[str, ...]
    adapter_types: tuple[str, ...]
    # What set the intrinsic scale beneath user strength, and its LoRA values.
    scale_source: str | None = None
    intrinsic_scales: tuple[float, ...] = ()

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "adapter_types": list(self.adapter_types),
            "fingerprint": self.fingerprint,
            "base_model_generation": self.base_model_generation,
            "declared_model_generation": self.declared_model_generation,
            "generation_mismatch": self.generation_mismatch,
            "strength": self.strength,
            "knockouts": self.knockouts,
            "complete_targets": self.complete_targets,
            "placed_targets": self.placed_targets,
            "structural_coverage": self.structural_coverage,
            "bare_parameter_targets": self.bare_parameter_targets,
            "target_categories": dict(self.target_categories),
            "skipped_reasons": dict(self.skipped_reasons),
            "effective": self.effective,
            "warnings": list(self.warnings),
            "scale_source": self.scale_source,
            "intrinsic_scales": list(self.intrinsic_scales),
        }


__all__ = ["LTX2AdapterReceipt"]
