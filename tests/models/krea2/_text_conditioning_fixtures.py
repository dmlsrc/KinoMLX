"""Shared real-fixture helpers for the Krea 2 text conditioning gates."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

RAW_REVISION = "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
TURBO_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
FIXTURE_SHA256 = "706e0cc127ecda9b79b177ad573e927f42981456f7955117e9dc64cbcfd8b04d"
REPORT_SHA256 = "a5171fc30784e54eddea101f48c9e3d2d48a5a497911e4b224e99254164095d6"
NORMALIZED_RMS_LIMIT = 0.03125
COSINE_FLOOR = 0.99951171875
MAX_SCALE_LIMIT = 0.125


@dataclass(frozen=True)
class TapMetrics:
    """Full-tensor comparison metrics for one selected Qwen state."""

    normalized_rms: float
    cosine: float
    max_scale: float

    def to_dict(self) -> dict[str, float]:
        return {
            "normalized_rms": self.normalized_rms,
            "cosine": self.cosine,
            "max_scale": self.max_scale,
        }


def sha256(path: Path) -> str:
    """Hash one fixture or report without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompts_from_report(report_path: Path) -> tuple[str, ...]:
    """Load fixture prompts while leaving them out of committed product tests."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        fixtures = report["batch"]["fixtures"]
        prompts = tuple(fixture["prompt"] for fixture in fixtures)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{report_path}: invalid Krea text fixture report") from exc
    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{report_path}: invalid Krea text fixture prompts")
    return prompts


def tap_metrics(candidate: mx.array, reference: mx.array) -> TapMetrics:
    """Measure one full selected state in FP32 without NumPy."""
    candidate = candidate.astype(mx.float32)
    reference = reference.astype(mx.float32)
    mx.eval(candidate, reference)
    delta = candidate - reference
    reference_max = float(mx.max(mx.abs(reference)).item())
    reference_rms = float(mx.sqrt(mx.mean(mx.square(reference))).item())
    scale = max(reference_max, reference_rms, 1e-12)
    normalized_rms = float(mx.sqrt(mx.mean(mx.square(delta))).item()) / max(
        reference_rms,
        scale * 2**-12,
    )
    max_scale = float(mx.max(mx.abs(delta)).item()) / scale
    candidate_flat = candidate.reshape(-1)
    reference_flat = reference.reshape(-1)
    cosine = float(
        mx.sum(candidate_flat * reference_flat).item()
        / (mx.linalg.norm(candidate_flat).item() * mx.linalg.norm(reference_flat).item())
    )
    return TapMetrics(
        normalized_rms=normalized_rms,
        cosine=cosine,
        max_scale=max_scale,
    )


__all__ = [
    "COSINE_FLOOR",
    "FIXTURE_SHA256",
    "MAX_SCALE_LIMIT",
    "NORMALIZED_RMS_LIMIT",
    "RAW_REVISION",
    "REPORT_SHA256",
    "TURBO_REVISION",
    "TapMetrics",
    "prompts_from_report",
    "sha256",
    "tap_metrics",
]
