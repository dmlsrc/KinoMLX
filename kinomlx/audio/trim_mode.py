"""Dependency-free parsing for sequence-start audio onset policy."""

from __future__ import annotations

import math

DEFAULT_TRIM_MS = 120.0
"""Default leading region zero-filled after automatic spike detection."""


def parse_trim_mode(spec: str) -> tuple[str, float]:
    """Map ``auto``, ``off``, or milliseconds to encoder policy values."""
    value_text = spec.strip().lower()
    if value_text == "auto":
        return ("auto", DEFAULT_TRIM_MS)
    if value_text in {"off", "none"}:
        return ("off", 0.0)
    try:
        value = float(value_text)
    except ValueError:
        raise ValueError(
            f"Invalid --audio-onset-trim value {spec!r}. Expected 'auto', 'off', "
            "or a duration in milliseconds (e.g. '120')."
        ) from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"--audio-onset-trim must be finite and non-negative; got {value}")
    if value == 0:
        return ("off", 0.0)
    return ("force", value)


__all__ = ["DEFAULT_TRIM_MS", "parse_trim_mode"]
