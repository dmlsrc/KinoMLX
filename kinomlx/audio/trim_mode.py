"""Parsing and local field admission for sequence-start audio onset policy."""

import math
from dataclasses import dataclass

from kinomlx.config_constraints import LengthConstraint

DEFAULT_TRIM_MS = 120.0
"""Default leading region zero-filled after automatic spike detection."""


NAMED_TRIM_MODES = frozendict(
    {"auto": ("auto", DEFAULT_TRIM_MS), "off": ("off", 0.0), "none": ("off", 0.0)}
)


def parse_trim_mode(spec: str) -> tuple[str, float]:
    """Map ``auto``, ``off``, or milliseconds to encoder policy values."""
    value_text = spec.strip().lower()
    named = NAMED_TRIM_MODES.get(value_text)
    if named is not None:
        return named
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


@dataclass(frozen=True)
class AudioOnsetTrimConstraint(LengthConstraint):
    """Own the existing named-mode or millisecond grammar as a local field constraint."""

    minimum: int = 1

    def canonicalize(self, value: object) -> object:
        """Spell a bare millisecond count the way the field stores it.

        TOML and ``--set`` deliver ``120`` as an integer; the field is a string so
        that ``auto`` and ``off`` share it. Both spellings mean the same thing, and
        the saved configuration keeps the string. A negative or non-finite number
        is passed through as text so ``validate`` names the actual problem.
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            return value
        if isinstance(value, int) or float(value).is_integer():
            return str(int(value)) if math.isfinite(value) else repr(float(value))
        return repr(float(value))

    def validate(self, value: object) -> None:
        if not isinstance(value, str):
            raise ValueError("must be an audio onset mode or millisecond string")
        super().validate(value)
        parse_trim_mode(value)

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "kind": "audio_onset_trim",
            "named_modes": list(NAMED_TRIM_MODES),
            "numeric_unit": "milliseconds",
            "numeric_minimum": 0,
            "finite": True,
        }


__all__ = ["DEFAULT_TRIM_MS", "NAMED_TRIM_MODES", "AudioOnsetTrimConstraint", "parse_trim_mode"]
