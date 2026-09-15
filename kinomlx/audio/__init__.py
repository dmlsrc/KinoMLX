"""Audio post-processing utilities."""

from .onset import (
    DEFAULT_DETECT_THRESHOLD_RATIO,
    DEFAULT_DETECT_WINDOW_MS,
    DEFAULT_SILENCE_END_MS,
    DEFAULT_SILENCE_RATIO,
    DEFAULT_SILENCE_START_MS,
    OnsetTrimResult,
    detect_onset_spike,
    mitigate_onset,
    trim_onset,
)
from .trim_mode import DEFAULT_TRIM_MS, parse_trim_mode

__all__ = [
    "DEFAULT_DETECT_THRESHOLD_RATIO",
    "DEFAULT_DETECT_WINDOW_MS",
    "DEFAULT_SILENCE_END_MS",
    "DEFAULT_SILENCE_RATIO",
    "DEFAULT_SILENCE_START_MS",
    "DEFAULT_TRIM_MS",
    "OnsetTrimResult",
    "detect_onset_spike",
    "mitigate_onset",
    "parse_trim_mode",
    "trim_onset",
]
