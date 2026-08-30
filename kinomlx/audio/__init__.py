"""Audio post-processing utilities with lazy MLX-backed exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

_EXPORT_MODULES = {
    "DEFAULT_DETECT_THRESHOLD_RATIO": "onset",
    "DEFAULT_DETECT_WINDOW_MS": "onset",
    "DEFAULT_SILENCE_END_MS": "onset",
    "DEFAULT_SILENCE_RATIO": "onset",
    "DEFAULT_SILENCE_START_MS": "onset",
    "DEFAULT_TRIM_MS": "trim_mode",
    "OnsetTrimResult": "onset",
    "detect_onset_latent_spike": "onset",
    "detect_onset_spike": "onset",
    "mitigate_onset": "onset",
    "parse_trim_mode": "trim_mode",
    "trim_onset": "onset",
}

__all__ = sorted(_EXPORT_MODULES)

if TYPE_CHECKING:
    from . import onset as _onset
    from . import trim_mode as _trim_mode

    # Mypy cannot infer per-name types from the dynamic module __getattr__.
    # Mirror the lazy map statically without importing MLX at runtime.
    DEFAULT_DETECT_THRESHOLD_RATIO = _onset.DEFAULT_DETECT_THRESHOLD_RATIO
    DEFAULT_DETECT_WINDOW_MS = _onset.DEFAULT_DETECT_WINDOW_MS
    DEFAULT_SILENCE_END_MS = _onset.DEFAULT_SILENCE_END_MS
    DEFAULT_SILENCE_RATIO = _onset.DEFAULT_SILENCE_RATIO
    DEFAULT_SILENCE_START_MS = _onset.DEFAULT_SILENCE_START_MS
    DEFAULT_TRIM_MS = _trim_mode.DEFAULT_TRIM_MS
    OnsetTrimResult = _onset.OnsetTrimResult
    detect_onset_latent_spike = _onset.detect_onset_latent_spike
    detect_onset_spike = _onset.detect_onset_spike
    mitigate_onset = _onset.mitigate_onset
    parse_trim_mode = _trim_mode.parse_trim_mode
    trim_onset = _onset.trim_onset


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    return cast(object, getattr(module, name))


def __dir__() -> list[str]:
    """Expose lazy public names to completion and introspection tools."""
    return sorted(set(globals()) | set(__all__))
