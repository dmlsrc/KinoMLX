"""Community SDA diversity-adapter selection, independent of adapter algebra.

The adapter restores the sampling diversity Turbo's distillation removed. It is
trained at a single high-noise node and damages detail if it stays active for the
whole trajectory, so a gate is part of the selection rather than an option beside
it.

The repository ships two key spellings of the same factors. Only the trainer's
own flattened export maps completely onto the Krea graph; the diffusers-style
conversion mis-splits the two text-fusion module names and silently loses 32 of
264 targets. The profile pins the complete one so a first-class switch cannot
pick the lossy file by accident.
"""

from dataclasses import dataclass
from pathlib import Path

from kinomlx.adapters.huggingface import resolve_cached_adapter
from kinomlx.adapters.selection import (
    AdapterSelection,
    validate_adapter_gate,
)

from .inputs import profile_selection

SDA_ADAPTER_REPO = "F16/krea2-turbo-sda"
SDA_ADAPTER_FILENAME = "krea2_turbo_sda_v1.0_comfy.safetensors"

# The band the adapter records as its own training range. Selecting it reproduces
# the gate its model card prescribes on the eight-step schedule without naming a
# step, and follows the band rather than a step index at any other count.
SDA_TRAINED_BAND = (0.93, 0.97)
SDA_DEFAULT_SIGMA_WINDOW = "0.93"

# The configuration every number on the adapter's card was measured at.
SDA_PANEL_RECIPE = {"steps": 8, "guidance": 0.0, "mu": 1.15}


@dataclass(frozen=True)
class SDAProfile:
    """One profile's switches, owned here so no resolver needs the request type."""

    enabled: bool = False
    strength: float = 1.0
    exclusions: str = ""
    step_strengths: tuple[float, ...] = ()
    sigma_window: str = ""


def resolve_sda_adapter(
    hf_home: Path | None,
    *,
    enabled: bool,
    strength: float,
    sigma_window: str,
    step_strengths: tuple[float, ...],
    exclusions: str,
    additional: tuple[AdapterSelection, ...],
) -> tuple[tuple[AdapterSelection, ...], dict[str, object] | None]:
    """Prepend the gated diversity adapter without changing generic pairing."""
    if not enabled:
        return additional, None
    if hf_home is None:
        raise ValueError("The SDA diversity profile requires an HF cache location")
    validate_adapter_gate(
        has_window=bool(sigma_window),
        has_steps=bool(step_strengths),
        window_field="turbo_sda_sigma_window",
        steps_field="turbo_sda_step_strengths",
    )
    path, receipt = resolve_cached_adapter(
        hf_home, SDA_ADAPTER_REPO, SDA_ADAPTER_FILENAME, label="Krea SDA diversity"
    )
    if any(selection.path.resolve() == path.resolve() for selection in additional):
        raise ValueError(
            "The SDA adapter is also listed in --adapter; use --turbo-sda-strength "
            "and its gate controls instead"
        )
    selected = profile_selection(
        path,
        strength=strength,
        exclusions=exclusions,
        location="turbo_sda",
        step_strengths=step_strengths,
        sigma_window=sigma_window,
    )
    # An active step array owns the magnitude, so a resolved default strength is
    # not part of what ran and must not be reported as if it were.
    magnitude: dict[str, object] = {} if step_strengths else {"strength": strength}
    return (selected, *additional), {
        **receipt,
        "profile": "turbo-sda",
        **magnitude,
        "gate": (
            {"step_strengths": list(step_strengths)}
            if step_strengths
            else {"sigma_window": sigma_window}
            if sigma_window
            else None
        ),
        "exclusions": exclusions,
        "trained_band": list(SDA_TRAINED_BAND),
    }


def sda_pairing_warning(steps: int, sigmas: tuple[float, ...]) -> str | None:
    """Warn when no step of the resolved schedule reaches the adapter's band.

    The adapter was trained at one node of the eight-step schedule. A shorter
    trajectory, whether from the four-step profile or an explicit request, can
    leave no step inside its training band at all, which is a combination worth
    naming rather than silently running.
    """
    low, high = SDA_TRAINED_BAND
    levels = list(sigmas[:-1]) if sigmas else []
    if not levels or any(low <= value <= high for value in levels):
        return None
    closest = min(levels, key=lambda value: min(abs(value - low), abs(value - high)))
    return (
        f"The SDA adapter records a training band of {low:g} to {high:g}, but no step of "
        f"this {steps}-step schedule falls inside it (closest level {closest:.4f}). The "
        "pairing is unvalidated: treat its gate and strength as settings to explore, "
        "and see the denoise-schedule guide for combinations worth trying."
    )


__all__ = [
    "SDA_ADAPTER_FILENAME",
    "SDA_ADAPTER_REPO",
    "SDA_DEFAULT_SIGMA_WINDOW",
    "SDA_PANEL_RECIPE",
    "SDA_TRAINED_BAND",
    "SDAProfile",
    "resolve_sda_adapter",
    "sda_pairing_warning",
]
