"""Four-step community Turbo recipe selection, independent of adapter algebra."""

from dataclasses import dataclass
from pathlib import Path

from kinomlx.adapters.huggingface import resolve_cached_adapter
from kinomlx.adapters.selection import (
    AdapterSelection,
    validate_adapter_gate,
)

from .inputs import profile_selection

TURBO_ADAPTER_REPO = "lvladikov/Krea2-Turbo-Distill-4step-LoRA"
TURBO_4STEP_RECIPE = {"steps": 4, "guidance": 0.0, "mu": 1.15}
TURBO_ADAPTER_FILENAME = "krea2_turbo_4step_rank_64_lora.safetensors"


@dataclass(frozen=True)
class TurboFourStepProfile:
    """One profile's switches, owned here so no resolver needs the request type."""

    enabled: bool = False
    strength: float = 1.0
    exclusions: str = ""
    step_strengths: tuple[float, ...] = ()
    sigma_window: str = ""


def resolve_turbo_adapters(
    hf_home: Path | None,
    *,
    enabled: bool,
    strength: float,
    additional: tuple[AdapterSelection, ...],
    exclusions: str = "",
    step_strengths: tuple[float, ...] = (),
    sigma_window: str = "",
) -> tuple[tuple[AdapterSelection, ...], dict[str, object] | None]:
    """Prepend the profile adapter without changing generic strength pairing."""
    if not enabled:
        return additional, None
    if hf_home is None:
        raise ValueError("The four-step Turbo profile requires an HF cache location")
    validate_adapter_gate(
        has_window=bool(sigma_window),
        has_steps=bool(step_strengths),
        window_field="turbo_4step_sigma_window",
        steps_field="turbo_4step_step_strengths",
    )
    path, receipt = resolve_cached_adapter(
        hf_home, TURBO_ADAPTER_REPO, TURBO_ADAPTER_FILENAME, label="Krea four-step Turbo"
    )
    if any(selection.path.resolve() == path.resolve() for selection in additional):
        raise ValueError(
            "The four-step Turbo adapter is also listed in --adapter; "
            "use --turbo-4step-strength to control its strength"
        )
    selected = profile_selection(
        path,
        strength=strength,
        exclusions=exclusions,
        location="turbo_4step",
        step_strengths=step_strengths,
        sigma_window=sigma_window,
    )
    # An active step array owns the magnitude, so a resolved default strength is
    # not part of what ran and must not be reported as if it were.
    magnitude: dict[str, object] = {} if step_strengths else {"strength": strength}
    return (selected, *additional), {
        **receipt,
        "profile": "turbo-4step",
        **magnitude,
        "exclusions": exclusions,
        "gate": (
            {"step_strengths": list(step_strengths)}
            if step_strengths
            else {"sigma_window": sigma_window}
            if sigma_window
            else None
        ),
    }
