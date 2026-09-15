"""Resolution order for the community adapter profiles.

Both profiles prepend an adapter to a request's own selections. The distilled
four-step adapter leads because it changes what every remaining step has to do;
the diversity adapter follows it, and the user's own selections follow both.
Only ordering is decided here, never strength or gating.

The profiles keep their own settings records so this module needs no model
request type, which keeps the adapter package free of a cycle back into it.
"""

from pathlib import Path

from kinomlx.adapters.selection import AdapterSelection

from .sda_profile import SDAProfile, resolve_sda_adapter
from .turbo_profile import TurboFourStepProfile, resolve_turbo_adapters


def resolve_profile_adapters(
    hf_home: Path | None,
    *,
    four_step: TurboFourStepProfile,
    sda: SDAProfile,
    additional: tuple[AdapterSelection, ...],
) -> tuple[tuple[AdapterSelection, ...], dict[str, object] | None, dict[str, object] | None]:
    """Prepend whichever profiles are enabled, in their fixed order."""
    selections, sda_receipt = resolve_sda_adapter(
        hf_home if sda.enabled else None,
        enabled=sda.enabled,
        strength=sda.strength,
        sigma_window=sda.sigma_window,
        step_strengths=sda.step_strengths,
        exclusions=sda.exclusions,
        additional=additional,
    )
    selections, four_step_receipt = resolve_turbo_adapters(
        hf_home if four_step.enabled else None,
        enabled=four_step.enabled,
        strength=four_step.strength,
        exclusions=four_step.exclusions,
        step_strengths=four_step.step_strengths,
        sigma_window=four_step.sigma_window,
        additional=selections,
    )
    return selections, four_step_receipt, sda_receipt


__all__ = ["resolve_profile_adapters"]
