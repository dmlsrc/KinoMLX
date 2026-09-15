"""Community edit adapter selection, separate from native conditioning code."""

from pathlib import Path

from kinomlx.adapters.huggingface import resolve_cached_adapter
from kinomlx.adapters.selection import AdapterSelection

from .inputs import profile_selection

EDIT_ADAPTER_REPO = "conradlocke/krea2-identity-edit"
EDIT_ADAPTER_FILENAME = "krea2_identity_edit_v1_2_r128.safetensors"


def resolve_cached_edit_adapter(hf_home: Path) -> tuple[Path, dict[str, str]]:
    """Resolve the exact cached main ref; never download or guess another revision."""
    return resolve_cached_adapter(
        hf_home,
        EDIT_ADAPTER_REPO,
        EDIT_ADAPTER_FILENAME,
        label="Krea 2 editing",
        alternative=" Or supply --edit-adapter with a local file.",
    )


def resolve_edit_adapters(
    hf_home: Path,
    *,
    edit_path: Path | None,
    edit_strength: float,
    additional: tuple[AdapterSelection, ...],
    edit_exclusions: str = "",
) -> tuple[tuple[AdapterSelection, ...], dict[str, str]]:
    """Keep the edit capability independent of optional style/character adapters."""
    if edit_path is None:
        path, receipt = resolve_cached_edit_adapter(hf_home)
    else:
        path = edit_path.expanduser().absolute()
        receipt = {"selection": "explicit", "filename": path.name}
    if any(selection.path.expanduser().resolve() == path.resolve() for selection in additional):
        raise ValueError(
            "the edit adapter is also listed in --adapter; select it with --edit-adapter "
            "and control its strength with --edit-adapter-strength"
        )
    edit = profile_selection(
        path, strength=edit_strength, exclusions=edit_exclusions, location="edit_adapter"
    )
    return (edit, *additional), receipt
