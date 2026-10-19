"""Exact offline edit-adapter selection from the configured Hub cache."""

from pathlib import Path

import pytest

from kinomlx.huggingface import repo_cache_dir
from kinomlx.models.krea2.adapters.edit_profile import (
    EDIT_ADAPTER_FILENAME,
    EDIT_ADAPTER_REPO,
    resolve_cached_edit_adapter,
)


def test_edit_adapter_uses_exact_main_ref_in_custom_cache(tmp_path: Path) -> None:
    root = repo_cache_dir(tmp_path, EDIT_ADAPTER_REPO)
    revision = "1" * 40
    selected = root / "snapshots" / revision
    selected.mkdir(parents=True)
    (selected / EDIT_ADAPTER_FILENAME).write_bytes(b"selected")
    other = root / "snapshots" / ("2" * 40)
    other.mkdir()
    (other / EDIT_ADAPTER_FILENAME).write_bytes(b"other")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(revision)
    path, receipt = resolve_cached_edit_adapter(tmp_path)
    assert path == selected / EDIT_ADAPTER_FILENAME
    assert receipt["revision"] == revision
    assert receipt["selection"] == "cached-default"
    assert receipt["repo_id"] == EDIT_ADAPTER_REPO


def test_edit_adapter_missing_file_has_actionable_offline_error(tmp_path: Path) -> None:
    root = repo_cache_dir(tmp_path, EDIT_ADAPTER_REPO)
    revision = "1" * 40
    (root / "snapshots" / revision).mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(revision)
    with pytest.raises(FileNotFoundError, match="hf download"):
        resolve_cached_edit_adapter(tmp_path)


def test_edit_adapter_does_not_guess_snapshot_without_main(tmp_path: Path) -> None:
    selected = repo_cache_dir(tmp_path, EDIT_ADAPTER_REPO) / "snapshots" / ("1" * 40)
    selected.mkdir(parents=True)
    (selected / EDIT_ADAPTER_FILENAME).write_bytes(b"unused")
    with pytest.raises(FileNotFoundError, match="supply --edit-adapter"):
        resolve_cached_edit_adapter(tmp_path)


def test_default_edit_adapter_is_retained_with_additional_adapters(
    tmp_path: Path, monkeypatch
) -> None:
    from kinomlx.adapters.selection import AdapterSelection
    from kinomlx.models.krea2.adapters import edit_profile

    edit = tmp_path / "edit.safetensors"
    style = AdapterSelection(tmp_path / "style.safetensors", 0.35, ("text_fusion",))
    character = AdapterSelection(tmp_path / "character.safetensors", 0.8)
    monkeypatch.setattr(
        edit_profile,
        "resolve_cached_edit_adapter",
        lambda root: (edit, {"selection": "cached-default"}),
    )
    selections, receipt = edit_profile.resolve_edit_adapters(
        tmp_path,
        edit_path=None,
        edit_strength=1.2,
        additional=(style, character),
        edit_exclusions="to_k,ff",
    )
    assert selections == (AdapterSelection(edit, 1.2, ("to_k", "ff")), style, character)
    assert receipt["selection"] == "cached-default"


def test_explicit_edit_adapter_does_not_access_default_cache(tmp_path: Path, monkeypatch) -> None:
    from kinomlx.models.krea2.adapters import edit_profile

    def forbidden(_root):
        pytest.fail("an explicit edit adapter must not resolve the default")

    monkeypatch.setattr(edit_profile, "resolve_cached_edit_adapter", forbidden)
    edit = tmp_path / "custom.safetensors"
    selections, receipt = edit_profile.resolve_edit_adapters(
        tmp_path, edit_path=edit, edit_strength=0.9, additional=()
    )
    assert selections[0].path == edit
    assert selections[0].strength == 0.9
    assert receipt["selection"] == "explicit"


def test_edit_adapter_duplicate_is_rejected_instead_of_applied_twice(tmp_path: Path) -> None:
    from kinomlx.adapters.selection import AdapterSelection
    from kinomlx.models.krea2.adapters.edit_profile import resolve_edit_adapters

    edit = tmp_path / "edit.safetensors"
    alias = tmp_path / "alias.safetensors"
    alias.symlink_to(edit)
    with pytest.raises(ValueError, match="also listed"):
        resolve_edit_adapters(
            tmp_path, edit_path=edit, edit_strength=1, additional=(AdapterSelection(alias, 0.5),)
        )


def test_cli_edit_and_additional_adapter_strengths_are_independent(tmp_path: Path) -> None:
    from kinomlx.models.krea2.arguments import build_parser
    from kinomlx.models.krea2.types import Krea2Request

    args = build_parser().parse_args(
        [
            "--source-image",
            str(tmp_path / "source.png"),
            "--prompt",
            "test",
            "--edit-adapter",
            str(tmp_path / "edit.safetensors"),
            "--edit-adapter-strength",
            "0.9",
            "--edit-adapter-exclude",
            "text_fusion,to_k",
            "--adapter",
            str(tmp_path / "style.safetensors"),
            "--adapter-strength",
            "0.35",
        ]
    )
    request = Krea2Request(
        prompt="test",
        source_images=tuple(args.source_images),
        edit_adapter_path=args.edit_adapter_path,
        edit_adapter_strength=args.edit_adapter_strength,
        edit_adapter_exclusions=args.edit_adapter_exclusions,
        adapters=tuple(args.adapters),
    )
    assert request.edit_adapter_strength == 0.9
    assert request.resolved_adapters()[0].strength == 0.35


@pytest.mark.parametrize("exclusion", ["unknown", "to_k,unknown"])
def test_edit_knockout_rejects_unknown_category(exclusion: str) -> None:
    from kinomlx.models.krea2.types import Krea2Request

    with pytest.raises(ValueError, match="unknown"):
        Krea2Request(
            prompt="test", source_images=(Path("source.png"),), edit_adapter_exclusions=exclusion
        )
