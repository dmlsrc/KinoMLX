"""Source identities and local Hugging Face Hub discovery."""

from pathlib import Path

import pytest

from kinomlx.huggingface import cached_snapshots, repo_cache_dir, resolve_cached_snapshot
from kinomlx.io.identity import artifact_fingerprint


def test_hub_ref_resolution_is_local_exact_and_stale_refs_fail(tmp_path: Path) -> None:
    repo_id = "example/model"
    cache = repo_cache_dir(tmp_path, repo_id)
    old = cache / "snapshots" / ("1" * 40)
    current = cache / "snapshots" / ("2" * 40)
    old.mkdir(parents=True)
    current.mkdir()
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(current.name + "\n", encoding="utf-8")

    resolved = resolve_cached_snapshot(tmp_path, repo_id)
    assert resolved.revision == current.name
    assert resolved.path == current
    assert set(cached_snapshots(tmp_path, repo_id)) == {old, current}

    (cache / "refs" / "main").write_text("3" * 40, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="points at missing snapshot"):
        resolve_cached_snapshot(tmp_path, repo_id)


def test_small_artifact_fingerprint_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"same": true}\n', encoding="utf-8")
    second.write_bytes(first.read_bytes())
    assert artifact_fingerprint(first) == artifact_fingerprint(second)
    second.write_text('{"same": false}\n', encoding="utf-8")
    assert artifact_fingerprint(first) != artifact_fingerprint(second)


def test_directory_artifact_fingerprint_uses_its_config_signature(tmp_path: Path) -> None:
    source = tmp_path / "component"
    source.mkdir()
    config = source / "config.json"
    config.write_text('{"model": "example"}\n', encoding="utf-8")
    before = artifact_fingerprint(source)

    config.write_text('{"model": "changed"}\n', encoding="utf-8")
    assert artifact_fingerprint(source) != before


def test_hub_ref_must_be_one_path_segment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one path-safe segment"):
        resolve_cached_snapshot(tmp_path, "example/model", ref="../main")


def test_large_local_artifact_ref_survives_same_volume_move_not_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"large-local-fixture")
    before = artifact_fingerprint(source, content_hash_limit_bytes=0)

    moved = tmp_path / "nested" / "renamed.safetensors"
    moved.parent.mkdir()
    source.replace(moved)
    assert artifact_fingerprint(moved, content_hash_limit_bytes=0) == before

    copied = tmp_path / "copied.safetensors"
    copied.write_bytes(moved.read_bytes())
    assert artifact_fingerprint(copied, content_hash_limit_bytes=0) != before
