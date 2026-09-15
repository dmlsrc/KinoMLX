"""The pinned-snapshot helper resolves through the product's cache reader."""

from pathlib import Path

import pytest

from tests._snapshots import pinned_snapshot

_REPO_ID = "example/model"
_REVISION = "0123456789abcdef0123456789abcdef01234567"
_OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _cache(hf_home: Path, revision: str) -> Path:
    repository = hf_home / "hub" / "models--example--model"
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text(revision, encoding="utf-8")
    return snapshot


def test_pinned_snapshot_is_the_snapshot_the_main_ref_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshot = _cache(tmp_path, _REVISION)

    assert pinned_snapshot(_REPO_ID, _REVISION) == snapshot.resolve()


def test_pinned_snapshot_skips_an_uncached_repository(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    with pytest.raises(pytest.skip.Exception, match="example/model is not cached"):
        pinned_snapshot(_REPO_ID, _REVISION)


def test_pinned_snapshot_skips_a_main_ref_at_another_revision(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _cache(tmp_path, _OTHER)

    with pytest.raises(pytest.skip.Exception, match=f"main ref is {_OTHER}, fixtures are pinned"):
        pinned_snapshot(_REPO_ID, _REVISION)
