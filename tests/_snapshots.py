"""Pinned Hugging Face snapshots for real-asset tests."""

from pathlib import Path

import pytest

from kinomlx.huggingface import resolve_cached_snapshot
from kinomlx.settings import Settings


def pinned_snapshot(repo_id: str, revision: str) -> Path:
    """Return the cached snapshot of ``repo_id`` when its main ref is ``revision``.

    The product finds a repository in the local Hub cache through its own reader and
    never through a hand-built cache path. A test whose fixtures were captured at one
    revision asks the same reader and skips when the cache holds nothing or another
    revision, naming both.
    """
    try:
        snapshot = resolve_cached_snapshot(Settings.from_env().hf_home, repo_id)
    except FileNotFoundError, ValueError:
        pytest.skip(f"{repo_id} is not cached")
    if snapshot.revision != revision:
        pytest.skip(f"{repo_id} main ref is {snapshot.revision}, fixtures are pinned at {revision}")
    return snapshot.path
