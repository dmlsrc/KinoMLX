"""Offline selection of an exact community adapter release file."""

from pathlib import Path

from kinomlx.huggingface import download_hint, resolve_cached_snapshot


def resolve_cached_adapter(
    hf_home: Path, repo_id: str, filename: str, *, label: str, alternative: str = ""
) -> tuple[Path, dict[str, str]]:
    """Resolve cached main without downloads or filename guessing."""
    try:
        snapshot = resolve_cached_snapshot(hf_home, repo_id)
        path = snapshot.path / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing cached adapter: {path}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{label} needs a cached adapter. Download it with "
            f"{download_hint(repo_id, filename)}." + alternative
        ) from exc
    return path, {
        "selection": "cached-default",
        "repo_id": snapshot.repo_id,
        "revision": snapshot.revision,
        "filename": filename,
    }
