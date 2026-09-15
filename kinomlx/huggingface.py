"""Read-only discovery of already-local Hugging Face Hub snapshots."""

import re
from dataclasses import dataclass
from pathlib import Path

_REVISION = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class CachedSnapshot:
    """One exact local Hub ref resolution."""

    repo_id: str
    revision: str
    path: Path


def repo_url(repo_id: str) -> str:
    """Return the browsable page for one repository.

    Every acquisition hint carries it, because a user who cannot run the CLI -
    no network in this shell, no CLI installed, an access condition still to
    accept - needs somewhere to fetch the same files by hand.
    """
    if not repo_id or "/" not in repo_id:
        raise ValueError(f"invalid Hugging Face repository id: {repo_id!r}")
    return f"https://huggingface.co/{repo_id}"


def download_hint(repo_id: str, *arguments: str) -> str:
    """Return an acquisition command paired with the page it downloads from."""
    command = " ".join(("hf", "download", repo_id, *arguments))
    return f"`{command}` (or download by hand from {repo_url(repo_id)})"


def repo_cache_dir(hf_home: Path | str, repo_id: str) -> Path:
    """Return the ordinary Hub cache directory for one model repository."""
    if not repo_id or "/" not in repo_id:
        raise ValueError(f"invalid Hugging Face repository id: {repo_id!r}")
    return Path(hf_home).expanduser() / "hub" / f"models--{repo_id.replace('/', '--')}"


def cached_snapshots(hf_home: Path | str, repo_id: str) -> tuple[Path, ...]:
    """Return local snapshot directories, newest filesystem entry first."""
    root = repo_cache_dir(hf_home, repo_id) / "snapshots"
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )


def resolve_cached_snapshot(
    hf_home: Path | str,
    repo_id: str,
    *,
    ref: str = "main",
) -> CachedSnapshot:
    """Resolve one local ref exactly, without network or Hub-library imports."""
    if not ref or Path(ref).name != ref or any(character in ref for character in "/\\"):
        raise ValueError("Hugging Face ref must be one path-safe segment")
    cache = repo_cache_dir(hf_home, repo_id)
    ref_path = cache / "refs" / ref
    if not ref_path.is_file():
        raise FileNotFoundError(f"no cached {ref!r} ref for {repo_id} at {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    if _REVISION.fullmatch(revision) is None:
        raise ValueError(f"invalid cached revision in {ref_path}: {revision!r}")
    snapshot = cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"cached ref {ref_path} points at missing snapshot {snapshot}")
    return CachedSnapshot(repo_id=repo_id, revision=revision, path=snapshot.resolve())


__all__ = [
    "CachedSnapshot",
    "cached_snapshots",
    "repo_cache_dir",
    "resolve_cached_snapshot",
]
