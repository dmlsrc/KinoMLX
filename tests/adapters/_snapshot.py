"""Hugging Face cache layouts for adapter tests."""

import hashlib
from pathlib import Path


def link_as_snapshot(root: Path, *files: Path, folder: str = "") -> tuple[Path, ...]:
    """Move files into a hub-style blob store and return their snapshot links.

    The cache keeps each file once under its digest and links it, relatively,
    from ``snapshots/<revision>/[folder/]`` under its original name.
    """
    blobs = root / "blobs"
    snapshot = root / "snapshots" / "0123abcd"
    directory = snapshot / folder if folder else snapshot
    blobs.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    up = Path(*[".."] * (len(directory.relative_to(root).parts)))
    links = []
    for file in files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        file.replace(blobs / digest)
        link = directory / file.name
        link.symlink_to(up / "blobs" / digest)
        links.append(link)
    return tuple(links)
