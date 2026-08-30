"""Small content-fingerprint helpers for artifacts and receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path | str) -> str:
    """Return a namespaced SHA-256 fingerprint for one local file."""
    with Path(path).open("rb") as stream:
        return f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"


__all__ = ["file_sha256"]
