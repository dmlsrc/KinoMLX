"""Stable source and artifact provenance identities."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from kinomlx._json import JsonObject, JsonValue

_CONTENT_HASH_LIMIT_BYTES = 16 * 1024 * 1024
_SHA256_NAME = re.compile(r"[0-9a-f]{64}")


def file_signature(path: Path | str) -> JsonObject:
    """Return one cheap path, size, and timestamp source signature."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _canonical_bytes(payload: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_digest(payload: Mapping[str, JsonValue]) -> str:
    """Return the stable 20-hex provenance identity."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:20]


def object_digest(payload: Mapping[str, JsonValue]) -> str:
    """Return the full SHA-256 identity of canonical metadata."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _local_file_reference(path: Path) -> JsonObject:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def artifact_fingerprint(
    path: Path | str,
    *,
    content_hash_limit_bytes: int = _CONTENT_HASH_LIMIT_BYTES,
) -> str:
    """Return a content identity when cheap and a stable source identity otherwise.

    Hugging Face cache symlinks point at content-addressed blobs, so their 64-hex
    blob name is already the file SHA-256. Small ordinary files are hashed in
    full. Large non-Hub files use their filesystem identity, size, and mtime.
    That survives a move on the same volume without adding a model-sized
    hashing pass; copies and cross-volume moves receive a new local identity.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        return f"source:{source_fingerprint(source)}"
    if not source.is_file():
        raise FileNotFoundError(f"artifact does not exist: {source}")
    if source.is_symlink():
        target_name = source.resolve().name
        if _SHA256_NAME.fullmatch(target_name):
            return f"sha256:{target_name}"
    if source.stat().st_size <= content_hash_limit_bytes:
        return f"sha256:{_sha256_file(source)}"
    return f"local-file:{payload_digest({'file': _local_file_reference(source)})}"


def artifact_set_fingerprint(
    artifacts: Iterable[tuple[str, Path]],
) -> str:
    """Return one order-independent identity for named component artifacts."""
    entries = sorted(
        (
            {
                "role": role,
                "fingerprint": artifact_fingerprint(path),
            }
            for role, path in artifacts
        ),
        key=lambda entry: str(entry["role"]),
    )
    payload: JsonObject = {"artifacts": entries}
    return f"sha256:{object_digest(payload)}"


def source_fingerprint(path: Path | str) -> str:
    """Return the established cheap identity for a file or configured directory."""
    source = Path(path).expanduser()
    signature_path = source / "config.json" if source.is_dir() else source
    return payload_digest({"source": file_signature(signature_path)})


__all__ = [
    "artifact_fingerprint",
    "artifact_set_fingerprint",
    "file_signature",
    "object_digest",
    "payload_digest",
    "source_fingerprint",
]
