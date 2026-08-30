"""Atomic publication helpers for single-file product artifacts."""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _existing_regular_mode(path: Path) -> int | None:
    """Return a replaceable file's permission bits without following symlinks."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    return status.st_mode & 0o777 if stat.S_ISREG(status.st_mode) else None


def _open_atomic_temporary(destination: Path, temp_suffix: str) -> tuple[int, Path, int]:
    """Exclusively create a private random peer and return its publication mode."""
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    for _attempt in range(100):
        temporary = destination.parent / (
            f".{destination.stem}.{secrets.token_hex(8)}{temp_suffix}"
        )
        try:
            # First let the kernel apply the process umask to an ordinary 0666
            # output mode, then keep partial content private until publication.
            descriptor = os.open(temporary, flags, 0o666)
        except FileExistsError:
            continue
        try:
            publication_mode = os.fstat(descriptor).st_mode & 0o777
            os.fchmod(descriptor, 0o600)
        except BaseException:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        return descriptor, temporary, publication_mode
    raise FileExistsError(f"cannot allocate atomic temporary beside {destination}")


@contextmanager
def atomic_output_path(path: Path | str, *, temp_suffix: str) -> Iterator[Path]:
    """Yield a same-directory temporary and atomically publish it on success.

    New outputs receive ordinary ``0666 & ~umask`` permissions. Replacing an
    existing regular file preserves its permission bits across the inode swap.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = _existing_regular_mode(destination)
    descriptor, temporary, creation_mode = _open_atomic_temporary(destination, temp_suffix)
    publication_mode = creation_mode if existing_mode is None else existing_mode
    try:
        descriptor_to_close = descriptor
        descriptor = -1
        os.close(descriptor_to_close)
        yield temporary
        temporary.chmod(publication_mode)
        temporary.replace(destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace one text artifact after its complete write succeeds."""
    with atomic_output_path(path, temp_suffix=".tmp") as temporary:
        temporary.write_text(text, encoding=encoding)


def write_text_exclusive(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Write a complete text artifact while refusing any existing destination."""
    destination = Path(path)
    created = False
    try:
        with destination.open("x", encoding=encoding, newline="\n") as stream:
            created = True
            stream.write(text)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


__all__ = ["atomic_output_path", "write_text_atomic", "write_text_exclusive"]
