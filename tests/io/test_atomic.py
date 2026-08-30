"""Atomic single-file publication contracts."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from kinomlx.io.atomic import atomic_output_path, write_text_atomic


def _permission_bits(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _write_partial_then_fail(destination: Path) -> None:
    with atomic_output_path(destination, temp_suffix=".tmp.json") as temporary:
        assert temporary.is_file()
        temporary.write_text("partial", encoding="utf-8")
        raise RuntimeError("injected failure")


def test_atomic_output_preserves_previous_file_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("previous", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected"):
        _write_partial_then_fail(destination)

    assert destination.read_text(encoding="utf-8") == "previous"
    assert tuple(tmp_path.glob(".artifact.*.tmp.json")) == ()


def test_atomic_text_replaces_only_after_complete_write(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("previous", encoding="utf-8")
    write_text_atomic(destination, "complete")
    assert destination.read_text(encoding="utf-8") == "complete"
    assert tuple(tmp_path.glob(".artifact.*.tmp")) == ()


@pytest.mark.parametrize(
    ("creation_umask", "expected_mode"),
    [(0o002, 0o664), (0o027, 0o640)],
)
def test_new_atomic_output_uses_regular_umask_permissions(
    tmp_path: Path,
    creation_umask: int,
    expected_mode: int,
) -> None:
    destination = tmp_path / "artifact.json"
    previous_umask = os.umask(creation_umask)
    try:
        with atomic_output_path(destination, temp_suffix=".tmp") as temporary:
            assert _permission_bits(temporary) == 0o600
            temporary.write_text("complete", encoding="utf-8")
    finally:
        os.umask(previous_umask)

    assert _permission_bits(destination) == expected_mode


def test_atomic_replacement_preserves_existing_permissions(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("previous", encoding="utf-8")
    destination.chmod(0o664)

    write_text_atomic(destination, "complete")

    assert destination.read_text(encoding="utf-8") == "complete"
    assert _permission_bits(destination) == 0o664
