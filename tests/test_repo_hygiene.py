"""Repository text hygiene.

Every tracked text file stays plain ASCII so diffs, grep, and terminals
behave identically everywhere. Intentional Unicode belongs in escape
sequences (the tokenizer fixtures use "\\ufeff"-style escapes), never in
raw source bytes; a file that genuinely needs raw Unicode joins _ALLOWED
with its reason recorded beside it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# Tracked files permitted to carry raw non-ASCII bytes. Empty on purpose:
# nothing in the repository currently qualifies.
_ALLOWED: frozenset[str] = frozenset()


def _tracked_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if listing.returncode != 0:
        pytest.skip("not a git work tree; tracked-file listing unavailable")
    names = [name for name in listing.stdout.decode("utf-8").split("\0") if name]
    return [_REPO / name for name in names]


def test_tracked_text_files_are_plain_ascii() -> None:
    findings: list[str] = []
    for path in _tracked_files():
        relative = path.relative_to(_REPO).as_posix()
        if relative in _ALLOWED or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data or data.isascii():
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}: not valid UTF-8")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(
                f"{relative}:{line_number}:{column}: U+{ord(char):04X}"
                for column, char in enumerate(line, start=1)
                if ord(char) > 127
            )
    assert not findings, "non-ASCII bytes in tracked files:\n" + "\n".join(findings[:40])
