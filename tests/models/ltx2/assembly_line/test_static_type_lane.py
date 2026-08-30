"""Focused mypy lane for assembly-line structural contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]


def test_assembly_line_static_contracts() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr
