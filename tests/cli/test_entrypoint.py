"""Installed console-script metadata and import-target contracts."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path


def test_console_script_targets_a_callable_in_the_concrete_module() -> None:
    repo = Path(__file__).resolve().parents[2]
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    target = project["project"]["scripts"]["kinomlx"]
    assert target == "kinomlx.cli.main:main"

    module_name, attribute = target.split(":", 1)
    entry = getattr(importlib.import_module(module_name), attribute)
    assert callable(entry)


def test_cli_package_does_not_reexport_the_colliding_main_name() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kinomlx.cli as package; assert 'main' not in vars(package)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
