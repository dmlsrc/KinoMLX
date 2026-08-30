"""Static contract: package modules and developer tools use logging."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "kinomlx", REPO / "scripts")


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_source_uses_logging_instead_of_direct_terminal_output() -> None:
    offenders: list[str] = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _dotted_name(node.func)
                direct = name == "print" or name in {
                    "sys.stdout.write",
                    "sys.stderr.write",
                }
                console_write = name is not None and name.endswith(".print")
                if direct or console_write:
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}")
    assert offenders == []
