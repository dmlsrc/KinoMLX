"""Repository text stays portable and free of typographic Unicode drift."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT_ROOTS = tuple(REPO / name for name in ("kinomlx", "tests", "scripts", "docs", "benches"))
ROOT_FILES = (
    REPO / "README.md",
    REPO / "THIRD_PARTY_LICENSES.md",
    REPO / "LICENSE",
    REPO / "pyproject.toml",
)
TEXT_SUFFIXES = frozenset({".py", ".md", ".toml"})


def test_public_project_text_is_ascii() -> None:
    paths = list(ROOT_FILES)
    for root in TEXT_ROOTS:
        paths.extend(
            path for path in root.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES
        )

    offenders = []
    for path in sorted(set(paths)):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            chars = sorted({character for character in line if ord(character) > 127})
            if chars:
                offenders.append(
                    f"{path.relative_to(REPO)}:{line_number}: "
                    + " ".join(f"U+{ord(character):04X}" for character in chars)
                )
    assert offenders == []


def test_wheel_declares_project_and_third_party_license_files() -> None:
    metadata = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))

    assert set(metadata["project"]["license-files"]) == {
        "LICENSE",
        "THIRD_PARTY_LICENSES.md",
    }
