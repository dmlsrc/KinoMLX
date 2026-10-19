"""Relative documentation links resolve, including links inside HTML argument tables.

Markdown lint checks Markdown links, but not the ``href`` attributes of the HTML tables the
argument catalogs use, so this test reads both forms.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = sorted(
    {
        REPO / "README.md",
        REPO / "CHANGELOG.md",
        *REPO.glob("docs/**/*.md"),
        *REPO.glob("kinomlx/**/*.md"),
        *REPO.glob("weights-src/**/*.md"),
    }
)

_FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)\)")
_HTML_LINK = re.compile(r"""\bhref="([^"]+)\"""")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)


def _prose(text: str) -> str:
    """Drop fenced blocks and code spans, which may show link syntax literally."""
    return _CODE_SPAN.sub("", _FENCE.sub("", text))


def _slug(heading: str) -> str:
    """Return GitHub's anchor for an ASCII heading."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.lower().replace("`", "")
    text = re.sub(r"[^a-z0-9 _-]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for match in _HEADING.finditer(text):
        base = _slug(match.group(2))
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        anchors.add(base if seen == 0 else f"{base}-{seen}")
    return anchors


def _targets(path: Path) -> list[str]:
    text = _prose(path.read_text(encoding="utf-8"))
    return _MARKDOWN_LINK.findall(text) + _HTML_LINK.findall(text)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: str(path.relative_to(REPO)))
def test_relative_links_and_anchors_resolve(document: Path) -> None:
    broken = []
    for target in _targets(document):
        if _EXTERNAL.match(target):
            continue
        location, _, fragment = target.partition("#")
        resolved = (document.parent / location).resolve() if location else document
        if not resolved.exists():
            broken.append(f"{target}: missing file")
            continue
        if fragment and resolved.suffix == ".md" and fragment not in _anchors(resolved):
            broken.append(f"{target}: missing anchor")
    assert broken == []
