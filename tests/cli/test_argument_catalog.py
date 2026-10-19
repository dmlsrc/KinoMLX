"""Each argument catalog lists exactly the arguments its parser accepts."""

import argparse
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from kinomlx.cli.config_init import build_config_parser
from kinomlx.models.flux2.arguments import build_parser as build_flux2_parser
from kinomlx.models.gmnet.arguments import build_parser as build_gmnet_parser
from kinomlx.models.gmnet.converter_cli import build_parser as build_gmnet_converter_parser
from kinomlx.models.krea2.arguments import build_parser as build_krea2_parser
from kinomlx.models.ltx2.arguments import build_parser as build_ltx2_parser
from kinomlx.weights.cli import build_generic_convert_parser

_DOCS = Path(__file__).resolve().parents[2] / "docs"
# The arguments every model shares are catalogued once, on the configuration page.
_SHARED = _DOCS / "CONFIG.md"
_FIRST_CELL = re.compile(r"<tr><td[^>]*>(.*?)</td>")
_FLAG = re.compile(r"<code>(--?[a-z0-9][a-z0-9-]*)</code>")

_CATALOGS: tuple[tuple[str, Callable[[], argparse.ArgumentParser], Path], ...] = (
    ("ltx2", build_ltx2_parser, _DOCS / "ltx2" / "arguments.md"),
    ("krea2", build_krea2_parser, _DOCS / "krea2" / "arguments.md"),
    ("flux2", build_flux2_parser, _DOCS / "flux2" / "arguments.md"),
    ("gmnet", build_gmnet_parser, _DOCS / "gmnet" / "arguments.md"),
    ("gmnet-convert", build_gmnet_converter_parser, _DOCS / "gmnet" / "weights.md"),
    ("weights-convert", build_generic_convert_parser, _DOCS / "WEIGHTS.md"),
)


def _accepted(parser: argparse.ArgumentParser) -> set[str]:
    flags: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                flags |= _accepted(child)
        else:
            flags.update(action.option_strings)
    return flags - {"-h", "--help"}


def _catalogued(page: Path) -> set[str]:
    """Return the flags named in the argument column of a page's tables."""
    flags: set[str] = set()
    for cell in _FIRST_CELL.findall(page.read_text(encoding="utf-8")):
        flags.update(_FLAG.findall(cell))
    return flags


@pytest.mark.parametrize(
    ("build", "page"),
    [(build, page) for _name, build, page in _CATALOGS],
    ids=[name for name, _build, _page in _CATALOGS],
)
def test_catalog_lists_every_accepted_argument(
    build: Callable[[], argparse.ArgumentParser], page: Path
) -> None:
    missing = _accepted(build()) - _catalogued(page) - _catalogued(_SHARED)
    assert missing == set()


@pytest.mark.parametrize(
    ("build", "page"),
    [(build, page) for _name, build, page in _CATALOGS],
    ids=[name for name, _build, _page in _CATALOGS],
)
def test_catalog_lists_only_accepted_arguments(
    build: Callable[[], argparse.ArgumentParser], page: Path
) -> None:
    assert _catalogued(page) - _accepted(build()) == set()


def test_shared_catalog_lists_config_command_arguments() -> None:
    assert _accepted(build_config_parser()) - _catalogued(_SHARED) == set()


def test_shared_catalog_lists_only_accepted_arguments() -> None:
    parsers = [build() for _name, build, _page in _CATALOGS] + [build_config_parser()]
    accepted = set().union(*(_accepted(parser) for parser in parsers))
    assert _catalogued(_SHARED) - accepted == set()
