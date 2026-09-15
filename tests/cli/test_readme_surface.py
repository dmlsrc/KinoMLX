"""Public documentation remains linked to executable public surfaces."""

import argparse
import re
from pathlib import Path

from kinomlx.cli.config_init import build_config_parser
from kinomlx.model_registry import config_registry
from kinomlx.models.flux2.arguments import build_parser as build_flux2_parser
from kinomlx.models.gmnet.arguments import build_parser as build_gmnet_parser
from kinomlx.models.gmnet.converter_cli import build_parser as build_gmnet_converter_parser
from kinomlx.models.krea2.arguments import build_parser as build_krea2_parser
from kinomlx.models.ltx2.arguments import build_parser
from kinomlx.weights.cli import build_generic_convert_parser, build_weights_parser

_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_DOCUMENTS = (
    _ROOT / "README.md",
    _ROOT / "docs" / "WEIGHTS.md",
    _ROOT / "docs" / "PERFORMANCE.md",
    _ROOT / "docs" / "ADAPTERS.md",
    _ROOT / "docs" / "LTX2.md",
    _ROOT / "docs" / "GMNET.md",
    _ROOT / "docs" / "KREA2.md",
    _ROOT / "docs" / "FLUX2.md",
)
_LONG_FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")
_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    result = {option for action in parser._actions for option in action.option_strings}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                result.update(_option_strings(child))
    return result


def _documented_flags(readme: str) -> set[str]:
    flags = set(re.findall(r"`(--[a-z0-9][a-z0-9-]*)", readme))
    for block in _FENCED_BLOCK.findall(readme):
        logical_lines = re.sub(r"\\[ \t]*\n[ \t]*", " ", block)
        for line in logical_lines.splitlines():
            command = line.strip()
            if command == "kinomlx" or command.startswith("kinomlx "):
                flags.update(_LONG_FLAG.findall(command))
    return flags


def _public_documentation() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _PUBLIC_DOCUMENTS)


def test_documented_flags_and_environment_names_are_executable() -> None:
    documentation = _public_documentation()
    documented_flags = _documented_flags(documentation)
    documented_environment = set(re.findall(r"`((?:KINO_[A-Z0-9_]+|HF_HOME))`", documentation))
    parsers = (
        build_parser(),
        build_gmnet_parser(),
        build_krea2_parser(),
        build_flux2_parser(),
        build_config_parser(),
        build_weights_parser(),
        build_generic_convert_parser(),
        build_gmnet_converter_parser(),
    )
    executable_flags = set().union(*(_option_strings(parser) for parser in parsers))

    assert documented_flags <= executable_flags
    assert documented_environment <= set(config_registry().environment_variables())
    assert "--prompt" in documented_flags
    assert "--python" not in documented_flags
    assert "--exclude" not in documented_flags
