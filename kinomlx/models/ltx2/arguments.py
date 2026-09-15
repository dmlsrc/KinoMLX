"""Complete, abbreviation-free LTX-2 argument parser."""

import argparse

from kinomlx.cli._registry import validate_model_parser
from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.cli.invocation import add_invocation_arguments, model_config_epilog
from kinomlx.config_record_cli import ConfigArgumentParser
from kinomlx.model_registry import config_registry, model_choices


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser without importing a model runtime."""
    config_schema = config_registry().model("ltx2")
    parser = ConfigArgumentParser(
        prog="kinomlx",
        description="Generate distilled LTX-2 video on Apple Silicon (--model ltx2).",
        epilog=model_config_epilog("ltx2"),
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help=config_schema.cli_help("model"),
    )
    add_config_arguments(parser, config_schema)
    validate_model_parser(parser, "ltx2")
    return parser


__all__ = ["build_parser"]
