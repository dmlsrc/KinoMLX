"""Complete GMNet argument parser."""

import argparse

from kinomlx.cli._registry import validate_model_parser
from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.cli.invocation import add_invocation_arguments, model_config_epilog
from kinomlx.config_record_cli import ConfigArgumentParser
from kinomlx.model_registry import config_registry, model_choices


def build_parser() -> argparse.ArgumentParser:
    """Build the complete GMNet parser."""
    config_schema = config_registry().model("gmnet")
    help_for = config_schema.cli_help
    parser = ConfigArgumentParser(
        prog="kinomlx",
        description="GMNet SDR-to-HDR still expansion (--model gmnet).",
        epilog=model_config_epilog("gmnet"),
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help=help_for("model"),
    )

    add_config_arguments(parser, config_schema)
    validate_model_parser(parser, "gmnet")
    return parser


__all__ = ["build_parser"]
