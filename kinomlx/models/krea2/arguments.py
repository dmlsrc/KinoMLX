"""Complete Krea 2 argument parser."""

import argparse

from kinomlx.cli._registry import validate_model_parser
from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.cli.invocation import add_invocation_arguments, model_config_epilog
from kinomlx.config_record_cli import ConfigArgumentParser
from kinomlx.model_registry import config_registry, model_choices


def build_parser() -> argparse.ArgumentParser:
    """Build the complete Krea 2 parser."""
    config_schema = config_registry().model("krea2")
    help_for = config_schema.cli_help
    parser = ConfigArgumentParser(
        prog="kinomlx",
        description=(
            "Generate Krea 2 Raw or Turbo still images, restart from saved text or "
            "final latents, or prepare resources."
        ),
        epilog=model_config_epilog("krea2"),
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help=help_for("model"),
    )

    add_config_arguments(parser, config_schema)

    validate_model_parser(parser, "krea2")
    return parser


__all__ = ["build_parser"]
