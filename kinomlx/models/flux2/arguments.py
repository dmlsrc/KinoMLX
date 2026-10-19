"""Klein uses the common image CLI with model-specific CFG and preparation controls."""

import argparse

from kinomlx.cli.config_arguments import add_config_arguments
from kinomlx.cli.invocation import add_invocation_arguments, model_config_epilog
from kinomlx.cli.parser_contract import validate_model_parser
from kinomlx.config_record_cli import ConfigArgumentParser
from kinomlx.model_catalog import model_choices

from .config_schema import CONFIG_SCHEMA


def build_parser() -> argparse.ArgumentParser:
    schema = CONFIG_SCHEMA
    parser = ConfigArgumentParser(
        prog="kinomlx",
        allow_abbrev=False,
        description="Generate FLUX.2 Klein base, distilled, or KV images, or repeat a saved stage.",
        epilog=model_config_epilog("flux2"),
    )
    add_invocation_arguments(parser, choices=model_choices(), model_help=schema.cli_help("model"))
    add_config_arguments(parser, schema)

    validate_model_parser(parser, "flux2")
    return parser
