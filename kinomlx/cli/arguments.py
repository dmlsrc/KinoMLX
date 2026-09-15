"""Compact, model-neutral KinoMLX argument parser."""

import argparse

from kinomlx.model_registry import DEFAULT_MODEL, model_choices

from ._registry import model_summary
from .invocation import add_invocation_arguments


def build_root_parser() -> argparse.ArgumentParser:
    """Build the compact model-neutral discovery surface."""
    parser = argparse.ArgumentParser(
        prog="kinomlx",
        description="Native multimodal MLX inference on Apple Silicon.",
        epilog=(
            "Models:\n"
            f"{model_summary()}\n\n"
            "Utilities:\n"
            "  config init              write an annotated model-specific config\n"
            "  config inspect           inspect registered field bindings and policies\n"
            "  weights convert          generic checkpoint re-serialization\n"
            "  weights convert gmnet    validated GMNet conversion\n\n"
            "Use 'kinomlx --model <name> --help' for model-specific arguments, "
            "'kinomlx config --help' for config utilities, or "
            "'kinomlx weights --help' for conversion help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_invocation_arguments(
        parser,
        choices=model_choices(),
        model_help=f"Model family (default execution model: {DEFAULT_MODEL}).",
    )
    return parser


__all__ = ["build_root_parser"]
