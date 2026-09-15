"""Shared CLI host services for generic and model-specific converters."""

import argparse

from kinomlx.settings import Settings, add_argparse_args, settings_from_args


def add_conversion_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose infrastructure controls used by conversion."""
    add_argparse_args(parser, skip={"check_adapter_weights"})


def add_checkpoint_safety_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the scan and overwrite decisions every converter has to offer."""
    parser.add_argument(
        "--allow-suspicious-globals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Proceed past static-scan findings for a trusted file; the restricted "
            "unpickler still refuses non-allowlisted globals."
        ),
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace an existing regular output; otherwise refuse before loading.",
    )


def bootstrap_json_output(options: object) -> bool:
    """Select machine output before settings compose, so errors can honor it."""
    try:
        base = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        base = Settings()
    explicit = getattr(options, "json_output", None)
    return base.json_output if explicit is None else bool(explicit)


def resolve_conversion_settings(options: argparse.Namespace) -> Settings:
    """Compose and validate infrastructure settings for a converter command."""
    settings = settings_from_args(options, Settings.from_env())
    settings.validate()
    return settings


__all__ = [
    "resolve_conversion_settings",
]
