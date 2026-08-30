"""GMNet-specific extension of the generic weights-conversion command."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path

from kinomlx.cli.common import format_elapsed, render_error
from kinomlx.errors import KinoMLXError
from kinomlx.settings import Settings, add_argparse_args
from kinomlx.weights.host import (
    apply_mlx_cache_limit,
    conversion_reporter,
    resolve_conversion_settings,
)

_log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the GMNet converter parser without importing MLX."""
    parser = argparse.ArgumentParser(
        prog="kinomlx weights convert gmnet",
        description=(
            "Convert and validate an upstream GMNet generator checkpoint. This "
            "model-specific path checks every key, identifies the numeric variant, "
            "records provenance metadata, and verifies the result with the GMNet loader."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="GMNet .pth checkpoint; bare names also resolve under weights-src/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Exact .safetensors output (default: model weights in an editable "
            "checkout, otherwise the configured KinoMLX cache)."
        ),
    )
    parser.add_argument(
        "--declare-variant",
        choices=("realworld", "synthetic"),
        default=None,
        help="Declare the numeric contract of a non-published checkpoint.",
    )
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
    add_argparse_args(parser)
    return parser


def _bootstrap_json(options: object) -> bool:
    try:
        base = Settings.from_env_fields("json_output")
    except TypeError, ValueError:
        base = Settings()
    explicit = getattr(options, "json_output", None)
    return base.json_output if explicit is None else bool(explicit)


def run_gmnet_convert_command(argv: list[str]) -> int:
    """Run the GMNet converter and return a process exit code."""
    from kinomlx.ui import configure_logging, configure_logging_from_settings

    configure_logging()
    options = build_parser().parse_args(argv)
    json_output = _bootstrap_json(options)
    try:
        settings = resolve_conversion_settings(options)
    except (TypeError, ValueError) as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    configure_logging_from_settings(settings)

    started = time.perf_counter()
    try:
        apply_mlx_cache_limit(settings)
        from kinomlx.models.gmnet.catalog import GMNetVariant
        from kinomlx.models.gmnet.convert import convert_checkpoint

        with ExitStack() as stack:
            reporter = conversion_reporter(stack, settings)
            reporter.memory_checkpoint("conversion_start")
            receipt = convert_checkpoint(
                options.input,
                options.output,
                declared_variant=(
                    None
                    if options.declare_variant is None
                    else GMNetVariant(options.declare_variant)
                ),
                allow_suspicious=options.allow_suspicious_globals,
                force=options.force,
                cache_dir=settings.cache_dir,
                reporter=reporter,
            )
            reporter.memory_checkpoint("conversion_complete")
            diagnostics = reporter.memory_to_dict()
            timings = reporter.to_dict()
    except (KinoMLXError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return render_error(
            f"GMNet weights conversion failed: {exc}",
            json_output=settings.json_output,
        )

    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "status": "ok",
        "action": "weights.convert",
        "converter": "gmnet",
        "output": str(receipt.output),
        "source": str(receipt.source),
        "variant": None if receipt.variant is None else receipt.variant.value,
        "source_sha256": receipt.source_sha256,
        "tensor_count": receipt.tensor_count,
        "parameter_count": receipt.parameter_count,
        "flagged_globals": list(receipt.flagged_globals),
        "elapsed_seconds": elapsed,
        "timings": timings,
        "diagnostics": {"memory": diagnostics} if diagnostics else {},
    }
    if settings.json_output:
        from kinomlx.cli.output import emit_json

        emit_json(payload)
    else:
        _log.info("Source: %s (sha256 %s...)", receipt.source, receipt.source_sha256[:12])
        _log.info("Output: %s", receipt.output)
        _log.info(
            "Variant: %s",
            "unknown" if receipt.variant is None else receipt.variant.value,
        )
        _log.info(
            "Converted %s GMNet tensors (%.3fM parameters)",
            receipt.tensor_count,
            receipt.parameter_count / 1e6,
        )
        if receipt.flagged_globals:
            _log.warning(
                "Static scan findings explicitly allowed: %s",
                list(receipt.flagged_globals),
            )
        _log.info("Total runtime: %s", format_elapsed(elapsed))
    return 0


__all__ = ["build_parser", "run_gmnet_convert_command"]
