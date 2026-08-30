"""Generic checkpoint-conversion command and model-specific dispatch."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path

from kinomlx.cli.common import format_elapsed, render_error
from kinomlx.errors import KinoMLXError
from kinomlx.settings import Settings, add_argparse_args

from .host import apply_mlx_cache_limit, conversion_reporter, resolve_conversion_settings

_log = logging.getLogger(__name__)


def build_weights_parser() -> argparse.ArgumentParser:
    """Build the model-neutral discovery parser without importing MLX."""
    parser = argparse.ArgumentParser(
        prog="kinomlx weights",
        description="Safely convert tensor-only PyTorch checkpoints to safetensors.",
        epilog=(
            "Converters:\n"
            "  convert INPUT          generic value/layout-preserving conversion\n"
            "  convert gmnet INPUT    GMNet keys, variant metadata, and loader validation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("action", choices=("convert",), help="Weights action.")
    return parser


def build_generic_convert_parser() -> argparse.ArgumentParser:
    """Build the generic conversion parser without importing MLX."""
    parser = argparse.ArgumentParser(
        prog="kinomlx weights convert",
        description=(
            "Re-serialize a plain tensor state dict without changing tensor values "
            "or layouts. Use a model-specific converter when one is available."
        ),
        epilog=(
            "For GMNet use 'kinomlx weights convert gmnet INPUT'; its converter "
            "validates the exact generator contract and artifact metadata."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("input", type=Path, help="Zip-format .pth or .pt checkpoint.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .safetensors path (default: replace the input suffix).",
    )
    parser.add_argument(
        "--param-key",
        default=None,
        help=(
            "Nested checkpoint mapping to extract. Required when inference intent "
            "is ambiguous, such as checkpoints with both params and params_ema."
        ),
    )
    parser.add_argument(
        "--strip-prefix",
        default="module.",
        help="Key prefix to strip when present (default: module.; pass '' to keep).",
    )
    parser.add_argument(
        "--only-prefix",
        default="",
        help="Keep only tensor keys beginning with this prefix.",
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


def run_generic_convert(argv: list[str]) -> int:
    """Run the generic converter and return a process exit code."""
    from kinomlx.ui import configure_logging, configure_logging_from_settings

    configure_logging()
    options = build_generic_convert_parser().parse_args(argv)
    json_output = _bootstrap_json(options)
    try:
        settings = resolve_conversion_settings(options)
    except (TypeError, ValueError) as exc:
        return render_error(f"config error: {exc}", json_output=json_output)
    configure_logging_from_settings(settings)

    started = time.perf_counter()
    try:
        apply_mlx_cache_limit(settings)
        from kinomlx.weights.convert import convert_checkpoint

        with ExitStack() as stack:
            reporter = conversion_reporter(stack, settings)
            reporter.memory_checkpoint("conversion_start")
            receipt = convert_checkpoint(
                options.input,
                options.output,
                param_key=options.param_key,
                strip_prefix=options.strip_prefix,
                only_prefix=options.only_prefix,
                allow_suspicious=options.allow_suspicious_globals,
                force=options.force,
                reporter=reporter,
            )
            reporter.memory_checkpoint("conversion_complete")
            diagnostics = reporter.memory_to_dict()
            timings = reporter.to_dict()
    except (KinoMLXError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return render_error(
            f"weights conversion failed: {exc}",
            json_output=settings.json_output,
        )

    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "status": "ok",
        "action": "weights.convert",
        "converter": "generic",
        "output": str(receipt.output),
        "source": str(receipt.source),
        "source_sha256": receipt.source_sha256,
        "tensor_count": receipt.tensor_count,
        "parameter_count": receipt.parameter_count,
        "stripped_keys": receipt.stripped_keys,
        "filtered_keys": receipt.filtered_keys,
        "dropped_entries": list(receipt.dropped_entries),
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
            "Converted %s tensors (%.3fM parameters)",
            receipt.tensor_count,
            receipt.parameter_count / 1e6,
        )
        if receipt.dropped_entries:
            _log.warning("Dropped non-tensor entries: %s", list(receipt.dropped_entries))
        if receipt.flagged_globals:
            _log.warning(
                "Static scan findings explicitly allowed: %s",
                list(receipt.flagged_globals),
            )
        _log.info("Total runtime: %s", format_elapsed(elapsed))
    return 0


def run_weights_command(argv: list[str]) -> int:
    """Dispatch generic and model-specific weights commands."""
    parser = build_weights_parser()
    if not argv:
        parser.print_help()
        return 2
    if argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if argv[0] != "convert":
        parser.parse_args(argv)
        return 2
    convert_arguments = argv[1:]
    if convert_arguments and convert_arguments[0] == "gmnet":
        from kinomlx.models.gmnet.converter_cli import run_gmnet_convert_command

        return run_gmnet_convert_command(convert_arguments[1:])
    return run_generic_convert(convert_arguments)


__all__ = [
    "build_generic_convert_parser",
    "build_weights_parser",
    "run_generic_convert",
    "run_weights_command",
]
