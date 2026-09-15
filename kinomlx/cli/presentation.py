"""Model-neutral human presentation and typed error rendering."""

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence

from rich.cells import cell_len

from kinomlx.adapters.selection import AdapterSelection
from kinomlx.errors import log_debug_exception
from kinomlx.model_registry import MODEL_SPECS
from kinomlx.ui.logging import console_message_width

from .machine_output import emit_json

_log = logging.getLogger(__name__)


def format_elapsed(seconds: float) -> str:
    """Format one non-negative wall-clock duration for human CLI output."""
    rounded_tenths = int(max(0.0, seconds) * 10.0 + 0.5)
    hours, remainder = divmod(rounded_tenths, 36_000)
    minutes, second_tenths = divmod(remainder, 600)
    rendered_seconds = second_tenths / 10.0
    if hours:
        return f"{hours}h {minutes}m {rendered_seconds:.1f}s"
    if minutes:
        return f"{minutes}m {rendered_seconds:.1f}s"
    return f"{rendered_seconds:.1f}s"


def model_display_name(model: str) -> str:
    """Return the canonical user-facing name for one registered model."""
    spec = MODEL_SPECS.get(model)
    return model if spec is None else spec.display_name


_SECTION_INDENT = "  "
_FIELD_SEPARATOR = " | "


def pack_fields(
    fields: Iterable[tuple[str, object]],
    *,
    width: int,
) -> tuple[str, ...]:
    """Pack ``label: value`` items left to right into lines of at most ``width`` cells.

    Items keep their order and are never split or shortened: an item wider than
    a line stands alone, and a value with line breaks keeps its own line.
    """
    lines: list[str] = []
    current = ""
    for label, value in fields:
        item = f"{label}: {value}"
        if "\n" in item:
            if current:
                lines.append(current)
                current = ""
            lines.append(item)
        elif not current:
            current = item
        elif cell_len(current) + len(_FIELD_SEPARATOR) + cell_len(item) <= width:
            current = f"{current}{_FIELD_SEPARATOR}{item}"
        else:
            lines.append(current)
            current = item
    if current:
        lines.append(current)
    return tuple(lines)


def log_section(
    logger: logging.Logger,
    heading: str,
    fields: Iterable[tuple[str, object]],
) -> None:
    """Log one section: its heading, then its fields packed to the console width."""
    values = tuple(fields)
    if not values:
        return
    logger.info("%s:", heading)
    width = console_message_width() - len(_SECTION_INDENT)
    for line in pack_fields(values, width=width):
        logger.info("%s%s", _SECTION_INDENT, line)


def log_run_header(
    logger: logging.Logger,
    *,
    model: str,
    action: str,
    resolution: str,
    prompts: int,
    images: int,
    seed: int,
    extra: Iterable[tuple[str, object]] = (),
) -> None:
    """Present common generation identity in the same order for image hosts."""
    log_section(
        logger,
        "Run",
        (
            ("Model", model),
            ("Action", action),
            ("Resolution", resolution),
            ("Prompts", prompts),
            ("Images", images),
            ("Seed", seed),
            *extra,
        ),
    )


def log_prompt_blocks(
    logger: logging.Logger,
    prompts: Iterable[str],
    *,
    label: str = "Prompt",
) -> None:
    """Log copyable prompt bodies below a single or numbered label."""
    values = tuple(prompts)
    for index, prompt in enumerate(values, start=1):
        heading = label if len(values) == 1 else f"{label} {index}"
        logger.info("%s:\n%s", heading, prompt or "(empty user prompt)")


def log_adapter_selections(
    logger: logging.Logger,
    selections: Iterable[AdapterSelection],
    *,
    strength_details: Callable[[AdapterSelection], Iterable[str]] | None = None,
) -> None:
    """Log ordered adapter paths and every effective user control.

    ``strength_details`` renders the strength fragments of one selection for a
    model that resolves more than the common scalar; the default names that
    scalar, which is then what every step applies.
    """
    values = tuple(selections)
    fields: list[tuple[str, object]] = []
    for index, selection in enumerate(values, start=1):
        strengths = (
            (f"strength={selection.strength:g}",)
            if strength_details is None
            else tuple(strength_details(selection))
        )
        parts = [str(selection.path), *strengths]
        parts.append(f"exclude={','.join(selection.exclude) or 'none'}")
        label = "Adapter" if len(values) == 1 else f"Adapter {index}"
        fields.append((label, "; ".join(parts)))
    log_section(logger, "Adapters", fields)


def _output_fields(outputs: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    fields: list[tuple[str, object]] = []
    for name, value in outputs.items():
        label = str(name).replace("_", " ").upper()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            fields.extend((f"{label} {index}", path) for index, path in enumerate(value, start=1))
        elif value is not None:
            fields.append((label, value))
    return tuple(fields)


def _same_output_fields(planned: Mapping[str, object], actual: Mapping[str, object]) -> bool:
    """Return True when the completed outputs name exactly the planned files, in any order."""

    def rendered(outputs: Mapping[str, object]) -> list[tuple[str, str]]:
        return sorted((label, str(value)) for label, value in _output_fields(outputs))

    return rendered(planned) == rendered(actual)


def log_output_paths(
    logger: logging.Logger,
    outputs: Mapping[str, object],
    *,
    heading: str = "Outputs",
) -> None:
    """Log one output mapping without model-specific footer formatting."""
    log_section(logger, heading, _output_fields(outputs))


def model_failure_message(model: str, action: str, error: object) -> str:
    """Apply one model/action failure prefix without retaining an older action prefix."""
    display_name = model_display_name(model)
    prefix = f"{display_name} {action} failed"
    detail = str(error)
    if detail.casefold().startswith(f"{prefix}:".casefold()):
        return detail
    detail = re.sub(
        rf"^{re.escape(display_name)} [^:\n]+ failed:\s*",
        "",
        detail,
        count=1,
        flags=re.IGNORECASE,
    )
    return f"{prefix}: {detail}"


def render_model_error(
    model: str,
    action: str,
    error: object,
    *,
    json_output: bool,
) -> int:
    """Render one consistently named model execution failure."""
    if isinstance(error, BaseException):
        log_debug_exception(
            _log,
            f"Unhandled {model_display_name(model)} {action} failure",
            error,
        )
    return render_error(
        model_failure_message(model, action, error),
        json_output=json_output,
    )


def render_run_success(
    logger: logging.Logger,
    *,
    model: str,
    action: str,
    outputs: Mapping[str, object],
    elapsed_seconds: float,
    timings: Mapping[str, object],
    diagnostics: Mapping[str, object],
    json_output: bool,
    generation: Mapping[str, object] | None = None,
    warnings: Sequence[str] = (),
    sidecar_errors: Sequence[Mapping[str, str]] = (),
    extra: Mapping[str, object] | None = None,
    planned_outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Render and return the common successful-run envelope used by every model.

    When ``planned_outputs`` names exactly the completed outputs, the console
    says so in one line instead of repeating the planned-outputs block.
    """
    payload: dict[str, object] = {
        "status": "ok",
        "model": model,
        "action": action,
        "outputs": dict(outputs),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "timings": dict(timings),
        "diagnostics": dict(diagnostics),
        "warnings": list(warnings),
        "sidecar_errors": [dict(error) for error in sidecar_errors],
    }
    if generation is not None:
        payload["generation"] = dict(generation)
    if extra:
        overlap = payload.keys() & extra.keys()
        if overlap:
            raise ValueError(f"run result repeats common fields: {', '.join(sorted(overlap))}")
        payload.update(extra)
    if planned_outputs is not None and _same_output_fields(planned_outputs, outputs):
        logger.info("Outputs: as planned")
    else:
        log_output_paths(logger, outputs)
    logger.info("Total runtime: %s", format_elapsed(elapsed_seconds))
    if json_output:
        emit_json(payload)
    return payload


def render_error(message: str, *, json_output: bool) -> int:
    """Render one expected failure through the selected human or JSON surface."""
    if json_output:
        emit_json({"status": "error", "error": message})
    else:
        _log.error("%s", message)
    return 2


def render_canceled(
    *,
    json_output: bool,
    outputs: Mapping[str, object] | None = None,
) -> int:
    """Render an interrupted command, naming whatever it finished first."""
    payload: dict[str, object] = {"status": "canceled"}
    if outputs:
        payload["outputs"] = dict(outputs)
    if json_output:
        emit_json(payload)
    else:
        _log.warning("Canceled.")
        if outputs:
            log_output_paths(_log, outputs)
    return 130


__all__ = [
    "format_elapsed",
    "log_adapter_selections",
    "log_output_paths",
    "log_prompt_blocks",
    "log_section",
    "model_display_name",
    "model_failure_message",
    "pack_fields",
    "render_canceled",
    "render_error",
    "render_model_error",
    "render_run_success",
]
