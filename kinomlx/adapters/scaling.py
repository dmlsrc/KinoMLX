"""Report the intrinsic scale an adapter applies beneath user strength."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from kinomlx._json import JsonObject, JsonValue

from .algebra import intrinsic_scale
from .formats import AdapterKind, AdapterOperation

ALPHA_TENSOR_SOURCE = "alpha tensors"
UNKNOWN_SCALE_WARNING = (
    "PEFT export without its adapter_config.json, beside the selected path or a link it "
    "follows; its LoRA scale (lora_alpha / r) is unknown, so strength applies unscaled"
)
# Configuration fields that set a LoRA-family scale, in the order they are reported.
_SCALE_FIELDS = ("lora_alpha", "alpha", "r", "use_rslora", "alpha_pattern", "rank_pattern")


@dataclass(frozen=True)
class ScaleDetermination:
    """Where an adapter's intrinsic scale came from, and the LoRA scales it applies."""

    source: str | None = None
    scales: tuple[float, ...] = ()
    terms: str = ""


def lora_intrinsic_scales(operations: Iterable[AdapterOperation]) -> tuple[float, ...]:
    """Return the distinct scales ordinary LoRA weight operations apply before strength."""
    return tuple(
        sorted(
            {
                intrinsic_scale(operation, int(operation.factors["down"].shape[0]))
                for operation in operations
                if operation.kind == AdapterKind.LORA and operation.part == "weight"
            }
        )
    )


def _term(name: str, value: JsonValue) -> str | None:
    if value is None or value is False or (isinstance(value, dict) and not value):
        return None
    if value is True:
        return name
    if isinstance(value, dict):
        return f"{name} ({len(value)} {'entry' if len(value) == 1 else 'entries'})"
    if isinstance(value, int | float):
        return f"{name}={value:g}"
    return f"{name}={value}"


def determine_scale(
    config_source: str | None, config: JsonObject, operations: Iterable[AdapterOperation]
) -> ScaleDetermination:
    """Name what set an adapter's intrinsic scale, or nothing when none applies.

    A declared configuration always counts, even at unit scale, because it is the
    evidence that the export's own rule was read. Without one, only alpha tensors
    that move some LoRA scale away from one count.
    """
    scales = lora_intrinsic_scales(operations)
    if config_source is not None:
        terms = (_term(name, config[name]) for name in _SCALE_FIELDS if name in config)
        return ScaleDetermination(config_source, scales, ", ".join(t for t in terms if t))
    if any(scale != 1.0 for scale in scales):
        return ScaleDetermination(ALPHA_TENSOR_SOURCE, scales)
    return ScaleDetermination()


def log_scale_determination(
    logger: logging.Logger,
    *,
    model: str,
    name: str,
    determination: ScaleDetermination,
    strength: float,
) -> None:
    """State the intrinsic scale applied beneath user strength, when one was applied."""
    if determination.source is None:
        return
    origin = determination.source + (f" ({determination.terms})" if determination.terms else "")
    scales = determination.scales
    if not scales:
        logger.info("%s adapter %s: scale from %s", model, name, origin)
    elif len(scales) == 1:
        logger.info(
            "%s adapter %s: scale %g from %s; effective %g at strength %g",
            model,
            name,
            scales[0],
            origin,
            strength * scales[0],
            strength,
        )
    else:
        logger.info(
            "%s adapter %s: scales %g to %g from %s; effective %g to %g at strength %g",
            model,
            name,
            scales[0],
            scales[-1],
            origin,
            strength * scales[0],
            strength * scales[-1],
            strength,
        )


__all__ = [
    "ALPHA_TENSOR_SOURCE",
    "UNKNOWN_SCALE_WARNING",
    "ScaleDetermination",
    "determine_scale",
    "log_scale_determination",
    "lora_intrinsic_scales",
]
