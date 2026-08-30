"""Exact validation for KinoMLX-owned transformer cache parameter graphs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

import mlx.core as mx

from .schema import LAYOUT_KEY_PREFIX, QUANT_KEY_PREFIX


class _ExpectedShapes(Protocol):
    def __call__(self, *, include_audio: bool) -> dict[str, tuple[int, ...]]: ...


def _is_audio_parameter(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in ("audio", "av_ca", "a2v", "v2a"))


def _graph_error(*details: str) -> ValueError:
    return ValueError("Transformer cache parameter graph mismatch: " + "; ".join(details))


def validate_model_cache_graph(
    model: object,
    cached_weights: Mapping[str, mx.array],
    *,
    include_audio: bool | None = None,
    require_graph: bool = True,
) -> None:
    """Reject an internal cache that cannot bind the selected model exactly.

    Raw community checkpoints are intentionally not judged by this contract.
    It applies only after conversion, where KinoMLX owns every logical key.
    Models must expose the graph-introspection surface unless an intentionally
    partial unit fixture opts out explicitly.
    """
    expected_shapes_method = getattr(model, "expected_parameter_shapes", None)
    if not callable(expected_shapes_method):
        if require_graph:
            raise TypeError(
                "Transformer cache graph validation requires a callable "
                "model.expected_parameter_shapes()"
            )
        return

    logical_keys = []
    for key in cached_weights:
        if key.startswith(LAYOUT_KEY_PREFIX):
            logical_keys.append(key[len(LAYOUT_KEY_PREFIX) :])
        elif key.startswith(QUANT_KEY_PREFIX):
            logical_keys.append(key[len(QUANT_KEY_PREFIX) :])
        else:
            logical_keys.append(key)
    selected_audio = (
        any(_is_audio_parameter(key) for key in logical_keys)
        if include_audio is None
        else include_audio
    )
    expected = cast(_ExpectedShapes, expected_shapes_method)(include_audio=selected_audio)

    represented: dict[str, tuple[str, tuple[int, ...] | None]] = {}
    quant_parameters: dict[str, set[str]] = {}

    def record(
        target: str,
        representation: str,
        shape: tuple[int, ...] | None,
    ) -> None:
        previous = represented.get(target)
        if previous is not None:
            raise _graph_error(
                f"target {target} is represented by both {previous[0]} and {representation}"
            )
        represented[target] = (representation, shape)

    for key, value in cached_weights.items():
        if key.startswith(LAYOUT_KEY_PREFIX):
            logical = key[len(LAYOUT_KEY_PREFIX) :]
            if not logical.endswith(".weight_t"):
                raise _graph_error(f"unsupported layout tensor {logical}")
            target = logical[: -len(".weight_t")] + ".weight"
            record(target, "layout", tuple(reversed(value.shape)))
            continue
        if key.startswith(QUANT_KEY_PREFIX):
            logical = key[len(QUANT_KEY_PREFIX) :]
            base, separator, parameter = logical.rpartition(".")
            if not separator or parameter not in {"weight", "scales", "biases"}:
                raise _graph_error(f"unsupported quantized tensor {logical}")
            quant_parameters.setdefault(base, set()).add(parameter)
            continue
        record(key, "normal", tuple(value.shape))

    for base, parameters in quant_parameters.items():
        missing_parts = sorted({"weight", "scales"} - parameters)
        if missing_parts:
            raise _graph_error(f"quantized target {base} is missing {', '.join(missing_parts)}")
        record(f"{base}.weight", "quantized", None)

    actual = set(represented)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual)
    extra = sorted(actual - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing[:3]}")
        if extra:
            details.append(f"unexpected {extra[:3]}")
        raise _graph_error(*details)

    for target, (_representation, shape) in represented.items():
        if shape is not None and shape != expected[target]:
            raise _graph_error(f"target {target} has shape {shape}, expected {expected[target]}")


__all__ = ["validate_model_cache_graph"]
