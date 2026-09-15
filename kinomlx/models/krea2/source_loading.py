"""Bind Krea component sources using the shared preparation policy."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import inspect_safetensors
from kinomlx.modeling.loading import WeightBinding, WeightTarget, bind_weight_targets
from kinomlx.reporting import Reporter
from kinomlx.tokenizers.byte_bpe import QwenByteBPETokenizer

from .preparation import prepare_weight
from .resource_types import Krea2ComponentResource, Krea2TensorBinding

SOURCE_DTYPES = frozenset({"BF16", "F16", "F32"})


def load_component_tokenizer(component: Krea2ComponentResource) -> QwenByteBPETokenizer:
    """Construct a tokenizer from resource-owned native tables."""
    return QwenByteBPETokenizer(component.tokenizer_tables)


def bind_source_weights(
    targets: Mapping[str, WeightTarget],
    bindings: tuple[Krea2TensorBinding, ...],
    *,
    dtype: mx.Dtype | None,
    phase: str,
    reporter: Reporter | None,
) -> tuple[int, tuple[Path, ...]]:
    """Preflight source bindings and attach dtype and layout transforms.

    A None dtype preserves each source tensor's storage dtype.
    """
    mapping = {binding.target_key: binding for binding in bindings}
    if len(mapping) != len(bindings) or not targets.keys() <= mapping.keys():
        raise ValueError("Krea source bindings do not cover the target graph")
    selected = {key: mapping[key] for key in targets}
    files = tuple(dict.fromkeys(binding.source_file for binding in selected.values()))
    headers = {file: {t.name: t for t in inspect_safetensors(file).tensors} for file in files}
    source_targets = {}
    for key, target in targets.items():
        binding = selected[key]
        tensor = headers[binding.source_file].get(binding.source_key)
        if tensor is None:
            raise ValueError(
                f"Krea source {binding.source_file} has no tensor {binding.source_key!r}"
            )
        if tensor.dtype not in SOURCE_DTYPES:
            raise ValueError(
                f"Unsupported Krea source dtype {tensor.dtype} for {binding.source_key!r}"
            )
        source_targets[key] = replace(target, shape=tensor.shape)

    def prepare(binding: WeightBinding, value: mx.array) -> mx.array:
        converted = prepare_weight(value, selected[binding.logical_key], dtype=dtype)
        expected = targets[binding.logical_key].shape
        if converted.shape != expected:
            raise ValueError(
                f"Krea tensor {binding.logical_key} has prepared shape {converted.shape}, "
                f"expected {expected}"
            )
        return converted

    count = bind_weight_targets(
        files,
        source_targets,
        lambda key: (selected[key].source_key,),
        component="Krea 2",
        phase=phase,
        reporter=reporter,
        prepare=prepare,
    )
    return count, files


__all__ = [
    "SOURCE_DTYPES",
    "bind_source_weights",
    "load_component_tokenizer",
]
