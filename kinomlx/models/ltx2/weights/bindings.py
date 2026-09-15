"""Header-only selection of consumed source tensors and FP8 scale companions."""

from collections.abc import Mapping
from pathlib import Path

from kinomlx.io.safetensors import SafetensorsBinding, SafetensorsInventory, read_dtypes

from .keys import (
    DIFFUSION_PREFIX,
    canonical_weight_family_key,
    convert_checkpoint_key,
    weight_family_for_key,
)
from .tensors import fp8_scale_companions

FP8_SCALE_TARGET_PREFIX = "__fp8_weight_scale__."


def family_source_bindings(
    inventory: SafetensorsInventory,
    *,
    family: str,
    source_component: str | None,
) -> tuple[SafetensorsBinding, ...]:
    """Return one family's consumed tensors under their canonical family keys."""
    source = inventory.path
    bindings = tuple(
        SafetensorsBinding(
            source_file=source,
            source_key=tensor.name,
            target_key=canonical_weight_family_key(family, tensor.name),
        )
        for tensor in inventory.tensors
        if weight_family_for_key(tensor.name, source_component=source_component) == family
    )
    if not bindings:
        raise ValueError(f"Checkpoint contains no tensors for requested {family} family")
    return bindings


def transformer_source_bindings(
    source: Path,
    *,
    include_audio: bool,
    source_bindings: Mapping[str, str] | None,
) -> tuple[tuple[SafetensorsBinding, ...], bool]:
    dtypes = read_dtypes(source)
    fp8_keys, weight_scales, input_scales, comfy_tags = fp8_scale_companions(dtypes)
    selected: list[SafetensorsBinding] = []
    source_has_fp8 = False
    for source_key in sorted(dtypes):
        if source_key in weight_scales or source_key in input_scales or source_key in comfy_tags:
            continue
        if source_bindings is None:
            if not source_key.startswith(DIFFUSION_PREFIX):
                continue
            target_key = convert_checkpoint_key(source_key, include_audio=include_audio)
        else:
            target_key = source_bindings.get(source_key)
        if target_key is None:
            continue
        source_has_fp8 = source_has_fp8 or source_key in fp8_keys
        selected.append(
            SafetensorsBinding(
                source_file=source,
                source_key=source_key,
                target_key=target_key,
            )
        )
        scale_key = f"{source_key}_scale"
        if source_key in fp8_keys and scale_key in weight_scales:
            selected.append(
                SafetensorsBinding(
                    source_file=source,
                    source_key=scale_key,
                    target_key=f"{FP8_SCALE_TARGET_PREFIX}{target_key}",
                )
            )
    if not selected:
        raise ValueError(f"Checkpoint {source} contains no compatible transformer tensors")
    return tuple(selected), source_has_fp8
