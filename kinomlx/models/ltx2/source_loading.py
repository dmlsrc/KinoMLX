"""Reloadable lazy transformer source preparation for full and streamed execution."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from kinomlx.adapters.fusion import (
    WeightedAdapterOperation,
    fuse_weight,
    half_accumulation_fallbacks,
)
from kinomlx.adapters.lora import LoRAProfile
from kinomlx.io.safetensors import load_weights
from kinomlx.modeling.conversion import SourceTensor
from kinomlx.models.ltx2.weights.policy import normalize_transformer_layout_policy
from kinomlx.models.ltx2.weights.schema import TRANSFORMER_QUANTIZE_OFF
from kinomlx.models.ltx2.weights.tensors import quant_mode_for_key

from .adapters.loading import community_adapter_operations
from .metadata import resolve_transformer_bindings, transformer_parameter_shapes
from .preparation import family_source_preparation, prepare_transformer_source
from .resources import ComponentKind, LTX2Resources
from .weights.residency import retains_tensor


@dataclass(frozen=True)
class TransformerSourceLoader:
    """A reloadable source provider and logical adapter targets prepared before binding."""

    load: Callable[[int | None], dict[str, mx.array]]
    prepared_targets: dict[str, tuple[int, ...]]
    streamed_targets: frozenset[str] = frozenset()


def transformer_source_loader(
    resources: LTX2Resources,
    *,
    adapters: LoRAProfile = (),
) -> TransformerSourceLoader:
    """Prepare lazy weights with source-owned fusion for packed and streamed targets.

    Quantized targets cannot take adapters after packing, so their adapters are
    applied to the logical source weight first with the runtime fuser's exact
    arithmetic. Streamed block weights use the same source-owned fusion on each
    reload; other dense targets keep the live fusion pass.
    """
    source = resources.require(ComponentKind.TRANSFORMER).source_path
    config = resources.transformer_config
    if config is None:
        raise ValueError("Transformer source requires a constructor config")
    policy = resources.weight_policy
    dtype = resources.dtype_policy.transformer
    layout = normalize_transformer_layout_policy(
        include_audio=policy.include_audio,
        transformer_dtype=dtype,
        video_ff_dtype=None,
        audio_ff_dtype=None,
        video_ff_layout_specs=policy.video_ff_layout_specs,
        video_ff_layout_layers=policy.video_ff_layout_layers,
        video_attn_layout_specs=policy.video_attn_layout_specs,
        video_attn_layout_layers=policy.video_attn_layout_layers,
        audio_ff_layout_specs=policy.audio_ff_layout_specs,
        audio_ff_layout_layers=policy.audio_ff_layout_layers,
        audio_attn_layout_specs=policy.audio_attn_layout_specs,
        audio_attn_layout_layers=policy.audio_attn_layout_layers,
        video_ff_quantize_specs=policy.video_ff_quantize_specs,
        video_ff_quantize_layers=policy.video_ff_quantize_layers,
        layouts_enabled=policy.transformer_quantize == TRANSFORMER_QUANTIZE_OFF,
    )
    resolved_bindings = resolve_transformer_bindings(
        source, config, include_audio=policy.include_audio
    )
    bindings = {b.source_key: b.target_key for b in resolved_bindings}
    source_dtypes = {b.target_key: b.dtype for b in resolved_bindings} if adapters else {}
    half_targets: set[str] = set()
    dense_block_targets: set[str] = set()
    source_shapes: dict[str, tuple[int, ...]] = {}
    if adapters:
        expected = transformer_parameter_shapes(config, include_audio=policy.include_audio)
        for target_key in bindings.values():
            mode = quant_mode_for_key(
                target_key,
                transformer_quantize=policy.transformer_quantize,
                transformer_quantize_exclude_layers=policy.transformer_quantize_exclude_layers,
                video_ff_quantize_specs=policy.video_ff_quantize_specs,
                video_ff_quantize_layers=layout.video_ff_quantize_layers,
            )
            if mode is None and (
                source_dtypes[target_key] == "F16"
                or (dtype == mx.float16 and source_dtypes[target_key] != "F32")
            ):
                half_targets.add(target_key)
            streamed = policy.resident_blocks is not None and target_key.startswith(
                "transformer_blocks."
            )
            if mode is None and streamed and target_key.endswith(".weight"):
                dense_block_targets.add(target_key.removesuffix(".weight"))
            if (mode is not None or streamed) and target_key.endswith(".weight"):
                source_shapes[target_key[: -len(".weight")]] = tuple(expected[target_key])
    operations: dict[str, tuple[WeightedAdapterOperation, ...]] = community_adapter_operations(
        adapters, target_shapes=source_shapes, include_audio=policy.include_audio
    )
    prepared_targets = {
        base: shape for base, shape in source_shapes.items() if f"{base}.weight" in operations
    }

    def adjust_quantized(mlx_key: str, value: mx.array) -> mx.array:
        selected = operations.get(mlx_key)
        return value if selected is None else fuse_weight(value, selected)

    accumulation = policy.adapter_accumulation
    fallbacks = (
        half_accumulation_fallbacks(
            {key: group for key, group in operations.items() if key in half_targets}
        )
        if accumulation == "float16"
        else frozenset()
    )

    def adjust_dense(mlx_key: str, value: mx.array) -> mx.array:
        selected = operations.get(mlx_key)
        if selected is None:
            return value
        return fuse_weight(
            value, selected, accumulation="float32" if mlx_key in fallbacks else accumulation
        )

    preparation = prepare_transformer_source(
        source,
        source_bindings=bindings,
        transformer_dtype=dtype,
        include_audio=policy.include_audio,
        video_ff_layout_specs=layout.video_ff_specs,
        video_ff_layout_layers=layout.video_ff_layers,
        video_attn_layout_specs=layout.video_attn_specs,
        video_attn_layout_layers=layout.video_attn_layers,
        audio_ff_layout_specs=layout.audio_ff_specs,
        audio_ff_layout_layers=layout.audio_ff_layers,
        audio_attn_layout_specs=layout.audio_attn_specs,
        audio_attn_layout_layers=layout.audio_attn_layers,
        adaln_pretranspose=policy.adaln_pretranspose,
        transformer_quantize=policy.transformer_quantize,
        transformer_quantize_exclude_layers=policy.transformer_quantize_exclude_layers,
        video_ff_quantize_specs=policy.video_ff_quantize_specs,
        video_ff_quantize_layers=layout.video_ff_quantize_layers,
        video_ff_quantize_group_size=policy.video_ff_quantize_group_size,
        video_ff_quantize_bits=policy.video_ff_quantize_bits,
        adjust_quantized=adjust_quantized if operations else None,
        adjust_dense=adjust_dense if operations and policy.resident_blocks is not None else None,
    )

    selected = tuple(item for item in preparation.sources if item.convert)
    shapes = (
        {b.target_key: tuple(b.shape) for b in resolved_bindings}
        if policy.transformer_retain
        else {}
    )
    retained_sources = {
        item.target_key
        for item in selected
        if policy.transformer_retain
        and retains_tensor(item.target_key, shapes[item.target_key], policy.transformer_retain)
    }
    for selector in policy.transformer_retain:
        if not any(retains_tensor(key, shape, (selector,)) for key, shape in shapes.items()):
            raise ValueError(f"transformer_retain selector matches no active tensors: {selector!r}")
    retained: dict[int, dict[str, mx.array]] = {}
    pending_retained = {
        item.target_key: item for item in selected if item.target_key in retained_sources
    }
    block_sources: dict[int, list[SourceTensor]] = {}
    for item in selected:
        if item.target_key.startswith("transformer_blocks."):
            index = int(item.target_key.split(".", 2)[1])
            block_sources.setdefault(index, [])
            if item.target_key not in retained_sources:
                block_sources[index].append(item)
    window_size = policy.resident_blocks or 1
    windows = {
        start: tuple(
            item
            for index in range(start, start + window_size)
            for item in block_sources.get(index, ())
        )
        for start in range(0, len(block_sources), window_size)
    }

    def load(block_index: int | None = None) -> dict[str, mx.array]:
        if block_index is None:
            entries = tuple(
                item
                for item in selected
                if item.target_key not in retained_sources or item.target_key in pending_retained
            )
        else:
            start = block_index // window_size * window_size
            entries = windows[start] + tuple(
                item
                for item in pending_retained.values()
                if start <= int(item.target_key.split(".", 2)[1]) < start + window_size
            )
        raw = load_weights(source) if entries else {}
        weights: dict[str, mx.array] = {}
        for item in entries:
            converted = dict(preparation.convert(item, raw.pop(item.source_key)).tensors)
            weights.update(converted)
            if item.target_key in retained_sources:
                index = int(item.target_key.split(".", 2)[1])
                retained.setdefault(index, {}).update(converted)
                # The lazy fused graph owns factors until evaluation; subsequent
                # reloads reuse the result, so the operation no longer needs them.
                operations.pop(item.target_key, None)
                pending_retained.pop(item.target_key, None)
        if block_index is None:
            for tensors in retained.values():
                weights.update(tensors)
        else:
            start = block_index // window_size * window_size
            for index in range(start, start + window_size):
                weights.update(retained.get(index, {}))
        return weights

    return TransformerSourceLoader(
        load,
        prepared_targets,
        frozenset(prepared_targets.keys() & dense_block_targets),
    )


_FAMILY_SOURCES: dict[ComponentKind, tuple[str, str]] = {
    ComponentKind.VIDEO_VAE: ("video_vae", "video_vae"),
    ComponentKind.AUDIO_VAE: ("audio_vae", "audio_vae_vocoder"),
    ComponentKind.VOCODER: ("vocoder", "audio_vae_vocoder"),
}


def family_source_weights(
    resources: LTX2Resources,
    kind: ComponentKind,
    *,
    select: Callable[[str], bool] | None = None,
) -> dict[str, mx.array]:
    """Select and prepare one weight family lazily from its original source file.

    Only the family's consumed tensors are converted, under the canonical keys
    and native execution layouts; every other entry of a possibly
    monolithic checkpoint is dropped without evaluating its payload.
    """
    try:
        family, source_component = _FAMILY_SOURCES[kind]
    except KeyError:
        raise ValueError(f"{kind.value} has no auxiliary weight family to load") from None
    source = resources.require(kind).source_path
    preparation = family_source_preparation(
        source, family=family, source_component=source_component, select=select
    )
    raw = load_weights(source)
    weights: dict[str, mx.array] = {}
    try:
        for item in preparation.sources:
            value = raw.pop(item.source_key, None)
            if value is None:
                raise ValueError(f"{source}: source tensor {item.source_key!r} disappeared")
            weights.update(preparation.convert(item, value).tensors)
    finally:
        raw.clear()
    return weights
