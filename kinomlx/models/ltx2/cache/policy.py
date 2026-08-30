"""Canonical transformer cache layout policy."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

LayoutSpecs = tuple[tuple[str, str], ...]

DEFAULT_VIDEO_FF_LAYOUT_SPECS = (("project_out", "pretranspose"),)
DEFAULT_VIDEO_ATTN_LAYOUT_SPECS: LayoutSpecs = ()
DEFAULT_TRANSFORMER_LAYOUT_LAYERS = tuple(range(48))


@dataclass(frozen=True)
class TransformerLayoutPolicy:
    """Canonical layout and targeted-quantization selections."""

    video_ff_specs: LayoutSpecs
    video_ff_layers: tuple[int, ...]
    video_attn_specs: LayoutSpecs
    video_attn_layers: tuple[int, ...]
    audio_ff_specs: LayoutSpecs
    audio_ff_layers: tuple[int, ...]
    audio_attn_specs: LayoutSpecs
    audio_attn_layers: tuple[int, ...]
    video_ff_quantize_layers: tuple[int, ...]


def ensure_ff_pretranspose_for_dtype(
    specs: LayoutSpecs,
    dtype: mx.Dtype | None,
) -> LayoutSpecs:
    """Add both FF pretranspose targets required by an FP16 FF path."""
    if dtype is None or dtype != mx.float16:
        return tuple(specs)
    existing = {target for target, _layout in specs}
    additions = tuple(
        (target, "pretranspose")
        for target in ("project_in", "project_out")
        if target not in existing
    )
    return additions + tuple(specs)


def normalize_layout_layers(
    specs: LayoutSpecs,
    layers: tuple[int, ...],
) -> tuple[int, ...]:
    """Expand an implicit all-layer selection for stable cache identity."""
    if specs and not layers:
        return DEFAULT_TRANSFORMER_LAYOUT_LAYERS
    return tuple(layers)


def normalize_transformer_layout_policy(
    *,
    include_audio: bool,
    transformer_dtype: mx.Dtype | None,
    video_ff_dtype: mx.Dtype | None,
    audio_ff_dtype: mx.Dtype | None,
    video_ff_layout_specs: LayoutSpecs,
    video_ff_layout_layers: tuple[int, ...],
    video_attn_layout_specs: LayoutSpecs,
    video_attn_layout_layers: tuple[int, ...],
    audio_ff_layout_specs: LayoutSpecs | None,
    audio_ff_layout_layers: tuple[int, ...] | None,
    audio_attn_layout_specs: LayoutSpecs | None,
    audio_attn_layout_layers: tuple[int, ...] | None,
    video_ff_quantize_specs: LayoutSpecs,
    video_ff_quantize_layers: tuple[int, ...],
    layouts_enabled: bool,
) -> TransformerLayoutPolicy:
    """Resolve defaults, audio mirrors, FP16 requirements, and layer aliases.

    ``None`` audio selections mirror their corresponding video selection.
    Explicit empty tuples disable the audio transform. Whole-block quantized
    caches disable the FF/attention layout selections but keep the independent
    top-level AdaLN option outside this policy.
    """
    if layouts_enabled:
        resolved_video_ff = ensure_ff_pretranspose_for_dtype(
            tuple(video_ff_layout_specs),
            (transformer_dtype if transformer_dtype is not None else video_ff_dtype),
        )
        resolved_video_attn = tuple(video_attn_layout_specs)
    else:
        resolved_video_ff = ()
        resolved_video_attn = ()

    resolved_video_ff_layers = normalize_layout_layers(
        resolved_video_ff,
        tuple(video_ff_layout_layers),
    )
    resolved_video_attn_layers = normalize_layout_layers(
        resolved_video_attn,
        tuple(video_attn_layout_layers),
    )

    if include_audio and layouts_enabled:
        resolved_audio_ff = (
            resolved_video_ff if audio_ff_layout_specs is None else tuple(audio_ff_layout_specs)
        )
        resolved_audio_attn = (
            resolved_video_attn
            if audio_attn_layout_specs is None
            else tuple(audio_attn_layout_specs)
        )
        resolved_audio_ff = ensure_ff_pretranspose_for_dtype(
            resolved_audio_ff,
            (transformer_dtype if transformer_dtype is not None else audio_ff_dtype),
        )
        raw_audio_ff_layers = (
            resolved_video_ff_layers
            if audio_ff_layout_layers is None
            else tuple(audio_ff_layout_layers)
        )
        raw_audio_attn_layers = (
            resolved_video_attn_layers
            if audio_attn_layout_layers is None
            else tuple(audio_attn_layout_layers)
        )
        resolved_audio_ff_layers = normalize_layout_layers(
            resolved_audio_ff,
            raw_audio_ff_layers,
        )
        resolved_audio_attn_layers = normalize_layout_layers(
            resolved_audio_attn,
            raw_audio_attn_layers,
        )
    else:
        resolved_audio_ff = ()
        resolved_audio_ff_layers = ()
        resolved_audio_attn = ()
        resolved_audio_attn_layers = ()

    return TransformerLayoutPolicy(
        video_ff_specs=resolved_video_ff,
        video_ff_layers=resolved_video_ff_layers,
        video_attn_specs=resolved_video_attn,
        video_attn_layers=resolved_video_attn_layers,
        audio_ff_specs=resolved_audio_ff,
        audio_ff_layers=resolved_audio_ff_layers,
        audio_attn_specs=resolved_audio_attn,
        audio_attn_layers=resolved_audio_attn_layers,
        video_ff_quantize_layers=normalize_layout_layers(
            tuple(video_ff_quantize_specs),
            tuple(video_ff_quantize_layers),
        ),
    )


__all__ = [
    "DEFAULT_TRANSFORMER_LAYOUT_LAYERS",
    "DEFAULT_VIDEO_ATTN_LAYOUT_SPECS",
    "DEFAULT_VIDEO_FF_LAYOUT_SPECS",
    "LayoutSpecs",
    "TransformerLayoutPolicy",
    "ensure_ff_pretranspose_for_dtype",
    "normalize_layout_layers",
    "normalize_transformer_layout_policy",
]
