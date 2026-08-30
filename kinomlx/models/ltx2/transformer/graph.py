"""Pure transformer parameter-graph and checkpoint-key contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class TransformerGraphConfig(Protocol):
    """Constructor fields that determine the parameter graph and its shapes."""

    @property
    def num_layers(self) -> int: ...

    @property
    def video_in_channels(self) -> int: ...

    @property
    def video_out_channels(self) -> int: ...

    @property
    def video_heads(self) -> int: ...

    @property
    def video_head_dim(self) -> int: ...

    @property
    def audio_heads(self) -> int: ...

    @property
    def audio_head_dim(self) -> int: ...

    @property
    def audio_out_channels(self) -> int: ...

    @property
    def video_context_dim(self) -> int: ...

    @property
    def audio_context_dim(self) -> int: ...

    @property
    def ff_bias(self) -> bool: ...

    @property
    def audio_ff_bias(self) -> bool: ...

    @property
    def use_keyframes_abs_pos_embedding(self) -> bool: ...

    @property
    def use_prompt_adaln_single(self) -> bool: ...


@dataclass(frozen=True)
class TransformerGraphSpec:
    """Concrete runtime copy of the fields that select the parameter graph."""

    num_layers: int
    video_in_channels: int
    audio_in_channels: int
    video_out_channels: int
    video_heads: int
    video_head_dim: int
    audio_heads: int
    audio_head_dim: int
    audio_out_channels: int
    video_context_dim: int
    audio_context_dim: int
    ff_bias: bool
    audio_ff_bias: bool
    use_keyframes_abs_pos_embedding: bool
    use_prompt_adaln_single: bool


@dataclass(frozen=True)
class ConvertedTransformerKey:
    """One normalized source key and its deterministic alias priority."""

    target_key: str
    priority: int


DIFFUSION_PREFIX = "model.diffusion_model."
_DIFFUSION_PREFIXES = (DIFFUSION_PREFIX, "diffusion_model.", "")
_TRANSFORMER_ROOTS = (
    "adaln_single.",
    "audio_adaln_single.",
    "audio_patchify_proj.",
    "audio_proj_out.",
    "audio_prompt_adaln_single.",
    "audio_scale_shift_table",
    "av_ca_a2v_gate_adaln_single.",
    "av_ca_audio_scale_shift_adaln_single.",
    "av_ca_v2a_gate_adaln_single.",
    "av_ca_video_scale_shift_adaln_single.",
    "keyframes_abs_pos_embedding",
    "norm_out.",
    "audio_norm_out.",
    "patchify_proj.",
    "proj_out.",
    "prompt_adaln_single.",
    "scale_shift_table",
    "transformer_blocks.",
)
_VIDEO_ONLY_SKIP_FRAGMENTS = ("av_ca", "a2v", "audio")
_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.to_out\.0\."), ".to_out."),
    (re.compile(r"(^|\.)ff\.net\.0\.proj\."), r"\1ff.project_in.proj."),
    (re.compile(r"(^|\.)ff\.net\.2\."), r"\1ff.project_out."),
    (
        re.compile(r"(^|\.)audio_ff\.net\.0\.proj\."),
        r"\1audio_ff.project_in.proj.",
    ),
    (re.compile(r"(^|\.)audio_ff\.net\.2\."), r"\1audio_ff.project_out."),
)


def _linear(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    input_dims: int,
    output_dims: int,
    *,
    bias: bool,
) -> None:
    shapes[f"{prefix}.weight"] = (output_dims, input_dims)
    if bias:
        shapes[f"{prefix}.bias"] = (output_dims,)


def _rms_norm(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    dims: int,
) -> None:
    shapes[f"{prefix}.weight"] = (dims,)


def _adaln(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    dims: int,
    modulations: int,
) -> None:
    _linear(
        shapes,
        f"{prefix}.emb.timestep_embedder.linear_1",
        256,
        dims,
        bias=True,
    )
    _linear(
        shapes,
        f"{prefix}.emb.timestep_embedder.linear_2",
        dims,
        dims,
        bias=True,
    )
    _linear(
        shapes,
        f"{prefix}.linear",
        dims,
        modulations * dims,
        bias=True,
    )


def _attention(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    *,
    query_dim: int,
    source_dim: int,
    heads: int,
    head_dim: int,
) -> None:
    inner_dim = heads * head_dim
    _rms_norm(shapes, f"{prefix}.q_norm", inner_dim)
    _rms_norm(shapes, f"{prefix}.k_norm", inner_dim)
    _linear(shapes, f"{prefix}.to_q", query_dim, inner_dim, bias=True)
    _linear(shapes, f"{prefix}.to_k", source_dim, inner_dim, bias=True)
    _linear(shapes, f"{prefix}.to_v", source_dim, inner_dim, bias=True)
    _linear(shapes, f"{prefix}.to_out", inner_dim, query_dim, bias=True)
    _linear(shapes, f"{prefix}.to_gate_logits", query_dim, heads, bias=True)


def _feed_forward(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    dims: int,
    *,
    bias: bool,
) -> None:
    inner_dim = dims * 4
    _linear(shapes, f"{prefix}.project_in.proj", dims, inner_dim, bias=bias)
    _linear(shapes, f"{prefix}.project_out", inner_dim, dims, bias=bias)


def transformer_parameter_shapes(
    config: TransformerGraphConfig,
    *,
    include_audio: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Return every consumed MLX target and its checkpoint-layout shape."""
    video_dim = config.video_heads * config.video_head_dim
    audio_dim = config.audio_heads * config.audio_head_dim
    audio_in_channels = getattr(
        config,
        "audio_in_channels",
        config.video_in_channels,
    )
    shapes: dict[str, tuple[int, ...]] = {}

    _linear(
        shapes,
        "patchify_proj",
        config.video_in_channels,
        video_dim,
        bias=True,
    )
    _adaln(shapes, "adaln_single", video_dim, 9)
    if config.use_prompt_adaln_single:
        _adaln(shapes, "prompt_adaln_single", video_dim, 2)
    shapes["scale_shift_table"] = (2, video_dim)
    _linear(shapes, "proj_out", video_dim, config.video_out_channels, bias=True)
    if config.use_keyframes_abs_pos_embedding:
        shapes["keyframes_abs_pos_embedding"] = (1, video_dim)

    if include_audio:
        _linear(
            shapes,
            "audio_patchify_proj",
            audio_in_channels,
            audio_dim,
            bias=True,
        )
        _adaln(shapes, "audio_adaln_single", audio_dim, 9)
        if config.use_prompt_adaln_single:
            _adaln(shapes, "audio_prompt_adaln_single", audio_dim, 2)
        shapes["audio_scale_shift_table"] = (2, audio_dim)
        _linear(
            shapes,
            "audio_proj_out",
            audio_dim,
            config.audio_out_channels,
            bias=True,
        )
        _adaln(shapes, "av_ca_video_scale_shift_adaln_single", video_dim, 4)
        _adaln(shapes, "av_ca_a2v_gate_adaln_single", video_dim, 1)
        _adaln(shapes, "av_ca_audio_scale_shift_adaln_single", audio_dim, 4)
        _adaln(shapes, "av_ca_v2a_gate_adaln_single", audio_dim, 1)

    for index in range(config.num_layers):
        block = f"transformer_blocks.{index}"
        _attention(
            shapes,
            f"{block}.attn1",
            query_dim=video_dim,
            source_dim=video_dim,
            heads=config.video_heads,
            head_dim=config.video_head_dim,
        )
        _attention(
            shapes,
            f"{block}.attn2",
            query_dim=video_dim,
            source_dim=config.video_context_dim,
            heads=config.video_heads,
            head_dim=config.video_head_dim,
        )
        _feed_forward(shapes, f"{block}.ff", video_dim, bias=config.ff_bias)
        shapes[f"{block}.scale_shift_table"] = (9, video_dim)
        shapes[f"{block}.prompt_scale_shift_table"] = (2, video_dim)
        if not include_audio:
            continue

        _attention(
            shapes,
            f"{block}.audio_attn1",
            query_dim=audio_dim,
            source_dim=audio_dim,
            heads=config.audio_heads,
            head_dim=config.audio_head_dim,
        )
        _attention(
            shapes,
            f"{block}.audio_attn2",
            query_dim=audio_dim,
            source_dim=config.audio_context_dim,
            heads=config.audio_heads,
            head_dim=config.audio_head_dim,
        )
        _feed_forward(
            shapes,
            f"{block}.audio_ff",
            audio_dim,
            bias=config.audio_ff_bias,
        )
        _attention(
            shapes,
            f"{block}.audio_to_video_attn",
            query_dim=video_dim,
            source_dim=audio_dim,
            heads=config.audio_heads,
            head_dim=config.audio_head_dim,
        )
        _attention(
            shapes,
            f"{block}.video_to_audio_attn",
            query_dim=audio_dim,
            source_dim=video_dim,
            heads=config.audio_heads,
            head_dim=config.audio_head_dim,
        )
        shapes[f"{block}.audio_scale_shift_table"] = (9, audio_dim)
        shapes[f"{block}.audio_prompt_scale_shift_table"] = (2, audio_dim)
        shapes[f"{block}.scale_shift_table_a2v_ca_audio"] = (5, audio_dim)
        shapes[f"{block}.scale_shift_table_a2v_ca_video"] = (5, video_dim)
    return shapes


def _strip_source_wrapper(checkpoint_key: str) -> tuple[str, int] | None:
    for priority, prefix in enumerate(_DIFFUSION_PREFIXES[:-1]):
        if checkpoint_key.startswith(prefix):
            return checkpoint_key.removeprefix(prefix), priority
    for offset, prefix in enumerate(_DIFFUSION_PREFIXES[:-1], start=3):
        marker = "." + prefix
        position = checkpoint_key.find(marker)
        if position >= 0:
            return checkpoint_key[position + len(marker) :], offset
    if checkpoint_key.startswith(_TRANSFORMER_ROOTS):
        return checkpoint_key, 2
    for root in _TRANSFORMER_ROOTS:
        marker = "." + root
        position = checkpoint_key.find(marker)
        if position >= 0:
            return checkpoint_key[position + 1 :], 6
    return None


def convert_checkpoint_key(
    checkpoint_key: str,
    *,
    include_audio: bool = False,
) -> ConvertedTransformerKey | None:
    """Normalize a transformer source key without granting wrappers authority."""
    stripped = _strip_source_wrapper(checkpoint_key)
    if stripped is None:
        return None
    logical_key, priority = stripped
    if "embeddings_connector." in logical_key:
        return None
    lowered = logical_key.lower()
    if not include_audio and any(fragment in lowered for fragment in _VIDEO_ONLY_SKIP_FRAGMENTS):
        return None
    converted = logical_key
    for pattern, replacement in _REPLACEMENTS:
        converted = pattern.sub(replacement, converted)
    return ConvertedTransformerKey(converted, priority)


__all__ = [
    "ConvertedTransformerKey",
    "DIFFUSION_PREFIX",
    "TransformerGraphConfig",
    "TransformerGraphSpec",
    "convert_checkpoint_key",
    "transformer_parameter_shapes",
]
