"""Translate community LTX LoRA prefixes to KinoMLX transformer paths."""

from __future__ import annotations

from ...types import LORA_EXCLUDE_CATEGORIES as _PUBLIC_LORA_EXCLUDE_CATEGORIES
from .transformer import convert_pytorch_key_to_mlx

_LORA_BRANCH_TAGS = frozenset({"video", "audio", "cross"})
_LORA_TYPE_TAGS = frozenset({"attn", "gate", "ff"})
_LORA_MODULE_TAGS = frozenset(
    {
        "attn1",
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "video_to_audio_attn",
        "audio_to_video_attn",
        "ff",
        "audio_ff",
    }
)
_LORA_PROJECTION_TAGS = frozenset(
    {
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "to_gate_logits",
        "project_in",
        "project_out",
    }
)
_LORA_CONTROL_TAGS = frozenset(
    {
        "adaln",
        "prompt_adaln",
        "scale_shift",
        "prompt_scale_shift",
        "gate_adaln",
        "av_ca",
        "cross_control",
        "distill_control",
    }
)
_LORA_CROSS_CONTROL_TAGS = frozenset({"adaln", "attn2", "audio_attn2", "cross"})
_LORA_DISTILL_CONTROL_TAGS = frozenset(
    {
        "cross",
        "gate",
        "adaln",
        "prompt_adaln",
        "scale_shift",
        "prompt_scale_shift",
        "gate_adaln",
        "av_ca",
    }
)
LORA_EXCLUDE_CATEGORIES = _PUBLIC_LORA_EXCLUDE_CATEGORIES

LORA_PAIR_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A",
    ".lora_B",
    ".lora_down",
    ".lora_up",
)


def split_lora_key(key: str) -> tuple[str, str] | None:
    """Split a supported LoRA tensor key into base and adapter suffix."""
    for suffix in LORA_PAIR_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], ".alpha"
    return None


def convert_lora_base_to_mlx(
    base: str,
    *,
    include_audio: bool = True,
) -> str | None:
    """Convert Diffusers/Comfy LTX LoRA base naming to an MLX weight prefix."""
    normalized = base
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    converted = convert_pytorch_key_to_mlx(
        f"{normalized}.weight",
        include_audio=include_audio,
    )
    if converted is None:
        return None
    return converted.removesuffix(".weight")


def lora_key_categories(mlx_key: str) -> frozenset[str]:
    """Return all overlapping knockout categories for one MLX target key."""
    categories: set[str] = set()
    if "video_to_audio" in mlx_key or "audio_to_video" in mlx_key:
        categories.add("cross")
    elif "audio_" in mlx_key:
        categories.add("audio")
    else:
        categories.add("video")

    if "to_gate_logits" in mlx_key:
        categories.add("gate")
    elif any(token in mlx_key for token in ("to_q", "to_k", "to_v", "to_out")):
        categories.add("attn")
    elif "ff" in mlx_key:
        categories.add("ff")

    if "av_ca" in mlx_key or "a2v" in mlx_key or "v2a" in mlx_key:
        categories.add("av_ca")
    if "adaln" in mlx_key:
        categories.add("adaln")
    if "prompt_adaln" in mlx_key:
        categories.update(("prompt_adaln", "prompt_scale_shift"))
    if "scale_shift" in mlx_key:
        categories.add("scale_shift")
    if "prompt_scale_shift" in mlx_key:
        categories.add("prompt_scale_shift")
    if "gate_adaln" in mlx_key:
        categories.add("gate_adaln")

    parts = mlx_key.split(".")
    if "transformer_blocks" in parts:
        block_marker = parts.index("transformer_blocks")
        block_parts = parts[block_marker + 2 :]
        if block_parts and block_parts[0] in _LORA_MODULE_TAGS:
            categories.add(block_parts[0])
        projection = next(
            (part for part in block_parts if part in _LORA_PROJECTION_TAGS),
            None,
        )
        if projection is not None:
            categories.add(projection)

    if categories & _LORA_CROSS_CONTROL_TAGS:
        categories.add("cross_control")
    if categories & _LORA_DISTILL_CONTROL_TAGS:
        categories.add("distill_control")
    return frozenset(categories)


def validate_lora_exclusions(exclude: tuple[str, ...]) -> frozenset[str]:
    """Validate and canonicalize LTX LoRA target knockout categories."""
    requested = frozenset(exclude)
    unknown = sorted(requested - LORA_EXCLUDE_CATEGORIES)
    if unknown:
        valid = ", ".join(sorted(LORA_EXCLUDE_CATEGORIES))
        raise ValueError(f"Unknown LoRA exclude categories {unknown}; valid values: {valid}")
    return requested


__all__ = [
    "LORA_PAIR_SUFFIXES",
    "LORA_EXCLUDE_CATEGORIES",
    "convert_lora_base_to_mlx",
    "lora_key_categories",
    "split_lora_key",
    "validate_lora_exclusions",
]
