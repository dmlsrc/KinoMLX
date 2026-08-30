"""Data-only LTX-2 resource, transformer, and cache-policy settings."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from kinomlx.settings import EnvironmentSettings

_FF_LAYOUT_TARGETS = frozenset({"project_in", "project_out"})
_ATTN_LAYOUT_TARGETS = frozenset({"to_q", "to_k", "to_v", "to_out", "to_gate_logits"})
_LAYOUT_MODES = frozenset({"pretranspose"})
_FF_QUANTIZATION_MODES = frozenset({"affine", "mxfp4", "mxfp8", "nvfp4"})
GEMMA_VARIANT_CHOICES = ("qat", "plain")
MODEL_GENERATION_CHOICES = ("2.3", "2.5")
VIDEO_VAE_CHOICES = ("conv", "diffusion")
TRANSFORMER_DTYPE_CHOICES = ("bfloat16", "float16", "float32")
TRANSFORMER_CACHE_QUANTIZE_CHOICES = (
    "off",
    "mxfp8-blocks",
    "mxfp8-blocks-pretranspose",
)
STREAM_TRANSFORMER_RESIDENT_BLOCKS = 16
STREAM_TRANSFORMER_COMPILE_GROUP_SIZE = 4


def _parse_specs(
    values: tuple[str, ...],
    *,
    label: str,
    targets: frozenset[str],
    modes: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    """Parse and validate ordered ``target:mode`` selectors."""
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{label} entries must be strings")
        target, separator, mode = value.strip().partition(":")
        if not separator or not target or not mode:
            raise ValueError(f"{label} entry {value!r} must be written as target:mode")
        if target not in targets:
            valid = ", ".join(sorted(targets))
            raise ValueError(f"{label} target {target!r} is unsupported; expected one of: {valid}")
        if mode not in modes:
            valid = ", ".join(sorted(modes))
            raise ValueError(f"{label} mode {mode!r} is unsupported; expected one of: {valid}")
        if target in seen:
            raise ValueError(f"{label} repeats target {target!r}")
        seen.add(target)
        parsed.append((target, mode))
    return tuple(parsed)


def parse_ff_layout_specs(values: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Parse LTX-2 feed-forward cache-layout selectors."""
    return _parse_specs(
        values,
        label="FF layout specs",
        targets=_FF_LAYOUT_TARGETS,
        modes=_LAYOUT_MODES,
    )


def parse_attention_layout_specs(
    values: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Parse LTX-2 attention cache-layout selectors."""
    return _parse_specs(
        values,
        label="attention layout specs",
        targets=_ATTN_LAYOUT_TARGETS,
        modes=_LAYOUT_MODES,
    )


def parse_ff_quantize_specs(
    values: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Parse LTX-2 targeted feed-forward quantization selectors."""
    return _parse_specs(
        values,
        label="video FF quantize specs",
        targets=_FF_LAYOUT_TARGETS,
        modes=_FF_QUANTIZATION_MODES,
    )


@dataclass(frozen=True)
class LTX2Settings(EnvironmentSettings):
    """Checkpoint and execution policy for the selected LTX-2 model family."""

    # Discovery intent is independent of declared checkpoint metadata. When no
    # primary path is supplied, None preserves the established 2.3 default.
    model_generation: str | None = field(
        default=None,
        metadata={
            "env": "{{KINO_LTX_GENERATION}}",
            "cli": ("--model-generation", "--ltx-generation"),
        },
    )
    # ``weights_path`` selects a monolithic baseline or an LTX-2.5 pack
    # directory. Any explicit component path replaces only that component while
    # retaining compatible baseline parts. Components may also be selected
    # directly without a baseline for either generation.
    weights_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_WEIGHTS_PATH}}"},
    )
    gemma_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_GEMMA_PATH}}"},
    )
    # Gemma-3 source for LTX-2.3 cache discovery. The official 2.3 release
    # pairs the QAT-unquantized encoder, so "qat" is the default; "plain"
    # selects the vanilla instruction-tuned release. Explicit gemma_path /
    # text_encoder_path selections bypass discovery and win outright.
    gemma_variant: str = field(
        default="qat",
        metadata={"env": "{{KINO_GEMMA_VARIANT}}"},
    )
    spatial_upscaler_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_SPATIAL_UPSCALER_PATH}}"},
    )
    # ``transformer_path`` selects a componentized primary source or replaces
    # the transformer in ``weights_path``. Every other path is a
    # generation-neutral logical-component override and does not select
    # packaging by itself.
    transformer_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_TRANSFORMER_PATH}}"},
    )
    text_encoder_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_TEXT_ENCODER_PATH}}"},
    )
    video_vae: str = field(
        default="conv",
        metadata={"env": "{{KINO_VIDEO_VAE}}"},
    )
    video_vae_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_VIDEO_VAE_PATH}}"},
    )
    audio_vae_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_AUDIO_VAE_PATH}}"},
    )
    temporal_latent_upscaler_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_TEMPORAL_LATENT_UPSCALER_PATH}}"},
    )
    duration_head_path: Path | None = field(
        default=None,
        metadata={"env": "{{KINO_DURATION_HEAD_PATH}}"},
    )

    # Lets all 48 transformer blocks accumulate in one lazy graph.
    fast_mode: bool = field(default=True, metadata={"env": "{{KINO_FAST_MODE}}"})
    transformer_dtype: str = field(
        default="bfloat16",
        metadata={"env": "{{KINO_TRANSFORMER_DTYPE}}"},
    )
    steel_attention: bool = field(
        default=True,
        metadata={"env": "{{KINO_STEEL_ATTENTION}}"},
    )
    steel_attention_d64: bool = field(
        default=True,
        metadata={"env": "{{KINO_STEEL_ATTENTION_D64}}"},
    )
    steel_attention_probe: bool = field(
        default=False,
        metadata={"env": "{{KINO_STEEL_ATTENTION_PROBE}}"},
    )
    compile_attention: bool = field(
        default=True,
        metadata={"env": "{{KINO_COMPILE_ATTENTION}}"},
    )

    video_ff_layout_specs: tuple[str, ...] = field(
        default=("project_out:pretranspose",),
        metadata={"env": "{{KINO_VIDEO_FF_LAYOUT_SPECS}}"},
    )
    video_ff_layout_layers: tuple[int, ...] = field(
        default=(),
        metadata={"env": "{{KINO_VIDEO_FF_LAYOUT_LAYERS}}"},
    )
    video_attn_layout_specs: tuple[str, ...] = field(
        default=(),
        metadata={"env": "{{KINO_VIDEO_ATTN_LAYOUT_SPECS}}"},
    )
    video_attn_layout_layers: tuple[int, ...] = field(
        default=(),
        metadata={"env": "{{KINO_VIDEO_ATTN_LAYOUT_LAYERS}}"},
    )
    audio_layout_mirror: bool = field(
        default=True,
        metadata={"env": "{{KINO_AUDIO_LAYOUT_MIRROR}}"},
    )
    audio_ff_layout_specs: tuple[str, ...] = field(
        default=(),
        metadata={"env": "{{KINO_AUDIO_FF_LAYOUT_SPECS}}"},
    )
    audio_ff_layout_layers: tuple[int, ...] = field(
        default=(),
        metadata={"env": "{{KINO_AUDIO_FF_LAYOUT_LAYERS}}"},
    )
    audio_attn_layout_specs: tuple[str, ...] = field(
        default=(),
        metadata={"env": "{{KINO_AUDIO_ATTN_LAYOUT_SPECS}}"},
    )
    audio_attn_layout_layers: tuple[int, ...] = field(
        default=(),
        metadata={"env": "{{KINO_AUDIO_ATTN_LAYOUT_LAYERS}}"},
    )
    adaln_pretranspose: bool = field(
        default=False,
        metadata={"env": "{{KINO_ADALN_PRETRANSPOSE}}"},
    )

    transformer_cache_quantize: str = field(
        default="off",
        metadata={"env": "{{KINO_TRANSFORMER_CACHE_QUANTIZE}}"},
    )
    video_ff_quantize_specs: tuple[str, ...] = field(
        default=(),
        metadata={"env": "{{KINO_VIDEO_FF_QUANTIZE_SPECS}}"},
    )
    video_ff_quantize_layers: tuple[int, ...] = field(
        default=(),
        metadata={"env": "{{KINO_VIDEO_FF_QUANTIZE_LAYERS}}"},
    )
    video_ff_quantize_group_size: int | None = field(
        default=None,
        metadata={"env": "{{KINO_VIDEO_FF_QUANTIZE_GROUP_SIZE}}"},
    )
    video_ff_quantize_bits: int | None = field(
        default=None,
        metadata={"env": "{{KINO_VIDEO_FF_QUANTIZE_BITS}}"},
    )

    stream_transformer: bool = field(
        default=False,
        metadata={"env": "{{KINO_STREAM_TRANSFORMER}}"},
    )
    transformer_resident_blocks: int | None = field(
        default=None,
        metadata={"env": "{{KINO_TRANSFORMER_RESIDENT_BLOCKS}}"},
    )
    transformer_compile_group_size: int | None = field(
        default=None,
        metadata={"env": "{{KINO_TRANSFORMER_COMPILE_GROUP_SIZE}}"},
    )
    compile_block_groups: int | None = field(
        default=None,
        metadata={"env": "{{KINO_COMPILE_BLOCK_GROUPS}}"},
    )

    def validate(self) -> None:
        """Validate LTX-2 transformer and prepared-cache policy."""
        if self.gemma_path is not None and self.text_encoder_path is not None:
            raise ValueError("gemma_path and text_encoder_path cannot be combined")
        if self.gemma_variant not in GEMMA_VARIANT_CHOICES:
            valid = ", ".join(GEMMA_VARIANT_CHOICES)
            raise ValueError(f"gemma_variant must be one of: {valid}")
        if (
            self.model_generation is not None
            and self.model_generation not in MODEL_GENERATION_CHOICES
        ):
            valid = ", ".join(MODEL_GENERATION_CHOICES)
            raise ValueError(f"model_generation must be one of: {valid}")
        if self.video_vae not in VIDEO_VAE_CHOICES:
            valid = ", ".join(VIDEO_VAE_CHOICES)
            raise ValueError(f"video_vae must be one of: {valid}")
        if self.transformer_dtype not in TRANSFORMER_DTYPE_CHOICES:
            valid = ", ".join(TRANSFORMER_DTYPE_CHOICES)
            raise ValueError(f"transformer_dtype must be one of: {valid}")
        if self.steel_attention_probe and not self.steel_attention:
            raise ValueError("steel_attention_probe requires steel_attention=true")
        if self.transformer_cache_quantize not in TRANSFORMER_CACHE_QUANTIZE_CHOICES:
            valid = ", ".join(TRANSFORMER_CACHE_QUANTIZE_CHOICES)
            raise ValueError("transformer_cache_quantize must be one of: " + valid)

        parse_ff_layout_specs(self.video_ff_layout_specs)
        parse_attention_layout_specs(self.video_attn_layout_specs)
        parse_ff_quantize_specs(self.video_ff_quantize_specs)
        parse_ff_layout_specs(self.audio_ff_layout_specs)
        parse_attention_layout_specs(self.audio_attn_layout_specs)

        if self.audio_layout_mirror and any(
            (
                self.audio_ff_layout_specs,
                self.audio_ff_layout_layers,
                self.audio_attn_layout_specs,
                self.audio_attn_layout_layers,
            )
        ):
            raise ValueError("explicit audio layout settings require audio_layout_mirror=false")

        for name in (
            "video_ff_layout_layers",
            "video_attn_layout_layers",
            "audio_ff_layout_layers",
            "audio_attn_layout_layers",
            "video_ff_quantize_layers",
        ):
            layers = getattr(self, name)
            if len(set(layers)) != len(layers):
                raise ValueError(f"{name} must not repeat a layer")
            invalid = tuple(layer for layer in layers if not 0 <= layer < 48)
            if invalid:
                raise ValueError(f"{name} layers must be between 0 and 47, got {invalid[0]}")

        if self.transformer_cache_quantize != "off" and self.video_ff_quantize_specs:
            raise ValueError(
                "whole-transformer and targeted FF cache quantization cannot be combined"
            )
        if self.video_ff_quantize_group_size is not None and not self.video_ff_quantize_specs:
            raise ValueError("video_ff_quantize_group_size requires video_ff_quantize_specs")
        if self.video_ff_quantize_bits is not None and not self.video_ff_quantize_specs:
            raise ValueError("video_ff_quantize_bits requires video_ff_quantize_specs")

        for name in (
            "video_ff_quantize_group_size",
            "video_ff_quantize_bits",
            "transformer_resident_blocks",
            "transformer_compile_group_size",
            "compile_block_groups",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.transformer_resident_blocks is not None and self.transformer_resident_blocks > 48:
            raise ValueError("transformer_resident_blocks cannot exceed 48")
        if (
            self.transformer_compile_group_size is not None
            and self.transformer_resident_blocks is None
        ):
            raise ValueError("transformer_compile_group_size requires transformer_resident_blocks")
        if (
            self.transformer_compile_group_size is not None
            and self.transformer_resident_blocks is not None
            and self.transformer_compile_group_size > self.transformer_resident_blocks
        ):
            raise ValueError(
                "transformer_compile_group_size cannot exceed transformer_resident_blocks"
            )
        if self.compile_block_groups is not None and self.compile_block_groups > 48:
            raise ValueError("compile_block_groups cannot exceed 48")

    @property
    def uses_split_checkpoint(self) -> bool:
        """Whether the primary transformer is selected as a component."""
        return self.transformer_path is not None

    def resolve_presets(self) -> LTX2Settings:
        """Expand the constrained-memory preset into low-level values."""
        if not self.stream_transformer:
            return self
        resident = (
            STREAM_TRANSFORMER_RESIDENT_BLOCKS
            if self.transformer_resident_blocks is None
            else self.transformer_resident_blocks
        )
        compile_group = (
            min(STREAM_TRANSFORMER_COMPILE_GROUP_SIZE, resident)
            if self.transformer_compile_group_size is None
            else self.transformer_compile_group_size
        )
        return dataclasses.replace(
            self,
            transformer_resident_blocks=resident,
            transformer_compile_group_size=compile_group,
        )


__all__ = [
    "GEMMA_VARIANT_CHOICES",
    "LTX2Settings",
    "MODEL_GENERATION_CHOICES",
    "STREAM_TRANSFORMER_COMPILE_GROUP_SIZE",
    "STREAM_TRANSFORMER_RESIDENT_BLOCKS",
    "TRANSFORMER_CACHE_QUANTIZE_CHOICES",
    "TRANSFORMER_DTYPE_CHOICES",
    "VIDEO_VAE_CHOICES",
    "parse_attention_layout_specs",
    "parse_ff_layout_specs",
    "parse_ff_quantize_specs",
]
