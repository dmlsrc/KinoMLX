"""Data-only LTX-2 resource, transformer, and weight-policy settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

from kinomlx.config_constraints import (
    ChoiceConstraint,
    IntegerConstraint,
    LengthConstraint,
    NumberConstraint,
    NumberSequenceConstraint,
)
from kinomlx.config_records import config_record, derive_config
from kinomlx.modeling.dtypes import FLOAT_DTYPE_CHOICES, FloatDType
from kinomlx.settings import EnvironmentSettings

from .settings_rules import (
    ADAPTER_PRECISION,
    AUDIO_LAYOUT,
    COMPILE_GROUPS,
    QUANTIZATION,
    RETENTION,
)
from .video_vae.kinds import VIDEO_VAE_KIND_CHOICES
from .vocabularies import (
    ADAPTER_ACCUMULATION_CHOICES,
    GEMMA_VARIANT_CHOICES,
    MODEL_GENERATION_CHOICES,
    GemmaVariant,
    LTX2AdapterAccumulation,
    LTX2ModelGeneration,
)
from .weights.schema import TRANSFORMER_QUANTIZE_MODES, TransformerQuantize

_FF_LAYOUT_TARGETS = frozenset({"project_in", "project_out"})
_ATTN_LAYOUT_TARGETS = frozenset({"to_q", "to_k", "to_v", "to_out", "to_gate_logits"})
_LAYOUT_MODES = frozenset({"pretranspose"})
_FF_QUANTIZATION_MODES = frozenset({"affine", "mxfp4", "mxfp8", "nvfp4"})
VIDEO_VAE_CHOICES = VIDEO_VAE_KIND_CHOICES
# Two groups of four resident: one group's source reads overlap the other's
# computation. More resident blocks only add memory once reads are overlapped.
# The groups stay uncompiled: compiled block groups measured no faster and round
# bfloat16 differently from eager blocks.
STREAM_TRANSFORMER_RESIDENT_BLOCKS = 8


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
    """Parse LTX-2 feed-forward weight-layout selectors."""
    return _parse_specs(
        values,
        label="FF layout specs",
        targets=_FF_LAYOUT_TARGETS,
        modes=_LAYOUT_MODES,
    )


def parse_attention_layout_specs(
    values: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Parse LTX-2 attention weight-layout selectors."""
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


@dataclass(frozen=True, kw_only=True)
class SpecSequenceConstraint(LengthConstraint):
    label: str
    targets: frozenset[str]
    modes: frozenset[str]

    def validate(self, value: object) -> None:
        if not isinstance(value, (tuple, list)):
            raise ValueError("must be an array of specifications")
        super().validate(value)
        _parse_specs(
            cast(tuple[str, ...], tuple(value)),
            label=self.label,
            targets=self.targets,
            modes=self.modes,
        )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "kind": "target_mode_specs",
            "targets": sorted(self.targets),
            "modes": sorted(self.modes),
            "unique_targets": True,
        }


@dataclass(frozen=True)
class RetainPatternsConstraint(LengthConstraint):
    def validate(self, value: object) -> None:
        if not isinstance(value, (tuple, list)) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("transformer_retain entries must not be empty")
        super().validate(value)

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "kind": "retain_patterns", "nonempty_items": True}


def _specs_field(
    default: tuple[str, ...] = (), *, label: str, targets: frozenset[str], modes: frozenset[str]
) -> tuple[str, ...]:
    return field(
        default=default,
        metadata={
            "config_constraints": (
                SpecSequenceConstraint(label=label, targets=targets, modes=modes),
            )
        },
    )


@dataclass(frozen=True)
class LayerIndicesConstraint(NumberSequenceConstraint):
    """Own the ordered, unique LTX transformer layer-index contract."""

    item: NumberConstraint = IntegerConstraint(minimum=0, maximum=47)

    def validate(self, value: object) -> None:
        super().validate(value)
        if isinstance(value, (list, tuple)) and len(set(value)) != len(value):
            raise ValueError("must not repeat a layer")

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "unique": True}


def _layer_indices_field() -> tuple[int, ...]:
    return field(default=(), metadata={"config_constraints": (LayerIndicesConstraint(),)})


@config_record
@dataclass(frozen=True)
class LTX2Settings(EnvironmentSettings):
    """Checkpoint and execution policy for the selected LTX-2 model family."""

    environment_prefix: ClassVar[str] = "KINO_LTX2_"
    config_table: ClassVar[str] = "[model_settings]"

    # Discovery intent is independent of declared checkpoint metadata. When no
    # primary path is supplied, None preserves the established 2.3 default.
    model_generation: LTX2ModelGeneration | None = field(
        default=None,
        metadata={
            "cli": ("--model-generation", "--ltx-generation"),
            "config_constraints": (ChoiceConstraint(MODEL_GENERATION_CHOICES),),
        },
    )
    # ``weights_path`` selects a monolithic baseline or an LTX-2.5 pack
    # directory. Any explicit component path replaces only that component while
    # retaining compatible baseline parts. Components may also be selected
    # directly without a baseline for either generation.
    weights_path: Path | None = None
    # Gemma-3 source for LTX-2.3 cache discovery. The official 2.3 release
    # pairs the QAT-unquantized encoder, so "qat" is the default; "plain"
    # selects the vanilla instruction-tuned release. An explicit
    # text_encoder_path bypasses discovery and wins outright.
    gemma_variant: GemmaVariant = field(
        default=GemmaVariant.QAT,
        metadata={"config_constraints": (ChoiceConstraint(GEMMA_VARIANT_CHOICES),)},
    )
    spatial_upscaler_path: Path | None = None
    # ``transformer_path`` selects a componentized primary source or replaces
    # the transformer in ``weights_path``. Every other path is a
    # generation-neutral logical-component override and does not select
    # packaging by itself.
    transformer_path: Path | None = None
    text_encoder_path: Path | None = None
    video_vae: str = field(
        default="conv", metadata={"config_constraints": (ChoiceConstraint(VIDEO_VAE_CHOICES),)}
    )
    video_vae_path: Path | None = None
    audio_vae_path: Path | None = None
    temporal_latent_upscaler_path: Path | None = None
    duration_head_path: Path | None = None

    # Lets all 48 transformer blocks accumulate in one lazy graph.
    fast_mode: bool = True
    # Measured against an FP32 reference, FP16 forwards came 3 to 16 times closer than BF16
    # on LTX-2.3 and LTX-2.5, and faster; the largest hidden state, about 23,600 on LTX-2.3,
    # stays 2.8 times below FP16's range, and the step check stops a run that overflows.
    transformer_dtype: FloatDType = field(
        default=FloatDType.FLOAT16,
        metadata={"config_constraints": (ChoiceConstraint(FLOAT_DTYPE_CHOICES),)},
    )
    adapter_accumulation: LTX2AdapterAccumulation = field(
        default=LTX2AdapterAccumulation.AUTO,
        metadata={
            "config_constraints": (ChoiceConstraint(ADAPTER_ACCUMULATION_CHOICES),),
            "config_rules": (ADAPTER_PRECISION,),
        },
    )
    compile_attention: bool = True
    dense_attention_kernel: bool = True

    video_ff_layout_specs: tuple[str, ...] = _specs_field(
        ("project_out:pretranspose",),
        label="FF layout specs",
        targets=_FF_LAYOUT_TARGETS,
        modes=_LAYOUT_MODES,
    )
    video_ff_layout_layers: tuple[int, ...] = _layer_indices_field()
    video_attn_layout_specs: tuple[str, ...] = _specs_field(
        (), label="attention layout specs", targets=_ATTN_LAYOUT_TARGETS, modes=_LAYOUT_MODES
    )
    video_attn_layout_layers: tuple[int, ...] = _layer_indices_field()
    audio_layout_mirror: bool = field(default=True, metadata={"config_rules": (AUDIO_LAYOUT,)})
    audio_ff_layout_specs: tuple[str, ...] = _specs_field(
        (), label="FF layout specs", targets=_FF_LAYOUT_TARGETS, modes=_LAYOUT_MODES
    )
    audio_ff_layout_layers: tuple[int, ...] = _layer_indices_field()
    audio_attn_layout_specs: tuple[str, ...] = _specs_field(
        (), label="attention layout specs", targets=_ATTN_LAYOUT_TARGETS, modes=_LAYOUT_MODES
    )
    audio_attn_layout_layers: tuple[int, ...] = _layer_indices_field()
    adaln_pretranspose: bool = False

    transformer_quantize: TransformerQuantize = field(
        default=TransformerQuantize.OFF,
        metadata={
            "config_constraints": (ChoiceConstraint(TRANSFORMER_QUANTIZE_MODES),),
            "config_rules": (QUANTIZATION,),
        },
    )
    transformer_quantize_exclude_layers: tuple[int, ...] = _layer_indices_field()
    video_ff_quantize_specs: tuple[str, ...] = _specs_field(
        (),
        label="video FF quantize specs",
        targets=_FF_LAYOUT_TARGETS,
        modes=_FF_QUANTIZATION_MODES,
    )
    video_ff_quantize_layers: tuple[int, ...] = _layer_indices_field()
    video_ff_quantize_group_size: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )
    video_ff_quantize_bits: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1),)}
    )

    stream_transformer: bool = False
    transformer_retain: tuple[str, ...] = field(
        default=(),
        metadata={
            "config_constraints": (RetainPatternsConstraint(),),
            "config_rules": (RETENTION,),
        },
    )
    transformer_resident_blocks: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1, maximum=48),)}
    )
    transformer_compile_group_size: int | None = field(
        default=None,
        metadata={
            "config_constraints": (IntegerConstraint(minimum=1),),
            "config_rules": (COMPILE_GROUPS,),
        },
    )
    compile_block_group_size: int | None = field(
        default=None, metadata={"config_constraints": (IntegerConstraint(minimum=1, maximum=48),)}
    )

    @property
    def uses_split_checkpoint(self) -> bool:
        """Whether the primary transformer is selected as a component."""
        return self.transformer_path is not None

    def resolved_adapter_accumulation(self) -> FloatDType:
        """Return the dtype adapter fusion accumulates in on this transformer.

        ``auto`` resolves here, so every consumer downstream holds a dtype
        rather than a vocabulary that still contains an instruction.
        """
        if self.adapter_accumulation != LTX2AdapterAccumulation.AUTO:
            return FloatDType(self.adapter_accumulation)
        streamed = self.resolve_presets().transformer_resident_blocks is not None
        streamed_fp16 = streamed and self.transformer_dtype == FloatDType.FLOAT16
        return FloatDType.FLOAT16 if streamed_fp16 else FloatDType.FLOAT32

    def resolve_presets(self) -> LTX2Settings:
        """Expand the constrained-memory preset into low-level values."""
        if not self.stream_transformer:
            return self
        resident = (
            STREAM_TRANSFORMER_RESIDENT_BLOCKS
            if self.transformer_resident_blocks is None
            else self.transformer_resident_blocks
        )
        compile_group = self.transformer_compile_group_size
        return derive_config(
            self,
            transformer_resident_blocks=resident,
            transformer_compile_group_size=compile_group,
        )


__all__ = [
    "GEMMA_VARIANT_CHOICES",
    "MODEL_GENERATION_CHOICES",
    "STREAM_TRANSFORMER_RESIDENT_BLOCKS",
    "VIDEO_VAE_CHOICES",
    "LTX2Settings",
    "parse_attention_layout_specs",
    "parse_ff_layout_specs",
    "parse_ff_quantize_specs",
]
