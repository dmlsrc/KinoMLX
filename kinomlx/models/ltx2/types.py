"""LTX-2.3 type constants and shape derivations.

These are LTX-2.3 facts - native frame rate, VAE compression ratios,
audio encoder geometry, latent channel counts - and the shape
derivations that depend on them.  Top-level :mod:`kinomlx.types`
carries only model-agnostic primitives so future models can reuse
them without inheriting LTX-2's numbers.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, NamedTuple

import mlx.core as mx

from kinomlx.adapters.lora import LoRAProfile
from kinomlx.config_constraints import (
    ChoiceConstraint,
    IntegerConstraint,
    NumberConstraint,
    NumberSequenceConstraint,
)
from kinomlx.config_records import config_record
from kinomlx.config_rules import validate_record
from kinomlx.request_rules import UNCONDITIONAL_PROMPT
from kinomlx.samplers.config import SigmaScheduleConstraint
from kinomlx.seeds import resolve_seeds
from kinomlx.types import (
    DEFAULT_NOISE_BACKEND,
    NOISE_BACKEND_CHOICES,
    VAE_TILING_MODE_CHOICES,
    NoiseBackend,
    SpatioTemporalScaleFactors,
    VAETilingMode,
    VideoLatentShape,
    VideoPixelShape,
)

from .adapters.inputs import LTX2AdapterConfig
from .adapters.selection import LTX2AdapterSelection, lora_profile_for_stage
from .request_rules import (
    DISTILLED_CAPABILITIES,
    HDR_SOURCES,
    IMAGE_FRAME_BOUNDS,
    KEYFRAME_POSITIONS,
    VAE_TILING,
    custom_temporal_tile_frames,
)
from .video_vae.tiling import SpatialTilingConfig, TemporalChunkConfig, TilingConfig

# ---------------------------------------------------------------------------
# Video model constants
# ---------------------------------------------------------------------------


# Native generation frame rate.  Matches ``frame_rate`` in the official
# Lightricks ``PipelineParams`` -
# https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
NATIVE_FPS: float = 24.0

# Video VAE compression: 8x temporal, 32x spatial.
VIDEO_VAE_SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)


def resolved_frame_count_for_duration(duration: float, fps: float) -> int:
    """Round one validated duration up to the exact LTX-2 frame lattice."""
    requested_frames = math.ceil(duration * fps)
    return ((max(1, requested_frames) - 1 + 7) // 8) * 8 + 1


# Latent channels at the video VAE bottleneck.
VIDEO_LATENT_CHANNELS: int = 128

# ``auto`` preserves the checkpoint generation's published sampler behavior.
# Explicit modes are useful for controlled A/B diagnostics without changing
# checkpoint selection or the literal sigma schedules.
DISTILLED_SAMPLER_CHOICES = ("auto", "deterministic", "ancestral")
DistilledSampler = Literal["auto", "deterministic", "ancestral"]


@dataclass(frozen=True)
class LatentState:
    """State of an LTX-2 latent tensor during diffusion denoising.

    Fields:
        latent: The currently noisy latent being denoised.
        denoise_mask: Per-token denoising strength (1 = full, 0 = none).
        positions: Positional indices used for positional embeddings.
        clean_latent: Initial latent before denoising; may include image or
            keyframe conditioning latents.
        uniform_mask: Python-side flag that is true iff every mask entry is 1.
            It is not derived in the hot path because that would force an
            evaluation boundary. LTX-2 uses it to select scalar versus
            per-token timestep work and preserve A/V cross-attention semantics.

    When replacing ``denoise_mask`` with a mixed mask, callers must explicitly
    set ``uniform_mask=False``. The flag does not update itself::

        state = dataclasses.replace(
            state,
            denoise_mask=new_mask,
            uniform_mask=False,
        )
    """

    latent: mx.array
    denoise_mask: mx.array
    positions: mx.array
    clean_latent: mx.array
    uniform_mask: bool = True


# HDR selects both the explicit EXR condition interpretation and the output
# sequence authoring space in the first public HDR profile. The model's working
# transfer is resolved independently from checkpoint generation and a supported
# recipe: 2.5 is image-to-video from an HDR EXR condition, while 2.3 is
# video-to-video from an SDR reference plus a metadata-declared LogC3 adapter.
# Unconditioned T2V HDR is refused: neither generation has a validated way to
# anchor the HDR signal from a prompt alone.
HDR_AUTHORING_CHOICES = ("SRGB_LINEAR", "ACESCG", "ACESCCT")
HDRAuthoring = Literal["SRGB_LINEAR", "ACESCG", "ACESCCT"]

# ``auto`` keeps the precision contract recipe-aware: BF16 for ordinary SDR
# decode and FP32 for HDR decode. Explicit values are controlled diagnostics
# and are also useful for decode-only restarts.
VAE_DECODE_DTYPE_CHOICES = ("auto", "bfloat16", "float16", "float32")
VideoVAEDecodeDType = Literal["auto", "bfloat16", "float16", "float32"]


@config_record
@dataclass(frozen=True)
class ImageConditioningConfig:
    """One image-conditioning input for the distilled pipeline."""

    path: Path
    frame_index: int = field(
        default=0, metadata={"config_constraints": (IntegerConstraint(minimum=0),)}
    )
    strength: float = field(
        default=0.95, metadata={"config_constraints": (NumberConstraint(0, 1),)}
    )


@config_record
@dataclass(frozen=True)
class HDRReferenceConditioningConfig:
    """One SDR reference video for the explicit LTX-2.3 HDR IC-LoRA recipe."""

    path: Path
    strength: float = field(default=1.0, metadata={"config_constraints": (NumberConstraint(0, 1),)})


@dataclass(frozen=True, kw_only=True)
class TilingGridConstraint(IntegerConstraint):
    """Tile and overlap values follow a VAE grid, with zero as an admitted sentinel."""

    minimum: float | None = 0
    grid: int
    nonzero_minimum: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.grid) is not int or self.grid < 1:
            raise ValueError("tiling grid must be a positive integer")
        if type(self.nonzero_minimum) is not int or self.nonzero_minimum < 0:
            raise ValueError("nonzero tile minimum must be a nonnegative integer")

    def validate(self, value: object) -> None:
        super().validate(value)
        if isinstance(value, int) and value and (value < self.nonzero_minimum or value % self.grid):
            raise ValueError(
                f"must be zero or at least {self.nonzero_minimum} and divisible by {self.grid}"
            )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "multiple_of": self.grid,
            "nonzero_minimum": self.nonzero_minimum,
            "zero_allowed": True,
        }


@config_record
@dataclass(frozen=True)
class VideoVAETilingConfig:
    """Resolved user policy for memory-bounded video VAE decoding.

    ``auto`` lets the native decoder select a memory-aware plan, ``single``
    forces one decode, and ``custom`` builds the requested temporal/spatial
    plan. Local field constraints admit tile values; final rules validate
    relationships before runtime conversion.
    """

    mode: VAETilingMode = field(
        default="auto",
        metadata={
            "config_constraints": (ChoiceConstraint(VAE_TILING_MODE_CHOICES),),
            "config_rules": (VAE_TILING,),
        },
    )
    temporal_tile_frames: int | None = field(
        default=None,
        metadata={
            "config_constraints": (
                TilingGridConstraint(grid=VIDEO_VAE_SCALE.time, nonzero_minimum=16),
            )
        },
    )
    temporal_overlap_frames: int = field(
        default=24,
        metadata={
            "config_constraints": (
                TilingGridConstraint(grid=VIDEO_VAE_SCALE.time, nonzero_minimum=0),
            )
        },
    )
    spatial_tile_pixels: int | None = field(
        default=None,
        metadata={
            "config_constraints": (
                TilingGridConstraint(grid=VIDEO_VAE_SCALE.height, nonzero_minimum=64),
            )
        },
    )
    spatial_overlap_pixels: int = field(
        default=64,
        metadata={
            "config_constraints": (
                TilingGridConstraint(grid=VIDEO_VAE_SCALE.height, nonzero_minimum=0),
            )
        },
    )

    def to_runtime_config(self) -> TilingConfig | None:
        """Build the decoder's native tiling config for this policy."""
        validate_record(self, location="[generate].vae_tiling")
        if self.mode == "auto":
            return None
        if self.mode == "single":
            # A non-None empty plan bypasses decode_streaming's auto planner.
            return TilingConfig()

        temporal_size = custom_temporal_tile_frames(self.temporal_tile_frames)
        temporal = (
            TemporalChunkConfig(temporal_size, self.temporal_overlap_frames)
            if temporal_size > 0
            else None
        )
        spatial_size = self.spatial_tile_pixels
        spatial = (
            SpatialTilingConfig(spatial_size, self.spatial_overlap_pixels)
            if spatial_size not in (None, 0)
            else None
        )
        return TilingConfig(spatial_config=spatial, temporal_config=temporal)


@dataclass(frozen=True)
class VideoDimensionConstraint(IntegerConstraint):
    """LTX two-stage dimensions follow the combined spatial grid."""

    minimum: float | None = 1

    def validate(self, value: object) -> None:
        super().validate(value)
        if isinstance(value, int) and value % 64:
            raise ValueError("two-stage resolution must be divisible by 64")

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "multiple_of": 64}


@dataclass(frozen=True)
class VideoFrameCountConstraint(IntegerConstraint):
    """LTX frame counts align with temporal VAE compression."""

    minimum: float | None = 1

    def validate(self, value: object) -> None:
        super().validate(value)
        if isinstance(value, int) and value % VIDEO_VAE_SCALE.time != 1:
            raise ValueError(f"frames must be 8*k + 1 for LTX-2, got {value}")

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "modulus": VIDEO_VAE_SCALE.time, "remainder": 1}


@config_record
@dataclass(frozen=True)
class DistilledRequest:
    """Resolved public distilled LTX-2 generation parameters."""

    config_table: ClassVar[str] = "[generate]"

    prompt: str = ""
    prompt_unconditional: bool = field(
        default=False, metadata={"config_rules": (UNCONDITIONAL_PROMPT,)}
    )
    width: int = field(default=1024, metadata={"config_constraints": (VideoDimensionConstraint(),)})
    height: int = field(default=576, metadata={"config_constraints": (VideoDimensionConstraint(),)})
    frames: int | None = field(
        default=121, metadata={"config_constraints": (VideoFrameCountConstraint(),)}
    )
    duration: float | None = field(
        default=None,
        metadata={"config_constraints": (NumberConstraint(minimum=0, exclusive_minimum=True),)},
    )
    fps: float = field(
        default=NATIVE_FPS,
        metadata={"config_constraints": (NumberConstraint(minimum=0, exclusive_minimum=True),)},
    )
    seed: int = field(
        default=42,
        metadata={"config_constraints": (IntegerConstraint(minimum=0, maximum=2**64 - 1),)},
    )
    seeds: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "config_constraints": (
                NumberSequenceConstraint(
                    minimum_length=1, item=IntegerConstraint(minimum=0, maximum=2**64 - 1)
                ),
            )
        },
    )
    noise_backend: NoiseBackend = field(
        default=DEFAULT_NOISE_BACKEND,
        metadata={"config_constraints": (ChoiceConstraint(NOISE_BACKEND_CHOICES),)},
    )
    sampler: DistilledSampler = field(
        default="auto",
        metadata={"config_constraints": (ChoiceConstraint(DISTILLED_SAMPLER_CHOICES),)},
    )
    stage_1_sigmas: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (SigmaScheduleConstraint(),)}
    )
    stage_2_sigmas: tuple[float, ...] = field(
        default=(), metadata={"config_constraints": (SigmaScheduleConstraint(),)}
    )
    hdr: HDRAuthoring | None = field(
        default=None,
        metadata={
            "config_constraints": (ChoiceConstraint(HDR_AUTHORING_CHOICES),),
            "config_rules": (HDR_SOURCES,),
        },
    )
    generate_audio: bool = field(
        default=False, metadata={"config_rules": (DISTILLED_CAPABILITIES,)}
    )
    reference_aligned_audio: bool = False
    image: ImageConditioningConfig | None = field(
        default=None, metadata={"config_rules": (IMAGE_FRAME_BOUNDS,)}
    )
    hdr_reference: HDRReferenceConditioningConfig | None = None
    generated_keyframes: int = field(
        default=0,
        metadata={
            "config_constraints": (IntegerConstraint(minimum=0),),
            "config_rules": (KEYFRAME_POSITIONS,),
        },
    )
    text_conditioning: Path | None = None
    vae_decode_dtype: VideoVAEDecodeDType = field(
        default="auto",
        metadata={"config_constraints": (ChoiceConstraint(VAE_DECODE_DTYPE_CHOICES),)},
    )
    vae_tiling: VideoVAETilingConfig = field(default_factory=VideoVAETilingConfig)
    pad_prompt_to_max: bool = True
    adapters: tuple[LTX2AdapterConfig, ...] = ()

    def selected_seeds(self) -> tuple[int, ...]:
        """Ordered seeds for each prompt."""
        return resolve_seeds(self.seed, self.seeds)

    def __post_init__(self) -> None:
        if self.duration is not None:
            resolved_frames = resolved_frame_count_for_duration(self.duration, self.fps)
            object.__setattr__(self, "frames", resolved_frames)
        self.resolved_adapters()

    def resolved_adapters(self) -> tuple[LTX2AdapterSelection, ...]:
        """Resolve ordered records with independent common and stage strengths."""
        return tuple(
            adapter.selection(location=f"[generate].adapters[{index}]")
            for index, adapter in enumerate(self.adapters)
        )

    @property
    def adapter_stage1_profile(self) -> LoRAProfile:
        """The effective first-stage profile, including zero-strength filtering."""
        return lora_profile_for_stage(self.resolved_adapters(), 1)

    @property
    def adapter_stage_profiles(self) -> tuple[LoRAProfile, LoRAProfile]:
        """Both completed-stage profiles used by decode restart policy."""
        selections = self.resolved_adapters()
        return lora_profile_for_stage(selections, 1), lora_profile_for_stage(selections, 2)

    def validate_for_generation(self) -> None:
        """Validate fields that may be omitted while printing a template."""
        validate_record(self, location="[generate]")
        if (
            self.text_conditioning is None
            and not self.prompt_unconditional
            and (not isinstance(self.prompt, str) or not self.prompt.strip())
        ):
            raise ValueError("prompt or text_conditioning is required")
        if self.text_conditioning is not None and not self.text_conditioning.is_file():
            raise ValueError(f"text conditioning does not exist: {self.text_conditioning}")
        if self.image is not None and not self.image.path.is_file():
            raise ValueError(f"image does not exist: {self.image.path}")
        if self.hdr_reference is not None and not self.hdr_reference.path.is_file():
            raise ValueError(f"HDR reference video does not exist: {self.hdr_reference.path}")
        for selection in self.resolved_adapters():
            if not selection.path.is_file():
                raise ValueError(f"adapter does not exist: {selection.path}")


def video_latent_shape_from_pixel(pixel: VideoPixelShape) -> VideoLatentShape:
    """Project a pixel-space video shape into LTX-2.3 VAE latent space."""
    return VideoLatentShape(
        batch=pixel.batch,
        channels=VIDEO_LATENT_CHANNELS,
        frames=(pixel.frames - 1) // VIDEO_VAE_SCALE.time + 1,
        height=pixel.height // VIDEO_VAE_SCALE.height,
        width=pixel.width // VIDEO_VAE_SCALE.width,
    )


# ---------------------------------------------------------------------------
# Audio model constants and shape
# ---------------------------------------------------------------------------


# Audio VAE encoder geometry: 16 kHz waveform, 160-sample hop, additional
# 4x downsample on the spectrogram, 8 latent channels, 16 mel bins.
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_HOP_LENGTH: int = 160
AUDIO_LATENT_DOWNSAMPLE: int = 4
AUDIO_LATENT_CHANNELS: int = 8
AUDIO_MEL_BINS: int = 16

# Audio latents per second of waveform, derived from the encoder geometry.
AUDIO_LATENTS_PER_SECOND: float = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_LATENT_DOWNSAMPLE


class AudioLatentShape(NamedTuple):
    """Dimensions of an LTX-2.3 audio latent: ``(batch, channels, frames, mel_bins)``.

    LTX-2.3-specific - every dimension assumes the model's audio VAE
    encoder geometry (16 kHz / 160 hop / 4x downsample / 8 channels /
    16 mel bins).
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.mel_bins)

    @classmethod
    def from_tuple(cls, shape: tuple[int, int, int, int]) -> AudioLatentShape:
        return cls(*shape)

    @classmethod
    def from_duration(
        cls,
        batch: int,
        duration_seconds: float,
        *,
        reference_aligned: bool = False,
    ) -> AudioLatentShape:
        latent_frames = duration_seconds * AUDIO_LATENTS_PER_SECOND
        # The reference uses nearest-integer rounding on the nominal 25 Hz audio
        # latent grid. KinoMLX biases upward by default because the causal decoder
        # emits ``4 * latent_frames - 3`` mel frames, so rounding down can leave
        # the decoded waveform short of the requested video timeline.
        return cls(
            batch=batch,
            channels=AUDIO_LATENT_CHANNELS,
            frames=round(latent_frames) if reference_aligned else math.ceil(latent_frames),
            mel_bins=AUDIO_MEL_BINS,
        )

    @classmethod
    def from_video(
        cls,
        pixel: VideoPixelShape,
        fps: float = NATIVE_FPS,
        *,
        reference_aligned: bool = False,
    ) -> AudioLatentShape:
        """Audio shape sized to a pixel-video shape's duration at ``fps``."""
        return cls.from_duration(
            batch=pixel.batch,
            duration_seconds=pixel.frames / fps,
            reference_aligned=reference_aligned,
        )
