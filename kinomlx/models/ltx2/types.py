"""LTX-2.3 type constants and shape derivations.

These are LTX-2.3 facts - native frame rate, VAE compression ratios,
audio encoder geometry, latent channel counts - and the shape
derivations that depend on them.  Top-level :mod:`kinomlx.types`
carries only model-agnostic primitives so future models can reuse
them without inheriting LTX-2's numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

if TYPE_CHECKING:
    from .settings import LTX2Settings
    from .video_vae.tiling import TilingConfig

from kinomlx.types import (
    DEFAULT_NOISE_BACKEND,
    NOISE_BACKEND_CHOICES,
    NoiseBackend,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)

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

# RGB output of the video VAE decode.
VIDEO_PIXEL_CHANNELS: int = 3

# ``auto`` preserves the checkpoint generation's published sampler behavior.
# Explicit modes are useful for controlled A/B diagnostics without changing
# checkpoint selection or the literal sigma schedules.
DISTILLED_SAMPLER_CHOICES = ("auto", "deterministic", "ancestral")
DistilledSampler = Literal["auto", "deterministic", "ancestral"]

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
VAE_DECODE_DTYPE_CHOICES = ("auto", "bfloat16", "float32")
VideoVAEDecodeDType = Literal["auto", "bfloat16", "float32"]

VAE_TILING_MODE_CHOICES = ("auto", "single", "custom")

LORA_EXCLUDE_CATEGORIES = frozenset(
    {
        "video",
        "audio",
        "cross",
        "attn",
        "gate",
        "ff",
        "attn1",
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "video_to_audio_attn",
        "audio_to_video_attn",
        "audio_ff",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "to_gate_logits",
        "project_in",
        "project_out",
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


@dataclass(frozen=True)
class ImageConditioningConfig:
    """One image-conditioning input for the distilled pipeline."""

    path: Path
    frame_index: int = 0
    strength: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            object.__setattr__(self, "path", Path(self.path))
        if self.frame_index < 0:
            raise ValueError("image frame_index must be non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("image strength must be between 0 and 1")


@dataclass(frozen=True)
class HDRReferenceConditioningConfig:
    """One SDR reference video for the explicit LTX-2.3 HDR IC-LoRA recipe."""

    path: Path
    strength: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            object.__setattr__(self, "path", Path(self.path))
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("HDR reference strength must be between 0 and 1")


@dataclass(frozen=True)
class VideoVAETilingConfig:
    """Resolved user policy for memory-bounded video VAE decoding.

    ``auto`` lets the native decoder select a memory-aware plan, ``single``
    forces one decode, and ``custom`` builds the requested temporal/spatial
    plan. Runtime tiling types are imported only when decoding so importing the
    data-only CLI schema does not initialize MLX.
    """

    mode: Literal["auto", "single", "custom"] = "auto"
    temporal_tile_frames: int | None = None
    temporal_overlap_frames: int = 24
    spatial_tile_pixels: int | None = None
    spatial_overlap_pixels: int = 64

    def __post_init__(self) -> None:
        if self.mode not in VAE_TILING_MODE_CHOICES:
            valid = ", ".join(VAE_TILING_MODE_CHOICES)
            raise ValueError(f"VAE tiling mode must be one of: {valid}")
        for name in (
            "temporal_tile_frames",
            "temporal_overlap_frames",
            "spatial_tile_pixels",
            "spatial_overlap_pixels",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"VAE {name} must be an integer")
            if value is not None and value < 0:
                raise ValueError(f"VAE {name} must be non-negative")

        temporal_size = self.temporal_tile_frames
        if self.mode == "custom" and temporal_size is None:
            temporal_size = 256
        if temporal_size not in (None, 0):
            if temporal_size < 16 or temporal_size % VIDEO_VAE_SCALE.time:
                raise ValueError("VAE temporal_tile_frames must be at least 16 and divisible by 8")
            if self.temporal_overlap_frames >= temporal_size:
                raise ValueError("VAE temporal overlap must be smaller than its tile size")
        if self.temporal_overlap_frames % VIDEO_VAE_SCALE.time:
            raise ValueError("VAE temporal_overlap_frames must be divisible by 8")

        spatial_size = self.spatial_tile_pixels
        if spatial_size not in (None, 0):
            if spatial_size < 64 or spatial_size % VIDEO_VAE_SCALE.height:
                raise ValueError("VAE spatial_tile_pixels must be at least 64 and divisible by 32")
            if self.spatial_overlap_pixels >= spatial_size:
                raise ValueError("VAE spatial overlap must be smaller than its tile size")
        if self.spatial_overlap_pixels % VIDEO_VAE_SCALE.height:
            raise ValueError("VAE spatial_overlap_pixels must be divisible by 32")

        if self.mode == "custom" and temporal_size == 0 and spatial_size in (None, 0):
            raise ValueError("custom VAE tiling needs a temporal or spatial tile size")

    def to_runtime_config(self) -> TilingConfig | None:
        """Build the decoder's native tiling config for this policy."""
        from .video_vae.tiling import (
            SpatialTilingConfig,
            TemporalChunkConfig,
            TilingConfig,
        )

        if self.mode == "auto":
            return None
        if self.mode == "single":
            # A non-None empty plan bypasses decode_streaming's auto planner.
            return TilingConfig()

        temporal_size = 256 if self.temporal_tile_frames is None else self.temporal_tile_frames
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
class LoRASelection:
    """One fully expanded per-run community adapter selection."""

    path: Path
    strength: float
    stage_1_strength: float | None
    stage_2_strength: float | None
    exclude: tuple[str, ...]


@dataclass(frozen=True)
class DistilledRequest:
    """Resolved public distilled LTX-2 generation parameters."""

    prompt: str = ""
    width: int = 1024
    height: int = 576
    frames: int | None = 121
    duration: float | None = None
    fps: float = NATIVE_FPS
    seed: int = 42
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND
    sampler: DistilledSampler = "auto"
    hdr: HDRAuthoring | None = None
    generate_audio: bool = False
    reference_aligned_audio: bool = False
    image: ImageConditioningConfig | None = None
    hdr_reference: HDRReferenceConditioningConfig | None = None
    generated_keyframes: int = 0
    text_conditioning: Path | None = None
    vae_decode_dtype: VideoVAEDecodeDType = "auto"
    vae_tiling: VideoVAETilingConfig = field(default_factory=VideoVAETilingConfig)
    pad_prompt_to_max: bool = True
    lora_paths: tuple[Path, ...] = ()
    lora_strengths: tuple[float, ...] = ()
    lora_stage1_strengths: tuple[float, ...] = ()
    lora_stage2_strengths: tuple[float, ...] = ()
    lora_exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.seed >= 2**64:
            raise ValueError("seed must be smaller than 2**64")
        if self.noise_backend not in NOISE_BACKEND_CHOICES:
            choices = ", ".join(NOISE_BACKEND_CHOICES)
            raise ValueError(f"noise_backend must be one of: {choices}")
        if self.sampler not in DISTILLED_SAMPLER_CHOICES:
            choices = ", ".join(DISTILLED_SAMPLER_CHOICES)
            raise ValueError(f"sampler must be one of: {choices}")
        if self.hdr not in (None, *HDR_AUTHORING_CHOICES):
            choices = ", ".join(HDR_AUTHORING_CHOICES)
            raise ValueError(f"hdr must be one of: {choices}")
        if self.vae_decode_dtype not in VAE_DECODE_DTYPE_CHOICES:
            choices = ", ".join(VAE_DECODE_DTYPE_CHOICES)
            raise ValueError(f"vae_decode_dtype must be one of: {choices}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if self.duration is not None:
            if not math.isfinite(self.duration) or self.duration <= 0:
                raise ValueError("duration must be finite and positive")
            resolved_frames = resolved_frame_count_for_duration(self.duration, self.fps)
            object.__setattr__(self, "frames", resolved_frames)
        if isinstance(self.text_conditioning, str):
            object.__setattr__(self, "text_conditioning", Path(self.text_conditioning))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 64 != 0 or self.height % 64 != 0:
            raise ValueError(
                f"two-stage resolution {self.width}x{self.height} must be divisible by 64"
            )
        if self.frames is not None:
            if isinstance(self.frames, bool) or not isinstance(self.frames, int):
                raise ValueError("frames must be an integer or None for auto-duration")
            if self.frames < 1 or self.frames % VIDEO_VAE_SCALE.time != 1:
                raise ValueError(f"frames must be 8*k + 1 for LTX-2, got {self.frames}")
        if (
            isinstance(self.generated_keyframes, bool)
            or not isinstance(self.generated_keyframes, int)
            or self.generated_keyframes < 0
        ):
            raise ValueError("generated_keyframes must be a non-negative integer")
        if (
            self.image is not None
            and self.frames is not None
            and self.image.frame_index >= self.frames
        ):
            raise ValueError(
                f"image frame_index {self.image.frame_index} is outside {self.frames} frames"
            )
        if not self.lora_paths and any(
            (
                self.lora_strengths,
                self.lora_stage1_strengths,
                self.lora_stage2_strengths,
                self.lora_exclusions,
            )
        ):
            raise ValueError("LoRA strengths or exclusions require lora_paths")
        self.resolved_loras()

    @staticmethod
    @overload
    def _expand_lora_values(
        values: tuple[float, ...],
        count: int,
        *,
        label: str,
        default: float,
    ) -> tuple[float, ...]: ...

    @staticmethod
    @overload
    def _expand_lora_values(
        values: tuple[float, ...],
        count: int,
        *,
        label: str,
        default: None,
    ) -> tuple[float | None, ...]: ...

    @staticmethod
    def _expand_lora_values(
        values: tuple[float, ...],
        count: int,
        *,
        label: str,
        default: float | None,
    ) -> tuple[float | None, ...]:
        if not values:
            return (default,) * count
        if len(values) == 1:
            return values * count
        if len(values) != count:
            raise ValueError(
                f"{label} must contain one value or one per LoRA ({count}), got {len(values)}"
            )
        return values

    @staticmethod
    def _parse_lora_exclusion(value: str) -> tuple[str, ...]:
        if value.strip().lower() in {"", "none", "off"}:
            return ()
        categories = tuple(category.strip() for category in value.split(",") if category.strip())
        unknown = sorted(set(categories) - LORA_EXCLUDE_CATEGORIES)
        if unknown:
            valid = ", ".join(sorted(LORA_EXCLUDE_CATEGORIES))
            raise ValueError(f"unknown LoRA exclusion categories {unknown}; valid values: {valid}")
        return categories

    def resolved_loras(self) -> tuple[LoRASelection, ...]:
        """Expand one-or-N LoRA values into ordered adapter selections."""
        count = len(self.lora_paths)
        if count == 0:
            return ()
        strengths = self._expand_lora_values(
            self.lora_strengths,
            count,
            label="lora_strengths",
            default=1.0,
        )
        stage_1 = self._expand_lora_values(
            self.lora_stage1_strengths,
            count,
            label="lora_stage1_strengths",
            default=None,
        )
        stage_2 = self._expand_lora_values(
            self.lora_stage2_strengths,
            count,
            label="lora_stage2_strengths",
            default=None,
        )
        exclusions: tuple[tuple[str, ...], ...]
        if not self.lora_exclusions:
            exclusions = ((),) * count
        elif len(self.lora_exclusions) == 1:
            exclusions = (self._parse_lora_exclusion(self.lora_exclusions[0]),) * count
        elif len(self.lora_exclusions) == count:
            exclusions = tuple(self._parse_lora_exclusion(value) for value in self.lora_exclusions)
        else:
            raise ValueError(
                "lora_exclusions must contain one value or one per LoRA "
                f"({count}), got {len(self.lora_exclusions)}"
            )

        selections = []
        for path, strength, stage_1_strength, stage_2_strength, exclude in zip(
            self.lora_paths,
            strengths,
            stage_1,
            stage_2,
            exclusions,
            strict=True,
        ):
            if not isinstance(path, Path):
                path = Path(path)
            normalized_strength = float(strength)
            normalized_stage_1 = None if stage_1_strength is None else float(stage_1_strength)
            normalized_stage_2 = None if stage_2_strength is None else float(stage_2_strength)
            for label, value in (
                ("strength", normalized_strength),
                ("stage 1 strength", normalized_stage_1),
                ("stage 2 strength", normalized_stage_2),
            ):
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"LoRA {label} must be finite, got {value}")
            selections.append(
                LoRASelection(
                    path=path,
                    strength=normalized_strength,
                    stage_1_strength=normalized_stage_1,
                    stage_2_strength=normalized_stage_2,
                    exclude=exclude,
                )
            )
        return tuple(selections)

    def validate_for_generation(self) -> None:
        """Validate fields that may be omitted while printing a template."""
        if self.text_conditioning is None and (
            not isinstance(self.prompt, str) or not self.prompt.strip()
        ):
            raise ValueError("prompt or text_conditioning is required")
        if self.text_conditioning is not None and not self.text_conditioning.is_file():
            raise ValueError(f"text conditioning does not exist: {self.text_conditioning}")
        if self.image is not None and not self.image.path.is_file():
            raise ValueError(f"image does not exist: {self.image.path}")
        if self.hdr_reference is not None and not self.hdr_reference.path.is_file():
            raise ValueError(f"HDR reference video does not exist: {self.hdr_reference.path}")
        for selection in self.resolved_loras():
            if not selection.path.is_file():
                raise ValueError(f"LoRA does not exist: {selection.path}")

    def validate_with_settings(self, settings: LTX2Settings) -> None:
        """Validate per-run features against the selected load/cache mode."""
        if not self.lora_paths:
            return
        if settings.stream_transformer or settings.transformer_resident_blocks is not None:
            raise ValueError("LoRA fusion cannot be combined with transformer block streaming")
        if settings.transformer_cache_quantize != "off":
            raise ValueError("LoRA fusion requires an unquantized transformer cache")
        if settings.video_ff_quantize_specs:
            raise ValueError("LoRA fusion cannot be combined with targeted FF quantization")


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
