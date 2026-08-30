"""Public LTX-2 component leases and the default native provider."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx

from kinomlx.components import ComponentLease
from kinomlx.lora.loading import LoRAProfile
from kinomlx.reporting import NullReporter, Reporter

from .audio_vae.decoder import AudioDecoder, create_audio_decoder_from_checkpoint
from .audio_vae.loading import load_audio_decoder_weights
from .audio_vae.vocoder import VocoderWithBWE, create_vocoder_from_checkpoint
from .audio_vae.vocoder_loading import load_vocoder_weights
from .cache import (
    LoRAAdapterReceipt,
    bind_transformer_cache,
    fuse_community_loras_into_model,
)
from .duration import DurationHead
from .duration import load_duration_head as _load_duration_head_model
from .resources import ComponentKind, LTX2Resources
from .transformer import LTXAVModel, Modality, X0Model
from .upscaler import SpatialUpscaler, TemporalUpscaler
from .upscaler import load_spatial_upscaler as _load_spatial_upscaler_model
from .upscaler import load_temporal_upscaler as _load_temporal_upscaler_model
from .video_vae.decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from .video_vae.diffusion_decoder import (
    NativeDiffusionVideoDecoder,
    load_diffusion_video_decoder_weights,
)
from .video_vae.encoder import (
    NativeConv3dVideoEncoder,
    load_native_vae_encoder_statistics,
    load_native_vae_encoder_weights,
)
from .video_vae.ops import PerChannelStatistics

_log = logging.getLogger(__name__)


class TransformerPort(Protocol):
    """Denoised-prediction transformer surface consumed by recipes."""

    lora_receipts: tuple[LoRAAdapterReceipt, ...]

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None = None,
    ) -> tuple[mx.array | None, mx.array | None]: ...


class LatentStatisticsPort(Protocol):
    """Affine normalization surface shared by encoder and upscaler ports."""

    def normalize(self, latent: mx.array) -> mx.array: ...

    def denormalize(self, latent: mx.array) -> mx.array: ...


class _VideoEncoderCallablePort(Protocol):
    """Callable-only VAE encoder surface used by conditioning helpers."""

    def __call__(self, video: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class VideoEncoderPort(_VideoEncoderCallablePort, Protocol):
    @property
    def per_channel_statistics(self) -> LatentStatisticsPort: ...


class SpatialUpscalerPort(Protocol):
    @property
    def per_channel_statistics(self) -> LatentStatisticsPort: ...

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class TemporalUpscalerPort(Protocol):
    @property
    def per_channel_statistics(self) -> LatentStatisticsPort: ...

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class DurationPredictorPort(Protocol):
    def predict_num_frames(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
        *,
        frame_rate: float,
        temporal_compression_ratio: int,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int: ...


class AudioDecoderPort(Protocol):
    def __call__(
        self,
        latent: mx.array,
        /,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array: ...


class VocoderPort(Protocol):
    @property
    def output_sample_rate(self) -> int: ...

    def __call__(self, mel: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class VideoDecoderPort(Protocol):
    def __call__(
        self,
        latent: mx.array,
        *,
        timestep: float | None = 0.05,
        causal: bool | None = None,
        reporter: Reporter | None = None,
    ) -> mx.array: ...


class TransformerProvider(Protocol):
    def transformer(
        self,
        resources: LTX2Resources,
        profile: LoRAProfile = (),
    ) -> ComponentLease[TransformerPort]: ...


class VideoEncoderProvider(Protocol):
    def video_encoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[VideoEncoderPort]: ...


class SpatialUpscalerProvider(Protocol):
    def spatial_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[SpatialUpscalerPort]: ...


class TemporalUpscalerProvider(Protocol):
    def temporal_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[TemporalUpscalerPort]: ...


class DurationPredictorProvider(Protocol):
    def duration_predictor(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[DurationPredictorPort]: ...


class AudioDecoderProvider(Protocol):
    def audio_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[AudioDecoderPort]: ...


class VocoderProvider(Protocol):
    def vocoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[VocoderPort]: ...


class VideoDecoderProvider(Protocol):
    def video_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[VideoDecoderPort]: ...


class DistilledComponents(
    TransformerProvider,
    VideoEncoderProvider,
    SpatialUpscalerProvider,
    DurationPredictorProvider,
    AudioDecoderProvider,
    VocoderProvider,
    VideoDecoderProvider,
    Protocol,
):
    """Native or injected component surface needed by the distilled recipe."""


def _cleanup_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _apply_cache_limit(resources: LTX2Resources) -> None:
    limit = resources.execution_policy.mlx_cache_limit_bytes
    if limit is not None:
        mx.set_cache_limit(limit)


def _cache_path(resources: LTX2Resources, kind: ComponentKind) -> Path:
    path = resources.require(kind).cache_path
    if path is None:
        raise LookupError(f"prepared {kind.value} component has no cache path")
    return path


def _source_path(resources: LTX2Resources, kind: ComponentKind) -> Path:
    return resources.require(kind).source_path


def load_transformer(
    resources: LTX2Resources,
    loras: LoRAProfile = (),
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[TransformerPort]:
    """Load one pristine prepared-cache transformer and fuse one full profile."""
    del reporter
    _apply_cache_limit(resources)
    execution = resources.execution_policy
    cache = resources.cache_policy
    config = resources.transformer_config
    if config is None:
        raise ValueError("prepared resources have no transformer constructor config")
    profile = tuple(loras)
    model: LTXAVModel | None = None
    try:
        model = LTXAVModel.from_config(
            config,
            compute_dtype=resources.dtype_policy.transformer,
            use_steel_attention=execution.use_steel_attention,
            compile_attention=execution.compile_attention,
            steel_attention_d64=execution.steel_attention_d64,
            steel_attention_probe=execution.steel_attention_probe,
            fast_mode=execution.fast_mode,
            compile_block_groups=execution.compile_block_groups,
            transformer_compile_group_size=execution.transformer_compile_group_size,
        )
        bind_transformer_cache(
            model,
            resources.transformer_cache_path,
            transformer_dtype=resources.dtype_policy.transformer,
            include_audio=cache.include_audio,
            resident_blocks=cache.resident_blocks,
            transformer_cache_quantize=cache.transformer_cache_quantize,
            video_ff_quantize_specs=cache.video_ff_quantize_specs,
            video_ff_quantize_group_size=cache.video_ff_quantize_group_size,
            video_ff_quantize_bits=cache.video_ff_quantize_bits,
        )
        lora_receipts: tuple[LoRAAdapterReceipt, ...] = ()
        if profile:
            lora_receipts = fuse_community_loras_into_model(
                model,
                profile,
                model_generation=config.model_generation,
                transformer_cache_path=resources.transformer_cache_path,
            )
        loaded_model: LTXAVModel = model
        wrapped: TransformerPort = X0Model(loaded_model)
        # Receipts are run metadata, not MLX module state. Bypass nn.Module's
        # tuple registration while keeping the field explicit on the port.
        object.__setattr__(wrapped, "lora_receipts", lora_receipts)
        return ComponentLease(
            wrapped,
            close_component=lambda _wrapped: loaded_model.close_streamer(),
            cleanup=_cleanup_mlx,
        )
    except BaseException:
        if model is not None:
            try:
                model.close_streamer()
            except Exception:
                _log.warning(
                    "Failed to close a partially loaded transformer streamer",
                    exc_info=True,
                )
        model = None
        _cleanup_mlx()
        raise


def load_video_encoder(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[VideoEncoderPort]:
    """Load the native Conv3d encoder from the prepared video-family cache."""
    del reporter
    _apply_cache_limit(resources)
    model: NativeConv3dVideoEncoder | None = None
    try:
        model = NativeConv3dVideoEncoder(
            resources.video_vae_config,
            compute_dtype=resources.dtype_policy.video_vae,
        )
        load_native_vae_encoder_weights(
            model,
            _cache_path(resources, ComponentKind.VIDEO_VAE),
        )
        mx.eval(model.parameters())
        return ComponentLease(model, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


@dataclass
class _SpatialUpscalerComponent:
    model: SpatialUpscaler
    per_channel_statistics: PerChannelStatistics

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        return self.model(latent, reporter=reporter)


@dataclass
class _TemporalUpscalerComponent:
    model: TemporalUpscaler
    per_channel_statistics: PerChannelStatistics

    def __call__(self, latent: mx.array, *, reporter: Reporter | None = None) -> mx.array:
        return self.model(latent, reporter=reporter)


def load_spatial_upscaler(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[SpatialUpscalerPort]:
    """Load the upscaler plus only the VAE statistics it consumes."""
    _apply_cache_limit(resources)
    model: SpatialUpscaler | None = None
    try:
        model = _load_spatial_upscaler_model(
            resources.spatial_upscaler_path,
            compute_dtype=resources.dtype_policy.spatial_upscaler,
            reporter=reporter,
        )
        statistics = load_native_vae_encoder_statistics(
            _cache_path(resources, ComponentKind.VIDEO_VAE),
            latent_channels=resources.video_vae_config.latent_channels,
        )
        component: SpatialUpscalerPort = _SpatialUpscalerComponent(
            model=model,
            per_channel_statistics=statistics.per_channel_statistics,
        )
        return ComponentLease(component, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_temporal_upscaler(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[TemporalUpscalerPort]:
    """Load the optional temporal x2 model plus its VAE statistics payload."""
    _apply_cache_limit(resources)
    model: TemporalUpscaler | None = None
    try:
        model = _load_temporal_upscaler_model(
            resources.temporal_upscaler_path,
            compute_dtype=resources.dtype_policy.temporal_upscaler,
            reporter=reporter,
        )
        statistics = load_native_vae_encoder_statistics(
            _cache_path(resources, ComponentKind.VIDEO_VAE),
            latent_channels=resources.video_vae_config.latent_channels,
        )
        component: TemporalUpscalerPort = _TemporalUpscalerComponent(
            model=model,
            per_channel_statistics=statistics.per_channel_statistics,
        )
        return ComponentLease(component, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_duration_predictor(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[DurationPredictorPort]:
    """Load the optional natural-duration head as a bounded component lease."""
    _apply_cache_limit(resources)
    model: DurationHead | None = None
    try:
        model = _load_duration_head_model(
            resources.duration_head_path,
            compute_dtype=resources.dtype_policy.duration_head,
            reporter=reporter,
        )
        return ComponentLease(model, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_audio_decoder(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[AudioDecoderPort]:
    """Load one audio VAE decoder from the prepared family cache."""
    _apply_cache_limit(resources)
    model: AudioDecoder | None = None
    try:
        model = create_audio_decoder_from_checkpoint(
            _source_path(resources, ComponentKind.AUDIO_VAE),
            compute_dtype=resources.dtype_policy.audio_vae,
        )
        load_audio_decoder_weights(
            model,
            _cache_path(resources, ComponentKind.AUDIO_VAE),
            reporter=reporter,
        )
        mx.eval(model.parameters())
        return ComponentLease(model, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_vocoder(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[VocoderPort]:
    """Load one vocoder from the prepared family cache."""
    _apply_cache_limit(resources)
    model: VocoderWithBWE | None = None
    try:
        model = create_vocoder_from_checkpoint(_source_path(resources, ComponentKind.VOCODER))
        load_vocoder_weights(
            model,
            _cache_path(resources, ComponentKind.VOCODER),
            reporter=reporter,
        )
        mx.eval(model.parameters())
        return ComponentLease(model, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise


def load_video_decoder(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[VideoDecoderPort]:
    """Load the metadata-selected native decoder from the video-family cache."""
    _apply_cache_limit(resources)
    model: NativeConv3dVideoDecoder | NativeDiffusionVideoDecoder | None = None
    sink = reporter if reporter is not None else NullReporter()
    phase = (
        "load diffusion video decoder"
        if resources.video_vae_config.decoder_kind == "diffusion-na"
        else "load video decoder"
    )
    sink.phase_start(phase, total=1, unit="file")
    try:
        if resources.video_vae_config.decoder_kind == "diffusion-na":
            model = NativeDiffusionVideoDecoder(
                resources.video_vae_config,
                compute_dtype=resources.dtype_policy.video_vae,
            )
            load_diffusion_video_decoder_weights(
                model,
                _cache_path(resources, ComponentKind.VIDEO_VAE),
            )
        else:
            model = NativeConv3dVideoDecoder(
                resources.video_vae_config,
                compute_dtype=resources.dtype_policy.video_vae,
            )
            load_native_vae_decoder_weights(
                model,
                _cache_path(resources, ComponentKind.VIDEO_VAE),
            )
        # The lease boundary and its VAE-entry receipt must describe resident
        # decoder assets, not a lazy graph that allocates weights on first pull.
        mx.eval(model.parameters())
        sink.phase_advance(phase)
        return ComponentLease(model, cleanup=_cleanup_mlx)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise
    finally:
        sink.phase_end(phase)


@dataclass(frozen=True)
class NativeLTX2Components:
    """Default provider implementing the public structural component surface."""

    reporter: Reporter | None = None

    def transformer(
        self,
        resources: LTX2Resources,
        profile: LoRAProfile = (),
    ) -> ComponentLease[TransformerPort]:
        return load_transformer(
            resources,
            profile,
            reporter=self.reporter,
        )

    def video_encoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[VideoEncoderPort]:
        return load_video_encoder(resources, reporter=self.reporter)

    def spatial_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[SpatialUpscalerPort]:
        return load_spatial_upscaler(resources, reporter=self.reporter)

    def temporal_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[TemporalUpscalerPort]:
        return load_temporal_upscaler(resources, reporter=self.reporter)

    def duration_predictor(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[DurationPredictorPort]:
        return load_duration_predictor(resources, reporter=self.reporter)

    def audio_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[AudioDecoderPort]:
        return load_audio_decoder(resources, reporter=self.reporter)

    def vocoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[VocoderPort]:
        return load_vocoder(resources, reporter=self.reporter)

    def video_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[VideoDecoderPort]:
        return load_video_decoder(resources, reporter=self.reporter)


__all__ = [
    "AudioDecoderPort",
    "AudioDecoderProvider",
    "DistilledComponents",
    "DurationPredictorPort",
    "DurationPredictorProvider",
    "LatentStatisticsPort",
    "NativeLTX2Components",
    "SpatialUpscalerPort",
    "SpatialUpscalerProvider",
    "TemporalUpscalerPort",
    "TemporalUpscalerProvider",
    "TransformerPort",
    "TransformerProvider",
    "VideoDecoderPort",
    "VideoDecoderProvider",
    "VideoEncoderPort",
    "VideoEncoderProvider",
    "VocoderPort",
    "VocoderProvider",
    "load_audio_decoder",
    "load_duration_predictor",
    "load_spatial_upscaler",
    "load_transformer",
    "load_temporal_upscaler",
    "load_video_decoder",
    "load_video_encoder",
    "load_vocoder",
]
