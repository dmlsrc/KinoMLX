"""Public LTX-2 component leases and the default native provider."""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx

from kinomlx.adapters.lora import LoRAProfile
from kinomlx.adapters.validation import AdapterWeightCheck
from kinomlx.components import ComponentLease
from kinomlx.reporting import NullReporter, Reporter

from .adapters.loading import fuse_community_loras_into_model, streamed_adapter_check
from .adapters.receipt import LTX2AdapterReceipt
from .audio_vae.decoder import AudioDecoder, create_audio_decoder_from_checkpoint
from .audio_vae.loading import load_audio_decoder_weights
from .audio_vae.vocoder import VocoderWithBWE, create_vocoder_from_checkpoint
from .audio_vae.vocoder_loading import load_vocoder_weights
from .duration import DurationHead
from .duration import load_duration_head as _load_duration_head_model
from .resources import ComponentKind, LTX2Resources
from .source_loading import family_source_weights, transformer_source_loader
from .transformer import LTXAVModel, Modality, X0Model
from .upscaler import SpatialUpscaler, TemporalUpscaler
from .upscaler import load_spatial_upscaler as _load_spatial_upscaler_model
from .upscaler import load_temporal_upscaler as _load_temporal_upscaler_model
from .upscaler.ports import (
    LTX2LatentStatisticsPort,
    LTX2SpatialUpscalerPort,
    LTX2TemporalUpscalerPort,
)
from .video_vae.decoder import (
    Conv3dVideoDecoder,
    load_vae_decoder_weights,
)
from .video_vae.diffusion_decoder import (
    DiffusionVideoDecoder,
    load_diffusion_video_decoder_weights,
)
from .video_vae.encoder import (
    Conv3dVideoEncoder,
    WeightSource,
    load_vae_encoder_statistics,
    load_vae_encoder_weights,
)
from .video_vae.kinds import VideoVAEKind
from .video_vae.ops import PerChannelStatistics
from .weights.loading import bind_transformer

_log = logging.getLogger(__name__)


class LTX2TransformerPort(Protocol):
    """Denoised-prediction transformer surface consumed by recipes."""

    adapter_receipts: tuple[LTX2AdapterReceipt, ...]

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None = None,
    ) -> tuple[mx.array | None, mx.array | None]: ...


class _VideoEncoderCallablePort(Protocol):
    """Callable-only VAE encoder surface used by conditioning helpers."""

    def __call__(self, video: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class LTX2VideoEncoderPort(_VideoEncoderCallablePort, Protocol):
    @property
    def per_channel_statistics(self) -> LTX2LatentStatisticsPort: ...


class LTX2DurationPredictorPort(Protocol):
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


class LTX2AudioDecoderPort(Protocol):
    def __call__(
        self,
        latent: mx.array,
        /,
        *,
        reporter: Reporter | None = None,
    ) -> mx.array: ...


class LTX2VocoderPort(Protocol):
    @property
    def output_sample_rate(self) -> int: ...

    def __call__(self, mel: mx.array, *, reporter: Reporter | None = None) -> mx.array: ...


class LTX2VideoDecoderPort(Protocol):
    def __call__(
        self,
        latent: mx.array,
        *,
        timestep: float | None = 0.05,
        causal: bool | None = None,
        reporter: Reporter | None = None,
    ) -> mx.array: ...


class LTX2TransformerProvider(Protocol):
    def transformer(
        self,
        resources: LTX2Resources,
        profile: LoRAProfile = (),
    ) -> ComponentLease[LTX2TransformerPort]: ...


class LTX2VideoEncoderProvider(Protocol):
    def video_encoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2VideoEncoderPort]: ...


class LTX2SpatialUpscalerProvider(Protocol):
    def spatial_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2SpatialUpscalerPort]: ...


class LTX2TemporalUpscalerProvider(Protocol):
    def temporal_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2TemporalUpscalerPort]: ...


class LTX2DurationPredictorProvider(Protocol):
    def duration_predictor(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2DurationPredictorPort]: ...


class LTX2AudioDecoderProvider(Protocol):
    def audio_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2AudioDecoderPort]: ...


class LTX2VocoderProvider(Protocol):
    def vocoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2VocoderPort]: ...


class LTX2VideoDecoderProvider(Protocol):
    def video_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2VideoDecoderPort]: ...


class LTX2ComponentsProvider(
    LTX2TransformerProvider,
    LTX2VideoEncoderProvider,
    LTX2SpatialUpscalerProvider,
    LTX2DurationPredictorProvider,
    LTX2AudioDecoderProvider,
    LTX2VocoderProvider,
    LTX2VideoDecoderProvider,
    Protocol,
):
    """Native or injected component surface needed by the distilled recipe."""


def _cleanup_mlx() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _cleanup_video_decoder() -> None:
    # The decode stream retains allocations across tiles and returns the
    # process cache limit to its prior value when it closes.
    gc.collect()
    mx.synchronize()


def _family_weights(resources: LTX2Resources, kind: ComponentKind) -> WeightSource:
    """Load the component family from its source tensors."""
    return family_source_weights(resources, kind)


def _source_path(resources: LTX2Resources, kind: ComponentKind) -> Path:
    return resources.require(kind).source_path


def load_transformer(
    resources: LTX2Resources,
    loras: LoRAProfile = (),
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[LTX2TransformerPort]:
    """Load one pristine transformer from source and fuse one full profile."""
    execution = resources.execution_policy
    policy = resources.weight_policy
    config = resources.transformer_config
    if config is None:
        raise ValueError("prepared resources have no transformer constructor config")
    profile = tuple(loras)
    model: LTXAVModel | None = None
    try:
        model = LTXAVModel.from_config(
            config,
            compute_dtype=resources.dtype_policy.transformer,
            compile_attention=execution.compile_attention,
            dense_attention_kernel=execution.dense_attention_kernel,
            fast_mode=execution.fast_mode,
            compile_block_group_size=execution.compile_block_group_size,
            transformer_compile_group_size=execution.transformer_compile_group_size,
        )
        source_loader = transformer_source_loader(resources, adapters=profile)
        bind_transformer(
            model,
            weight_loader=source_loader.load,
            transformer_dtype=resources.dtype_policy.transformer,
            include_audio=policy.include_audio,
            resident_blocks=policy.resident_blocks,
            transformer_quantize=policy.transformer_quantize,
            video_ff_quantize_specs=policy.video_ff_quantize_specs,
            video_ff_quantize_group_size=policy.video_ff_quantize_group_size,
            video_ff_quantize_bits=policy.video_ff_quantize_bits,
        )
        pending = AdapterWeightCheck(enabled=execution.check_adapter_weights)
        adapter_receipts: tuple[LTX2AdapterReceipt, ...] = ()
        if profile:
            adapter_receipts = fuse_community_loras_into_model(
                model,
                profile,
                model_generation=config.model_generation,
                pending=pending,
                prepared_targets=source_loader.prepared_targets,
                accumulation=policy.adapter_accumulation,
            )
        if model.transformer_block_streamer is None and execution.compile_block_group_size is None:
            model.enable_mpp_weight_conversion()
        if source_loader.streamed_targets and model.transformer_block_streamer is not None:
            model.transformer_block_streamer.on_evaluated = streamed_adapter_check(
                source_loader.streamed_targets,
                enabled=execution.check_adapter_weights,
                reporter=reporter or NullReporter(),
            )
        loaded_model: LTXAVModel = model
        wrapped: LTX2TransformerPort = X0Model(loaded_model)
        # Receipts are run metadata, not MLX module state. Bypass nn.Module's
        # tuple registration while keeping the field explicit on the port.
        object.__setattr__(wrapped, "adapter_receipts", adapter_receipts)
        object.__setattr__(wrapped, "adapter_weight_check", pending)
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
) -> ComponentLease[LTX2VideoEncoderPort]:
    """Load the native Conv3d encoder from the video family source."""
    del reporter
    model: Conv3dVideoEncoder | None = None
    try:
        model = Conv3dVideoEncoder(
            resources.video_vae_config,
            compute_dtype=resources.dtype_policy.video_vae,
        )
        load_vae_encoder_weights(
            model,
            _family_weights(resources, ComponentKind.VIDEO_VAE),
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


def _load_vae_statistics(resources: LTX2Resources) -> PerChannelStatistics:
    return load_vae_encoder_statistics(
        family_source_weights(
            resources,
            ComponentKind.VIDEO_VAE,
            select=lambda key: ".per_channel_statistics." in f".{key}",
        ),
        latent_channels=resources.video_vae_config.latent_channels,
    ).per_channel_statistics


def load_spatial_upscaler(
    resources: LTX2Resources,
    *,
    reporter: Reporter | None = None,
) -> ComponentLease[LTX2SpatialUpscalerPort]:
    """Load the upscaler plus only the VAE statistics it consumes."""
    model: SpatialUpscaler | None = None
    try:
        model = _load_spatial_upscaler_model(
            resources.spatial_upscaler_path,
            compute_dtype=resources.dtype_policy.spatial_upscaler,
            reporter=reporter,
        )
        statistics = _load_vae_statistics(resources)
        component: LTX2SpatialUpscalerPort = _SpatialUpscalerComponent(
            model=model,
            per_channel_statistics=statistics,
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
) -> ComponentLease[LTX2TemporalUpscalerPort]:
    """Load the optional temporal x2 model plus its VAE statistics payload."""
    model: TemporalUpscaler | None = None
    try:
        model = _load_temporal_upscaler_model(
            resources.temporal_upscaler_path,
            compute_dtype=resources.dtype_policy.temporal_upscaler,
            reporter=reporter,
        )
        statistics = _load_vae_statistics(resources)
        component: LTX2TemporalUpscalerPort = _TemporalUpscalerComponent(
            model=model,
            per_channel_statistics=statistics,
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
) -> ComponentLease[LTX2DurationPredictorPort]:
    """Load the optional natural-duration head as a bounded component lease."""
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
) -> ComponentLease[LTX2AudioDecoderPort]:
    """Load one audio VAE decoder from the audio family source."""
    model: AudioDecoder | None = None
    try:
        model = create_audio_decoder_from_checkpoint(
            _source_path(resources, ComponentKind.AUDIO_VAE),
            compute_dtype=resources.dtype_policy.audio_vae,
        )
        load_audio_decoder_weights(
            model,
            _family_weights(resources, ComponentKind.AUDIO_VAE),
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
) -> ComponentLease[LTX2VocoderPort]:
    """Load one vocoder from the audio family source."""
    model: VocoderWithBWE | None = None
    try:
        model = create_vocoder_from_checkpoint(_source_path(resources, ComponentKind.VOCODER))
        load_vocoder_weights(
            model,
            _family_weights(resources, ComponentKind.VOCODER),
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
) -> ComponentLease[LTX2VideoDecoderPort]:
    """Load the metadata-selected native decoder from the video family source."""
    model: Conv3dVideoDecoder | DiffusionVideoDecoder | None = None
    sink = reporter if reporter is not None else NullReporter()
    phase = (
        "load diffusion video decoder"
        if resources.video_vae_config.decoder_kind == VideoVAEKind.DIFFUSION
        else "load video decoder"
    )
    sink.phase_start(phase, total=1, unit="file")
    try:
        if resources.video_vae_config.decoder_kind == VideoVAEKind.DIFFUSION:
            model = DiffusionVideoDecoder(
                resources.video_vae_config,
                compute_dtype=resources.dtype_policy.video_vae,
            )
            load_diffusion_video_decoder_weights(
                model,
                _family_weights(resources, ComponentKind.VIDEO_VAE),
            )
        else:
            model = Conv3dVideoDecoder(
                resources.video_vae_config,
                compute_dtype=resources.dtype_policy.video_vae,
            )
            load_vae_decoder_weights(
                model,
                _family_weights(resources, ComponentKind.VIDEO_VAE),
            )
        # The lease boundary and its VAE-entry receipt must describe resident
        # decoder assets, not a lazy graph that allocates weights on first pull.
        mx.eval(model.parameters())
        sink.phase_advance(phase)
        return ComponentLease(model, cleanup=_cleanup_video_decoder)
    except BaseException:
        model = None
        _cleanup_mlx()
        raise
    finally:
        sink.phase_end(phase)


@dataclass(frozen=True)
class LTX2Components:
    """Default provider implementing the public structural component surface."""

    reporter: Reporter | None = None

    def transformer(
        self,
        resources: LTX2Resources,
        profile: LoRAProfile = (),
    ) -> ComponentLease[LTX2TransformerPort]:
        return load_transformer(
            resources,
            profile,
            reporter=self.reporter,
        )

    def video_encoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2VideoEncoderPort]:
        return load_video_encoder(resources, reporter=self.reporter)

    def spatial_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2SpatialUpscalerPort]:
        return load_spatial_upscaler(resources, reporter=self.reporter)

    def temporal_upscaler(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2TemporalUpscalerPort]:
        return load_temporal_upscaler(resources, reporter=self.reporter)

    def duration_predictor(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2DurationPredictorPort]:
        return load_duration_predictor(resources, reporter=self.reporter)

    def audio_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2AudioDecoderPort]:
        return load_audio_decoder(resources, reporter=self.reporter)

    def vocoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2VocoderPort]:
        return load_vocoder(resources, reporter=self.reporter)

    def video_decoder(
        self,
        resources: LTX2Resources,
    ) -> ComponentLease[LTX2VideoDecoderPort]:
        return load_video_decoder(resources, reporter=self.reporter)


__all__ = [
    "LTX2AudioDecoderPort",
    "LTX2AudioDecoderProvider",
    "LTX2Components",
    "LTX2ComponentsProvider",
    "LTX2DurationPredictorPort",
    "LTX2DurationPredictorProvider",
    "LTX2SpatialUpscalerProvider",
    "LTX2TemporalUpscalerProvider",
    "LTX2TransformerPort",
    "LTX2TransformerProvider",
    "LTX2VideoDecoderPort",
    "LTX2VideoDecoderProvider",
    "LTX2VideoEncoderPort",
    "LTX2VideoEncoderProvider",
    "LTX2VocoderPort",
    "LTX2VocoderProvider",
    "load_audio_decoder",
    "load_duration_predictor",
    "load_spatial_upscaler",
    "load_temporal_upscaler",
    "load_transformer",
    "load_video_decoder",
    "load_video_encoder",
    "load_vocoder",
]
