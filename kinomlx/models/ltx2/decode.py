"""Pure audio operations and LTX-specific lazy video decode composition."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from kinomlx.components import ComponentLease
from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.signals import VideoSignalSpec
from kinomlx.reporting import Reporter
from kinomlx.samplers.noise import NoiseStreamState, noise_compatibility_profile
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend

from .components import AudioDecoderPort, VideoDecoderPort, VocoderPort
from .signals import validate_ltx23_sdr_signal, validate_ltx_hdr_working_signal
from .video_vae.tiling import TilingConfig, TilingPlanReceipt, decode_streaming


@dataclass(frozen=True)
class VAEEntryMemoryReceipt:
    """Measured live set and decode budget immediately after decoder load."""

    active_bytes: int
    cache_bytes: int
    peak_bytes: int
    total_bytes: int | None
    recommended_working_set_bytes: int | None
    available_bytes: int
    planner_budget_bytes: int
    latent_bytes: int = 0
    decoder_parameter_bytes: int = 0
    live_assets: frozenset[str] = frozenset({"final_video_latent", "video_decoder"})

    @property
    def planner_budget_gb(self) -> float:
        return self.planner_budget_bytes / 1_000_000_000

    @property
    def accounted_asset_bytes(self) -> int:
        return self.latent_bytes + self.decoder_parameter_bytes

    @property
    def unaccounted_active_bytes(self) -> int:
        return max(0, self.active_bytes - self.accounted_asset_bytes)

    def to_dict(self) -> dict[str, object]:
        """Return the measured allocator budget as stable receipt data."""
        return {
            "active_bytes": self.active_bytes,
            "cache_bytes": self.cache_bytes,
            "peak_bytes": self.peak_bytes,
            "total_bytes": self.total_bytes,
            "recommended_working_set_bytes": self.recommended_working_set_bytes,
            "available_bytes": self.available_bytes,
            "planner_budget_bytes": self.planner_budget_bytes,
            "latent_bytes": self.latent_bytes,
            "decoder_parameter_bytes": self.decoder_parameter_bytes,
            "accounted_asset_bytes": self.accounted_asset_bytes,
            "unaccounted_active_bytes": self.unaccounted_active_bytes,
            "live_assets": sorted(self.live_assets),
        }


def _mlx_memory_bytes(name: str) -> int:
    getter = getattr(mx, name, None)
    if getter is None:
        return 0
    try:
        return max(0, int(getter()))
    except RuntimeError, TypeError, ValueError:
        return 0


def _owned_array_bytes(*values: object) -> int:
    """Count unique logical MLX payload bytes in nested owned products."""
    seen: set[int] = set()

    def visit(value: object) -> int:
        if isinstance(value, mx.array):
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        return 0

    return sum(visit(value) for value in values)


def capture_vae_entry_memory_receipt(
    *,
    latent: mx.array | None = None,
    decoder: object | None = None,
) -> VAEEntryMemoryReceipt:
    """Capture reclaim-aware memory available to the VAE station.

    The recommended working set is the hard ceiling when MLX reports one.
    Half of the remaining capacity is offered to the calibrated Conv3d
    planner; the other half covers output assembly, native sinks, and error
    in the fitted model.
    """
    try:
        info = dict(mx.device_info())
    except AttributeError, RuntimeError, TypeError, ValueError:
        info = {}
    total = info.get("memory_size")
    recommended = info.get("max_recommended_working_set_size")
    total_bytes = int(total) if isinstance(total, (int, float)) and total > 0 else None
    recommended_bytes = (
        int(recommended) if isinstance(recommended, (int, float)) and recommended > 0 else None
    )
    active = _mlx_memory_bytes("get_active_memory")
    cache = _mlx_memory_bytes("get_cache_memory")
    peak = _mlx_memory_bytes("get_peak_memory")
    ceilings = [value for value in (total_bytes, recommended_bytes) if value is not None]
    # A 12 GB fallback preserves the prior conservative behavior on runtimes
    # that do not report unified-memory limits.
    ceiling = min(ceilings) if ceilings else 12_000_000_000
    available = max(0, ceiling - active - cache)
    planner_budget = available // 2
    decoder_parameters = getattr(decoder, "parameters", None)
    decoder_tree = decoder_parameters() if callable(decoder_parameters) else ()
    return VAEEntryMemoryReceipt(
        active_bytes=active,
        cache_bytes=cache,
        peak_bytes=peak,
        total_bytes=total_bytes,
        recommended_working_set_bytes=recommended_bytes,
        available_bytes=available,
        planner_budget_bytes=planner_budget,
        latent_bytes=_owned_array_bytes(latent),
        decoder_parameter_bytes=_owned_array_bytes(decoder_tree),
    )


def decode_video(
    latent: mx.array,
    decoder: VideoDecoderPort,
    *,
    tiling_config: TilingConfig | None = None,
    reporter: Reporter | None = None,
) -> mx.array:
    """Materialize a low-level decode for callers explicitly requesting it.

    Generation recipes use :func:`decode_ltx23_sdr_frames` instead.
    """
    chunks = list(decode_streaming(latent, decoder, tiling_config=tiling_config, reporter=reporter))
    if not chunks:
        raise RuntimeError("video VAE decoder returned no chunks")
    video = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
    mx.eval(video)
    return video


def postprocess_ltx23_sdr_frame(decoded_frame: mx.array) -> mx.array:
    """Convert one LTX-2.3 CHW model frame to owned float16 HWC SDR RGB."""
    if decoded_frame.ndim != 3 or decoded_frame.shape[0] != 3:
        raise ValueError(
            f"LTX-2.3 decoded frame must have shape (3, H, W), got {tuple(decoded_frame.shape)}"
        )
    rgb = mx.clip((decoded_frame.astype(mx.float32) + 1.0) * 0.5, 0.0, 1.0)
    rgb = mx.contiguous(rgb.transpose(1, 2, 0)).astype(mx.float16)
    mx.eval(rgb)
    return rgb


def postprocess_ltx_hdr_working_frame(decoded_frame: mx.array) -> mx.array:
    """Convert one LTX CHW model frame to bounded float32 HWC working codes."""
    if decoded_frame.ndim != 3 or decoded_frame.shape[0] != 3:
        raise ValueError(
            f"LTX HDR decoded frame must have shape (3, H, W), got {tuple(decoded_frame.shape)}"
        )
    codes = mx.clip((decoded_frame.astype(mx.float32) + 1.0) * 0.5, 0.0, 1.0)
    codes = mx.contiguous(codes.transpose(1, 2, 0))
    mx.eval(codes)
    return codes


def _decode_ltx_frames(
    latent: mx.array,
    decoder_provider: Callable[[], ComponentLease[VideoDecoderPort]],
    *,
    spec: VideoSignalSpec,
    frame_count: int,
    postprocess: Callable[[mx.array], mx.array],
    contract_name: str,
    tiling_config: TilingConfig | None = None,
    auto_tiling: bool = False,
    tiling_mode: Literal["auto", "single", "custom"] | None = None,
    reporter: Reporter | None = None,
    decoder_seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
) -> VideoFrameStream:
    """Compose one typed lazy LTX decode without choosing its signal meaning."""
    receipts: dict[str, object] = {}
    owned_latent: mx.array | None = latent

    def produce() -> Iterator[mx.array]:
        nonlocal owned_latent
        try:
            if owned_latent is None:
                raise RuntimeError("video frame stream has already released its latent")
            with decoder_provider() as decoder:
                receipt = capture_vae_entry_memory_receipt(latent=owned_latent, decoder=decoder)
                receipts["vae_entry"] = receipt
                receipts["vae_load"] = getattr(decoder, "load_receipt", None)
                if getattr(decoder, "decoder_kind", "native-conv3d") == "diffusion-na":
                    receipts["vae_noise_backend"] = noise_backend
                    receipts["vae_noise_compatibility_profile"] = noise_compatibility_profile(
                        noise_backend
                    )
                resolved_tiling = tiling_config
                if auto_tiling:
                    if getattr(decoder, "decoder_kind", "native-conv3d") == "diffusion-na":
                        planned_tiling = TilingConfig.auto_diffusion(
                            spec.height,
                            spec.width,
                            frame_count,
                            memory_budget_gb=receipt.planner_budget_gb,
                        )
                    else:
                        planned_tiling = TilingConfig.auto_native_conv3d(
                            spec.height,
                            spec.width,
                            frame_count,
                            memory_budget_gb=receipt.planner_budget_gb,
                            compute_dtype=getattr(
                                decoder,
                                "compute_dtype",
                                mx.bfloat16,
                            ),
                        )
                    resolved_tiling = TilingConfig() if planned_tiling is None else planned_tiling
                requested_mode = tiling_mode
                if requested_mode is None:
                    if auto_tiling or resolved_tiling is None:
                        requested_mode = "auto"
                    elif (
                        resolved_tiling.temporal_config is None
                        and resolved_tiling.spatial_config is None
                    ):
                        requested_mode = "single"
                    else:
                        requested_mode = "custom"
                receipts["vae_tiling_mode"] = requested_mode
                receipts["vae_tiling"] = resolved_tiling
                receipts["vae_decoder_seed"] = decoder_seed

                def record_plan(plan: TilingPlanReceipt) -> None:
                    receipts["vae_plan"] = plan

                def record_noise_state(state: NoiseStreamState) -> None:
                    receipts["vae_noise_state"] = state

                chunks = decode_streaming(
                    owned_latent,
                    decoder,
                    tiling_config=resolved_tiling,
                    reporter=reporter,
                    plan_callback=record_plan,
                    noise_state_callback=record_noise_state,
                    seed=decoder_seed,
                    noise_backend=noise_backend,
                )
                try:
                    for chunk in chunks:
                        if chunk.ndim != 5 or tuple(chunk.shape[:2]) != (1, 3):
                            raise RuntimeError(
                                f"{contract_name} decoder returned incompatible BCFHW chunk "
                                f"{tuple(chunk.shape)}"
                            )
                        if tuple(chunk.shape[3:]) != (spec.height, spec.width):
                            raise RuntimeError(
                                f"{contract_name} decoder returned chunk geometry "
                                f"{tuple(chunk.shape[3:])}, expected {(spec.height, spec.width)}"
                            )
                        for index in range(int(chunk.shape[2])):
                            yield postprocess(chunk[0, :, index])
                        del chunk
                        mx.clear_cache()
                finally:
                    close = getattr(chunks, "close", None)
                    if close is not None:
                        close()
        finally:
            owned_latent = None

    return VideoFrameStream(produce, spec=spec, frame_count=frame_count, receipts=receipts)


def decode_ltx23_sdr_frames(
    latent: mx.array,
    decoder_provider: Callable[[], ComponentLease[VideoDecoderPort]],
    *,
    spec: VideoSignalSpec,
    frame_count: int,
    tiling_config: TilingConfig | None = None,
    auto_tiling: bool = False,
    tiling_mode: Literal["auto", "single", "custom"] | None = None,
    reporter: Reporter | None = None,
    decoder_seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
) -> VideoFrameStream:
    """Return a lazy stream that owns the final latent and decoder lifetime."""
    validate_ltx23_sdr_signal(spec)
    return _decode_ltx_frames(
        latent,
        decoder_provider,
        spec=spec,
        frame_count=frame_count,
        postprocess=postprocess_ltx23_sdr_frame,
        contract_name="LTX-2.3 SDR",
        tiling_config=tiling_config,
        auto_tiling=auto_tiling,
        tiling_mode=tiling_mode,
        reporter=reporter,
        decoder_seed=decoder_seed,
        noise_backend=noise_backend,
    )


def decode_ltx_hdr_working_frames(
    latent: mx.array,
    decoder_provider: Callable[[], ComponentLease[VideoDecoderPort]],
    *,
    spec: VideoSignalSpec,
    frame_count: int,
    tiling_config: TilingConfig | None = None,
    auto_tiling: bool = False,
    tiling_mode: Literal["auto", "single", "custom"] | None = None,
    reporter: Reporter | None = None,
    decoder_seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
) -> VideoFrameStream:
    """Return a lazy bounded float32 HDR working-code stream."""
    validate_ltx_hdr_working_signal(spec)
    return _decode_ltx_frames(
        latent,
        decoder_provider,
        spec=spec,
        frame_count=frame_count,
        postprocess=postprocess_ltx_hdr_working_frame,
        contract_name="LTX HDR",
        tiling_config=tiling_config,
        auto_tiling=auto_tiling,
        tiling_mode=tiling_mode,
        reporter=reporter,
        decoder_seed=decoder_seed,
        noise_backend=noise_backend,
    )


def video_decode_diagnostics(stream: VideoFrameStream) -> dict[str, object]:
    """Return stable VAE-entry and resolved-plan diagnostics after stream use."""
    receipts = stream.receipts
    entry = receipts.get("vae_entry")
    config = receipts.get("vae_tiling")
    plan = receipts.get("vae_plan")
    mode = receipts.get("vae_tiling_mode")
    decoder_seed = receipts.get("vae_decoder_seed")
    noise_backend = receipts.get("vae_noise_backend")
    noise_profile = receipts.get("vae_noise_compatibility_profile")
    noise_state = receipts.get("vae_noise_state")
    load_receipt = receipts.get("vae_load")
    if not isinstance(entry, VAEEntryMemoryReceipt):
        return {}

    tiling: dict[str, object] = {
        "requested_mode": mode if isinstance(mode, str) else None,
    }
    if isinstance(decoder_seed, int) and not isinstance(decoder_seed, bool):
        tiling["decoder_seed"] = decoder_seed
    if isinstance(plan, TilingPlanReceipt):
        tiling.update(plan.to_dict())
    elif isinstance(config, TilingConfig):
        tiling.update(
            {
                "latent_shape": None,
                "decoded_shape": None,
                "temporal_tiles": None,
                "spatial_height_tiles": None,
                "spatial_width_tiles": None,
                "total_tiles": None,
                "resolved_config": config.to_dict(),
            }
        )
    result: dict[str, object] = {
        "vae_decode": {
            "entry_memory": entry.to_dict(),
            "tiling": tiling,
        }
    }
    if isinstance(noise_state, NoiseStreamState):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["noise"] = noise_state.to_metadata()
    elif isinstance(noise_backend, str) and isinstance(noise_profile, str):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["noise"] = {
                "backend": noise_backend,
                "compatibility_profile": noise_profile,
            }
    to_dict = getattr(load_receipt, "to_dict", None)
    if callable(to_dict):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["decoder_load"] = to_dict()
    return result


def decode_audio(
    latent: mx.array,
    audio_decoder: AudioDecoderPort,
    vocoder: VocoderPort,
    *,
    reporter: Reporter | None = None,
) -> mx.array:
    """Decode a normalized audio latent through mel space to waveform."""
    return vocode_audio(
        decode_audio_mel(latent, audio_decoder, reporter=reporter),
        vocoder,
        reporter=reporter,
    )


def decode_audio_mel(
    latent: mx.array,
    audio_decoder: AudioDecoderPort,
    *,
    reporter: Reporter | None = None,
) -> mx.array:
    """Decode a normalized audio latent to stereo log-mel."""
    mel = audio_decoder(latent, reporter=reporter)
    mx.eval(mel)
    return mel


def vocode_audio(
    mel: mx.array,
    vocoder: VocoderPort,
    *,
    reporter: Reporter | None = None,
) -> mx.array:
    """Synthesize a waveform from stereo log-mel."""
    waveform = vocoder(mel, reporter=reporter)
    mx.eval(waveform)
    return waveform


def video_frames(video: mx.array) -> Iterator[mx.array]:
    """Iterate normalized SDR frames from an explicitly materialized decode."""
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"decoded video must have shape (1, 3, T, H, W), got {tuple(video.shape)}")
    for index in range(video.shape[2]):
        yield postprocess_ltx23_sdr_frame(video[0, :, index])


__all__ = [
    "VAEEntryMemoryReceipt",
    "capture_vae_entry_memory_receipt",
    "decode_audio",
    "decode_audio_mel",
    "decode_ltx23_sdr_frames",
    "decode_ltx_hdr_working_frames",
    "decode_video",
    "postprocess_ltx23_sdr_frame",
    "postprocess_ltx_hdr_working_frame",
    "video_decode_diagnostics",
    "video_frames",
    "vocode_audio",
]
