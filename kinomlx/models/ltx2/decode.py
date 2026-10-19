"""Pure audio operations and LTX-specific lazy video decode composition."""

import logging
import math
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import mlx.core as mx

from kinomlx.components import ComponentLease
from kinomlx.errors import KinoMLXError
from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.signals import VideoSignalSpec
from kinomlx.modeling.dtypes import float_dtype_name
from kinomlx.modeling.memory import capture_decode_memory_budget
from kinomlx.reporting import Reporter
from kinomlx.samplers.noise import NoiseStreamState, noise_compatibility_profile
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend, VAETilingMode

from .components import LTX2AudioDecoderPort, LTX2VideoDecoderPort, LTX2VocoderPort
from .signals import validate_ltx23_sdr_signal, validate_ltx_hdr_working_signal
from .video_vae.kinds import VideoVAEKind
from .video_vae.sharpening import ContextSharpening
from .video_vae.tiling import (
    TilingConfig,
    TilingPlanReceipt,
    decode_streaming,
    estimated_diffusion_seconds,
)

_log = logging.getLogger(__name__)

VideoDecoderProvider = Callable[[], ComponentLease[LTX2VideoDecoderPort]]


class LTX2DecodeError(KinoMLXError, RuntimeError):
    """A video decode produced nonfinite frames."""


def _nonfinite_decode(dtype: mx.Dtype, *, frames_written: int) -> LTX2DecodeError:
    message = f"LTX-2 video VAE decode produced nonfinite frames in {float_dtype_name(dtype)}"
    message += f" after {frames_written} frames were written." if frames_written else "."
    if dtype == mx.float16:
        message += (
            " The float16 decoder's activations may have overflowed; decode with"
            " --vae-decode-dtype bfloat16 (or float32), from this run's record with --restart"
            " when it saved its latents (--save-all-sidecars)."
        )
    return LTX2DecodeError(message)


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
    system_reclaimable_bytes: int | None = None
    memory_limit_bytes: int | None = None
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
            "system_reclaimable_bytes": self.system_reclaimable_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "latent_bytes": self.latent_bytes,
            "decoder_parameter_bytes": self.decoder_parameter_bytes,
            "accounted_asset_bytes": self.accounted_asset_bytes,
            "unaccounted_active_bytes": self.unaccounted_active_bytes,
            "live_assets": sorted(self.live_assets),
        }


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
    """Capture the shared reclaim-aware budget after decoder materialization."""
    budget = capture_decode_memory_budget()
    decoder_parameters = getattr(decoder, "parameters", None)
    decoder_tree = decoder_parameters() if callable(decoder_parameters) else ()
    return VAEEntryMemoryReceipt(
        active_bytes=budget.active_bytes,
        cache_bytes=budget.cache_bytes,
        peak_bytes=int(mx.get_peak_memory()),
        total_bytes=budget.total_bytes,
        recommended_working_set_bytes=budget.recommended_working_set_bytes,
        available_bytes=budget.available_bytes,
        planner_budget_bytes=budget.available_bytes,
        system_reclaimable_bytes=budget.system_reclaimable_bytes,
        memory_limit_bytes=budget.limit_bytes,
        latent_bytes=_owned_array_bytes(latent),
        decoder_parameter_bytes=_owned_array_bytes(decoder_tree),
    )


def _tile_jobs(
    config: TilingConfig | None,
    *,
    height: int,
    width: int,
    frames: int,
) -> int:
    if config is None:
        return 1
    temporal = getattr(config, "temporal_config", None)
    spatial = getattr(config, "spatial_config", None)

    def count(length: int, size: int, overlap: int) -> int:
        return max(1, math.ceil((length - overlap) / (size - overlap)))

    temporal_jobs = (
        1
        if temporal is None
        else count(frames, temporal.chunk_size_in_frames, temporal.chunk_overlap_in_frames)
    )
    if spatial is None:
        return temporal_jobs
    return (
        temporal_jobs
        * count(height, spatial.tile_size_in_pixels, spatial.tile_overlap_in_pixels)
        * count(width, spatial.tile_size_in_pixels, spatial.tile_overlap_in_pixels)
    )


def _plan_vae_cache_and_tiling(
    *,
    available_bytes: int,
    requested_cache_gb: float | None,
    auto_tiling: bool,
    tiling_config: TilingConfig | None,
    decoder_kind: VideoVAEKind,
    compute_dtype: mx.Dtype,
    optimized_fp16: bool,
    height: int,
    width: int,
    frames: int,
    max_buffer_bytes: int | None = None,
) -> tuple[float, int, TilingConfig | None]:
    """Favor fewer decode jobs, then more cache, within the live headroom.

    The cache shrinks before the plan tiles: a plan with fewer jobs wins at any cache size, and
    among plans with as few jobs the largest cache wins. The FP16 convolutional decoder and the
    diffusion decoder's FP16 kernel route reuse full-size buffers from block to block and get
    the larger ladder. A diffusion plan the device or its kernels cannot address is never
    chosen, and a requested one is refused (``TilingConfig.addressable``; ``max_buffer_bytes``
    stands in for the device's buffer limit).
    """
    if requested_cache_gb is not None and (
        not math.isfinite(requested_cache_gb) or requested_cache_gb < 0
    ):
        raise ValueError("vae_cache_limit_gb must be finite and non-negative")
    if decoder_kind == VideoVAEKind.DIFFUSION and not auto_tiling:
        requested = TilingConfig() if tiling_config is None else tiling_config
        if not requested.addressable(
            height, width, frames, compute_dtype=compute_dtype, max_buffer_bytes=max_buffer_bytes
        ):
            raise ValueError(
                "the requested VAE tiling has a tile this device cannot address; decode with "
                "temporal or spatial tiles"
            )
    large_cache = optimized_fp16 or (
        decoder_kind == VideoVAEKind.DIFFUSION and compute_dtype == mx.float16
    )
    candidates = (
        (requested_cache_gb,)
        if requested_cache_gb is not None
        else ((16.0, 12.0, 8.0, 4.0, 1.0, 0.0) if large_cache else (4.0, 1.0, 0.0))
    )
    if auto_tiling and large_cache and decoder_kind == VideoVAEKind.DIFFUSION:
        return _plan_fp16_diffusion(
            available_bytes=available_bytes,
            requested_cache_gb=requested_cache_gb,
            compute_dtype=compute_dtype,
            height=height,
            width=width,
            frames=frames,
            max_buffer_bytes=max_buffer_bytes,
        )
    best: tuple[int, float, int, TilingConfig | None] | None = None
    for cache_gb in candidates:
        # Retained allocations can briefly exceed the nominal MLX cache cap.
        reserved_bytes = int(cache_gb * 1024**3) + (1 << 30)
        tile_budget_bytes = available_bytes - reserved_bytes
        if tile_budget_bytes <= 0:
            continue
        planned_tiling = tiling_config
        if auto_tiling:
            try:
                if decoder_kind == VideoVAEKind.DIFFUSION:
                    planned_tiling = TilingConfig.auto_diffusion(
                        height,
                        width,
                        frames,
                        memory_budget_gb=tile_budget_bytes / 1_000_000_000,
                        compute_dtype=compute_dtype,
                    )
                else:
                    planned_tiling = TilingConfig.auto_conv3d(
                        height,
                        width,
                        frames,
                        memory_budget_gb=tile_budget_bytes / 1_000_000_000,
                        compute_dtype=compute_dtype,
                        optimized_fp16=optimized_fp16,
                    )
            except MemoryError:
                continue
        elif (
            tiling_config is not None
            and tiling_config.estimated_peak_gb(
                height,
                width,
                frames,
                decoder_kind=decoder_kind,
                compute_dtype=compute_dtype,
                optimized_fp16=optimized_fp16,
            )
            > tile_budget_bytes / 1_000_000_000
        ):
            continue
        jobs = _tile_jobs(planned_tiling, height=height, width=width, frames=frames)
        if best is None or (jobs, -cache_gb) < (best[0], -best[1]):
            best = (jobs, cache_gb, tile_budget_bytes, planned_tiling)
    if best is None:
        raise MemoryError("No VAE cache and tile plan fits the available decode memory budget")
    return best[1], best[2], best[3]


# Temporal tiles keep a 24-frame overlap, so a shorter tile recomputes most of what it advances
# (a 32-frame tile moves 8 frames). The time estimate was fitted on 128- and 192-frame tiles;
# shorter ones are chosen only when no longer temporal tile fits.
_SHORTEST_RANKED_TILE = 64
# How far an FP16 kernel-route decode's footprint exceeds its active peak: the allocator cache
# holds at most a block's five feature buffers beside about 2 GiB of tile temporaries, and
# three of those features are live at the peak, so the footprint is bounded by
# peak + max(0, min(cache, 5F) - 3F) + 3 GiB. Measured as active plus cache at phase boundaries
# against that bound: a 768x448x121 pass 13.1 GB (15.4 bounded), 1024x576 tiles of 128, 224 and
# 264 frames at a 16 GiB cache 22.9, 34.8 and 35.0 (30.9, 44.1, 47.5), 264-frame tiles with no
# cache limit 42.1 (55.3); a 16 GiB cache held 15 to 17 GB at a block's start and an unlimited
# one the whole set, 25.0 GB, with 4.3 GB more at most.
_CACHE_SET_FEATURES = 5
_CACHE_LIVE_FEATURES = 3
_CACHE_TEMPORARIES_BYTES = 3 << 30
_CACHE_SET_EXTRA_BYTES = 4 << 30


def _kernel_route_footprint_bytes(peak_bytes: int, feature_bytes: int, cache_bytes: int) -> int:
    """The bound on an FP16 kernel-route decode's footprint: its active peak plus what the cache
    holds beyond the features live at the peak."""
    cached = max(
        0,
        min(cache_bytes, _CACHE_SET_FEATURES * feature_bytes)
        - _CACHE_LIVE_FEATURES * feature_bytes,
    )
    return peak_bytes + cached + _CACHE_TEMPORARIES_BYTES


def _plan_fp16_diffusion(
    *,
    available_bytes: int,
    requested_cache_gb: float | None,
    compute_dtype: mx.Dtype,
    height: int,
    width: int,
    frames: int,
    max_buffer_bytes: int | None = None,
) -> tuple[float, int, TilingConfig | None]:
    """Choose an FP16 diffusion decode's cache and tiles.

    Every plan (one pass, temporal tiles by count, then spatial tiles) gets the one cache it
    wants: the smallest that holds its tile's whole block set (five features and the extra
    above; a larger cache holds no more) when the plan's bounded peak plus that cache's
    footprint over it (`_kernel_route_footprint_bytes`) leaves 1 GiB of headroom, else the
    largest whole GiB that does. One pass wins whenever it fits. Otherwise the fastest plan by
    estimated time (fewer tiles recompute less overlap, a larger cache faults in fewer pages):
    on a 1024x576x721 decode, 128-frame tiles at 16 GiB took 154.6 s
    against 157.9 for 192-frame tiles at 8 GiB; beside the same 16 GiB cache, 192-frame tiles
    took 145.4 s, 224-frame ones 141.9 and three 264-frame ones 138.3, and those with no cache
    limit 137.1. Tiles shorter than 64 frames only when no longer one fits, and spatial tiles
    only when no temporal plan fits. A plan the kernels or the device cannot address is never
    proposed (``TilingConfig.diffusion_candidates``).
    """
    gib = 1 << 30
    # Ranked tiers: one pass, temporal tiles of at least _SHORTEST_RANKED_TILE frames, shorter
    # temporal tiles, then spatial tiles; within a tier the fastest.
    ranked: dict[int, tuple[float, float, int, TilingConfig | None]] = {}
    for option in TilingConfig.diffusion_candidates(
        height, width, frames, max_buffer_bytes=max_buffer_bytes
    ):
        plan = TilingConfig() if option is None else option
        peak_bytes = int(
            plan.estimated_peak_gb(
                height,
                width,
                frames,
                decoder_kind=VideoVAEKind.DIFFUSION,
                compute_dtype=compute_dtype,
            )
            * 1_000_000_000
        )
        feature_bytes = int(plan.kernel_route_feature_gb(height, width, frames) * 1_000_000_000)
        if requested_cache_gb is not None:
            cache_gb = requested_cache_gb
        else:
            # The footprint grows with the cache only between three and five features, so the
            # largest cache that fits is what the headroom leaves past the peak, plus the three
            # live features; the block set caps it.
            block_set = _CACHE_SET_FEATURES * feature_bytes + _CACHE_SET_EXTRA_BYTES
            room = available_bytes - gib - peak_bytes - _CACHE_TEMPORARIES_BYTES
            if room < 0:
                continue
            cache_gb = float(min(block_set, room + _CACHE_LIVE_FEATURES * feature_bytes) // gib)
        temporal = plan.temporal_config
        if option is None:
            tier = -1
        elif plan.spatial_config is not None:
            tier = 2
        elif temporal is not None and temporal.chunk_size_in_frames < _SHORTEST_RANKED_TILE:
            tier = 1
        else:
            tier = 0
        footprint = _kernel_route_footprint_bytes(peak_bytes, feature_bytes, int(cache_gb * gib))
        if footprint + gib > available_bytes:
            continue
        tile_budget_bytes = available_bytes - (footprint - peak_bytes) - gib
        seconds = estimated_diffusion_seconds(option, height, width, frames, cache_gb=cache_gb)
        if tier not in ranked or seconds < ranked[tier][0]:
            ranked[tier] = (seconds, cache_gb, tile_budget_bytes, option)
    for tier in (-1, 0, 1, 2):
        if tier in ranked:
            _seconds, cache_gb, tile_budget_bytes, option = ranked[tier]
            return cache_gb, tile_budget_bytes, option
    raise MemoryError("No VAE cache and tile plan fits the available decode memory budget")


@contextmanager
def _vae_cache_scope(limit_gb: float) -> Iterator[None]:
    previous = mx.set_cache_limit(int(limit_gb * 1024**3))
    try:
        yield
    finally:
        mx.set_cache_limit(previous)


def decode_video(
    latent: mx.array,
    decoder: LTX2VideoDecoderPort,
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
    decoder_provider: VideoDecoderProvider,
    *,
    spec: VideoSignalSpec,
    frame_count: int,
    postprocess: Callable[[mx.array], mx.array],
    contract_name: str,
    tiling_config: TilingConfig | None = None,
    auto_tiling: bool = False,
    vae_cache_limit_gb: float | None = None,
    tiling_mode: VAETilingMode | None = None,
    reporter: Reporter | None = None,
    decoder_seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    overflow_fallback: VideoDecoderProvider | None = None,
    context_sharpening: ContextSharpening | None = None,
) -> VideoFrameStream:
    """Compose one typed lazy LTX decode without choosing its signal meaning.

    Each decoded chunk must be finite before its frames are emitted. When the first chunk is
    not, the decode runs again with ``overflow_fallback``'s decoder if one is given, so no
    nonfinite frame reaches the consumer; a nonfinite later chunk, or one without a fallback,
    raises :class:`LTX2DecodeError`.
    """
    receipts: dict[str, object] = {}
    owned_latent: mx.array | None = latent

    def decode_with(provider: VideoDecoderProvider, source: mx.array) -> Generator[mx.array]:
        with provider() as decoder:
            receipt = capture_vae_entry_memory_receipt(latent=source, decoder=decoder)
            receipts["vae_entry"] = receipt
            receipts["vae_load"] = getattr(decoder, "load_receipt", None)
            if getattr(decoder, "decoder_kind", VideoVAEKind.CONV) == VideoVAEKind.DIFFUSION:
                receipts["vae_noise_backend"] = noise_backend
                receipts["vae_noise_compatibility_profile"] = noise_compatibility_profile(
                    noise_backend
                )
            decoder_kind = getattr(decoder, "decoder_kind", VideoVAEKind.CONV)
            compute_dtype = getattr(decoder, "compute_dtype", mx.bfloat16)
            receipts["vae_compute_dtype"] = compute_dtype
            optimized_fp16 = (
                decoder_kind == VideoVAEKind.CONV
                and compute_dtype == mx.float16
                and getattr(decoder, "causal", True) is False
            )
            cache_gb, tile_budget_bytes, planned_tiling = _plan_vae_cache_and_tiling(
                available_bytes=receipt.available_bytes,
                requested_cache_gb=vae_cache_limit_gb,
                auto_tiling=auto_tiling or tiling_config is None,
                tiling_config=tiling_config,
                decoder_kind=decoder_kind,
                compute_dtype=compute_dtype,
                optimized_fp16=optimized_fp16,
                height=spec.height,
                width=spec.width,
                frames=frame_count,
            )
            receipt = replace(receipt, planner_budget_bytes=tile_budget_bytes)
            receipts["vae_entry"] = receipt
            receipts["vae_cache"] = {
                "requested_limit_gb": vae_cache_limit_gb,
                "selected_limit_gb": cache_gb,
                "tile_budget_bytes": tile_budget_bytes,
            }
            resolved_tiling = TilingConfig() if planned_tiling is None else planned_tiling
            requested_mode = tiling_mode
            if requested_mode is None:
                if auto_tiling or resolved_tiling is None:
                    requested_mode = VAETilingMode.AUTO
                elif (
                    resolved_tiling.temporal_config is None
                    and resolved_tiling.spatial_config is None
                ):
                    requested_mode = VAETilingMode.SINGLE
                else:
                    requested_mode = VAETilingMode.CUSTOM
            receipts["vae_tiling_mode"] = requested_mode
            receipts["vae_tiling"] = resolved_tiling
            receipts["vae_decoder_seed"] = decoder_seed
            if context_sharpening is not None:
                receipts["vae_context_sharpening"] = context_sharpening

            def record_plan(plan: TilingPlanReceipt) -> None:
                receipts["vae_plan"] = plan

            def record_noise_state(state: NoiseStreamState) -> None:
                receipts["vae_noise_state"] = state

            written = 0
            with _vae_cache_scope(cache_gb):
                chunks = decode_streaming(
                    source,
                    decoder,
                    tiling_config=resolved_tiling,
                    reporter=reporter,
                    plan_callback=record_plan,
                    noise_state_callback=record_noise_state,
                    seed=decoder_seed,
                    noise_backend=noise_backend,
                    context_sharpening=context_sharpening,
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
                        if not bool(mx.all(mx.isfinite(chunk)).item()):
                            raise _nonfinite_decode(compute_dtype, frames_written=written)
                        for index in range(int(chunk.shape[2])):
                            yield postprocess(chunk[0, :, index])
                            written += 1
                        del chunk
                finally:
                    close = getattr(chunks, "close", None)
                    if close is not None:
                        close()

    def produce() -> Iterator[mx.array]:
        nonlocal owned_latent
        try:
            if owned_latent is None:
                raise RuntimeError("video frame stream has already released its latent")
            source = owned_latent
            frames = decode_with(decoder_provider, source)
            emitted = 0
            fallback: VideoDecoderProvider | None = None
            try:
                for frame in frames:
                    emitted += 1
                    yield frame
            except LTX2DecodeError:
                if overflow_fallback is None or emitted:
                    raise
                fallback = overflow_fallback
            finally:
                frames.close()
            if fallback is None:
                return
            failed = receipts.get("vae_compute_dtype")
            if isinstance(failed, mx.Dtype):
                receipts["vae_overflow_fallback"] = failed
                _log.warning(
                    "Video VAE decode produced nonfinite %s frames; decoding again with the "
                    "overflow fallback precision.",
                    float_dtype_name(failed),
                )
            frames = decode_with(fallback, source)
            try:
                yield from frames
            finally:
                frames.close()
        finally:
            owned_latent = None

    return VideoFrameStream(produce, spec=spec, frame_count=frame_count, receipts=receipts)


def decode_ltx23_sdr_frames(
    latent: mx.array,
    decoder_provider: VideoDecoderProvider,
    *,
    spec: VideoSignalSpec,
    frame_count: int,
    tiling_config: TilingConfig | None = None,
    auto_tiling: bool = False,
    vae_cache_limit_gb: float | None = None,
    tiling_mode: VAETilingMode | None = None,
    reporter: Reporter | None = None,
    decoder_seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    overflow_fallback: VideoDecoderProvider | None = None,
    context_sharpening: ContextSharpening | None = None,
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
        vae_cache_limit_gb=vae_cache_limit_gb,
        tiling_mode=tiling_mode,
        reporter=reporter,
        decoder_seed=decoder_seed,
        noise_backend=noise_backend,
        overflow_fallback=overflow_fallback,
        context_sharpening=context_sharpening,
    )


def decode_ltx_hdr_working_frames(
    latent: mx.array,
    decoder_provider: VideoDecoderProvider,
    *,
    spec: VideoSignalSpec,
    frame_count: int,
    tiling_config: TilingConfig | None = None,
    auto_tiling: bool = False,
    vae_cache_limit_gb: float | None = None,
    tiling_mode: VAETilingMode | None = None,
    reporter: Reporter | None = None,
    decoder_seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
    overflow_fallback: VideoDecoderProvider | None = None,
    context_sharpening: ContextSharpening | None = None,
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
        vae_cache_limit_gb=vae_cache_limit_gb,
        tiling_mode=tiling_mode,
        reporter=reporter,
        decoder_seed=decoder_seed,
        noise_backend=noise_backend,
        overflow_fallback=overflow_fallback,
        context_sharpening=context_sharpening,
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
    cache = receipts.get("vae_cache")
    if isinstance(cache, dict):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["cache"] = cache
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
    sharpening = receipts.get("vae_context_sharpening")
    if isinstance(sharpening, ContextSharpening):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["context_sharpening"] = sharpening.to_dict()
    to_dict = getattr(load_receipt, "to_dict", None)
    if callable(to_dict):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["decoder_load"] = to_dict()
    failed = receipts.get("vae_overflow_fallback")
    compute = receipts.get("vae_compute_dtype")
    if isinstance(failed, mx.Dtype) and isinstance(compute, mx.Dtype):
        vae_decode = result["vae_decode"]
        if isinstance(vae_decode, dict):
            vae_decode["overflow_fallback"] = {
                "from_dtype": float_dtype_name(failed),
                "to_dtype": float_dtype_name(compute),
            }
    return result


def decode_audio(
    latent: mx.array,
    audio_decoder: LTX2AudioDecoderPort,
    vocoder: LTX2VocoderPort,
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
    audio_decoder: LTX2AudioDecoderPort,
    *,
    reporter: Reporter | None = None,
) -> mx.array:
    """Decode a normalized audio latent to stereo log-mel."""
    mel = audio_decoder(latent, reporter=reporter)
    mx.eval(mel)
    return mel


def vocode_audio(
    mel: mx.array,
    vocoder: LTX2VocoderPort,
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
    "LTX2DecodeError",
    "VAEEntryMemoryReceipt",
    "VideoDecoderProvider",
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
