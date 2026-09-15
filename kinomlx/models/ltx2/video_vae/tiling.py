"""Memory-bounded tiled decoding for the native LTX video VAE."""

import gc
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from kinomlx.modeling.memory import capture_decode_memory_budget
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.samplers.noise import NoiseStreamState
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend

_log = logging.getLogger(__name__)

DEFAULT_TEMPORAL_SCALE = 8
DEFAULT_SPATIAL_SCALE = 32


class DecoderCallable(Protocol):
    """Video-decoder call surface used by the generic tiling engine."""

    def __call__(
        self,
        latent: mx.array,
        *,
        timestep: float | None = 0.05,
        reporter: Reporter | None = None,
    ) -> mx.array: ...


def _strict_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _estimate_native_conv3d_peak_gb(
    height: int,
    width: int,
    tile_frames: int,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
) -> float:
    """Estimate incremental Conv3d peak above the VAE-entry live set.

    The Phase D MLX 0.32.1 sweep covered 512x320, 768x448, and 1024x576
    at 33 and 65 decoded frames. The 121-frame 768x448 MLX 0.32.2 check
    reached 30.02 GB above the entry live set; a coefficient of 0.38 also
    covers that longer decode. FP32 doubles the activation/workspace
    estimate; tiled output assembly is accounted separately in FP32.
    """
    if height <= 0 or width <= 0 or tile_frames <= 0:
        raise ValueError("decode dimensions must be positive")
    # FP16 activations with the checkpoint's BF16 Conv3d weights promote to
    # FP32 in MLX, so FP16 is not a truthful low-precision planner input.
    if compute_dtype == mx.bfloat16:
        dtype_scale = 1.0
    elif compute_dtype == mx.float32:
        dtype_scale = 2.0
    else:
        raise ValueError(f"unsupported native Conv3d compute dtype {compute_dtype}")
    area_scale = (height * width) / (1024 * 576)
    return (4.2 + 0.38 * tile_frames * area_scale) * dtype_scale


def _estimate_tiled_assembly_gb(
    height: int,
    width: int,
    tile_frames: int,
    *,
    spatial_tiled: bool,
) -> float:
    """Conservatively bound the FP32 tiled-output assembly live set."""
    if height <= 0 or width <= 0 or tile_frames <= 0:
        raise ValueError("decode dimensions must be positive")
    # During temporal merging, one pending chunk, one new chunk, and a merged
    # buffer up to twice the chunk length can coexist. Spatial tiling also keeps
    # one FP32 weight plane beside each three-channel decoded buffer.
    channels = 4 if spatial_tiled else 3
    return height * width * tile_frames * channels * 4 * 4 / 1_000_000_000


def _estimate_diffusion_peak_gb(
    height: int,
    width: int,
    frames: int,
    *,
    compute_dtype: mx.Dtype,
) -> float:
    """Bound full-resolution feature liveness plus attention/FF scratch.

    Stage 5 carries 256 channels at quarter spatial resolution. Eight live
    feature-sized buffers cover context, residuals, normalization, and the
    temporal QKV partition; 1 GiB covers the bounded attention/FF workspaces.
    Include the decoder's latent alignment and two trailing context frames.
    """
    if compute_dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError(f"unsupported diffusion VAE compute dtype {compute_dtype}")
    itemsize = 4 if compute_dtype == mx.float32 else 2
    padded_frames = math.ceil(frames / 8) * 8 + 16
    feature_bytes = padded_frames * math.ceil(height / 4) * math.ceil(width / 4) * 256 * itemsize
    return (8 * feature_bytes + 1024**3 * (itemsize // 2)) / 1_000_000_000


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> mx.array:
    """Create a one-dimensional overlap blend mask."""
    length = _strict_int(length, field="length", minimum=1)
    ramp_left = _strict_int(ramp_left, field="ramp_left", minimum=0)
    ramp_right = _strict_int(ramp_right, field="ramp_right", minimum=0)

    ramp_left = min(ramp_left, length)
    ramp_right = min(ramp_right, length)
    values = [1.0] * length

    if ramp_left:
        count = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade = [index / (count - 1) for index in range(count)][:-1]
        if not left_starts_from_0:
            fade = fade[1:]
        for index, value in enumerate(fade[:ramp_left]):
            values[index] *= value

    if ramp_right:
        fade = [(ramp_right + 1 - index) / (ramp_right + 1) for index in range(1, ramp_right + 1)]
        start = length - ramp_right
        for index, value in enumerate(fade):
            values[start + index] *= value

    return mx.clip(mx.array(values, dtype=mx.float32), 0.0, 1.0)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Spatial tile geometry in decoded-pixel coordinates."""

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        _strict_int(
            self.tile_size_in_pixels,
            field="tile_size_in_pixels",
            minimum=1,
        )
        _strict_int(
            self.tile_overlap_in_pixels,
            field="tile_overlap_in_pixels",
            minimum=0,
        )
        if self.tile_size_in_pixels < 64:
            raise ValueError(
                f"tile_size_in_pixels must be 64 or greater, received {self.tile_size_in_pixels}"
            )
        if self.tile_size_in_pixels % DEFAULT_SPATIAL_SCALE:
            raise ValueError(
                "tile_size_in_pixels must be divisible by "
                f"{DEFAULT_SPATIAL_SCALE}, got {self.tile_size_in_pixels}"
            )
        if self.tile_overlap_in_pixels % DEFAULT_SPATIAL_SCALE:
            raise ValueError(
                "tile_overlap_in_pixels must be divisible by "
                f"{DEFAULT_SPATIAL_SCALE}, got {self.tile_overlap_in_pixels}"
            )
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError("spatial overlap must be smaller than tile size")


@dataclass(frozen=True)
class TemporalChunkConfig:
    """Temporal tile geometry in decoded-frame coordinates."""

    chunk_size_in_frames: int
    chunk_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        _strict_int(
            self.chunk_size_in_frames,
            field="chunk_size_in_frames",
            minimum=1,
        )
        _strict_int(
            self.chunk_overlap_in_frames,
            field="chunk_overlap_in_frames",
            minimum=0,
        )
        if self.chunk_size_in_frames < 16:
            raise ValueError(
                f"chunk_size_in_frames must be at least 16, got {self.chunk_size_in_frames}"
            )
        if self.chunk_size_in_frames % DEFAULT_TEMPORAL_SCALE:
            raise ValueError(
                "chunk_size_in_frames must be divisible by "
                f"{DEFAULT_TEMPORAL_SCALE}, got {self.chunk_size_in_frames}"
            )
        if self.chunk_overlap_in_frames % DEFAULT_TEMPORAL_SCALE:
            raise ValueError(
                "chunk_overlap_in_frames must be divisible by "
                f"{DEFAULT_TEMPORAL_SCALE}, got {self.chunk_overlap_in_frames}"
            )
        if self.chunk_overlap_in_frames >= self.chunk_size_in_frames:
            raise ValueError("temporal overlap must be smaller than chunk size")


@dataclass(frozen=True)
class TilingConfig:
    """Spatial and temporal controls for tiled VAE decoding."""

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalChunkConfig | None = None

    @classmethod
    def default(cls) -> TilingConfig:
        """Return the balanced 512/64 spatial and 64/24 temporal plan."""
        return cls(
            spatial_config=SpatialTilingConfig(512, 64),
            temporal_config=TemporalChunkConfig(64, 24),
        )

    @classmethod
    def auto_diffusion(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        memory_budget_gb: float | None = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> TilingConfig | None:
        """Choose full or stage-tiled diffusion decode from current headroom."""
        height = _strict_int(height, field="height", minimum=1)
        width = _strict_int(width, field="width", minimum=1)
        num_frames = _strict_int(num_frames, field="num_frames", minimum=1)
        # The official decoder pads to a 3x7x7 latent before its first stage.
        # Planning tiny requested images must account for that working grid.
        height = max(224, math.ceil(height / 32) * 32)
        width = max(224, math.ceil(width / 32) * 32)
        num_frames = max(17, num_frames)
        budget_gb = (
            capture_decode_memory_budget().available_gb
            if memory_budget_gb is None
            else memory_budget_gb
        )
        if not math.isfinite(budget_gb) or budget_gb <= 0:
            raise ValueError("memory_budget_gb must be positive")
        full_peak = _estimate_diffusion_peak_gb(
            height, width, num_frames, compute_dtype=compute_dtype
        )
        if full_peak <= budget_gb:
            return None
        # Stage 1-3 context remains resident even when later stages are tiled.
        context_gb = height * width * (num_frames + 16) * 8 / 1_000_000_000
        if compute_dtype == mx.float32:
            context_gb *= 2
        temporal_candidates = (None, 512, 256, 128, 80, 64, 40, 32, 16)
        spatial_candidates = (None, 1024, 768, 512, 256, 128, 64)
        for spatial_size in spatial_candidates:
            if spatial_size is not None and spatial_size >= max(height, width):
                continue
            tile_height = height if spatial_size is None else min(height, spatial_size)
            tile_width = width if spatial_size is None else min(width, spatial_size)
            for temporal_size in temporal_candidates:
                if temporal_size is not None and temporal_size >= num_frames:
                    continue
                tile_frames = num_frames if temporal_size is None else temporal_size
                peak = _estimate_diffusion_peak_gb(
                    tile_height, tile_width, tile_frames, compute_dtype=compute_dtype
                )
                peak += context_gb + _estimate_tiled_assembly_gb(
                    height, width, tile_frames, spatial_tiled=spatial_size is not None
                )
                if peak > budget_gb:
                    continue
                temporal = (
                    None
                    if temporal_size is None
                    else TemporalChunkConfig(temporal_size, min(24, temporal_size - 8))
                )
                spatial = (
                    None
                    if spatial_size is None
                    else SpatialTilingConfig(spatial_size, min(64, spatial_size - 32))
                )
                return cls(spatial_config=spatial, temporal_config=temporal)
        raise MemoryError("No diffusion VAE tile fits the available decode memory budget")

    @classmethod
    def auto(
        cls,
        height: int,
        width: int,
        num_frames: int,
        memory_budget_gb: float | None = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> TilingConfig | None:
        """Select the native Conv3d plan for an output geometry."""
        return cls.auto_native_conv3d(
            height,
            width,
            num_frames,
            memory_budget_gb=memory_budget_gb,
            compute_dtype=compute_dtype,
        )

    @classmethod
    def auto_native_conv3d(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        memory_budget_gb: float | None = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
    ) -> TilingConfig | None:
        """Bound native Conv3d decode jobs by estimated peak memory."""
        height = _strict_int(height, field="height", minimum=1)
        width = _strict_int(width, field="width", minimum=1)
        num_frames = _strict_int(num_frames, field="num_frames", minimum=1)
        budget_gb = (
            memory_budget_gb
            if memory_budget_gb is not None
            else capture_decode_memory_budget().available_gb
        )
        if not math.isfinite(budget_gb) or budget_gb <= 0:
            raise ValueError("memory_budget_gb must be positive")
        full_peak_gb = _estimate_native_conv3d_peak_gb(
            height,
            width,
            num_frames,
            compute_dtype=compute_dtype,
        )
        if full_peak_gb <= budget_gb:
            return None

        for spatial_tile in (None, 512, 256, 128, 64):
            if spatial_tile is not None and spatial_tile >= max(height, width):
                continue
            effective_h = height if spatial_tile is None else min(height, spatial_tile)
            effective_w = width if spatial_tile is None else min(width, spatial_tile)
            temporal_candidates = (
                (256, 128, 64, 40, 32, 16)
                if spatial_tile is None
                else (None, 256, 128, 64, 40, 32, 16)
            )
            for tile_frames in temporal_candidates:
                if tile_frames is not None and tile_frames >= num_frames:
                    continue
                # A temporal tile can retain one leading causal-context frame.
                work_frames = (
                    num_frames if tile_frames is None else min(num_frames, tile_frames + 1)
                )
                peak_gb = _estimate_native_conv3d_peak_gb(
                    effective_h,
                    effective_w,
                    work_frames,
                    compute_dtype=compute_dtype,
                ) + _estimate_tiled_assembly_gb(
                    height,
                    width,
                    work_frames,
                    spatial_tiled=spatial_tile is not None,
                )
                if peak_gb <= budget_gb:
                    return cls(
                        spatial_config=(
                            None
                            if spatial_tile is None
                            else SpatialTilingConfig(spatial_tile, min(64, spatial_tile - 32))
                        ),
                        temporal_config=(
                            None if tile_frames is None else TemporalChunkConfig(tile_frames, 8)
                        ),
                    )
        raise MemoryError("No Conv3d VAE tile fits the available decode memory budget")

    @classmethod
    def temporal_only(
        cls,
        tile_size: int = 64,
        overlap: int = 24,
    ) -> TilingConfig:
        """Build a temporal-only decode plan."""
        return cls(
            temporal_config=TemporalChunkConfig(tile_size, overlap),
        )

    def to_dict(self) -> dict[str, int | None]:
        """Return decoded-coordinate settings suitable for run receipts."""
        temporal = self.temporal_config
        spatial = self.spatial_config
        return {
            "temporal_tile_frames": (None if temporal is None else temporal.chunk_size_in_frames),
            "temporal_overlap_frames": (
                None if temporal is None else temporal.chunk_overlap_in_frames
            ),
            "spatial_tile_pixels": None if spatial is None else spatial.tile_size_in_pixels,
            "spatial_overlap_pixels": (None if spatial is None else spatial.tile_overlap_in_pixels),
        }


@dataclass(frozen=True)
class AxisTiles:
    """Starts, ends, and overlap ramps for one latent axis."""

    starts: list[int]
    ends: list[int]
    left_ramps: list[int]
    right_ramps: list[int]


@dataclass(frozen=True)
class TilingPlanReceipt:
    """Resolved VAE tile geometry and job counts for one decode."""

    latent_shape: tuple[int, int, int]
    decoded_shape: tuple[int, int, int]
    temporal_tiles: int
    spatial_height_tiles: int
    spatial_width_tiles: int
    total_tiles: int
    resolved_config: TilingConfig
    decoder_kind: str = "native-conv3d"
    attention_tiling: dict[str, int | str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready plan record."""
        result: dict[str, object] = {
            "latent_shape": list(self.latent_shape),
            "decoded_shape": list(self.decoded_shape),
            "temporal_tiles": self.temporal_tiles,
            "spatial_height_tiles": self.spatial_height_tiles,
            "spatial_width_tiles": self.spatial_width_tiles,
            "total_tiles": self.total_tiles,
            "resolved_config": self.resolved_config.to_dict(),
        }
        if self.decoder_kind != "native-conv3d" or self.attention_tiling is not None:
            result["decoder_kind"] = self.decoder_kind
        if self.attention_tiling is not None:
            result["attention_tiling"] = self.attention_tiling
        return result


def _split_axis(length: int, tile_size: int, overlap: int) -> AxisTiles:
    """Split one latent axis into overlapping intervals."""
    if length <= 0:
        raise ValueError(f"axis length must be positive, got {length}")
    if tile_size <= 0:
        raise ValueError(f"tile size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"invalid overlap {overlap} for tile size {tile_size}")
    if length <= tile_size:
        return AxisTiles([0], [length], [0], [0])

    stride = tile_size - overlap
    starts: list[int] = []
    ends: list[int] = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        starts.append(start)
        ends.append(end)
        if end >= length:
            break
        start += stride

    left = [0] * len(starts)
    right = [0] * len(starts)
    for index in range(1, len(starts)):
        actual_overlap = ends[index - 1] - starts[index]
        left[index] = actual_overlap
        right[index - 1] = actual_overlap
    return AxisTiles(starts, ends, left, right)


def _split_temporal_axis(length: int, tile_size: int, overlap: int) -> AxisTiles:
    """Split time and add one causal context latent after the first tile."""
    base = _split_axis(length, tile_size, overlap)
    starts = list(base.starts)
    left = list(base.left_ramps)
    for index in range(1, len(starts)):
        if starts[index] > 0:
            starts[index] -= 1
            left[index] += 1
    return AxisTiles(starts, list(base.ends), left, list(base.right_ramps))


def _temporal_output_slice(
    start_latent: int,
    end_latent: int,
    left_ramp_latent: int,
    right_ramp_latent: int,
    scale: int,
) -> tuple[slice[int, int, int | None], mx.array]:
    start = 0 if start_latent == 0 else start_latent * scale
    stop = 1 if end_latent <= 1 else 1 + (end_latent - 1) * scale
    left = 0
    if left_ramp_latent > 0:
        left = 1 + (left_ramp_latent - 1) * scale
    right = right_ramp_latent * scale
    mask = compute_trapezoidal_mask_1d(
        stop - start,
        left,
        right,
        left_starts_from_0=True,
    )
    return slice(start, stop), mask


def _spatial_output_slice(
    start_latent: int,
    end_latent: int,
    left_ramp_latent: int,
    right_ramp_latent: int,
    scale: int,
) -> tuple[slice[int, int, int | None], mx.array]:
    start = start_latent * scale
    stop = end_latent * scale
    mask = compute_trapezoidal_mask_1d(
        stop - start,
        left_ramp_latent * scale,
        right_ramp_latent * scale,
    )
    return slice(start, stop), mask


def _assign_add_5d(
    array: mx.array,
    update: mx.array,
    time_slice: slice[int, int, int | None],
    height_slice: slice[int, int, int | None],
    width_slice: slice[int, int, int | None],
) -> mx.array:
    """Accumulate through direct slicing, avoiding scatter corruption."""
    current = array[:, :, time_slice, height_slice, width_slice]
    array[:, :, time_slice, height_slice, width_slice] = current + update
    return array


def _merge_temporal_pending(
    pending: mx.array | None,
    pending_weights: mx.array | None,
    pending_start: int,
    chunk: mx.array,
    chunk_weights: mx.array,
    chunk_start: int,
) -> tuple[mx.array, mx.array, int]:
    if pending is None or pending_weights is None:
        return chunk, chunk_weights, chunk_start

    pending_end = pending_start + pending.shape[2]
    chunk_end = chunk_start + chunk.shape[2]
    merged_start = min(pending_start, chunk_start)
    merged_end = max(pending_end, chunk_end)
    merged_time = merged_end - merged_start

    batch, channels, _frames, height, width = pending.shape
    weight_height = pending_weights.shape[3]
    weight_width = pending_weights.shape[4]
    merged = mx.zeros(
        (batch, channels, merged_time, height, width),
        dtype=mx.float32,
    )
    merged_weights = mx.zeros(
        (1, 1, merged_time, weight_height, weight_width),
        dtype=mx.float32,
    )
    pending_slice = slice(
        pending_start - merged_start,
        pending_end - merged_start,
    )
    chunk_slice = slice(chunk_start - merged_start, chunk_end - merged_start)
    full_height = slice(0, height)
    full_width = slice(0, width)
    weight_height_slice = slice(0, weight_height)
    weight_width_slice = slice(0, weight_width)

    merged = _assign_add_5d(
        merged,
        pending,
        pending_slice,
        full_height,
        full_width,
    )
    merged = _assign_add_5d(
        merged,
        chunk,
        chunk_slice,
        full_height,
        full_width,
    )
    merged_weights = _assign_add_5d(
        merged_weights,
        pending_weights,
        pending_slice,
        weight_height_slice,
        weight_width_slice,
    )
    merged_weights = _assign_add_5d(
        merged_weights,
        chunk_weights,
        chunk_slice,
        weight_height_slice,
        weight_width_slice,
    )
    mx.eval(merged, merged_weights)
    return merged, merged_weights, merged_start


def _as_bcfhw_latent(latent: mx.array) -> mx.array:
    if latent.ndim == 4:
        latent = latent[None]
    if latent.ndim != 5 or any(dimension <= 0 for dimension in latent.shape):
        raise ValueError(f"expected a non-empty BCFHW latent, got {latent.shape}")
    return latent


def _decoder_output_channels(decoder: object) -> int:
    config = getattr(decoder, "config", None)
    channels = getattr(config, "out_channels", 3)
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("decoder output channel count must be a positive integer")
    return channels


def _decoder_scale(decoder: object) -> tuple[int, int, int]:
    config = getattr(decoder, "config", None)
    scale = getattr(config, "decoder_scale", None)
    values = (
        getattr(scale, "time", DEFAULT_TEMPORAL_SCALE),
        getattr(scale, "height", DEFAULT_SPATIAL_SCALE),
        getattr(scale, "width", DEFAULT_SPATIAL_SCALE),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("decoder scale factors must be positive integers")
    return values


def _validate_decoder_output(
    decoded: mx.array,
    *,
    batch: int,
    channels: int,
) -> None:
    if decoded.ndim != 5 or decoded.shape[0] != batch or decoded.shape[1] != channels:
        raise ValueError(f"decoder returned incompatible BCFHW output {tuple(decoded.shape)}")
    if any(dimension <= 0 for dimension in decoded.shape):
        raise ValueError(f"decoder returned empty output {tuple(decoded.shape)}")


def decode_single_pass(
    latent: mx.array,
    decoder: DecoderCallable,
    *,
    timestep: float | None = 0.05,
    reporter: Reporter | None = None,
) -> mx.array:
    """Decode one complete latent without tiling."""
    latent = _as_bcfhw_latent(latent)
    return decoder(latent, timestep=timestep, reporter=reporter)


def decode_streaming(
    latent: mx.array,
    decoder_fn: DecoderCallable,
    tiling_config: TilingConfig | None = None,
    timestep: float | None = 0.05,
    *,
    reporter: Reporter | None = None,
    plan_callback: Callable[[TilingPlanReceipt], None] | None = None,
    noise_state_callback: Callable[[NoiseStreamState], None] | None = None,
    seed: int = 0,
    noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
) -> Iterator[mx.array]:
    """Decode and yield blended BCFHW chunks without retaining the full video."""
    latent = _as_bcfhw_latent(latent)
    specialized = getattr(decoder_fn, "decode_streaming", None)
    if callable(specialized):
        yield from specialized(
            latent,
            tiling_config=tiling_config,
            seed=seed,
            noise_backend=noise_backend,
            reporter=reporter,
            plan_callback=plan_callback,
            noise_state_callback=noise_state_callback,
        )
        return
    batch, _channels, latent_time, latent_height, latent_width = latent.shape
    scale_time, scale_height, scale_width = _decoder_scale(decoder_fn)

    if tiling_config is None:
        output_frames = 1 + (int(latent_time) - 1) * scale_time
        tiling_config = TilingConfig.auto_native_conv3d(
            int(latent_height) * scale_height,
            int(latent_width) * scale_width,
            output_frames,
            compute_dtype=getattr(decoder_fn, "compute_dtype", mx.bfloat16),
        )

    if tiling_config is None:
        time_tiles = AxisTiles([0], [latent_time], [0], [0])
        height_tiles = AxisTiles([0], [latent_height], [0], [0])
        width_tiles = AxisTiles([0], [latent_width], [0], [0])
        spatial_off = True
    else:
        if tiling_config.temporal_config is not None:
            temporal = tiling_config.temporal_config
            time_tiles = _split_temporal_axis(
                latent_time,
                temporal.chunk_size_in_frames // scale_time,
                temporal.chunk_overlap_in_frames // scale_time,
            )
        else:
            time_tiles = AxisTiles([0], [latent_time], [0], [0])

        if tiling_config.spatial_config is not None:
            spatial = tiling_config.spatial_config
            height_tiles = _split_axis(
                latent_height,
                spatial.tile_size_in_pixels // scale_height,
                spatial.tile_overlap_in_pixels // scale_height,
            )
            width_tiles = _split_axis(
                latent_width,
                spatial.tile_size_in_pixels // scale_width,
                spatial.tile_overlap_in_pixels // scale_width,
            )
        else:
            height_tiles = AxisTiles([0], [latent_height], [0], [0])
            width_tiles = AxisTiles([0], [latent_width], [0], [0])
        spatial_off = tiling_config.spatial_config is None

    temporal_count = len(time_tiles.starts)
    height_count = len(height_tiles.starts)
    width_count = len(width_tiles.starts)
    total_jobs = temporal_count * height_count * width_count
    if plan_callback is not None:
        plan_callback(
            TilingPlanReceipt(
                latent_shape=(int(latent_time), int(latent_height), int(latent_width)),
                decoded_shape=(
                    1 + (int(latent_time) - 1) * scale_time,
                    int(latent_height) * scale_height,
                    int(latent_width) * scale_width,
                ),
                temporal_tiles=temporal_count,
                spatial_height_tiles=height_count,
                spatial_width_tiles=width_count,
                total_tiles=total_jobs,
                resolved_config=(TilingConfig() if tiling_config is None else tiling_config),
            )
        )
    _log.info(
        "VAE decode plan: temporal=%d spatial=%dx%d total=%d",
        temporal_count,
        height_count,
        width_count,
        total_jobs,
    )

    sink = reporter if reporter is not None else NullReporter()
    phase = "VAE decode tiles"
    output_channels = _decoder_output_channels(decoder_fn)

    if total_jobs == 1:
        sink.phase_start(phase, total=1, unit="tile")
        try:
            decoded = decoder_fn(latent, timestep=timestep)
            _validate_decoder_output(
                decoded,
                batch=batch,
                channels=output_channels,
            )
            mx.eval(decoded)
            sink.phase_advance(phase)
        finally:
            sink.phase_end(phase)
        # The decoded tensor is fully materialized here. End the decoder-owned
        # phase before handing it to a downstream VSR/encoder; otherwise the
        # cooperative generator makes tile timing include frame consumption.
        yield decoded
        return

    sink.phase_start(phase, total=total_jobs, unit="tile")
    pending: mx.array | None = None
    pending_weights: mx.array | None = None
    pending_start = 0
    output_height = latent_height * scale_height
    output_width = latent_width * scale_width

    try:
        for time_index in range(temporal_count):
            time_start = time_tiles.starts[time_index]
            time_end = time_tiles.ends[time_index]
            output_time_slice, mask_time = _temporal_output_slice(
                time_start,
                time_end,
                time_tiles.left_ramps[time_index],
                time_tiles.right_ramps[time_index],
                scale_time,
            )
            chunk_time = output_time_slice.stop - output_time_slice.start

            if spatial_off:
                chunk = None
                chunk_weights = None
            else:
                chunk = mx.zeros(
                    (
                        batch,
                        output_channels,
                        chunk_time,
                        output_height,
                        output_width,
                    ),
                    dtype=mx.float32,
                )
                chunk_weights = mx.zeros(
                    (1, 1, chunk_time, output_height, output_width),
                    dtype=mx.float32,
                )
                mx.eval(chunk, chunk_weights)

            for height_index in range(height_count):
                height_start = height_tiles.starts[height_index]
                height_end = height_tiles.ends[height_index]
                output_height_slice, mask_height = _spatial_output_slice(
                    height_start,
                    height_end,
                    height_tiles.left_ramps[height_index],
                    height_tiles.right_ramps[height_index],
                    scale_height,
                )

                for width_index in range(width_count):
                    width_start = width_tiles.starts[width_index]
                    width_end = width_tiles.ends[width_index]
                    output_width_slice, mask_width = _spatial_output_slice(
                        width_start,
                        width_end,
                        width_tiles.left_ramps[width_index],
                        width_tiles.right_ramps[width_index],
                        scale_width,
                    )
                    tile_latent = latent[
                        :,
                        :,
                        time_start:time_end,
                        height_start:height_end,
                        width_start:width_end,
                    ]
                    decoded = decoder_fn(tile_latent, timestep=timestep)
                    _validate_decoder_output(
                        decoded,
                        batch=batch,
                        channels=output_channels,
                    )
                    mx.eval(decoded)

                    expected_height = output_height_slice.stop - output_height_slice.start
                    expected_width = output_width_slice.stop - output_width_slice.start
                    actual_time = min(decoded.shape[2], chunk_time)
                    actual_height = min(decoded.shape[3], expected_height)
                    actual_width = min(decoded.shape[4], expected_width)
                    decoded = decoded[:, :, :actual_time, :actual_height, :actual_width].astype(
                        mx.float32
                    )

                    local_time_slice = slice(0, actual_time)
                    actual_height_slice = slice(
                        output_height_slice.start,
                        output_height_slice.start + actual_height,
                    )
                    actual_width_slice = slice(
                        output_width_slice.start,
                        output_width_slice.start + actual_width,
                    )

                    if spatial_off:
                        mask = mask_time[:actual_time].reshape(1, 1, actual_time, 1, 1)
                        weighted = decoded * mask
                        if actual_time == chunk_time:
                            chunk = weighted
                            chunk_weights = mask
                        else:
                            chunk = mx.zeros(
                                (
                                    batch,
                                    output_channels,
                                    chunk_time,
                                    output_height,
                                    output_width,
                                ),
                                dtype=mx.float32,
                            )
                            chunk[
                                :,
                                :,
                                local_time_slice,
                                actual_height_slice,
                                actual_width_slice,
                            ] = weighted
                            chunk_weights = mx.zeros(
                                (1, 1, chunk_time, 1, 1),
                                dtype=mx.float32,
                            )
                            chunk_weights[:, :, local_time_slice, :, :] = mask
                    else:
                        if chunk is None or chunk_weights is None:
                            raise RuntimeError("spatial VAE tile assembly was not initialized")
                        mask = (
                            mask_time[:actual_time].reshape(1, 1, actual_time, 1, 1)
                            * mask_height[:actual_height].reshape(1, 1, 1, actual_height, 1)
                            * mask_width[:actual_width].reshape(1, 1, 1, 1, actual_width)
                        )
                        chunk = _assign_add_5d(
                            chunk,
                            decoded * mask,
                            local_time_slice,
                            actual_height_slice,
                            actual_width_slice,
                        )
                        chunk_weights = _assign_add_5d(
                            chunk_weights,
                            mask,
                            local_time_slice,
                            actual_height_slice,
                            actual_width_slice,
                        )

                    mx.eval(chunk, chunk_weights)
                    sink.phase_advance(phase)
                    del decoded, mask, tile_latent
                    mx.clear_cache()

            if chunk is None or chunk_weights is None:
                raise RuntimeError("VAE tile plan produced no decoded chunk")

            if pending is not None and pending_weights is not None:
                pending_end = pending_start + pending.shape[2]
                ready_time = max(
                    0,
                    min(output_time_slice.start, pending_end) - pending_start,
                )
                if ready_time:
                    ready = pending[:, :, :ready_time]
                    ready_weights = pending_weights[:, :, :ready_time]
                    ready = ready / mx.maximum(ready_weights, 1e-8)
                    mx.eval(ready)
                    yield ready
                    pending = pending[:, :, ready_time:]
                    pending_weights = pending_weights[:, :, ready_time:]
                    pending_start += ready_time
                    mx.eval(pending, pending_weights)

            pending, pending_weights, pending_start = _merge_temporal_pending(
                pending,
                pending_weights,
                pending_start,
                chunk,
                chunk_weights,
                output_time_slice.start,
            )
            del chunk, chunk_weights
            gc.collect()
            mx.clear_cache()

        if pending is not None and pending_weights is not None and pending.shape[2]:
            pending = pending / mx.maximum(pending_weights, 1e-8)
            mx.eval(pending)
            yield pending
    finally:
        sink.phase_end(phase)


__all__ = [
    "DEFAULT_SPATIAL_SCALE",
    "DEFAULT_TEMPORAL_SCALE",
    "SpatialTilingConfig",
    "TemporalChunkConfig",
    "TilingConfig",
    "TilingPlanReceipt",
    "compute_trapezoidal_mask_1d",
    "decode_single_pass",
    "decode_streaming",
]
