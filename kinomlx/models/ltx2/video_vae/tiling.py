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

from . import diffusion_fused
from .diffusion_fused import SWIGLU_TILE_TOKENS
from .kinds import VideoVAEKind
from .sharpening import ContextSharpening

_log = logging.getLogger(__name__)

DEFAULT_TEMPORAL_SCALE = 8
DEFAULT_SPATIAL_SCALE = 32

# Spatial tile sizes the diffusion tiler considers, largest first; temporal tiles are chosen
# by count (`_equalized_temporal_tiles`). The overlap each tile recomputes is 24 frames and 64
# pixels.
_DIFFUSION_SPATIAL_TILES = (None, 1024, 768, 512, 256, 128, 64)
_TEMPORAL_OVERLAP_FRAMES = 24
_SHORTEST_TEMPORAL_TILE = 16


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


def tile_intervals(
    length: int,
    tile_size: int,
    stride: int,
    min_size: int,
) -> list[tuple[int, int]]:
    """Overlapping ``[start, end)`` tiles of ``length``: starts every ``stride``, each
    ``tile_size`` long, and a last tile shorter than ``min_size`` merged into the one before."""
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [
        (starts[-1], length)
    ]


def _equalized_temporal_tiles(length: int) -> list[int]:
    """Temporal tile lengths (decoded frames, multiples of 8) that cut ``length`` stage-4 frames
    into 2, 3, 4, ... tiles of as nearly equal length as the stride allows, fewest tiles first.

    A plan's work is set by its tile count (every tile past the first recomputes the overlap),
    so for a given count the shortest tile that still gives that count has the smallest peak:
    for 721 decoded frames, 384, 264, 208 and 168 frames give 2, 3, 4 and 5 tiles, where a
    fixed ladder's 224-frame tiles also gave 4 with a short last one.
    """
    sizes: list[int] = []
    for count in range(2, length + 1):
        # The shortest multiple of 8 whose intervals number `count`: the count only falls as
        # the tile grows, so step up from the overlap-free length.
        size = max(_SHORTEST_TEMPORAL_TILE, -(-2 * length // (8 * count)) * 8)
        while True:
            tile = size // _STAGE4_TIME_SCALE
            if tile >= length:
                return sizes
            overlap = min(_TEMPORAL_OVERLAP_FRAMES, size - 8) // _STAGE4_TIME_SCALE
            tiles = len(tile_intervals(length, tile, tile - overlap, _STAGE4_MIN_TILE))
            if tiles <= count:
                break
            size += 8
        if tiles == count and (not sizes or size < sizes[-1]):
            sizes.append(size)
        if size <= _SHORTEST_TEMPORAL_TILE:
            return sizes
    return sizes


def _estimate_conv3d_peak_gb(
    height: int,
    width: int,
    tile_frames: int,
    *,
    compute_dtype: mx.Dtype = mx.bfloat16,
    optimized_fp16: bool = False,
) -> float:
    """Estimate incremental Conv3d peak above the VAE-entry live set.

    The Phase D MLX 0.32.1 sweep covered 512x320, 768x448, and 1024x576
    at 33 and 65 decoded frames. The 121-frame 768x448 MLX 0.32.2 check
    reached 30.02 GB above the entry live set; a coefficient of 0.38 also
    covers that longer decode. FP32 doubles the activation/workspace
    estimate. The eager FP16 Winograd decoder has a separate measured
    envelope; tiled output assembly is accounted separately in FP32.
    """
    if height <= 0 or width <= 0 or tile_frames <= 0:
        raise ValueError("decode dimensions must be positive")
    area_scale = (height * width) / (1024 * 576)
    if compute_dtype == mx.float16 and optimized_fp16:
        # The selected LTX-2.3/2.5 decoder uses eager FP16 Winograd bands.
        # This envelope covers measured 64/128/160/256-frame 768x448 tiles;
        # auto_conv3d adds the separate full-frame assembly allowance.
        return 4.0 + (5.0 + 0.10 * tile_frames) * area_scale
    if compute_dtype in (mx.bfloat16, mx.float16):
        dtype_scale = 1.0
    elif compute_dtype == mx.float32:
        dtype_scale = 2.0
    else:
        raise ValueError(f"unsupported native Conv3d compute dtype {compute_dtype}")
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


# The diffusion decoder's stage-5 width and SwiGLU hidden width, and on its FP16 kernel route
# the feature-sized buffers each half of a block holds at once. The attention half holds its
# residual, its normalized rows, Q, K and V; the attention and proj outputs take buffers those
# free. The SwiGLU half holds its residual, its normalized rows and its result beside the
# gate/up products and SiLU outputs of the few tiles MLX, evaluating the tiles' assignments
# into the result in order, has encoded ahead of the GPU (1.9 GB, about four tiles, measured
# on a 768x448x65 decode).
_STAGE5_CHANNELS = 256
_STAGE5_HIDDEN = 1024
_KERNEL_ROUTE_ATTENTION_FEATURES = 5
_KERNEL_ROUTE_SWIGLU_FEATURES = 3
_SWIGLU_TILES_IN_FLIGHT = 5
# The final stage's attention: four 64-wide heads over an 11x11x11 window. Its whole-volume
# kernels bound a tile (`diffusion_fused.addressable`); the earlier stages hold at most an
# eighth of its tokens.
_STAGE5_HEADS = 4
_STAGE5_WINDOW = (11, 11, 11)


def _diffusion_grid(height: int, width: int) -> tuple[int, int]:
    """The decoder's working grid for a requested size: whole 32-pixel latents, and at least
    the 3x7x7 latent the official decoder pads tiny images to."""
    return max(224, math.ceil(height / 32) * 32), max(224, math.ceil(width / 32) * 32)


def _stage5_grid(height: int, width: int, frames: int) -> tuple[int, int, int]:
    """The final stage's ``(T, H, W)`` token grid of a tile: full frame rate, a quarter of the
    pixels per axis."""
    return frames, math.ceil(height / 4), math.ceil(width / 4)


def _stage5_tokens(height: int, width: int, frames: int) -> int:
    return math.prod(_stage5_grid(height, width, frames))


def _latent_context_bytes(height: int, width: int, num_frames: int, compute_dtype: mx.Dtype) -> int:
    """The whole-clip context the tiles read, as the peak bound counts it: 8 bytes per output
    pixel and frame (16 in FP32), with 16 frames of latent alignment and trailing context."""
    itemsize = 4 if compute_dtype == mx.float32 else 2
    return height * width * (num_frames + 16) * 4 * itemsize


def device_buffer_limit() -> int:
    """The largest buffer this device allocates, in bytes: MLX refuses a larger one."""
    return int(mx.device_info()["max_buffer_length"])


def _estimate_kernel_route_peak_gb(height: int, width: int, frames: int) -> float:
    """Bound the FP16 kernel route's stage-5 live set above the resident weights and the
    stage-3 context: the tile's latent context, the larger of a block's two halves, and 1 GiB
    for the noise, the modulation and the kernels' workspaces.

    On an M1 Max, 768x448 decodes of 65 and 121 frames in one pass at a 16 GiB cache peaked
    5.00 and 8.44 GB above the resident weights (the SwiGLU half, then the attention half);
    this bound gives 6.12 and 9.07. The first 128-frame tile of a 1024x576x721 decode peaked
    at 19.2 GB in all, 22.8 bounded with the context and the tile assembly.
    """
    tokens = _stage5_tokens(height, width, frames)
    feature_bytes = tokens * _STAGE5_CHANNELS * 2
    tiles = -(-tokens // SWIGLU_TILE_TOKENS)
    tile_bytes = SWIGLU_TILE_TOKENS * (3 * _STAGE5_HIDDEN + _STAGE5_CHANNELS) * 2
    attention_bytes = _KERNEL_ROUTE_ATTENTION_FEATURES * feature_bytes
    swiglu_bytes = (
        _KERNEL_ROUTE_SWIGLU_FEATURES * feature_bytes
        + min(tiles, _SWIGLU_TILES_IN_FLIGHT) * tile_bytes
    )
    return (feature_bytes + max(attention_bytes, swiglu_bytes) + 1024**3) / 1_000_000_000


# What an FP16 decode's tiles cost under a plan, fitted on an M1 Max to the first tile of
# 1024x576x721 decodes (a tile's time is proportional to its frames, and the first tile
# reproduces the full decodes' tiles: 128-frame tiles at a 16 GiB cache took 20.7 s each and
# the decode 152.1 s, 192-frame tiles 31.2 s and 145.4 s). On the kernel route: 4.35 us per
# stage-5 token of every tile, overlaps included, about 0.3 s a tile for its noise, blend and
# cache clear, and, measured as system time, 0.25 s per tile for each gigabyte by which 4.5 of
# the tile's stage-5 features (the five an attention half holds, less what the cache keeps of
# them) exceed the allocator cache: 192-frame tiles took 32.4 s at 8 GiB and 34.5 s at 4 GiB,
# 128-frame tiles 21.3 s at 8 GiB, 224-frame tiles 37.0 s at 16 GiB, all within 0.6 s of the
# fit. The constants rank plans; they do not predict another machine's seconds.
_SECONDS_PER_STAGE5_TOKEN = 4.35e-6
_CHURN_FEATURES = 4.5
_CHURN_SECONDS_PER_GB = 0.25
_SECONDS_PER_TILE = 0.3
# The decoder tiles stage 4, which runs at half the output's frame rate and an eighth of its
# pixels per axis, and merges a last tile shorter than its windows need.
_STAGE4_TIME_SCALE = 2
_STAGE4_SPACE_SCALE = 8
_STAGE4_MIN_TILE = 6


def estimated_diffusion_seconds(
    config: TilingConfig | None,
    height: int,
    width: int,
    frames: int,
    *,
    cache_gb: float,
) -> float:
    """Rank an FP16 diffusion decode plan: the tiles' stage-4 and stage-5 work, overlaps
    included, plus the time to fault in the buffers a ``cache_gb`` allocator cache cannot
    hold. Stages 1 to 3 cost the same under every plan and are left out."""
    temporal = None if config is None else config.temporal_config
    spatial = None if config is None else config.spatial_config
    length_t = (frames - 1) // _STAGE4_TIME_SCALE + 1
    length_h = height // _STAGE4_SPACE_SCALE
    length_w = width // _STAGE4_SPACE_SCALE

    def lengths(length: int, size: int | None, overlap: int, scale: int) -> list[int]:
        if size is None:
            return [length]
        tile = size // scale
        stride = (size - overlap) // scale
        return [
            end - start for start, end in tile_intervals(length, tile, stride, _STAGE4_MIN_TILE)
        ]

    times = lengths(
        length_t,
        None if temporal is None else temporal.chunk_size_in_frames,
        0 if temporal is None else temporal.chunk_overlap_in_frames,
        _STAGE4_TIME_SCALE,
    )
    rows = lengths(
        length_h,
        None if spatial is None else spatial.tile_size_in_pixels,
        0 if spatial is None else spatial.tile_overlap_in_pixels,
        _STAGE4_SPACE_SCALE,
    )
    columns = lengths(
        length_w,
        None if spatial is None else spatial.tile_size_in_pixels,
        0 if spatial is None else spatial.tile_overlap_in_pixels,
        _STAGE4_SPACE_SCALE,
    )
    cache_bytes = cache_gb * 1024**3
    seconds = 0.0
    for t in times:
        for h in rows:
            for w in columns:
                tokens = 8 * t * h * w
                feature_bytes = tokens * _STAGE5_CHANNELS * 2
                churn = max(0.0, _CHURN_FEATURES * feature_bytes - cache_bytes)
                seconds += tokens * _SECONDS_PER_STAGE5_TOKEN + _SECONDS_PER_TILE
                seconds += churn / 1_000_000_000 * _CHURN_SECONDS_PER_GB
    return seconds


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
    FP16 decodes take the kernel route, whose whole-grid half-blocks are
    bounded separately.
    """
    if compute_dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError(f"unsupported diffusion VAE compute dtype {compute_dtype}")
    if compute_dtype == mx.float16:
        return _estimate_kernel_route_peak_gb(height, width, frames)
    itemsize = 4 if compute_dtype == mx.float32 else 2
    padded_frames = math.ceil(frames / 8) * 8 + 16
    feature_bytes = (
        padded_frames * math.ceil(height / 4) * math.ceil(width / 4) * _STAGE5_CHANNELS * itemsize
    )
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

    def estimated_peak_gb(
        self,
        height: int,
        width: int,
        num_frames: int,
        *,
        decoder_kind: VideoVAEKind,
        compute_dtype: mx.Dtype,
        optimized_fp16: bool = False,
    ) -> float:
        """Estimate a user-selected tile using the same bounds as auto tiling."""
        spatial = self.spatial_config
        temporal = self.temporal_config
        spatial_tiled = spatial is not None
        if decoder_kind == VideoVAEKind.DIFFUSION:
            height, width = _diffusion_grid(height, width)
        tile_height, tile_width, tile_frames = self._tile(height, width, num_frames)
        tiled = spatial_tiled or temporal is not None
        assembly_gb = (
            _estimate_tiled_assembly_gb(height, width, tile_frames, spatial_tiled=spatial_tiled)
            if tiled
            else 0.0
        )
        if decoder_kind == VideoVAEKind.DIFFUSION:
            context_gb = (
                _latent_context_bytes(height, width, num_frames, compute_dtype) / 1_000_000_000
            )
            return (
                _estimate_diffusion_peak_gb(
                    tile_height, tile_width, tile_frames, compute_dtype=compute_dtype
                )
                + context_gb
                + assembly_gb
            )
        return (
            _estimate_conv3d_peak_gb(
                tile_height,
                tile_width,
                tile_frames,
                compute_dtype=compute_dtype,
                optimized_fp16=optimized_fp16,
            )
            + assembly_gb
        )

    def _tile(self, height: int, width: int, num_frames: int) -> tuple[int, int, int]:
        """This plan's tile: the whole grid where an axis is untiled, else the tile size,
        and one frame more than a temporal tile for its trailing context."""
        spatial = self.spatial_config
        temporal = self.temporal_config
        tile_height = height if spatial is None else min(height, spatial.tile_size_in_pixels)
        tile_width = width if spatial is None else min(width, spatial.tile_size_in_pixels)
        tile_frames = (
            num_frames if temporal is None else min(num_frames, temporal.chunk_size_in_frames + 1)
        )
        return tile_height, tile_width, tile_frames

    def kernel_route_feature_gb(self, height: int, width: int, num_frames: int) -> float:
        """One stage-5 feature buffer of this plan's tile on the FP16 kernel route, in GB:
        the unit the attention half holds five of and the allocator cache fills with."""
        height, width = _diffusion_grid(height, width)
        tile_height, tile_width, tile_frames = self._tile(height, width, num_frames)
        tokens = _stage5_tokens(tile_height, tile_width, tile_frames)
        return tokens * _STAGE5_CHANNELS * 2 / 1_000_000_000

    def addressable(
        self,
        height: int,
        width: int,
        num_frames: int,
        *,
        compute_dtype: mx.Dtype,
        max_buffer_bytes: int | None = None,
    ) -> bool:
        """Whether this plan's decode can be addressed: the clip's latent context and each
        tile's stage-5 feature buffer within ``max_buffer_bytes`` (this device's buffer limit
        when ``None``; MLX refuses a larger buffer), and on the FP16 kernel route the tile's
        final-stage volume within its kernels' 32-bit launch counts
        (``diffusion_fused.addressable``: 134,217,696 tokens, a 68.7 GB feature). A temporal
        tile holds one frame more than its length."""
        limit = device_buffer_limit() if max_buffer_bytes is None else max_buffer_bytes
        height, width = _diffusion_grid(height, width)
        tile_height, tile_width, tile_frames = self._tile(height, width, num_frames)
        grid = _stage5_grid(tile_height, tile_width, tile_frames)
        itemsize = 4 if compute_dtype == mx.float32 else 2
        feature_bytes = math.prod(grid) * _STAGE5_CHANNELS * itemsize
        context_bytes = _latent_context_bytes(height, width, num_frames, compute_dtype)
        if max(feature_bytes, context_bytes) > limit:
            return False
        if compute_dtype != mx.float16:
            return True
        return diffusion_fused.addressable(grid, heads=_STAGE5_HEADS, window=_STAGE5_WINDOW)

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
        """Choose full or stage-tiled diffusion decode from current headroom: the largest plan
        that fits, a single pass when it does."""
        options = cls.diffusion_options(
            height,
            width,
            num_frames,
            memory_budget_gb=memory_budget_gb,
            compute_dtype=compute_dtype,
        )
        if not options:
            raise MemoryError("No diffusion VAE tile fits the available decode memory budget")
        return options[0]

    @classmethod
    def diffusion_options(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        memory_budget_gb: float | None = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
        max_buffer_bytes: int | None = None,
    ) -> list[TilingConfig | None]:
        """Every diffusion decode plan whose estimated peak fits the headroom, in the order of
        ``diffusion_candidates`` (which leaves out what the device cannot address): ``None``
        (a single pass) when it fits, then temporal tiles by count, then spatial tiles."""
        budget_gb = (
            capture_decode_memory_budget().available_gb
            if memory_budget_gb is None
            else memory_budget_gb
        )
        if not math.isfinite(budget_gb) or budget_gb <= 0:
            raise ValueError("memory_budget_gb must be positive")
        # The decoder pads a clip to its shortest working grid, 17 frames.
        num_frames = max(17, _strict_int(num_frames, field="num_frames", minimum=1))
        return [
            candidate
            for candidate in cls.diffusion_candidates(
                height,
                width,
                num_frames,
                compute_dtype=compute_dtype,
                max_buffer_bytes=max_buffer_bytes,
            )
            if (cls() if candidate is None else candidate).estimated_peak_gb(
                height,
                width,
                num_frames,
                decoder_kind=VideoVAEKind.DIFFUSION,
                compute_dtype=compute_dtype,
            )
            <= budget_gb
        ]

    @classmethod
    def diffusion_candidates(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        compute_dtype: mx.Dtype = mx.float16,
        max_buffer_bytes: int | None = None,
    ) -> list[TilingConfig | None]:
        """Every diffusion decode plan the device addresses, memory aside: ``None`` (a single
        pass), temporal tiles by count (fewest first, ``_equalized_temporal_tiles``), then each
        spatial tile size alone and with those temporal tiles. A plan whose tiles the kernels or
        the device cannot address (``addressable``) is left out, so no plan proposed can overflow
        a kernel's index range; ``max_buffer_bytes`` defaults to this device's buffer limit.
        Raises ``ValueError`` when the clip's latent context alone exceeds that limit."""
        height = _strict_int(height, field="height", minimum=1)
        width = _strict_int(width, field="width", minimum=1)
        num_frames = _strict_int(num_frames, field="num_frames", minimum=1)
        height, width = _diffusion_grid(height, width)
        num_frames = max(17, num_frames)
        candidates: list[TilingConfig | None] = [None]
        stage4_frames = (num_frames - 1) // _STAGE4_TIME_SCALE + 1
        temporal_sizes = _equalized_temporal_tiles(stage4_frames)
        for spatial_size in _DIFFUSION_SPATIAL_TILES:
            if spatial_size is not None and spatial_size >= max(height, width):
                continue
            # Spatial tiles with no temporal tiling first; one pass stands at the list's head.
            for temporal_size in ([None] if spatial_size is not None else []) + temporal_sizes:
                temporal = (
                    None
                    if temporal_size is None
                    else TemporalChunkConfig(
                        temporal_size, min(_TEMPORAL_OVERLAP_FRAMES, temporal_size - 8)
                    )
                )
                spatial = (
                    None
                    if spatial_size is None
                    else SpatialTilingConfig(spatial_size, min(64, spatial_size - 32))
                )
                candidates.append(cls(spatial_config=spatial, temporal_config=temporal))
        limit = device_buffer_limit() if max_buffer_bytes is None else max_buffer_bytes
        addressable = [
            candidate
            for candidate in candidates
            if (cls() if candidate is None else candidate).addressable(
                height, width, num_frames, compute_dtype=compute_dtype, max_buffer_bytes=limit
            )
        ]
        if not addressable:
            context = _latent_context_bytes(height, width, num_frames, compute_dtype)
            raise ValueError(
                f"a {width}x{height}x{num_frames} diffusion decode's {context:,}-byte latent "
                f"context exceeds this device's {limit:,}-byte buffer limit"
            )
        return addressable

    @classmethod
    def auto(
        cls,
        height: int,
        width: int,
        num_frames: int,
        memory_budget_gb: float | None = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
        *,
        optimized_fp16: bool = False,
    ) -> TilingConfig | None:
        """Select the native Conv3d plan for an output geometry."""
        return cls.auto_conv3d(
            height,
            width,
            num_frames,
            memory_budget_gb=memory_budget_gb,
            compute_dtype=compute_dtype,
            optimized_fp16=optimized_fp16,
        )

    @classmethod
    def auto_conv3d(
        cls,
        height: int,
        width: int,
        num_frames: int,
        *,
        memory_budget_gb: float | None = None,
        compute_dtype: mx.Dtype = mx.bfloat16,
        optimized_fp16: bool = False,
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
        full_peak_gb = _estimate_conv3d_peak_gb(
            height,
            width,
            num_frames,
            compute_dtype=compute_dtype,
            optimized_fp16=optimized_fp16,
        )
        # Longer FP16 Winograd tiles were slower even when they fit in memory.
        max_fast_tile = 128 if compute_dtype == mx.float16 and optimized_fp16 else None
        if full_peak_gb <= budget_gb and (max_fast_tile is None or num_frames <= max_fast_tile):
            return None

        for spatial_tile in (None, 512, 256, 128, 64):
            if spatial_tile is not None and spatial_tile >= max(height, width):
                continue
            effective_h = height if spatial_tile is None else min(height, spatial_tile)
            effective_w = width if spatial_tile is None else min(width, spatial_tile)
            if max_fast_tile is not None:
                temporal_candidates = (
                    (None,) if spatial_tile is not None and num_frames <= max_fast_tile else ()
                ) + (128, 64, 40, 32, 16)
            else:
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
                peak_gb = _estimate_conv3d_peak_gb(
                    effective_h,
                    effective_w,
                    work_frames,
                    compute_dtype=compute_dtype,
                    optimized_fp16=optimized_fp16,
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
    decoder_kind: VideoVAEKind = VideoVAEKind.CONV
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
        if self.decoder_kind != VideoVAEKind.CONV or self.attention_tiling is not None:
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
    context_sharpening: ContextSharpening | None = None,
) -> Iterator[mx.array]:
    """Decode and yield blended BCFHW chunks without retaining the full video.

    ``context_sharpening`` applies only to the diffusion decoder, which owns its streaming.
    """
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
            context_sharpening=context_sharpening,
        )
        return
    if context_sharpening is not None:
        raise ValueError("context sharpening applies only to the diffusion video VAE")
    batch, _channels, latent_time, latent_height, latent_width = latent.shape
    scale_time, scale_height, scale_width = _decoder_scale(decoder_fn)

    if tiling_config is None:
        output_frames = 1 + (int(latent_time) - 1) * scale_time
        tiling_config = TilingConfig.auto_conv3d(
            int(latent_height) * scale_height,
            int(latent_width) * scale_width,
            output_frames,
            compute_dtype=getattr(decoder_fn, "compute_dtype", mx.bfloat16),
            optimized_fp16=(
                getattr(decoder_fn, "compute_dtype", mx.bfloat16) == mx.float16
                and getattr(decoder_fn, "causal", True) is False
            ),
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
