"""Qwen Image VAE sample-space tiling policy and overlap blending."""

from dataclasses import dataclass

import mlx.core as mx

from kinomlx.modeling.memory import DecodeMemoryBudget, capture_decode_memory_budget

from ..metadata import Krea2VAEConfig


def estimate_decode_memory(
    config: Krea2VAEConfig,
    height: int,
    width: int,
    *,
    compute_dtype: mx.Dtype,
    output_scale: int = 1,
) -> tuple[int, int]:
    """Estimate incremental peak and largest buffer for one full still decode.

    The 384-wide bottleneck takes MLX's unfused FP32 attention path, so its
    score/softmax storage grows with the SQUARE of the latent pixel count.
    The convolution envelope includes expanded upsample inputs, workspace,
    residuals, and FP32 normalization intermediates. Neither estimate includes
    decoder weights: those are already in the measured live allocation set.
    """
    if compute_dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError(f"unsupported Krea VAE compute dtype {compute_dtype}")
    if (
        isinstance(output_scale, bool)
        or not isinstance(output_scale, int)
        or output_scale not in (1, 2)
    ):
        raise ValueError("Krea VAE output_scale must be 1 or 2")
    compression = config.spatial_compression
    if height <= 0 or width <= 0 or height % compression or width % compression:
        raise ValueError("decode dimensions must be positive and compression-aligned")
    itemsize = 4 if compute_dtype == mx.float32 else 2
    pixels = (height // compression) * (width // compression)
    tokens = pixels
    channels = config.base_dim * config.dim_mult[-1]
    score_bytes = tokens * tokens * 4
    attention_peak = 2 * score_bytes + tokens * channels * 24
    expanded_elements = 0
    for index, multiplier in enumerate(reversed(config.dim_mult)):
        channels = config.base_dim * multiplier
        if index < len(config.dim_mult) - 1:
            pixels *= 4
        expanded_elements = max(expanded_elements, pixels * channels)
    feature_bytes = expanded_elements * itemsize
    convolution_peak = feature_bytes * 10
    rgb_bytes = height * width * config.input_channels * output_scale**2 * itemsize
    return max(attention_peak, convolution_peak) + 2 * rgb_bytes, max(
        score_bytes, feature_bytes * 9
    )


@dataclass(frozen=True)
class Krea2VAETilingConfig:
    """Diffusers-compatible sample-space tiling geometry."""

    tile_sample_size: int = 256
    tile_sample_stride: int = 192

    @classmethod
    def auto(
        cls,
        config: Krea2VAEConfig,
        height: int,
        width: int,
        *,
        compute_dtype: mx.Dtype,
        output_scale: int = 1,
        budget: DecodeMemoryBudget | None = None,
        reserved_output_bytes: int = 0,
    ) -> Krea2VAETilingConfig | None:
        """Use a full image when it fits, otherwise the largest fitting tile."""
        budget = capture_decode_memory_budget() if budget is None else budget
        available = max(0, budget.available_bytes - reserved_output_bytes)

        def fits(tile: int, *, assembly: int = 0) -> bool:
            peak, buffer = estimate_decode_memory(
                config,
                min(height, tile),
                min(width, tile),
                compute_dtype=compute_dtype,
                output_scale=output_scale,
            )
            return peak + assembly <= available and (
                budget.max_buffer_bytes is None or buffer <= budget.max_buffer_bytes
            )

        if fits(max(height, width)):
            return None
        # Decoded tiles, blend temporaries, and the final RGB assembly coexist.
        assembly = height * width * config.input_channels * output_scale**2 * 4 * 6
        compression = config.spatial_compression
        lower, upper = 2, max(height, width) // compression
        best = 0
        while lower <= upper:
            middle = (lower + upper) // 2
            if fits(middle * compression, assembly=assembly):
                best = middle
                lower = middle + 1
            else:
                upper = middle - 1
        if best == 0:
            raise MemoryError("No Krea VAE tile fits the available decode memory budget")
        tile = best * compression
        overlap = max(1, best // 4) * compression
        return cls(tile_sample_size=tile, tile_sample_stride=tile - overlap)

    def __post_init__(self) -> None:
        if isinstance(self.tile_sample_size, bool) or not isinstance(self.tile_sample_size, int):
            raise TypeError("tile_sample_size must be an integer")
        if isinstance(self.tile_sample_stride, bool) or not isinstance(
            self.tile_sample_stride, int
        ):
            raise TypeError("tile_sample_stride must be an integer")
        if self.tile_sample_size <= 0 or self.tile_sample_stride <= 0:
            raise ValueError("VAE tile size and stride must be positive")
        if self.tile_sample_stride >= self.tile_sample_size:
            raise ValueError("VAE tile stride must be smaller than tile size")


def _blend_overlap(previous: mx.array, current: mx.array, *, axis: int) -> mx.array:
    extent = current.shape[axis]
    shape = [1] * current.ndim
    shape[axis] = extent
    # These short geometry-only tables match the reference's Python scalar
    # coefficients. Both tensor products use FP32 opmath and narrow before the
    # addition; rounding the ramp to BF16/FP16 first changes non-power-of-two blends.
    ratios = tuple(index / extent for index in range(extent))
    incoming = mx.array(ratios, dtype=mx.float32).reshape(shape)
    outgoing = mx.array([1.0 - ratio for ratio in ratios], dtype=mx.float32).reshape(shape)
    old = (previous.astype(mx.float32) * outgoing).astype(previous.dtype)
    new = (current.astype(mx.float32) * incoming).astype(current.dtype)
    return old + new


def blend_vertical(previous: mx.array, current: mx.array, extent: int) -> mx.array:
    """Blend the leading rows of one decoded tile with the tile above it."""
    extent = min(previous.shape[1], current.shape[1], extent)
    if extent <= 0:
        return current
    blended = _blend_overlap(previous[:, -extent:, :, :], current[:, :extent, :, :], axis=1)
    return mx.concatenate([blended, current[:, extent:, :, :]], axis=1)


def blend_horizontal(previous: mx.array, current: mx.array, extent: int) -> mx.array:
    """Blend the leading columns of one decoded tile with its left neighbor."""
    extent = min(previous.shape[2], current.shape[2], extent)
    if extent <= 0:
        return current
    blended = _blend_overlap(previous[:, :, -extent:, :], current[:, :, :extent, :], axis=2)
    return mx.concatenate([blended, current[:, :, extent:, :]], axis=2)


__all__ = ["Krea2VAETilingConfig", "blend_horizontal", "blend_vertical", "estimate_decode_memory"]
