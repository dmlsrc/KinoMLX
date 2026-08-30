"""Native MLX LTX-2.5 diffusion video decoder.

The graph follows the Apache-2.0 Diffusers implementation. Neighborhood
attention is evaluated with bounded query tiles and exact inward-shifted
windows on stock MLX scaled-dot-product attention; no NATTEN, Triton, CuTe, or
CUDA runtime is required.
"""

from __future__ import annotations

import gc
import logging
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlx.core as mx
from mlx.utils import tree_flatten

import kinomlx._mlx_nn as nn
from kinomlx._typing import TensorTree
from kinomlx.io.safetensors import load_weights
from kinomlx.kernels import rms_norm, silu
from kinomlx.reporting import NullReporter, Reporter
from kinomlx.samplers.noise import (
    NoiseStreamState,
    NormalNoiseStream,
    create_normal_noise_stream,
)
from kinomlx.types import DEFAULT_NOISE_BACKEND, NoiseBackend

from .config import VideoVAEConfig
from .ops import PerChannelStatistics

if TYPE_CHECKING:
    from .tiling import TilingConfig, TilingPlanReceipt

_log = logging.getLogger(__name__)

DEFAULT_ATTENTION_SCORE_BUDGET = 128 * 1024 * 1024
DEFAULT_SWIGLU_TOKEN_BUDGET = 16_384


def patchify_spatial(x: mx.array, patch_size: int) -> mx.array:
    """Space-to-depth on H/W with the LTX channel packing order."""
    batch, channels, frames, height, width = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"pixel dimensions {height}x{width} are not divisible by patch size {patch_size}"
        )
    x = x.reshape(
        batch,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.transpose(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(
        batch,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


def unpatchify_spatial(x: mx.array, patch_size: int) -> mx.array:
    """Depth-to-space inverse of :func:`patchify_spatial`."""
    batch, packed_channels, frames, height, width = x.shape
    divisor = patch_size * patch_size
    if packed_channels % divisor:
        raise ValueError(f"packed channel count {packed_channels} is not divisible by {divisor}")
    channels = packed_channels // divisor
    x = x.reshape(batch, channels, patch_size, patch_size, frames, height, width)
    x = x.transpose(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(
        batch,
        channels,
        frames,
        height * patch_size,
        width * patch_size,
    )


@dataclass(frozen=True)
class AttentionTilePlan:
    """One exact bounded neighborhood-attention query partition."""

    grid: tuple[int, int, int]
    kernel: tuple[int, int, int]
    query_tile: tuple[int, int, int]
    heads: int
    score_budget: int

    @property
    def tile_count(self) -> int:
        return math.prod(
            math.ceil(length / tile)
            for length, tile in zip(self.grid, self.query_tile, strict=True)
        )

    @property
    def max_query_tokens(self) -> int:
        return math.prod(self.query_tile)

    @property
    def max_key_tokens(self) -> int:
        return math.prod(
            min(length, tile + kernel - 1)
            for length, tile, kernel in zip(
                self.grid,
                self.query_tile,
                self.kernel,
                strict=True,
            )
        )

    @property
    def max_score_elements(self) -> int:
        return self.heads * self.max_query_tokens * self.max_key_tokens


def plan_attention_tiles(
    grid: tuple[int, int, int],
    kernel: tuple[int, int, int],
    heads: int,
    *,
    score_budget: int = DEFAULT_ATTENTION_SCORE_BUDGET,
) -> AttentionTilePlan:
    """Choose rectangular query tiles bounded by total per-head score work."""
    if any(length <= 0 for length in grid):
        raise ValueError(f"attention grid must be positive, got {grid}")
    if any(size <= 0 for size in kernel):
        raise ValueError(f"attention kernel must be positive, got {kernel}")
    if any(length < size for length, size in zip(grid, kernel, strict=True)):
        raise ValueError(f"neighborhood attention grid {grid} is smaller than kernel {kernel}")
    if heads <= 0 or score_budget <= 0:
        raise ValueError("attention heads and score budget must be positive")

    query_tile = list(grid)

    def score_elements() -> int:
        query_tokens = math.prod(query_tile)
        key_tokens = math.prod(
            min(length, tile + size - 1)
            for length, tile, size in zip(grid, query_tile, kernel, strict=True)
        )
        return heads * query_tokens * key_tokens

    while score_elements() > score_budget:
        candidates = [index for index, length in enumerate(query_tile) if length > 1]
        if not candidates:
            break
        axis = max(candidates, key=lambda index: query_tile[index] / kernel[index])
        query_tile[axis] = max(1, math.ceil(query_tile[axis] / 2))

    plan = AttentionTilePlan(
        grid=grid,
        kernel=kernel,
        query_tile=(query_tile[0], query_tile[1], query_tile[2]),
        heads=heads,
        score_budget=score_budget,
    )
    if plan.max_score_elements > score_budget:
        raise ValueError(
            "attention score budget is smaller than one neighborhood window: "
            f"{plan.max_score_elements} > {score_budget}"
        )
    return plan


@dataclass
class AttentionTilingStats:
    """Runtime receipt for all neighborhood-attention calls in one decode."""

    invocations: int = 0
    query_tiles: int = 0
    max_query_tokens: int = 0
    max_key_tokens: int = 0
    max_score_elements: int = 0
    score_budget: int = DEFAULT_ATTENTION_SCORE_BUDGET

    def reset(self) -> None:
        self.invocations = 0
        self.query_tiles = 0
        self.max_query_tokens = 0
        self.max_key_tokens = 0
        self.max_score_elements = 0

    def record(self, plan: AttentionTilePlan) -> None:
        self.invocations += 1
        self.query_tiles += plan.tile_count
        self.max_query_tokens = max(self.max_query_tokens, plan.max_query_tokens)
        self.max_key_tokens = max(self.max_key_tokens, plan.max_key_tokens)
        self.max_score_elements = max(self.max_score_elements, plan.max_score_elements)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "backend": "mlx-sdpa-exact-neighborhood",
            "invocations": self.invocations,
            "query_tiles": self.query_tiles,
            "score_budget": self.score_budget,
            "max_query_tokens": self.max_query_tokens,
            "max_key_tokens": self.max_key_tokens,
            "max_score_elements": self.max_score_elements,
            "max_boolean_mask_bytes": self.max_query_tokens * self.max_key_tokens,
        }


@dataclass(frozen=True)
class _AxisPadding:
    before: int = 0
    after: int = 0


@dataclass(frozen=True)
class _LatentPadding:
    time: _AxisPadding = _AxisPadding()
    height: _AxisPadding = _AxisPadding()
    width: _AxisPadding = _AxisPadding()


def _minimum_latent_shape(
    stage_kernels: tuple[tuple[int, int, int], ...],
    upsample_strides: tuple[tuple[int, int, int], ...],
    stage5_kernel: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Smallest latent grid that keeps every attention stage in bounds."""
    cumulative = [1, 1, 1]
    minimum = [1, 1, 1]
    for kernel, stride in zip(stage_kernels, upsample_strides, strict=True):
        for axis in range(3):
            minimum[axis] = max(
                minimum[axis],
                math.ceil(kernel[axis] / cumulative[axis]),
            )
            cumulative[axis] *= stride[axis]
    for axis in range(3):
        minimum[axis] = max(
            minimum[axis],
            math.ceil(stage5_kernel[axis] / cumulative[axis]),
        )
    return minimum[0], minimum[1], minimum[2]


def _edge_pad_axis(
    value: mx.array,
    *,
    axis: int,
    minimum: int,
    trailing_only: bool,
) -> tuple[mx.array, _AxisPadding]:
    length = int(value.shape[axis])
    needed = max(0, minimum - length)
    if not needed:
        return value, _AxisPadding()
    before = 0 if trailing_only else needed // 2
    after = needed - before
    parts = []
    if before:
        shape = list(value.shape)
        shape[axis] = before
        first = mx.take(value, mx.array([0]), axis=axis)
        parts.append(mx.broadcast_to(first, shape))
    parts.append(value)
    if after:
        shape = list(value.shape)
        shape[axis] = after
        last = mx.take(value, mx.array([length - 1]), axis=axis)
        parts.append(mx.broadcast_to(last, shape))
    return mx.concatenate(parts, axis=axis), _AxisPadding(before, after)


class DiffusionRMSNorm(nn.Module):
    """Learned RMSNorm with KinoMLX's central FP32-opmath policy."""

    def __init__(self, dims: int) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)

    def __call__(self, value: mx.array) -> mx.array:
        return rms_norm(value, self.weight, 1e-6)


class RotaryPosition3D:
    """Absolute 3D rotary embedding for one attention head."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        if head_dim % 8:
            raise ValueError(f"head_dim must be a multiple of 8, got {head_dim}")
        dim_t = (head_dim // 4) // 2 * 2
        dim_hw = (head_dim - dim_t) // 2
        if dim_hw % 2:
            dim_t -= 2
            dim_hw = (head_dim - dim_t) // 2
        self.dim_split = (dim_t, dim_hw, dim_hw)
        self.base = base

    def _inverse_frequencies(self, dims: int) -> mx.array:
        exponents = mx.arange(0, dims, 2, dtype=mx.float32) / dims
        return mx.power(mx.array(self.base, dtype=mx.float32), -exponents)

    def _rotate_axis(
        self,
        value: mx.array,
        *,
        axis: int,
        offset: int,
    ) -> mx.array:
        out_dtype = value.dtype
        pairs = value.reshape(*value.shape[:-1], value.shape[-1] // 2, 2)
        even = pairs[..., 0].astype(mx.float32)
        odd = pairs[..., 1].astype(mx.float32)
        positions = mx.arange(
            offset,
            offset + value.shape[axis],
            dtype=mx.float32,
        )
        inverse = self._inverse_frequencies(value.shape[-1])
        shape = [1, 1, 1, 1, 1, inverse.shape[0]]
        shape[axis] = positions.shape[0]
        angles = (positions[:, None] * inverse[None, :]).reshape(shape)
        cosine = mx.cos(angles)
        sine = mx.sin(angles)
        rotated = mx.stack(
            [even * cosine - odd * sine, even * sine + odd * cosine],
            axis=-1,
        )
        return rotated.reshape(value.shape).astype(out_dtype)

    def __call__(
        self,
        value: mx.array,
        *,
        offsets: tuple[int, int, int],
    ) -> mx.array:
        dim_t, dim_h, _dim_w = self.dim_split
        rotated_t = self._rotate_axis(value[..., :dim_t], axis=1, offset=offsets[0])
        rotated_h = self._rotate_axis(
            value[..., dim_t : dim_t + dim_h],
            axis=2,
            offset=offsets[1],
        )
        rotated_w = self._rotate_axis(
            value[..., dim_t + dim_h :],
            axis=3,
            offset=offsets[2],
        )
        return mx.concatenate([rotated_t, rotated_h, rotated_w], axis=-1)


def _axis_ranges(length: int, tile: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(start + tile, length)) for start in range(0, length, tile))


def _key_interval(
    query_start: int,
    query_end: int,
    *,
    length: int,
    kernel: int,
) -> tuple[int, int]:
    first = min(max(query_start - kernel // 2, 0), length - kernel)
    last = min(max(query_end - 1 - kernel // 2, 0), length - kernel)
    return first, last + kernel


def _neighborhood_mask(
    query_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    key_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    grid: tuple[int, int, int],
    kernel: tuple[int, int, int],
) -> mx.array:
    axis_masks = []
    for (q0, q1), (k0, k1), length, size in zip(
        query_ranges,
        key_ranges,
        grid,
        kernel,
        strict=True,
    ):
        queries = mx.arange(q0, q1, dtype=mx.int32)
        keys = mx.arange(k0, k1, dtype=mx.int32)
        starts = mx.minimum(
            mx.maximum(queries - size // 2, 0),
            length - size,
        )
        axis_masks.append(
            (keys[None, :] >= starts[:, None]) & (keys[None, :] < starts[:, None] + size)
        )
    mask_t, mask_h, mask_w = axis_masks
    mask = (
        mask_t[:, None, None, :, None, None]
        & mask_h[None, :, None, None, :, None]
        & mask_w[None, None, :, None, None, :]
    )
    query_tokens = math.prod(q1 - q0 for q0, q1 in query_ranges)
    key_tokens = math.prod(k1 - k0 for k0, k1 in key_ranges)
    return mask.reshape(1, 1, query_tokens, key_tokens)


class NeighborhoodAttention3D(nn.Module):
    """Exact inward-shifted 3D neighborhood attention on bounded tiles."""

    _stats: AttentionTilingStats

    def __init__(
        self,
        dims: int,
        kernel_size: tuple[int, int, int],
        *,
        head_dim: int,
        stats: AttentionTilingStats,
        score_budget: int,
    ) -> None:
        super().__init__()
        if dims % head_dim:
            raise ValueError(f"attention width {dims} is not divisible by head_dim {head_dim}")
        self.dims = dims
        self.heads = dims // head_dim
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.qkv = nn.Linear(dims, 3 * dims, bias=True)
        self.proj = nn.Linear(dims, dims, bias=True)
        self.q_norm = DiffusionRMSNorm(head_dim)
        self.k_norm = DiffusionRMSNorm(head_dim)
        self.rope = RotaryPosition3D(head_dim)
        self.score_budget = score_budget
        object.__setattr__(self, "_stats", stats)

    def _project(
        self,
        value: mx.array,
        start: int,
        end: int,
    ) -> mx.array:
        weight = self.qkv.weight[start:end]
        if self.qkv.bias is None:
            raise RuntimeError("diffusion attention QKV projection requires a bias")
        bias = self.qkv.bias[start:end]
        return mx.addmm(bias, value, weight.T)

    def _project_temporal_partition(
        self,
        hidden_states: mx.array,
        time_range: tuple[int, int],
        *,
        grid: tuple[int, int, int],
    ) -> tuple[mx.array, mx.array, mx.array, tuple[int, int]]:
        key_time = _key_interval(
            *time_range,
            length=grid[0],
            kernel=self.kernel_size[0],
        )
        query_source = hidden_states[:, time_range[0] : time_range[1]]
        key_source = hidden_states[:, key_time[0] : key_time[1]]
        query = self._project(query_source, 0, self.dims)
        key = self._project(key_source, self.dims, 2 * self.dims)
        value = self._project(key_source, 2 * self.dims, 3 * self.dims)
        query = query.reshape(*query.shape[:-1], self.heads, self.head_dim)
        key = key.reshape(*key.shape[:-1], self.heads, self.head_dim)
        value = value.reshape(*value.shape[:-1], self.heads, self.head_dim)
        query = self.q_norm(query) * (self.head_dim**-0.5)
        key = self.k_norm(key)
        query = self.rope(query, offsets=(time_range[0], 0, 0))
        key = self.rope(key, offsets=(key_time[0], 0, 0))
        mx.eval(query, key, value)
        return query, key, value, key_time

    def _projected_tile(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        query_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        key_time: tuple[int, int],
        *,
        grid: tuple[int, int, int],
    ) -> mx.array:
        key_height = _key_interval(
            *query_ranges[1],
            length=grid[1],
            kernel=self.kernel_size[1],
        )
        key_width = _key_interval(
            *query_ranges[2],
            length=grid[2],
            kernel=self.kernel_size[2],
        )
        key_ranges = (key_time, key_height, key_width)
        query = query[
            :,
            : query_ranges[0][1] - query_ranges[0][0],
            query_ranges[1][0] : query_ranges[1][1],
            query_ranges[2][0] : query_ranges[2][1],
        ]
        key = key[
            :,
            : key_time[1] - key_time[0],
            key_height[0] : key_height[1],
            key_width[0] : key_width[1],
        ]
        value = value[
            :,
            : key_time[1] - key_time[0],
            key_height[0] : key_height[1],
            key_width[0] : key_width[1],
        ]
        batch = query.shape[0]
        query_tokens = math.prod(end - start for start, end in query_ranges)
        key_tokens = math.prod(end - start for start, end in key_ranges)
        query = query.reshape(batch, query_tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, key_tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, key_tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        mask = _neighborhood_mask(
            query_ranges,
            key_ranges,
            grid=grid,
            kernel=self.kernel_size,
        )
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch,
            *(end - start for start, end in query_ranges),
            self.dims,
        )
        return self.proj(output)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if hidden_states.ndim != 5 or hidden_states.shape[-1] != self.dims:
            raise ValueError(
                f"neighborhood attention expects BFHWC width {self.dims}, got "
                f"{tuple(hidden_states.shape)}"
            )
        grid = (
            int(hidden_states.shape[1]),
            int(hidden_states.shape[2]),
            int(hidden_states.shape[3]),
        )
        plan = plan_attention_tiles(
            grid,
            self.kernel_size,
            self.heads,
            score_budget=self.score_budget,
        )
        self._stats.record(plan)
        time_ranges = _axis_ranges(grid[0], plan.query_tile[0])
        height_ranges = _axis_ranges(grid[1], plan.query_tile[1])
        width_ranges = _axis_ranges(grid[2], plan.query_tile[2])
        frame_groups = []
        for time_range in time_ranges:
            query, key, value, key_time = self._project_temporal_partition(
                hidden_states,
                time_range,
                grid=grid,
            )
            rows = []
            for height_range in height_ranges:
                row = [
                    self._projected_tile(
                        query,
                        key,
                        value,
                        (time_range, height_range, width_range),
                        key_time,
                        grid=grid,
                    )
                    for width_range in width_ranges
                ]
                joined = row[0] if len(row) == 1 else mx.concatenate(row, axis=3)
                mx.eval(joined)
                rows.append(joined)
            joined_rows = rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=2)
            mx.eval(joined_rows)
            frame_groups.append(joined_rows)
        result = frame_groups[0] if len(frame_groups) == 1 else mx.concatenate(frame_groups, axis=1)
        mx.eval(result)
        return result


class TiledSwiGLU(nn.Module):
    """Pointwise SwiGLU evaluated with a fixed token working set."""

    def __init__(self, dims: int, hidden_dims: int, *, token_budget: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dims, hidden_dims, bias=False)
        self.w_gate = nn.Linear(dims, hidden_dims, bias=False)
        self.w_down = nn.Linear(hidden_dims, dims, bias=False)
        self.token_budget = token_budget

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch = hidden_states.shape[0]
        token_shape = hidden_states.shape[1:-1]
        tokens = math.prod(token_shape)
        flat = hidden_states.reshape(batch, tokens, hidden_states.shape[-1])
        outputs = []
        for start in range(0, tokens, self.token_budget):
            tile = flat[:, start : start + self.token_budget]
            output = self.w_down(silu(self.w_gate(tile)) * self.w_up(tile))
            mx.eval(output)
            outputs.append(output)
        result = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
        return result.reshape(hidden_states.shape)


def _swiglu_hidden_dims(dims: int) -> int:
    return (int(dims * 4.0) + 15) // 16 * 16


class DeterministicNABlock(nn.Module):
    def __init__(
        self,
        dims: int,
        kernel: tuple[int, int, int],
        *,
        head_dim: int,
        stats: AttentionTilingStats,
        score_budget: int,
        swiglu_token_budget: int,
    ) -> None:
        super().__init__()
        self.norm1 = DiffusionRMSNorm(dims)
        self.attn = NeighborhoodAttention3D(
            dims,
            kernel,
            head_dim=head_dim,
            stats=stats,
            score_budget=score_budget,
        )
        self.norm2 = DiffusionRMSNorm(dims)
        self.mlp = TiledSwiGLU(
            dims,
            _swiglu_hidden_dims(dims),
            token_budget=swiglu_token_budget,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        mx.eval(hidden_states)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        mx.eval(hidden_states)
        return hidden_states


class SharedAdaLN(nn.Module):
    def __init__(self, dims: int, t_emb_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, 7 * dims, bias=True)

    def __call__(self, timestep_embedding: mx.array) -> tuple[mx.array, ...]:
        projected = self.proj(silu(timestep_embedding))
        return tuple(value[:, None, None, None, :] for value in mx.split(projected, 7, axis=-1))


class DiffusionNABlock(nn.Module):
    def __init__(
        self,
        dims: int,
        kernel: tuple[int, int, int],
        *,
        context_dims: int,
        head_dim: int,
        stats: AttentionTilingStats,
        score_budget: int,
        swiglu_token_budget: int,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_dims, dims, bias=True)
        self.scale_shift_table = mx.zeros((7, dims), dtype=mx.float32)
        self.norm1 = DiffusionRMSNorm(dims)
        self.attn = NeighborhoodAttention3D(
            dims,
            kernel,
            head_dim=head_dim,
            stats=stats,
            score_budget=score_budget,
        )
        self.norm2 = DiffusionRMSNorm(dims)
        self.mlp = TiledSwiGLU(
            dims,
            _swiglu_hidden_dims(dims),
            token_budget=swiglu_token_budget,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        latent_context: mx.array,
        modulation: tuple[mx.array, ...],
    ) -> mx.array:
        values = tuple(
            modulation[index] + self.scale_shift_table[index][None, None, None, None, :]
            for index in range(7)
        )
        scale_msa, shift_msa, _gate_msa, scale_mlp, shift_mlp, _gate_mlp, _gate_ctx = values
        hidden_states = hidden_states + self.context_proj(latent_context)
        mx.eval(hidden_states)
        normalized = self.norm1(hidden_states) * (1.0 + scale_msa) + shift_msa
        hidden_states = hidden_states + self.attn(normalized)
        mx.eval(hidden_states)
        normalized = self.norm2(hidden_states) * (1.0 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.mlp(normalized)
        mx.eval(hidden_states)
        return hidden_states


class LinearPixelShuffle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        reduction: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        product = math.prod(stride)
        projected_channels = product * in_channels // reduction
        self.out_channels = projected_channels // product
        self.proj = nn.Linear(in_channels, projected_channels, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        drop_leading_frame: bool = True,
    ) -> mx.array:
        batch, frames, height, width, _channels = hidden_states.shape
        stride_t, stride_h, stride_w = self.stride
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(
            batch,
            frames,
            height,
            width,
            self.out_channels,
            stride_t,
            stride_h,
            stride_w,
        )
        hidden_states = hidden_states.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.reshape(
            batch,
            frames * stride_t,
            height * stride_h,
            width * stride_w,
            self.out_channels,
        )
        if stride_t == 2 and drop_leading_frame:
            hidden_states = hidden_states[:, 1:]
        return hidden_states


class TimestepEmbedder(nn.Module):
    """Diffusers PixArt timestep projection without size conditions."""

    def __init__(self, output_dims: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, output_dims, bias=True)
        self.linear_2 = nn.Linear(output_dims, output_dims, bias=True)

    def __call__(self, timesteps: mx.array, dtype: mx.Dtype) -> mx.array:
        half = 128
        exponent = -math.log(10_000.0) * mx.arange(half, dtype=mx.float32) / half
        frequencies = mx.exp(exponent)
        angles = timesteps.astype(mx.float32)[:, None] * frequencies[None, :]
        embedding = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)
        embedding = embedding.astype(dtype)
        return self.linear_2(silu(self.linear_1(embedding)))


def _tile_intervals(
    length: int,
    tile_size: int,
    stride: int,
    min_size: int,
) -> list[tuple[int, int]]:
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [
        (starts[-1], length)
    ]


def _blend_axis(
    previous: mx.array,
    current: mx.array,
    *,
    axis: int,
    extent: int,
) -> mx.array:
    extent = min(previous.shape[axis], current.shape[axis], extent)
    if extent <= 0:
        return current
    previous_slice = [slice(None)] * current.ndim
    current_slice = [slice(None)] * current.ndim
    remainder_slice = [slice(None)] * current.ndim
    previous_slice[axis] = slice(previous.shape[axis] - extent, None)
    current_slice[axis] = slice(0, extent)
    remainder_slice[axis] = slice(extent, None)
    alpha_shape = [1] * current.ndim
    alpha_shape[axis] = extent
    alpha = (mx.arange(extent, dtype=mx.float32) / extent).reshape(alpha_shape)
    prefix = (
        previous[tuple(previous_slice)].astype(mx.float32) * (1.0 - alpha)
        + current[tuple(current_slice)].astype(mx.float32) * alpha
    ).astype(current.dtype)
    result = mx.concatenate([prefix, current[tuple(remainder_slice)]], axis=axis)
    mx.eval(result)
    return result


class NativeDiffusionVideoDecoder(nn.Module):
    """Decode LTX latents with deterministic context and pixel diffusion."""

    decoder_kind = "diffusion-na"
    attention_tiling_stats: AttentionTilingStats

    def __init__(
        self,
        config: VideoVAEConfig,
        *,
        compute_dtype: mx.Dtype = mx.bfloat16,
        attention_score_budget: int = DEFAULT_ATTENTION_SCORE_BUDGET,
        swiglu_token_budget: int = DEFAULT_SWIGLU_TOKEN_BUDGET,
    ) -> None:
        super().__init__()
        diffusion = config.diffusion_decoder
        if config.decoder_kind != "diffusion-na" or diffusion is None:
            raise ValueError("NativeDiffusionVideoDecoder requires diffusion VAE metadata")
        self.config = config
        self.diffusion_config = diffusion
        self.compute_dtype = compute_dtype
        self.per_channel_statistics = PerChannelStatistics(latent_channels=config.latent_channels)
        stats = AttentionTilingStats(score_budget=attention_score_budget)
        object.__setattr__(self, "attention_tiling_stats", stats)
        self.minimum_latent_shape = _minimum_latent_shape(
            diffusion.stage_kernels,
            diffusion.upsample_strides,
            diffusion.stage5_kernel,
        )
        self.trailing_pad_latent_frames = (diffusion.stage_kernels[0][0] // 2) * 2
        self.conv_in = nn.Linear(config.latent_channels, diffusion.stage_channels[0], bias=True)
        self.det_stages: list[list[DeterministicNABlock]] = []
        self.upsamples: list[LinearPixelShuffle] = []
        for stage_index, stride in enumerate(diffusion.upsample_strides):
            channels = diffusion.stage_channels[stage_index]
            self.det_stages.append(
                [
                    DeterministicNABlock(
                        channels,
                        diffusion.stage_kernels[stage_index],
                        head_dim=diffusion.head_dim,
                        stats=stats,
                        score_budget=attention_score_budget,
                        swiglu_token_budget=swiglu_token_budget,
                    )
                    for _index in range(diffusion.stage_depths[stage_index])
                ]
            )
            self.upsamples.append(
                LinearPixelShuffle(
                    channels,
                    stride,
                    diffusion.upsample_channel_reductions[stage_index],
                )
            )
        stage5_channels = diffusion.stage_channels[-1]
        pixel_channels = config.out_channels * config.patch_size**2
        self.t_embedder = TimestepEmbedder(diffusion.t_emb_dim)
        self.conv_in_x_t = nn.Linear(pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = SharedAdaLN(stage5_channels, diffusion.t_emb_dim)
        self.diff_blocks = [
            DiffusionNABlock(
                stage5_channels,
                diffusion.stage5_kernel,
                context_dims=stage5_channels,
                head_dim=diffusion.head_dim,
                stats=stats,
                score_budget=attention_score_budget,
                swiglu_token_budget=swiglu_token_budget,
            )
            for _index in range(diffusion.stage_depths[-1])
        ]
        self.norm_out = DiffusionRMSNorm(stage5_channels)
        self.conv_out = nn.Linear(stage5_channels, pixel_channels, bias=True)
        object.__setattr__(self, "load_receipt", None)

    def _validate_input(self, latent: mx.array) -> mx.array:
        if latent.ndim == 4:
            latent = latent[None]
        if latent.ndim != 5 or latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"expected BCFHW latent with {self.config.latent_channels} channels, got "
                f"{tuple(latent.shape)}"
            )
        if any(dimension <= 0 for dimension in latent.shape):
            raise ValueError(f"latent dimensions must be positive, got {latent.shape}")
        return latent

    def _pad_to_minimum_latent_shape(
        self,
        latent: mx.array,
    ) -> tuple[mx.array, _LatentPadding]:
        latent, time = _edge_pad_axis(
            latent,
            axis=2,
            minimum=self.minimum_latent_shape[0],
            trailing_only=True,
        )
        latent, height = _edge_pad_axis(
            latent,
            axis=3,
            minimum=self.minimum_latent_shape[1],
            trailing_only=False,
        )
        latent, width = _edge_pad_axis(
            latent,
            axis=4,
            minimum=self.minimum_latent_shape[2],
            trailing_only=False,
        )
        return latent, _LatentPadding(time, height, width)

    def _crop_to_content(
        self,
        pixels: mx.array,
        *,
        frames: int,
        height: int,
        width: int,
        padding: _LatentPadding,
    ) -> mx.array:
        start_h = padding.height.before * self.config.decoder_scale.height
        start_w = padding.width.before * self.config.decoder_scale.width
        if (
            frames > pixels.shape[2]
            or start_h + height > pixels.shape[3]
            or start_w + width > pixels.shape[4]
        ):
            raise RuntimeError(
                "diffusion VAE content crop exceeds decoded working shape: "
                f"target={(frames, height, width)} start={(0, start_h, start_w)} "
                f"work={tuple(pixels.shape[2:])}"
            )
        return pixels[
            :,
            :,
            :frames,
            start_h : start_h + height,
            start_w : start_w + width,
        ]

    def forward_stages_1_to_3(self, latent: mx.array) -> mx.array:
        trailing = latent[:, :, -1:]
        if self.trailing_pad_latent_frames:
            trailing = mx.broadcast_to(
                trailing,
                (*trailing.shape[:2], self.trailing_pad_latent_frames, *trailing.shape[3:]),
            )
            latent = mx.concatenate([latent, trailing], axis=2)
        hidden_states = latent.transpose(0, 2, 3, 4, 1)
        hidden_states = self.conv_in(hidden_states)
        mx.eval(hidden_states)
        for blocks, upsample in zip(self.det_stages[:-1], self.upsamples[:-1], strict=True):
            for block in blocks:
                hidden_states = block(hidden_states)
            hidden_states = upsample(hidden_states)
            mx.eval(hidden_states)
        return hidden_states

    def forward_stage_4(
        self,
        hidden_states: mx.array,
        *,
        drop_leading_frame: bool = True,
        crop_trailing_ghost: bool = True,
    ) -> mx.array:
        for block in self.det_stages[-1]:
            hidden_states = block(hidden_states)
        hidden_states = self.upsamples[-1](
            hidden_states,
            drop_leading_frame=drop_leading_frame,
        )
        if crop_trailing_ghost and self.trailing_pad_latent_frames:
            ghost_frames = (
                self.trailing_pad_latent_frames * self.diffusion_config.temporal_compression_ratio
            )
            work_frames = int(hidden_states.shape[1])
            removable_frames = max(
                work_frames - self.diffusion_config.stage5_kernel[0],
                0,
            )
            keep_frames = work_frames - min(ghost_frames, removable_frames)
            hidden_states = hidden_states[:, :keep_frames]
        mx.eval(hidden_states)
        return hidden_states

    def forward_diffusion_step(
        self,
        latent_context: mx.array,
        x_t: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        timestep_embedding = self.t_embedder(
            timestep * self.diffusion_config.timestep_scale_multiplier,
            latent_context.dtype,
        )
        modulation = self.shared_adaln(timestep_embedding)
        hidden_states = patchify_spatial(x_t, self.config.patch_size)
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        hidden_states = self.conv_in_x_t(hidden_states)
        mx.eval(hidden_states)
        for block in self.diff_blocks:
            hidden_states = block(hidden_states, latent_context, modulation)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_out(hidden_states).transpose(0, 4, 1, 2, 3)
        output = unpatchify_spatial(hidden_states, self.config.patch_size)
        mx.eval(output)
        return output

    def denoise(
        self,
        latent_context: mx.array,
        x_t: mx.array,
        *,
        num_inference_steps: int,
    ) -> mx.array:
        batch = latent_context.shape[0]
        timesteps = mx.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            dtype=mx.float32,
        )
        if num_inference_steps == 1 and self.diffusion_config.model_output_type == "x0":
            return self.forward_diffusion_step(
                latent_context,
                x_t,
                mx.broadcast_to(timesteps[:1], (batch,)),
            )
        for index in range(num_inference_steps):
            timestep = mx.broadcast_to(timesteps[index : index + 1], (batch,))
            next_timestep = timesteps[index + 1] if index + 1 < num_inference_steps else 0.0
            model_output = self.forward_diffusion_step(
                latent_context,
                x_t,
                timestep,
            ).astype(mx.float32)
            x_t_fp32 = x_t.astype(mx.float32)
            if self.diffusion_config.model_output_type == "x0":
                sigma = timestep.reshape(batch, 1, 1, 1, 1)
                model_output = (x_t_fp32 - model_output) / sigma
            delta = (timestep - next_timestep).reshape(batch, 1, 1, 1, 1)
            x_t = (x_t_fp32 - delta * model_output).astype(x_t.dtype)
            mx.eval(x_t)
        return x_t

    def _noise(
        self,
        shape: tuple[int, ...],
        *,
        stream: NormalNoiseStream,
    ) -> mx.array:
        return stream.normal(shape, self.compute_dtype)

    def __call__(
        self,
        latent: mx.array,
        *,
        timestep: float | None = 0.05,
        causal: bool | None = None,
        reporter: Reporter | None = None,
        seed: int = 0,
        noise_backend: NoiseBackend = DEFAULT_NOISE_BACKEND,
        noise_stream: NormalNoiseStream | None = None,
    ) -> mx.array:
        del timestep, causal
        latent = self._validate_input(latent)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("diffusion decoder seed must be an unsigned 64-bit integer")
        stream = (
            noise_stream
            if noise_stream is not None
            else create_normal_noise_stream(seed, backend=noise_backend)
        )
        sink = reporter if reporter is not None else NullReporter()
        phase = "diffusion VAE decode"
        sink.phase_start(phase, total=1, unit="decode")
        self.attention_tiling_stats.reset()
        try:
            content_frames = 1 + (int(latent.shape[2]) - 1) * self.config.decoder_scale.time
            content_height = int(latent.shape[3]) * self.config.decoder_scale.height
            content_width = int(latent.shape[4]) * self.config.decoder_scale.width
            latent, padding = self._pad_to_minimum_latent_shape(latent)
            latent = self.per_channel_statistics.denormalize(latent)
            if latent.dtype != self.compute_dtype:
                latent = latent.astype(self.compute_dtype)
            context = self.forward_stage_4(self.forward_stages_1_to_3(latent))
            pixel_shape = (
                latent.shape[0],
                self.config.out_channels,
                context.shape[1],
                context.shape[2] * self.config.patch_size,
                context.shape[3] * self.config.patch_size,
            )
            x_t = self._noise(pixel_shape, stream=stream)
            output = self.denoise(
                context,
                x_t,
                num_inference_steps=self.diffusion_config.default_num_inference_steps,
            )
            output = self._crop_to_content(
                output,
                frames=content_frames,
                height=content_height,
                width=content_width,
                padding=padding,
            )
            mx.eval(output)
            sink.phase_advance(phase)
            return output
        finally:
            sink.phase_end(phase)

    def decode_streaming(
        self,
        latent: mx.array,
        *,
        tiling_config: TilingConfig | None,
        seed: int,
        noise_backend: NoiseBackend,
        reporter: Reporter | None,
        plan_callback: Callable[[TilingPlanReceipt], None] | None,
        noise_state_callback: Callable[[NoiseStreamState], None] | None,
    ) -> Iterator[mx.array]:
        """Yield bounded stage-4/stage-5 temporal groups in BCFHW layout."""
        from .tiling import TilingConfig, TilingPlanReceipt

        latent = self._validate_input(latent)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("diffusion decoder seed must be an unsigned 64-bit integer")
        noise_stream = create_normal_noise_stream(seed, backend=noise_backend)
        content_latent_shape = (
            int(latent.shape[2]),
            int(latent.shape[3]),
            int(latent.shape[4]),
        )
        output_frames = 1 + (content_latent_shape[0] - 1) * self.config.decoder_scale.time
        output_height = content_latent_shape[1] * self.config.decoder_scale.height
        output_width = content_latent_shape[2] * self.config.decoder_scale.width
        if tiling_config is None:
            automatic = TilingConfig.auto_diffusion(
                output_height,
                output_width,
                output_frames,
            )
            tiling_config = TilingConfig() if automatic is None else automatic
        if not isinstance(tiling_config, TilingConfig):
            raise TypeError("tiling_config must be a TilingConfig")

        single = tiling_config.temporal_config is None and tiling_config.spatial_config is None
        self.attention_tiling_stats.reset()
        if single:
            initial_plan = TilingPlanReceipt(
                latent_shape=content_latent_shape,
                decoded_shape=(output_frames, output_height, output_width),
                temporal_tiles=1,
                spatial_height_tiles=1,
                spatial_width_tiles=1,
                total_tiles=1,
                resolved_config=tiling_config,
                decoder_kind=self.decoder_kind,
            )
            if plan_callback is not None:
                plan_callback(initial_plan)
            output = self(
                latent,
                seed=seed,
                noise_backend=noise_backend,
                noise_stream=noise_stream,
                reporter=reporter,
            )
            if noise_state_callback is not None:
                noise_state_callback(noise_stream.state)
            if plan_callback is not None:
                plan_callback(
                    replace(
                        initial_plan,
                        attention_tiling=self.attention_tiling_stats.to_dict(),
                    )
                )
            yield output
            return

        if self.diffusion_config.default_num_inference_steps != 1:
            raise ValueError(
                "bounded tiled diffusion decoding currently requires one inference step"
            )

        latent, padding = self._pad_to_minimum_latent_shape(latent)
        latent = self.per_channel_statistics.denormalize(latent)
        if latent.dtype != self.compute_dtype:
            latent = latent.astype(self.compute_dtype)
        sink = reporter if reporter is not None else NullReporter()
        context_phase = "diffusion VAE context"
        sink.phase_start(context_phase, total=1, unit="context")
        try:
            features = self.forward_stages_1_to_3(latent)
            sink.phase_advance(context_phase)
        finally:
            sink.phase_end(context_phase)
        last_stride = self.diffusion_config.upsample_strides[-1]
        scale_t = last_stride[0]
        scale_h = last_stride[1] * self.config.patch_size
        scale_w = last_stride[2] * self.config.patch_size
        ghost_frames = self.trailing_pad_latent_frames * math.prod(
            stride[0] for stride in self.diffusion_config.upsample_strides[:-1]
        )
        num_frames = int(features.shape[1]) - ghost_frames
        height = int(features.shape[2])
        width = int(features.shape[3])

        temporal = tiling_config.temporal_config
        spatial = tiling_config.spatial_config
        tile_t = num_frames if temporal is None else temporal.chunk_size_in_frames // scale_t
        stride_t = (
            tile_t
            if temporal is None
            else (temporal.chunk_size_in_frames - temporal.chunk_overlap_in_frames) // scale_t
        )
        tile_h = height if spatial is None else spatial.tile_size_in_pixels // scale_h
        stride_h = (
            tile_h
            if spatial is None
            else (spatial.tile_size_in_pixels - spatial.tile_overlap_in_pixels) // scale_h
        )
        tile_w = width if spatial is None else spatial.tile_size_in_pixels // scale_w
        stride_w = (
            tile_w
            if spatial is None
            else (spatial.tile_size_in_pixels - spatial.tile_overlap_in_pixels) // scale_w
        )
        min_sizes = tuple(
            max(stage4, math.ceil(stage5 / stride))
            for stage4, stage5, stride in zip(
                self.diffusion_config.stage_kernels[-1],
                self.diffusion_config.stage5_kernel,
                last_stride,
                strict=True,
            )
        )
        time_tiles = _tile_intervals(num_frames, tile_t, stride_t, min_sizes[0])
        height_tiles = _tile_intervals(height, tile_h, stride_h, min_sizes[1])
        width_tiles = _tile_intervals(width, tile_w, stride_w, min_sizes[2])
        blend_frames = (tile_t - stride_t) * scale_t
        blend_height = (tile_h - stride_h) * scale_h
        blend_width = (tile_w - stride_w) * scale_w
        total_tiles = len(time_tiles) * len(height_tiles) * len(width_tiles)
        base_plan = TilingPlanReceipt(
            latent_shape=content_latent_shape,
            decoded_shape=(output_frames, output_height, output_width),
            temporal_tiles=len(time_tiles),
            spatial_height_tiles=len(height_tiles),
            spatial_width_tiles=len(width_tiles),
            total_tiles=total_tiles,
            resolved_config=tiling_config,
            decoder_kind=self.decoder_kind,
        )
        if plan_callback is not None:
            plan_callback(base_plan)
        _log.info(
            "Diffusion VAE decode plan: temporal=%d spatial=%dx%d total=%d",
            len(time_tiles),
            len(height_tiles),
            len(width_tiles),
            total_tiles,
        )

        phase = "VAE decode tiles"
        sink.phase_start(phase, total=total_tiles, unit="tile")
        previous_raw_group: mx.array | None = None
        pending_group: mx.array | None = None
        pending_index = 0
        emitted_frames = 0
        try:
            for time_index, (time_start, time_end) in enumerate(time_tiles):
                origin = time_start == 0
                trailing = time_end == num_frames
                feature_end = int(features.shape[1]) if trailing else time_end
                result_rows = []
                above_raw: list[mx.array] | None = None
                for height_index, (height_start, height_end) in enumerate(height_tiles):
                    raw_row: list[mx.array] = []
                    result_row = []
                    raw_left: mx.array | None = None
                    for width_index, (width_start, width_end) in enumerate(width_tiles):
                        context = self.forward_stage_4(
                            features[
                                :,
                                time_start:feature_end,
                                height_start:height_end,
                                width_start:width_end,
                            ],
                            drop_leading_frame=origin,
                            crop_trailing_ghost=trailing,
                        )
                        pixel_shape = (
                            latent.shape[0],
                            self.config.out_channels,
                            context.shape[1],
                            context.shape[2] * self.config.patch_size,
                            context.shape[3] * self.config.patch_size,
                        )
                        raw_tile = self.denoise(
                            context,
                            self._noise(pixel_shape, stream=noise_stream),
                            num_inference_steps=1,
                        )
                        mx.eval(raw_tile)
                        tile = raw_tile
                        if above_raw is not None:
                            tile = _blend_axis(
                                above_raw[width_index],
                                tile,
                                axis=3,
                                extent=blend_height,
                            )
                        if raw_left is not None:
                            tile = _blend_axis(
                                raw_left,
                                tile,
                                axis=4,
                                extent=blend_width,
                            )
                        keep_height = (
                            stride_h * scale_h
                            if height_index < len(height_tiles) - 1
                            else tile.shape[3]
                        )
                        keep_width = (
                            stride_w * scale_w
                            if width_index < len(width_tiles) - 1
                            else tile.shape[4]
                        )
                        result_row.append(tile[:, :, :, :keep_height, :keep_width])
                        raw_row.append(raw_tile)
                        raw_left = raw_tile
                        sink.phase_advance(phase)
                        del context, tile
                        mx.clear_cache()
                    row = (
                        result_row[0]
                        if len(result_row) == 1
                        else mx.concatenate(result_row, axis=4)
                    )
                    mx.eval(row)
                    result_rows.append(row)
                    above_raw = raw_row
                raw_group = (
                    result_rows[0] if len(result_rows) == 1 else mx.concatenate(result_rows, axis=3)
                )
                raw_group = self._crop_to_content(
                    raw_group,
                    frames=int(raw_group.shape[2]),
                    height=output_height,
                    width=output_width,
                    padding=padding,
                )
                mx.eval(raw_group)
                processed_group = raw_group
                if previous_raw_group is not None:
                    processed_group = _blend_axis(
                        previous_raw_group,
                        raw_group,
                        axis=2,
                        extent=blend_frames,
                    )
                if pending_group is not None:
                    keep_frames = stride_t * scale_t
                    if pending_index == 0 and scale_t == 2:
                        keep_frames -= 1
                    frames_to_emit = min(keep_frames, output_frames - emitted_frames)
                    if frames_to_emit > 0:
                        finalized = pending_group[:, :, :frames_to_emit]
                        mx.eval(finalized)
                        emitted_frames += frames_to_emit
                        yield finalized
                previous_raw_group = raw_group
                pending_group = processed_group
                pending_index = time_index
            if pending_group is None:
                raise RuntimeError("diffusion VAE tiled decode produced no frame groups")
            frames_to_emit = min(int(pending_group.shape[2]), output_frames - emitted_frames)
            if frames_to_emit > 0:
                pending_group = pending_group[:, :, :frames_to_emit]
                mx.eval(pending_group)
                yield pending_group
        finally:
            if noise_state_callback is not None:
                noise_state_callback(noise_stream.state)
            if plan_callback is not None:
                plan_callback(
                    replace(
                        base_plan,
                        attention_tiling=self.attention_tiling_stats.to_dict(),
                    )
                )
            sink.phase_end(phase)


@dataclass(frozen=True)
class DiffusionDecoderLoadReceipt:
    """Declared binding outcome for one permissive community checkpoint."""

    loaded_tensors: int
    folded_gate_tensors: int
    ignored_decoder_tensors: tuple[str, ...]
    inferred_constructor_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decoder_kind": "diffusion-na",
            "loaded_tensors": self.loaded_tensors,
            "folded_gate_tensors": self.folded_gate_tensors,
            "ignored_decoder_tensors": list(self.ignored_decoder_tensors),
            "inferred_constructor_fields": list(self.inferred_constructor_fields),
        }


WeightSource = Mapping[str, mx.array] | Path | str


def _weights_from_source(source: WeightSource) -> Mapping[str, mx.array]:
    if isinstance(source, Mapping):
        return source
    return load_weights(source)


def _flatten_to_nested(flat: dict[str, mx.array]) -> dict[str, TensorTree]:
    nested: dict[str, TensorTree] = {}
    for key, value in flat.items():
        current = nested
        parts = key.split(".")
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"diffusion decoder parameter path collides at {part!r}")
            current = child
        current[parts[-1]] = value

    def lists(value: TensorTree) -> TensorTree:
        if not isinstance(value, dict):
            return value
        converted = {key: lists(item) for key, item in value.items()}
        if converted and all(key.isdigit() for key in converted):
            result: list[TensorTree] = [None] * (max(map(int, converted)) + 1)
            for key, item in converted.items():
                result[int(key)] = item
            return result
        return converted

    result = lists(nested)
    if not isinstance(result, dict):
        raise AssertionError("diffusion decoder parameter root must remain a mapping")
    return result


def _checkpoint_local_key(parameter_key: str) -> str:
    if parameter_key == "t_embedder.linear_1.weight":
        return "t_embedder.mlp.0.weight"
    if parameter_key == "t_embedder.linear_1.bias":
        return "t_embedder.mlp.0.bias"
    if parameter_key == "t_embedder.linear_2.weight":
        return "t_embedder.mlp.2.weight"
    if parameter_key == "t_embedder.linear_2.bias":
        return "t_embedder.mlp.2.bias"
    return parameter_key


def _find_weight(
    weights: Mapping[str, mx.array],
    *keys: str,
) -> tuple[str, mx.array] | None:
    for key in keys:
        if key in weights:
            return key, weights[key]
    for key in keys:
        matches = sorted(name for name in weights if name.endswith(f".{key}"))
        if matches:
            return matches[0], weights[matches[0]]
    return None


def _gate_for_parameter(
    weights: Mapping[str, mx.array],
    checkpoint_key: str,
) -> tuple[str, mx.array] | None:
    suffixes = {
        ".attn.proj.weight": ".gate_msa",
        ".attn.proj.bias": ".gate_msa",
        ".mlp.w_down.weight": ".gate_mlp",
        ".context_proj.weight": ".gate_ctx",
        ".context_proj.bias": ".gate_ctx",
    }
    for suffix, gate_suffix in suffixes.items():
        if checkpoint_key.endswith(suffix):
            gate_key = checkpoint_key[: -len(suffix)] + gate_suffix
            return _find_weight(
                weights,
                f"vae.decoder.{gate_key}",
                f"vae_decoder.{gate_key}",
                f"decoder.{gate_key}",
            )
    return None


def load_diffusion_video_decoder_weights(
    decoder: NativeDiffusionVideoDecoder,
    source: WeightSource,
) -> DiffusionDecoderLoadReceipt:
    """Bind every consumed diffusion-decoder target and report ignored baggage."""
    weights = _weights_from_source(source)
    flattened = cast(list[tuple[str, object]], tree_flatten(decoder.parameters()))
    parameters: dict[str, mx.array] = {}
    for key, value in flattened:
        if key.startswith("per_channel_statistics."):
            continue
        if not isinstance(value, mx.array):
            raise TypeError(f"diffusion decoder parameter {key!r} is not an MLX array")
        parameters[key] = value
    bound: dict[str, mx.array] = {}
    consumed: set[str] = set()
    missing: list[str] = []
    folded_gates: set[str] = set()

    for parameter_key, parameter in parameters.items():
        checkpoint_key = _checkpoint_local_key(parameter_key)
        statistic_candidates = (
            f"vae.decoder.{checkpoint_key}",
            f"vae_decoder.{checkpoint_key}",
            f"decoder.{checkpoint_key}",
        )
        found = _find_weight(weights, *statistic_candidates)
        if found is None:
            missing.append(checkpoint_key)
            continue
        source_key, value = found
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"diffusion decoder {checkpoint_key} has shape {tuple(value.shape)}; "
                f"expected {tuple(parameter.shape)}"
            )
        consumed.add(source_key)
        gate_entry = _gate_for_parameter(weights, checkpoint_key)
        if gate_entry is not None:
            gate_key, gate = gate_entry
            consumed.add(gate_key)
            folded_gates.add(gate_key)
            value_fp32 = value.astype(mx.float32)
            gate_fp32 = gate.astype(mx.float32)
            value = (
                value_fp32 * gate_fp32[:, None] if value.ndim == 2 else value_fp32 * gate_fp32
            ).astype(value.dtype)
        bound[parameter_key] = value

    for statistic, attribute in (
        ("mean-of-means", "mean_of_means"),
        ("std-of-means", "std_of_means"),
    ):
        candidates = (
            f"vae.per_channel_statistics.{statistic}",
            f"per_channel_statistics.{statistic}",
        )
        found = _find_weight(weights, *candidates)
        if found is None:
            missing.append(f"per_channel_statistics.{statistic}")
            continue
        source_key, value = found
        expected = (decoder.config.latent_channels,)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"per_channel_statistics.{statistic} has shape {tuple(value.shape)}; "
                f"expected {expected}"
            )
        setattr(decoder.per_channel_statistics, attribute, value)
        consumed.add(source_key)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", plus {len(missing) - 5} more"
        raise ValueError(f"diffusion video decoder weights are incomplete: {preview}{suffix}")

    decoder.update(_flatten_to_nested(bound))
    ignored = tuple(
        sorted(
            key
            for key in weights
            if key.startswith(("vae.decoder.", "vae_decoder.", "decoder.")) and key not in consumed
        )
    )
    receipt = DiffusionDecoderLoadReceipt(
        loaded_tensors=len(bound) + 2,
        folded_gate_tensors=len(folded_gates),
        ignored_decoder_tensors=ignored,
        inferred_constructor_fields=tuple(
            sorted(
                {
                    *decoder.config.inferred_fields,
                    *decoder.diffusion_config.inferred_fields,
                }
            )
        ),
    )
    object.__setattr__(decoder, "load_receipt", receipt)
    _log.info(
        "Loaded %d diffusion VAE decoder tensors; folded %d gates; ignored %d declared extras",
        receipt.loaded_tensors,
        receipt.folded_gate_tensors,
        len(receipt.ignored_decoder_tensors),
    )
    del weights
    gc.collect()
    mx.clear_cache()
    return receipt


__all__ = [
    "AttentionTilePlan",
    "AttentionTilingStats",
    "DEFAULT_ATTENTION_SCORE_BUDGET",
    "DEFAULT_SWIGLU_TOKEN_BUDGET",
    "DiffusionDecoderLoadReceipt",
    "NativeDiffusionVideoDecoder",
    "NeighborhoodAttention3D",
    "TiledSwiGLU",
    "load_diffusion_video_decoder_weights",
    "patchify_spatial",
    "plan_attention_tiles",
    "unpatchify_spatial",
]
