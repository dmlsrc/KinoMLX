"""Adapter-independent Qwen3-VL still-image vision backbone."""

import math
from dataclasses import dataclass

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.modeling.layers import embedding_shell, linear_shell


@dataclass(frozen=True)
class Qwen3VLVisionConfig:
    """Allocation and patch geometry for a Qwen3-VL vision checkpoint."""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24
    num_heads: int = 16
    out_hidden_size: int = 2560
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.intermediate_size,
            self.depth,
            self.num_heads,
            self.out_hidden_size,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
            self.num_position_embeddings,
        )
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in dimensions):
            raise ValueError("Qwen vision dimensions must be positive integers")
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 4:
            raise ValueError("Qwen vision head dimension must be divisible by four")
        if math.isqrt(self.num_position_embeddings) ** 2 != self.num_position_embeddings:
            raise ValueError("Qwen vision learned positions must form a square grid")
        if tuple(sorted(set(self.deepstack_visual_indexes))) != self.deepstack_visual_indexes:
            raise ValueError("Qwen vision DeepStack indexes must be sorted and unique")
        if any(i < 0 or i >= self.depth for i in self.deepstack_visual_indexes):
            raise ValueError("Qwen vision DeepStack index is outside the model")


@dataclass(frozen=True)
class Qwen3VLVisualFeatures:
    """Merged image embeddings and early-language-layer residual features."""

    embeddings: mx.array
    deepstack: tuple[mx.array, ...]
    grid: tuple[int, int]


class QwenVisionNorm(nn.Module):
    """Layer normalization with checkpoint-owned affine parameters."""

    def __init__(self, dims: int) -> None:
        super().__init__()
        self.dims = dims
        self.weight = mx.zeros((0,))
        self.bias = mx.zeros((0,))

    def __call__(self, value: mx.array) -> mx.array:
        return mx.fast.layer_norm(value, self.weight, self.bias, 1e-6)


def vision_gelu(value: mx.array, *, approximate: bool) -> mx.array:
    """Qwen uses tanh GELU in blocks and exact GELU in its mergers."""
    if approximate:
        return 0.5 * value * (1 + mx.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
    return 0.5 * value * (1 + mx.erf(value / math.sqrt(2)))


class QwenVisionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = linear_shell(bias=True)
        self.linear_fc2 = linear_shell(bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        return self.linear_fc2(vision_gelu(self.linear_fc1(value), approximate=True))


class QwenVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = linear_shell(bias=True)
        self.proj = linear_shell(bias=True)

    def __call__(self, value: mx.array, phases: mx.array) -> mx.array:
        length, width = value.shape
        qkv = self.qkv(value).reshape(length, 3, self.heads, self.head_dim)
        query, key, projected = (qkv[:, i].transpose(1, 0, 2)[None] for i in range(3))
        cos, sin = mx.cos(phases)[None, None], mx.sin(phases)[None, None]

        def rotate(x: mx.array) -> mx.array:
            first, last = mx.split(x.astype(mx.float32), 2, axis=-1)
            rotated = mx.concatenate([-last, first], axis=-1)
            return (x.astype(mx.float32) * cos + rotated * sin).astype(x.dtype)

        attended = mx.fast.scaled_dot_product_attention(
            rotate(query), rotate(key), projected, scale=self.head_dim**-0.5
        )
        return self.proj(attended[0].transpose(1, 0, 2).reshape(length, width))


class QwenVisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = QwenVisionNorm(config.hidden_size)
        self.norm2 = QwenVisionNorm(config.hidden_size)
        self.attn = QwenVisionAttention(config)
        self.mlp = QwenVisionMLP()

    def __call__(self, value: mx.array, phases: mx.array) -> mx.array:
        value = value + self.attn(self.norm1(value), phases)
        return value + self.mlp(self.norm2(value))


class QwenVisionMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, *, postshuffle: bool) -> None:
        super().__init__()
        self.width = config.hidden_size * config.spatial_merge_size**2
        self.postshuffle = postshuffle
        self.norm = QwenVisionNorm(self.width if postshuffle else config.hidden_size)
        self.linear_fc1 = linear_shell(bias=True)
        self.linear_fc2 = linear_shell(bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        if self.postshuffle:
            value = value.reshape(-1, self.width)
        value = self.norm(value).reshape(-1, self.width)
        return self.linear_fc2(vision_gelu(self.linear_fc1(value), approximate=False))


def vision_grid_positions(height: int, width: int, merge: int) -> mx.array:
    """Row/column patch positions in spatial-merge block order."""
    if min(height, width, merge) <= 0 or height % merge or width % merge:
        raise ValueError("Qwen vision grid must align to spatial merge size")
    rows = mx.broadcast_to(mx.arange(height)[:, None], (height, width))
    cols = mx.broadcast_to(mx.arange(width)[None, :], (height, width))
    return (
        mx.stack([rows, cols], axis=-1)
        .reshape(height // merge, merge, width // merge, merge, 2)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 2)
    )


class Qwen3VLVisionModel(nn.Module):
    """One image per forward, without cross-image attention or adapter policy."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        # A nonoverlapping Conv3d patch projection is exactly a flattened Linear.
        self.patch_projection = linear_shell(bias=True)
        self.pos_embed = embedding_shell()
        self.blocks = [QwenVisionBlock(config) for _ in range(config.depth)]
        self.merger = QwenVisionMerger(config, postshuffle=False)
        self.deepstack_merger_list = [
            QwenVisionMerger(config, postshuffle=True) for _ in config.deepstack_visual_indexes
        ]

    def __call__(self, patches: mx.array, *, grid: tuple[int, int]) -> Qwen3VLVisualFeatures:
        config = self.config
        height, width = grid
        positions = vision_grid_positions(height, width, config.spatial_merge_size)
        expected = (height * width, 3 * config.temporal_patch_size * config.patch_size**2)
        if patches.shape != expected:
            raise ValueError(f"Qwen vision patches have shape {patches.shape}, expected {expected}")
        value = self.patch_projection(patches)
        # Learned square positions use align-corners bilinear interpolation.
        side = math.isqrt(config.num_position_embeddings)
        coordinates = positions.astype(mx.float32) * mx.array(
            [(side - 1) / max(height - 1, 1), (side - 1) / max(width - 1, 1)]
        )
        low = mx.floor(coordinates).astype(mx.int32)
        high = mx.minimum(low + 1, side - 1)
        weight = coordinates - low
        positional = mx.zeros(value.shape, dtype=mx.float32)
        for y in range(2):
            for x in range(2):
                row = low[:, 0] if y == 0 else high[:, 0]
                col = low[:, 1] if x == 0 else high[:, 1]
                factor = (1 - weight[:, 0] if y == 0 else weight[:, 0]) * (
                    1 - weight[:, 1] if x == 0 else weight[:, 1]
                )
                positional = (
                    positional
                    + self.pos_embed(row * side + col).astype(mx.float32) * factor[:, None]
                )
        value = value + positional.astype(value.dtype)
        rotary_dim = config.hidden_size // config.num_heads // 2
        frequency = 1 / (10000 ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))
        angles = (positions[..., None] * frequency).reshape(height * width, -1)
        phases = mx.concatenate([angles, angles], axis=-1)
        deepstack: list[mx.array] = []
        for index, block in enumerate(self.blocks):
            value = block(value, phases)
            if index in config.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[len(deepstack)](value))
        embeddings = self.merger(value)
        mx.eval(embeddings, deepstack)
        return Qwen3VLVisualFeatures(
            embeddings,
            tuple(deepstack),
            (height // config.spatial_merge_size, width // config.spatial_merge_size),
        )
