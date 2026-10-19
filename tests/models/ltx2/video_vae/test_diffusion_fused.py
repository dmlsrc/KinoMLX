"""The diffusion decoder's FP16 kernel route: pieces, blocks, fallbacks and the receipt."""

import math
from collections.abc import Iterator

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import kinomlx._mlx_nn as nn
from kinomlx.models.ltx2.kernels import neighborhood_attention
from kinomlx.models.ltx2.video_vae import diffusion_fused
from kinomlx.models.ltx2.video_vae.diffusion_decoder import (
    AttentionTilingStats,
    DeterministicNABlock,
    DiffusionNABlock,
    DiffusionRMSNorm,
    NeighborhoodAttention3D,
    TiledSwiGLU,
    plan_attention_tiles,
)
from kinomlx.models.ltx2.vocabularies import DiffusionAttentionBackend, DiffusionKernelFallback

pytestmark = pytest.mark.requires_metal

Grid = tuple[int, int, int]
WINDOW = (3, 5, 5)


def _normal(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.float16)


def _randomize(module: nn.Module, seed: int) -> None:
    """FP16 parameters of a trained block's scale: unit-mean norm weights, fan-in products."""
    updates = []
    for index, (name, value) in enumerate(tree_flatten(module.parameters())):
        shape = tuple(value.shape)
        if "norm" in name:
            updates.append((name, 1.0 + _normal(shape, seed + index, 0.25)))
        elif name.endswith("scale_shift_table") or value.ndim == 1:
            updates.append((name, _normal(shape, seed + index, 0.1)))
        else:
            updates.append((name, _normal(shape, seed + index, shape[1] ** -0.5)))
    module.update(tree_unflatten(updates))


def _relative_l2(actual: mx.array, expected: mx.array) -> float:
    difference = actual.astype(mx.float32) - expected.astype(mx.float32)
    squared = (difference * difference).sum().item()
    reference = expected.astype(mx.float32)
    return math.sqrt(squared / (reference * reference).sum().item())


@pytest.fixture
def masked_only() -> Iterator[None]:
    neighborhood_attention.set_enabled(False)
    yield
    neighborhood_attention.set_enabled(True)


@pytest.mark.parametrize("rows", [300, 5120])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_linear_equals_addmm_then_the_residual_add(rows: int, bias: bool, residual: bool) -> None:
    # 5,120 rows by 512 outputs reach MPP's run-time-row form; 300 rows keep MLX's GEMM.
    value = _normal((rows, 256), rows)
    weight = _normal((512, 256), 1, 0.0625)
    offset = _normal((512,), 2) if bias else None
    added = _normal((rows, 512), 3) if residual else None
    expected = value @ weight.T if offset is None else mx.addmm(offset, value, weight.T)
    if added is not None:
        expected = added + expected
    actual = diffusion_fused.linear(value, weight, offset, residual=added)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("width", [256, 2048])
@pytest.mark.parametrize("modulated", [False, True])
def test_norm_equals_the_block_norm_and_modulation(width: int, modulated: bool) -> None:
    # 256 channels take the vector pass; 2,048 keep the threadgroup kernel and MLX's ops.
    norm = DiffusionRMSNorm(width)
    norm.weight = 1.0 + _normal((width,), 1, 0.25)
    states = _normal((1, 3, 4, 5, width), width, 2.0)
    expected = norm(states)
    scale1p = shift = None
    if modulated:
        scale = _normal((1, 1, 1, 1, width), 2, 0.25)
        offset = _normal((1, 1, 1, 1, width), 3, 0.25)
        expected = expected * (1.0 + scale) + offset
        scale1p, shift = 1.0 + scale.reshape(width), offset.reshape(width)
    actual = diffusion_fused.norm(states.reshape(-1, width), norm.weight, scale1p, shift)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected.reshape(actual.shape)).item()


@pytest.mark.parametrize("hidden", [512, 130])
def test_swiglu_equals_the_tiled_swiglu_then_the_residual_add(hidden: int) -> None:
    # Four 256-token tiles, the last partial; 130 hidden units keep MLX's SiLU and multiply.
    dims, tile = 128, 256
    mlp = TiledSwiGLU(dims, hidden, token_budget=tile)
    _randomize(mlp, hidden)
    states = _normal((1, 10, 10, 9, dims), 4)
    residual = _normal((1, 10, 10, 9, dims), 5)
    expected = residual + mlp(states)
    actual = diffusion_fused.swiglu(
        states.reshape(-1, dims),
        mlp.w_gate.weight,
        mlp.w_up.weight,
        mlp.w_down.weight,
        residual.reshape(-1, dims),
        tile_tokens=tile,
    )
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected.reshape(actual.shape)).item()


def _stats(block: DeterministicNABlock | DiffusionNABlock) -> AttentionTilingStats:
    return block.attn._stats


def _deterministic(dims: int, seed: int) -> DeterministicNABlock:
    block = DeterministicNABlock(
        dims,
        WINDOW,
        head_dim=64,
        stats=AttentionTilingStats(),
        score_budget=1 << 22,
        swiglu_token_budget=4096,
    )
    _randomize(block, seed)
    return block


def _diffusion(dims: int, seed: int) -> DiffusionNABlock:
    block = DiffusionNABlock(
        dims,
        WINDOW,
        context_dims=dims,
        head_dim=64,
        stats=AttentionTilingStats(),
        score_budget=1 << 22,
        swiglu_token_budget=4096,
    )
    _randomize(block, seed)
    return block


@pytest.mark.parametrize(
    "grid",
    [
        # 5,120 tokens: the fused Q/K pass and MPP's products. 576 tokens: the attention kernel
        # with MLX's products and the separate Q/K projection.
        (8, 16, 40),
        (4, 9, 16),
    ],
)
def test_deterministic_block_route_agrees_with_the_masked_path(grid: Grid) -> None:
    block = _deterministic(128, sum(grid))
    states = _normal((1, *grid, 128), 1)
    stats = _stats(block)
    kernel = block(states)
    mx.eval(kernel)
    assert stats.backend == DiffusionAttentionBackend.FP16_KERNELS
    assert stats.kernel_invocations == 1
    assert stats.max_score_elements == 0
    neighborhood_attention.set_enabled(False)
    try:
        masked = block(states)
        mx.eval(masked)
    finally:
        neighborhood_attention.set_enabled(True)
    assert stats.backend == DiffusionAttentionBackend.MIXED
    assert stats.fallback_reason == DiffusionKernelFallback.DISABLED
    assert kernel.shape == masked.shape == states.shape
    assert kernel.dtype == mx.float16
    assert _relative_l2(kernel, masked) < 1e-3


def test_diffusion_block_route_agrees_with_the_masked_path() -> None:
    grid = (8, 16, 40)
    block = _diffusion(128, 11)
    states = _normal((1, *grid, 128), 1)
    context = _normal((1, *grid, 128), 2)
    modulation = tuple(_normal((1, 1, 1, 1, 128), 3 + index, 0.1) for index in range(7))
    kernel = block(states, context, modulation)
    mx.eval(kernel)
    assert _stats(block).backend == DiffusionAttentionBackend.FP16_KERNELS
    neighborhood_attention.set_enabled(False)
    try:
        masked = block(states, context, modulation)
        mx.eval(masked)
    finally:
        neighborhood_attention.set_enabled(True)
    assert _relative_l2(kernel, masked) < 1e-3


def test_attention_names_why_a_block_keeps_the_masked_path(masked_only: None) -> None:
    attention = NeighborhoodAttention3D(
        128, WINDOW, head_dim=64, stats=AttentionTilingStats(), score_budget=1 << 22
    )
    states = mx.zeros((1, 8, 16, 40, 128), dtype=mx.float16)
    assert attention.kernel_fallback(states) == DiffusionKernelFallback.DISABLED


def test_kernel_fallback_reasons() -> None:
    def reason(states: mx.array, head_dim: int = 64) -> DiffusionKernelFallback | None:
        heads = int(states.shape[-1]) // head_dim
        return diffusion_fused.kernel_fallback(
            states, heads=heads, head_dim=head_dim, window=WINDOW
        )

    states = mx.zeros((1, 8, 16, 40, 128), dtype=mx.float16)
    assert reason(states) is None
    assert reason(states.astype(mx.bfloat16)) == DiffusionKernelFallback.DTYPE
    assert reason(mx.zeros((2, 8, 16, 40, 128), dtype=mx.float16)) == DiffusionKernelFallback.BATCH
    assert reason(states, head_dim=32) == DiffusionKernelFallback.HEAD_DIM
    small = mx.zeros((1, 2, 16, 40, 128), dtype=mx.float16)
    assert reason(small) == DiffusionKernelFallback.WINDOW
    # Volumes past 2**31 elements take the kernels too (never evaluated: only the shape is read).
    huge = mx.zeros((1, 1024, 1024, 64, 64), dtype=mx.float16)
    assert reason(huge) is None
    # Past the fused Q/K pass's launch grid (2**31 threads at 128 a 32-token tile and head) the
    # route raises rather than falling back.
    assert diffusion_fused.addressable((1024, 1024, 64), heads=2, window=WINDOW)
    assert not diffusion_fused.addressable((4096, 1024, 64), heads=2, window=WINDOW)
    vast = mx.zeros((1, 4096, 1024, 64, 128), dtype=mx.float16)
    with pytest.raises(ValueError, match="exceeds what the kernel route addresses"):
        reason(vast)


def test_receipt_names_the_route_of_every_call() -> None:
    stats = AttentionTilingStats()
    assert stats.to_dict()["backend"] == DiffusionAttentionBackend.MASKED_SDPA
    stats.record_kernel()
    receipt = stats.to_dict()
    assert receipt["backend"] == DiffusionAttentionBackend.FP16_KERNELS
    assert receipt["kernel_invocations"] == 1
    assert "fallback_reason" not in receipt
    stats.record_fallback(DiffusionKernelFallback.DTYPE)
    stats.record(plan_attention_tiles((4, 8, 8), (3, 3, 3), 2, score_budget=1 << 20))
    stats.record_fallback(DiffusionKernelFallback.BATCH)
    receipt = stats.to_dict()
    assert receipt["backend"] == DiffusionAttentionBackend.MIXED
    assert receipt["invocations"] == 2
    assert receipt["fallback_reason"] == DiffusionKernelFallback.DTYPE
    stats.reset()
    receipt = stats.to_dict()
    assert receipt["backend"] == DiffusionAttentionBackend.MASKED_SDPA
    assert receipt["invocations"] == receipt["kernel_invocations"] == 0
    assert "fallback_reason" not in receipt
