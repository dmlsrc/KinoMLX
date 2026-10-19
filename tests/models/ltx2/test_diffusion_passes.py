"""FP16 diffusion-decoder passes: bit for bit against the operations they replace."""

import mlx.core as mx
import pytest

from kinomlx.kernels import rms_norm, silu
from kinomlx.models.ltx2.kernels import diffusion_passes
from kinomlx.models.ltx2.kernels.diffusion_passes import QKWeights
from kinomlx.models.ltx2.video_vae.diffusion_decoder import (
    AttentionTilingStats,
    NeighborhoodAttention3D,
)

pytestmark = pytest.mark.requires_metal

Grid = tuple[int, int, int]


def _normal(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> mx.array:
    return (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(mx.float16)


def _spread_rows(rows: int, width: int, seed: int) -> mx.array:
    # Row magnitudes from 2^-4 to 2^4, so the norm's reduction sees a range of sums.
    exponents = (mx.arange(rows) % 9 - 4).astype(mx.float32)
    scale = mx.power(mx.array(2.0), exponents).astype(mx.float16)
    return _normal((rows, width), seed) * scale[:, None]


@pytest.mark.parametrize("width", [128, 256, 512, 1024])
@pytest.mark.parametrize("modulated", [False, True])
def test_rms_norm_equals_the_norm_and_modulation_ops(width: int, modulated: bool) -> None:
    # 777 rows leave a partial threadgroup of the eight rows each one takes.
    value = _spread_rows(777, width, width)
    weight = 1.0 + _normal((width,), 1, 0.25)
    expected = rms_norm(value, weight, 1e-6)
    scale1p = shift = None
    if modulated:
        scale1p = 1.0 + _normal((width,), 2, 0.25)
        shift = _normal((width,), 3, 0.25)
        expected = expected * scale1p + shift
    assert diffusion_passes.rms_norm_applies(value, weight, scale1p, shift)
    actual = diffusion_passes.rms_norm(value, weight, scale1p, shift)
    mx.eval(actual, expected)
    assert actual.dtype == mx.float16
    assert mx.array_equal(actual, expected).item()


def test_rms_norm_keeps_an_fp32_weight_in_fp32() -> None:
    value = _spread_rows(300, 256, 4)
    weight = (1.0 + mx.random.normal((256,), key=mx.random.key(5)) * 0.25).astype(mx.float32)
    actual = diffusion_passes.rms_norm(value, weight)
    expected = rms_norm(value, weight, 1e-6)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_rms_norm_applies_only_where_the_vector_kernel_holds() -> None:
    value = _normal((64, 256), 0)
    weight = mx.ones((256,), dtype=mx.float16)
    vector = mx.ones((256,), dtype=mx.float16)
    assert diffusion_passes.rms_norm_applies(value, weight)
    assert diffusion_passes.rms_norm_applies(value, weight, vector, vector)
    assert not diffusion_passes.rms_norm_applies(value.astype(mx.bfloat16), weight)
    assert not diffusion_passes.rms_norm_applies(value, weight, vector, None)
    assert not diffusion_passes.rms_norm_applies(value, weight, vector.astype(mx.float32), vector)
    assert not diffusion_passes.rms_norm_applies(value, mx.ones((128,), dtype=mx.float16))
    for width in (192, 2048):
        wide = _normal((64, width), 0)
        assert not diffusion_passes.rms_norm_applies(wide, mx.ones((width,), dtype=mx.float16))
    with pytest.raises(ValueError, match="does not take"):
        diffusion_passes.rms_norm(_normal((64, 192), 0), mx.ones((192,), dtype=mx.float16))


@pytest.mark.parametrize("hidden", [1024, 4096, 8192])
def test_silu_mul_equals_silu_times_up(hidden: int) -> None:
    gate_up = _normal((301, 2 * hidden), hidden, 4.0)
    expected = silu(gate_up[:, :hidden]) * gate_up[:, hidden:]
    actual = diffusion_passes.silu_mul(gate_up, hidden)
    mx.eval(actual, expected)
    assert actual.shape == (301, hidden)
    assert mx.array_equal(actual, expected).item()


def test_silu_mul_refuses_a_mismatched_product() -> None:
    assert diffusion_passes.silu_mul_applies(_normal((8, 2048), 0), 1024)
    assert not diffusion_passes.silu_mul_applies(_normal((8, 2048), 0), 512)
    assert not diffusion_passes.silu_mul_applies(_normal((8, 2 * 1026), 0), 1026)
    assert not diffusion_passes.silu_mul_applies(_normal((8, 2048), 0).astype(mx.float32), 1024)
    with pytest.raises(ValueError, match="does not take"):
        diffusion_passes.silu_mul(_normal((8, 2048), 0), 512)


def _attention(dims: int, seed: int) -> NeighborhoodAttention3D:
    attn = NeighborhoodAttention3D(
        dims, (3, 5, 5), head_dim=64, stats=AttentionTilingStats(), score_budget=1 << 30
    )
    attn.qkv.weight = _normal((3 * dims, dims), seed, dims**-0.5)
    attn.qkv.bias = _normal((3 * dims,), seed + 1, 0.1)
    attn.q_norm.weight = 1.0 + _normal((64,), seed + 2, 0.25)
    attn.k_norm.weight = 1.0 + _normal((64,), seed + 3, 0.25)
    return attn


def _qk_weights(attn: NeighborhoodAttention3D) -> QKWeights:
    width = attn.dims
    weight, bias = attn.qkv.weight, attn.qkv.bias
    return QKWeights(
        weight[:width],
        bias[:width],
        weight[width : 2 * width],
        bias[width : 2 * width],
        attn.q_norm.weight,
        attn.k_norm.weight,
    )


def _skip_unless_available() -> None:
    if not diffusion_passes.available():
        pytest.skip("the fused Q/K epilogue does not keep the separate passes' bits here")


@pytest.mark.parametrize(
    ("dims", "grid"),
    [
        # Two, four and eight heads; grids with tails on H and W and enough tiles that MLX sums
        # its products in MPP's order.
        (128, (8, 16, 40)),
        (256, (5, 16, 64)),
        (512, (4, 14, 24)),
    ],
)
def test_qk_norm_rope_equals_the_attention_projection(dims: int, grid: Grid) -> None:
    _skip_unless_available()
    attn = _attention(dims, dims)
    hidden = _normal((1, *grid, dims), sum(grid), 2.0)
    expected_q, expected_k, _value, key_time = attn._project_temporal_partition(
        hidden, (0, grid[0]), grid=grid
    )
    assert key_time == (0, grid[0])
    x = hidden.reshape(-1, dims)
    weights = _qk_weights(attn)
    split = attn.rope.dim_split
    assert diffusion_passes.qk_norm_rope_applies(x, weights, heads=attn.heads, dim_split=split)
    q, k = diffusion_passes.qk_norm_rope(
        x, weights, grid=grid, heads=attn.heads, dim_split=split, tables=attn.rope.tables(grid)
    )
    mx.eval(q, k)
    assert q.dtype == k.dtype == mx.float16
    assert mx.array_equal(q, expected_q.reshape(q.shape)).item()
    assert mx.array_equal(k, expected_k.reshape(k.shape)).item()


def test_qk_norm_rope_applies_refuses_what_it_does_not_compute() -> None:
    _skip_unless_available()
    attn = _attention(128, 7)
    weights = _qk_weights(attn)
    split = attn.rope.dim_split
    x = _normal((5120, 128), 1)

    def applies(
        rows: mx.array, qk: QKWeights = weights, heads: int = 2, dim_split: Grid = split
    ) -> bool:
        return diffusion_passes.qk_norm_rope_applies(rows, qk, heads=heads, dim_split=dim_split)

    assert applies(x)
    # A partial 32-token tile, BF16 rows, and too few output tiles for MLX's summation order.
    assert not applies(x[:5100])
    assert not applies(x.astype(mx.bfloat16))
    assert not applies(x[:2048])
    assert not applies(x, weights._replace(q_weight=weights.q_weight.astype(mx.float32)))
    assert not applies(x, weights._replace(k_norm=weights.k_norm.astype(mx.float32)))
    assert not applies(x, heads=3)
    assert not applies(x, dim_split=(16, 23, 25))
    # 128 threads a 32-token tile and head: 2**28 rows of two heads launch 2**31 threads, one
    # past MLX's signed 32-bit grid (never evaluated: only the shape is read).
    assert applies(mx.zeros((2**27, 128), dtype=mx.float16))
    assert not applies(mx.zeros((2**28, 128), dtype=mx.float16))
    assert diffusion_passes.rows_addressable(134_217_696, heads=4)
    assert not diffusion_passes.rows_addressable(134_217_728, heads=4)
    with pytest.raises(ValueError, match="token grid"):
        diffusion_passes.qk_norm_rope(
            x,
            weights,
            grid=(8, 16, 41),
            heads=2,
            dim_split=split,
            tables=attn.rope.tables((8, 16, 41)),
        )
    with pytest.raises(ValueError, match="rotary tables"):
        diffusion_passes.qk_norm_rope(
            x,
            weights,
            grid=(8, 16, 40),
            heads=2,
            dim_split=split,
            tables=attn.rope.tables((8, 40, 16)),
        )
