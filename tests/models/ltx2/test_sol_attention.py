"""Sol-Attn kernel: Morton order, preparation, routing, and parity with an FP32 reference."""

import math

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.kernels import sol_attention as sol

_HEADS = 2
# 384 grid tokens plus 33 appended tokens: seven blocks, the last one partial.
_GRID = (4, 8, 12)
_TOKENS = 417


def _python_morton(frames: int, height: int, width: int) -> list[int]:
    bits = max(frames, height, width).bit_length()

    def spread(value: int) -> int:
        code = 0
        for bit in range(bits):
            code |= ((value >> bit) & 1) << (3 * bit)
        return code

    codes = [
        spread(f) | spread(h) << 1 | spread(w) << 2
        for f in range(frames)
        for h in range(height)
        for w in range(width)
    ]
    return sorted(range(len(codes)), key=codes.__getitem__)


@pytest.mark.parametrize("grid", [(5, 3, 7), (4, 4, 4), (1, 6, 9), (16, 14, 24)])
def test_morton_order_matches_an_independent_interleave(grid: tuple[int, int, int]) -> None:
    assert sol.morton_order(*grid).tolist() == _python_morton(*grid)


def test_morton_blocks_are_cubes_on_a_multiple_of_four_grid() -> None:
    first = sol.morton_order(8, 8, 8)[:64].tolist()
    cells = {(index // 64, (index // 8) % 8, index % 8) for index in first}
    assert cells == {(f, h, w) for f in range(4) for h in range(4) for w in range(4)}


def test_morton_order_keeps_appended_tokens_after_the_grid() -> None:
    order = sol.morton_order(2, 3, 4, tokens=30)
    assert sorted(order[:24].tolist()) == list(range(24))
    assert order[24:].tolist() == list(range(24, 30))
    with pytest.raises(ValueError, match="cannot hold"):
        sol.morton_order(2, 3, 4, tokens=23)


def _inputs(dtype: mx.Dtype, seed: int) -> tuple[mx.array, mx.array, mx.array]:
    keys = mx.random.split(mx.random.key(seed), 4)
    # A smooth shared component gives blocks distinct means, so routing has choices to make.
    phase = mx.arange(_TOKENS, dtype=mx.float32)[:, None] * 0.05
    columns = mx.arange(_HEADS * sol.HEAD_DIM, dtype=mx.float32)[None, :]
    shared = 2.0 * mx.sin(phase + columns * 0.3)
    # Small queries flatten the softmax without changing the routing, which scales with the
    # scores, so unrouted blocks carry a quarter of the output: a kernel that dropped the
    # centroid pass would miss by about 0.2 in relative L2.
    q = 0.15 * (mx.random.normal((_TOKENS, _HEADS * sol.HEAD_DIM), key=keys[0]) + shared)
    k = mx.random.normal((_TOKENS, _HEADS * sol.HEAD_DIM), key=keys[1]) + shared
    v = mx.random.normal((_TOKENS, _HEADS * sol.HEAD_DIM), key=keys[2])
    return q.astype(dtype), k.astype(dtype), v.astype(dtype)


def _reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    routes: mx.array,
    key_mean: mx.array,
    value_mean: mx.array,
) -> mx.array:
    """FP32 Sol-Attn of routing-order inputs on a given routing, head by head."""
    tokens = q.shape[0]
    blocks = sol.block_count(tokens)
    lengths = mx.minimum(sol.BLOCK, tokens - mx.arange(blocks) * sol.BLOCK).astype(mx.float32)
    token_block = mx.arange(tokens) // sol.BLOCK
    scale = sol.HEAD_DIM**-0.5
    outputs = []
    for head in range(_HEADS):
        columns = slice(head * sol.HEAD_DIM, (head + 1) * sol.HEAD_DIM)
        q_head, k_head, v_head = (x[:, columns].astype(mx.float32) for x in (q, k, v))
        route_rows = routes[head].astype(mx.float32)[token_block]
        scores = (q_head @ k_head.T) * scale
        peak = scores.max(axis=-1, keepdims=True)
        exact = mx.exp(scores - peak) * route_rows[:, token_block]
        proxy = (
            mx.exp((q_head @ key_mean[head].astype(mx.float32).T) * scale - peak)
            * (1.0 - route_rows)
            * lengths
        )
        numerator = exact @ v_head + proxy @ value_mean[head].astype(mx.float32)
        outputs.append(numerator / (exact.sum(axis=-1) + proxy.sum(axis=-1))[:, None])
    return mx.concatenate(outputs, axis=-1)


def _relative_error(actual: mx.array, expected: mx.array) -> float:
    difference = actual.astype(mx.float32) - expected
    return math.sqrt((difference * difference).sum().item() / (expected * expected).sum().item())


_TOLERANCE = {mx.float16: 1e-3, mx.bfloat16: 6e-3}


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("tau", [-1.0, 0.5, 1.0, 1.5])
def test_kernel_matches_the_reference_on_its_own_routing(dtype: mx.Dtype, tau: float) -> None:
    q, k, v = _inputs(dtype, seed=11)
    order = sol.morton_order(*_GRID, tokens=_TOKENS)
    actual = sol.sol_attention(q, k, v, heads=_HEADS, tau=tau, order=order)
    gathered_q, gathered_k, gathered_v, query_mean, key_mean, value_mean = sol._prepare(
        q, k, v, heads=_HEADS, order=order
    )
    routes, _, counts, _ = sol.route_blocks(query_mean, key_mean, tau=tau)
    in_routing_order = _reference(
        gathered_q, gathered_k, gathered_v, routes=routes, key_mean=key_mean, value_mean=value_mean
    )
    expected = in_routing_order[mx.argsort(order)]
    mx.eval(actual, expected, counts)
    assert actual.dtype == dtype
    assert actual.shape == q.shape
    assert _relative_error(actual, expected) < _TOLERANCE[dtype]
    # Neighbors are always routed; the scenario must also leave some blocks unrouted.
    assert counts.min().item() >= 2
    if tau >= 0.5:
        assert counts.max().item() < sol.block_count(_TOKENS)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_routing_every_block_is_dense_attention(dtype: mx.Dtype) -> None:
    q, k, v = _inputs(dtype, seed=12)
    order = sol.morton_order(*_GRID, tokens=_TOKENS)
    actual = sol.sol_attention(q, k, v, heads=_HEADS, tau=-1e6, order=order)
    shaped = [
        x.astype(mx.float32).reshape(1, _TOKENS, _HEADS, sol.HEAD_DIM).transpose(0, 2, 1, 3)
        for x in (q, k, v)
    ]
    dense = mx.fast.scaled_dot_product_attention(*shaped, scale=sol.HEAD_DIM**-0.5)
    expected = dense.transpose(0, 2, 1, 3).reshape(_TOKENS, _HEADS * sol.HEAD_DIM)
    mx.eval(actual, expected)
    assert _relative_error(actual, expected) < _TOLERANCE[dtype]


def test_a_large_tau_routes_only_each_block_and_its_neighbors() -> None:
    q, k, v = _inputs(mx.float16, seed=13)
    order = sol.morton_order(*_GRID, tokens=_TOKENS)
    *_, query_mean, key_mean, _ = sol._prepare(q, k, v, heads=_HEADS, order=order)
    routes, lists, counts, routed_bits = sol.route_blocks(query_mean, key_mean, tau=1e6)
    blocks = sol.block_count(_TOKENS)
    index = mx.arange(blocks)
    band = mx.abs(index[:, None] - index[None, :]) <= 1
    assert mx.array_equal(routes, mx.broadcast_to(band, routes.shape)).item()
    assert counts[0].tolist() == [2] + [3] * (blocks - 2) + [2]
    assert lists[0, 3, :3].tolist() == [2, 3, 4]
    assert routed_bits[0, 3, 0].item() == 0b11100


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_preparation_gathers_converts_and_averages_blocks(dtype: mx.Dtype) -> None:
    q, k, v = _inputs(dtype, seed=14)
    order = sol.morton_order(*_GRID, tokens=_TOKENS)
    gathered_q, _, gathered_v, query_mean, _, value_mean = sol._prepare(
        q, k, v, heads=_HEADS, order=order
    )
    assert gathered_q.dtype == mx.float16
    assert mx.array_equal(gathered_q, q[order].astype(mx.float16)).item()
    assert mx.array_equal(gathered_v, v[order].astype(mx.float16)).item()
    blocks = sol.block_count(_TOKENS)
    padded = mx.pad(q[order].astype(mx.float32), [(0, blocks * sol.BLOCK - _TOKENS), (0, 0)])
    lengths = mx.minimum(sol.BLOCK, _TOKENS - mx.arange(blocks) * sol.BLOCK)
    sums = padded.reshape(blocks, sol.BLOCK, _HEADS, sol.HEAD_DIM).sum(axis=1)
    expected = (sums / lengths[:, None, None].astype(mx.float32)).transpose(1, 0, 2)
    assert mx.allclose(query_mean, expected, rtol=1e-5, atol=1e-6).item()
    # Value means are rounded through the input dtype before the kernel dtype.
    assert value_mean.dtype == mx.float16


def test_bfloat16_values_beyond_float16_saturate_instead_of_overflowing() -> None:
    q, k, v = _inputs(mx.bfloat16, seed=15)
    v = mx.concatenate([mx.full((1, v.shape[1]), 1e5, dtype=mx.bfloat16), v[1:]])
    order = sol.morton_order(*_GRID, tokens=_TOKENS)
    out = sol.sol_attention(q, k, v, heads=_HEADS, tau=1.0, order=order)
    assert mx.all(mx.isfinite(out)).item()


def test_inputs_outside_the_kernel_contract_are_refused() -> None:
    q, k, v = _inputs(mx.float16, seed=16)
    order = sol.morton_order(*_GRID, tokens=_TOKENS)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        sol.sol_attention(*(x.astype(mx.float32) for x in (q, k, v)), heads=2, tau=1.0, order=order)
    with pytest.raises(ValueError, match=r"\(tokens, 4 \* 128\)"):
        sol.sol_attention(q, k, v, heads=4, tau=1.0, order=order)
    with pytest.raises(ValueError, match="one index per token"):
        sol.sol_attention(q, k, v, heads=_HEADS, tau=1.0, order=order[:-1])
    with pytest.raises(ValueError, match="tau must be finite"):
        sol.sol_attention(q, k, v, heads=_HEADS, tau=math.inf, order=order)
