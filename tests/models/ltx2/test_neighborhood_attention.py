"""FP16 neighborhood attention kernels: agreement with the window formula, and their limits."""

import math

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.kernels import neighborhood_attention

pytestmark = pytest.mark.requires_metal

Grid = tuple[int, int, int]


def _starts(length: int, size: int) -> mx.array:
    return mx.clip(mx.arange(length, dtype=mx.int32) - size // 2, 0, length - size)


def _oracle(q: mx.array, k: mx.array, v: mx.array, window: Grid) -> mx.array:
    """FP32 attention of every query over the keys of its inward-shifted window, gathered."""
    batch, gt, gh, gw, heads, dim = q.shape
    t_idx = _starts(gt, window[0])[:, None] + mx.arange(window[0], dtype=mx.int32)[None, :]
    h_idx = _starts(gh, window[1])[:, None] + mx.arange(window[1], dtype=mx.int32)[None, :]
    w_idx = _starts(gw, window[2])[:, None] + mx.arange(window[2], dtype=mx.int32)[None, :]
    keys_of = (
        t_idx[:, None, None, :, None, None] * (gh * gw)
        + h_idx[None, :, None, None, :, None] * gw
        + w_idx[None, None, :, None, None, :]
    ).reshape(gt * gh * gw, math.prod(window))
    tokens = gt * gh * gw
    outputs = []
    for index in range(batch):
        qf = q[index].reshape(tokens, heads, dim).astype(mx.float32)
        kf = k[index].reshape(tokens, heads, dim).astype(mx.float32)
        vf = v[index].reshape(tokens, heads, dim).astype(mx.float32)
        rows = []
        for start in range(0, tokens, 128):
            gathered = keys_of[start : start + 128]
            scores = mx.einsum("nhd,nkhd->nhk", qf[start : start + 128], kf[gathered])
            weights = mx.softmax(scores, axis=-1, precise=True)
            rows.append(mx.einsum("nhk,nkhd->nhd", weights, vf[gathered]))
            mx.eval(rows[-1])
        outputs.append(mx.concatenate(rows, axis=0).reshape(gt, gh, gw, heads, dim))
    return mx.stack(outputs, axis=0)


def _inputs(
    batch: int, grid: Grid, heads: int, seed: int, q_scale: float
) -> tuple[mx.array, mx.array, mx.array]:
    shape = (batch, *grid, heads, neighborhood_attention.HEAD_DIM)
    keys = mx.random.split(mx.random.key(seed), 3)
    # The decoder's queries arrive scaled by D^-0.5.
    q = (mx.random.normal(shape, key=keys[0]) * 0.125 * q_scale).astype(mx.float16)
    k = mx.random.normal(shape, key=keys[1]).astype(mx.float16)
    v = mx.random.normal(shape, key=keys[2]).astype(mx.float16)
    return q, k, v


@pytest.mark.parametrize(
    ("batch", "grid", "heads", "window", "q_scale"),
    [
        # The final stage's window on the patches kernel: tails on every axis, batches, heads.
        (1, (19, 13, 15), 2, (11, 11, 11), 1.0),
        (2, (18, 12, 12), 3, (11, 11, 11), 1.0),
        (1, (20, 14, 13), 2, (11, 11, 11), 8.0),
        # The final window on a grid too short for the patches kernel's key box: runs kernel.
        (1, (14, 14, 16), 2, (11, 11, 11), 1.0),
        # The earlier stages' windows on the runs kernel, at and past the key box's size.
        (1, (4, 9, 16), 3, (3, 7, 7), 1.0),
        (2, (7, 11, 21), 2, (3, 7, 7), 4.0),
        (1, (3, 8, 16), 2, (3, 5, 5), 1.0),
        (1, (6, 9, 19), 4, (3, 5, 5), 1.0),
    ],
)
def test_kernels_match_the_window_formula(
    batch: int, grid: Grid, heads: int, window: Grid, q_scale: float
) -> None:
    q, k, v = _inputs(batch, grid, heads, sum(grid) + heads, q_scale)
    assert neighborhood_attention.applies(q, k, v, window=window)
    actual = neighborhood_attention.neighborhood_attention(q, k, v, window=window)
    expected = _oracle(q, k, v, window)
    mx.eval(actual, expected)
    assert actual.shape == q.shape
    assert actual.dtype == mx.float16
    assert mx.all(mx.isfinite(actual)).item()
    difference = actual.astype(mx.float32) - expected
    relative = math.sqrt(
        (difference * difference).sum().item() / (expected * expected).sum().item()
    )
    assert relative < 1e-3


def test_values_outside_a_window_do_not_reach_its_query() -> None:
    # The window of query (0, 0, 0) spans frames 0 to 10 of this 18-frame grid. Large values in
    # frames 11 and later must leave its output bit for bit and reach the last frame's queries.
    grid, window = (18, 12, 12), (11, 11, 11)
    q, k, v = _inputs(1, grid, 2, 5, 1.0)
    spike = mx.zeros_like(v).at[:, 11:].add(mx.array(1000.0, dtype=mx.float16))
    plain = neighborhood_attention.neighborhood_attention(q, k, v, window=window)
    spiked = neighborhood_attention.neighborhood_attention(q, k, v + spike, window=window)
    mx.eval(plain, spiked)
    assert mx.array_equal(spiked[0, 0, 0, 0], plain[0, 0, 0, 0]).item()
    assert not mx.array_equal(spiked[0, -1, 0, 0], plain[0, -1, 0, 0]).item()


@pytest.mark.parametrize("window", [(11, 11, 11), (3, 7, 7)])
def test_applies_refuses_what_the_kernels_do_not_compute(window: Grid) -> None:
    q, k, v = _inputs(1, (19, 13, 16), 2, 3, 1.0)
    assert neighborhood_attention.applies(q, k, v, window=window)
    bf16 = (x.astype(mx.bfloat16) for x in (q, k, v))
    assert not neighborhood_attention.applies(*bf16, window=window)
    assert not neighborhood_attention.applies(q, k[:, :-1], v, window=window)
    assert not neighborhood_attention.applies(q, k, v, window=(window[0], window[1], 4))
    assert not neighborhood_attention.applies(q, k, v, window=(window[0], window[1], 17))
    wide = mx.zeros((1, 19, 13, 16, 2, 128), dtype=mx.float16)
    assert not neighborhood_attention.applies(wide, wide, wide, window=window)
    small = mx.zeros((1, 2, 13, 16, 2, 64), dtype=mx.float16)
    assert not neighborhood_attention.applies(small, small, small, window=window)
    # Each threadgroup forms a 64-bit base, so arrays past 2**31 elements take the kernels (never
    # evaluated here).
    huge = mx.zeros((1, 1024, 1024, 64, 2, 64), dtype=mx.float16)
    assert neighborhood_attention.applies(huge, huge, huge, window=window)
    # The launch grid, 128 threads a 32-query block, is signed 32-bit in MLX.
    assert neighborhood_attention.launch_fits((1024, 1024, 64), window)
    assert not neighborhood_attention.launch_fits((65_536, 2048, 2048), window)
    vast = mx.zeros((1, 65_536, 2048, 2048, 2, 64), dtype=mx.float16)
    assert not neighborhood_attention.applies(vast, vast, vast, window=window)
    with pytest.raises(ValueError, match="does not take"):
        neighborhood_attention.neighborhood_attention(small, small, small, window=window)
