"""The one-pass split RoPE kernel against MLX's graph for the same rotation."""

import mlx.core as mx
import pytest

from kinomlx.kernels import split_rope
from kinomlx.models.ltx2.transformer import rope


def _tables(
    batch: int, heads: int, tokens: int, half: int, *, token_major: bool
) -> tuple[mx.array, mx.array]:
    count = batch * tokens * heads * half
    angles = mx.arange(count, dtype=mx.float32) * 0.2718281
    if token_major:
        # LTX's own layout: built token-major, then transposed to head-major.
        angles = angles.reshape(batch, tokens, heads, half).transpose(0, 2, 1, 3)
    else:
        angles = angles.reshape(batch, heads, tokens, half)
    return mx.cos(angles), mx.sin(angles)


def _graph(value: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    split_rope.set_enabled(False)
    try:
        return rope.apply_split_rotary_emb(value, cos, sin)
    finally:
        split_rope.set_enabled(True)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    ("batch", "table_batch", "token_major"),
    [(1, 1, True), (2, 1, True), (2, 2, False)],
)
def test_kernel_matches_the_graph(
    dtype: mx.Dtype, batch: int, table_batch: int, token_major: bool
) -> None:
    if not split_rope.available(dtype):
        pytest.skip("the RoPE kernel does not build or does not match MLX on this GPU")
    heads, tokens, half = 4, 13, 24
    value = mx.random.normal((batch, tokens, heads * 2 * half), key=mx.random.key(611))
    value = (value * 3.0).astype(dtype)
    cos, sin = _tables(table_batch, heads, tokens, half, token_major=token_major)
    assert split_rope.applies(value, cos, sin)
    rotated = rope.apply_split_rotary_emb(value, cos, sin)
    expected = _graph(value, cos, sin)
    mx.eval(rotated, expected)
    assert rotated.dtype == dtype
    assert mx.array_equal(rotated, expected).item()


def test_apply_split_rotary_emb_routes_through_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    if not split_rope.available(mx.float16):
        pytest.skip("the RoPE kernel does not build or does not match MLX on this GPU")
    calls = []
    original = split_rope.rotate

    def recording(value: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        calls.append(value.shape)
        return original(value, cos, sin)

    monkeypatch.setattr(split_rope, "rotate", recording)
    value = mx.zeros((1, 5, 2 * 2 * 8), dtype=mx.float16)
    cos, sin = _tables(1, 2, 5, 8, token_major=True)
    mx.eval(rope.apply_split_rotary_emb(value, cos, sin))
    assert calls == [(1, 5, 32)]
    split_rope.set_enabled(False)
    try:
        mx.eval(rope.apply_split_rotary_emb(value, cos, sin))
    finally:
        split_rope.set_enabled(True)
    assert len(calls) == 1


def test_kernel_refuses_what_the_graph_must_handle() -> None:
    value = mx.zeros((2, 5, 2 * 2 * 8), dtype=mx.float16)
    cos, sin = _tables(1, 2, 5, 8, token_major=True)
    assert not split_rope.applies(value, cos.astype(mx.float16), sin.astype(mx.float16))
    assert not split_rope.applies(value, cos, sin[..., :4])
    assert not split_rope.applies(value.reshape(2, 5, 2, 16).transpose(0, 2, 1, 3), cos, sin)
    assert not split_rope.applies(value[:, :4], cos, sin)
    three_tables = _tables(3, 2, 5, 8, token_major=True)
    assert not split_rope.applies(value, *three_tables)
    with pytest.raises(ValueError, match="does not apply"):
        split_rope.rotate(value, cos.astype(mx.float16), sin.astype(mx.float16))
    split_rope.set_enabled(False)
    try:
        assert not split_rope.applies(value, cos, sin)
    finally:
        split_rope.set_enabled(True)
