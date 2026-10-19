"""FP16 dense attention kernel: parity with FP32 attention and where the model uses it."""

import math

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.kernels import dense_attention
from kinomlx.models.ltx2.transformer import LTXAVModel, Modality
from kinomlx.models.ltx2.transformer import attention as attention_module

from .transformer._synthetic import build_shaped_attention, build_shaped_ltx_model

_HEADS = 2
_TOLERANCE = {mx.float16: 1e-3, mx.bfloat16: 6e-3}


def _tensor(tokens: int, dim: int, dtype: mx.Dtype, seed: int) -> mx.array:
    return mx.random.normal((tokens, _HEADS * dim), key=mx.random.key(seed)).astype(dtype)


def _inputs(
    query_tokens: int, key_tokens: int, dim: int, dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array, mx.array]:
    return (
        _tensor(query_tokens, dim, dtype, seed),
        _tensor(key_tokens, dim, dtype, seed + 1),
        _tensor(key_tokens, dim, dtype, seed + 2),
    )


def _reference(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    dim = q.shape[1] // _HEADS

    def split(x: mx.array) -> mx.array:
        heads_first = x.astype(mx.float32).reshape(1, x.shape[0], _HEADS, dim)
        return heads_first.transpose(0, 2, 1, 3)

    out = mx.fast.scaled_dot_product_attention(split(q), split(k), split(v), scale=dim**-0.5)
    return out.transpose(0, 2, 1, 3).reshape(q.shape[0], _HEADS * dim)


def _relative_error(actual: mx.array, expected: mx.array) -> float:
    difference = actual.astype(mx.float32) - expected.astype(mx.float32)
    reference = expected.astype(mx.float32)
    return math.sqrt((difference * difference).sum().item() / (reference * reference).sum().item())


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize(
    ("query_tokens", "key_tokens"),
    [(7, 7), (16, 16), (33, 33), (517, 517), (300, 1024), (1, 300), (517, 33), (40, 5)],
)
def test_kernel_matches_fp32_attention_across_tile_edges(
    dtype: mx.Dtype, dim: int, query_tokens: int, key_tokens: int
) -> None:
    q, k, v = _inputs(query_tokens, key_tokens, dim, dtype, seed=query_tokens + key_tokens + dim)
    actual = dense_attention.dense_attention(q, k, v, heads=_HEADS)
    expected = _reference(q, k, v)
    mx.eval(actual, expected)
    assert actual.dtype == dtype
    assert actual.shape == q.shape
    assert _relative_error(actual, expected) < _TOLERANCE[dtype]


def test_keys_past_a_partial_tile_take_no_weight() -> None:
    # A key that only a wrong mask could reach would dominate every row: it matches every
    # query strongly and carries a large value. It sits one past the real keys, in the same
    # buffer, so a kernel that read past the last key would find it.
    q, k, v = _inputs(9, 17, 128, mx.float16, seed=3)
    padded_k = mx.concatenate([k, mx.full((1, k.shape[1]), 8.0, dtype=mx.float16)])
    padded_v = mx.concatenate([v, mx.full((1, v.shape[1]), 1000.0, dtype=mx.float16)])
    actual = dense_attention.dense_attention(q, padded_k[:17], padded_v[:17], heads=_HEADS)
    mx.eval(actual)
    assert _relative_error(actual, _reference(q, k, v)) < _TOLERANCE[mx.float16]


def test_bfloat16_values_beyond_float16_saturate_instead_of_overflowing() -> None:
    q, k, v = _inputs(64, 64, 128, mx.bfloat16, seed=4)
    v = mx.concatenate([mx.full((1, v.shape[1]), 1e5, dtype=mx.bfloat16), v[1:]])
    out = dense_attention.dense_attention(q, k, v, heads=_HEADS)
    assert mx.all(mx.isfinite(out)).item()


def test_inputs_outside_the_kernel_contract_are_refused() -> None:
    q, k, v = _inputs(32, 20, 128, mx.float16, seed=5)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        dense_attention.dense_attention(*(x.astype(mx.float32) for x in (q, k, v)), heads=_HEADS)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        dense_attention.dense_attention(q, k, v.astype(mx.bfloat16), heads=_HEADS)
    with pytest.raises(ValueError, match="64 or 128"):
        dense_attention.dense_attention(q, k, v, heads=3)
    with pytest.raises(ValueError, match="64 or 128"):
        dense_attention.dense_attention(q, k, v, heads=8)
    with pytest.raises(ValueError, match="64 or 128"):
        dense_attention.dense_attention(q, k[:, :128], v[:, :128], heads=_HEADS)
    with pytest.raises(ValueError, match="64 or 128"):
        dense_attention.dense_attention(q, k, v[:-1], heads=_HEADS)


def test_only_64_or_128_wide_float16_or_bfloat16_heads_apply() -> None:
    def zeros(tokens: int, width: int, dtype: mx.Dtype = mx.float16) -> mx.array:
        return mx.zeros((1, tokens, width), dtype=dtype)

    assert dense_attention.applies(zeros(5, 256), zeros(9, 256), heads=2)
    assert dense_attention.applies(zeros(512, 128), zeros(512, 128), heads=2)
    assert dense_attention.applies(zeros(5, 256, mx.bfloat16), zeros(5, 256, mx.bfloat16), heads=2)
    assert not dense_attention.applies(
        zeros(5, 256, mx.float32), zeros(5, 256, mx.float32), heads=2
    )
    assert not dense_attention.applies(zeros(5, 256), zeros(5, 256, mx.bfloat16), heads=2)
    assert not dense_attention.applies(zeros(5, 96), zeros(5, 96), heads=2)
    assert not dense_attention.applies(zeros(5, 256), zeros(5, 128), heads=2)
    assert not dense_attention.applies(zeros(0, 256), zeros(5, 256), heads=2)
    assert not dense_attention.applies(zeros(5, 256), zeros(0, 256), heads=2)
    assert not dense_attention.applies(mx.zeros((5, 256)), mx.zeros((5, 256)), heads=2)


def test_64_wide_calls_take_the_kernel_from_512_tokens_on_both_sides() -> None:
    def zeros(tokens: int) -> mx.array:
        return mx.zeros((1, tokens, 128), dtype=mx.float16)

    assert dense_attention.applies(zeros(512), zeros(512), heads=2)
    assert dense_attention.applies(zeros(30576), zeros(752), heads=2)
    assert not dense_attention.applies(zeros(511), zeros(30576), heads=2)
    assert not dense_attention.applies(zeros(30576), zeros(511), heads=2)
    assert not dense_attention.applies(zeros(127), zeros(127), heads=2)


def _count_dense_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = attention_module.dense_kernel_attention

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(attention_module, "dense_kernel_attention", counted)
    return calls


def _attention(
    *, context_dim: int | None = None, dtype: mx.Dtype, dense_kernel: bool, dim_head: int = 128
):
    attention = build_shaped_attention(
        2 * dim_head,
        heads=2,
        dim_head=dim_head,
        context_dim=context_dim,
        apply_gated_attention=True,
    )
    attention.dense_kernel = dense_kernel
    attention.set_dtype(dtype)
    return attention


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("context_dim", [None, 64])
def test_attention_module_matches_its_previous_path(
    dtype: mx.Dtype, context_dim: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_dense_calls(monkeypatch)
    value = (mx.random.normal((1, 300, 256), key=mx.random.key(6)) * 0.5).astype(dtype)
    context = None
    if context_dim is not None:
        context = (mx.random.normal((1, 77, context_dim), key=mx.random.key(7)) * 0.5).astype(dtype)
    outputs = {
        dense_kernel: _attention(context_dim=context_dim, dtype=dtype, dense_kernel=dense_kernel)(
            value, context=context
        )
        for dense_kernel in (True, False)
    }
    mx.eval(*outputs.values())
    assert calls == [1]
    assert _relative_error(outputs[True], outputs[False]) < _TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_64_wide_attention_takes_the_kernel_from_512_tokens(
    dtype: mx.Dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_dense_calls(monkeypatch)
    for tokens, routed in ((512, [1]), (511, [])):
        calls.clear()
        value = (mx.random.normal((1, tokens, 128), key=mx.random.key(12)) * 0.5).astype(dtype)
        outputs = {
            dense_kernel: _attention(dtype=dtype, dense_kernel=dense_kernel, dim_head=64)(value)
            for dense_kernel in (True, False)
        }
        mx.eval(*outputs.values())
        assert calls == routed
        assert _relative_error(outputs[True], outputs[False]) < _TOLERANCE[dtype]


def test_masked_attention_keeps_its_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_dense_calls(monkeypatch)
    masked = _attention(dtype=mx.float16, dense_kernel=True)
    value = mx.zeros((1, 40, 256), dtype=mx.float16)
    mx.eval(masked(value, mask=mx.ones((1, 40, 40), dtype=mx.bool_)))
    assert calls == []


def _model(dense_attention_kernel: bool) -> LTXAVModel:
    def factory() -> LTXAVModel:
        return LTXAVModel(
            num_layers=2,
            video_heads=2,
            video_head_dim=128,
            audio_heads=2,
            audio_head_dim=64,
            video_in_channels=3,
            video_out_channels=3,
            audio_in_channels=2,
            audio_out_channels=2,
            video_context_dim=256,
            audio_context_dim=128,
            video_max_pos=(8, 8, 8),
            audio_max_pos=(8,),
            compute_dtype=mx.bfloat16,
            double_precision_rope=True,
            dense_attention_kernel=dense_attention_kernel,
        )

    return build_shaped_ltx_model(factory, seed=9)


def _video(tokens: int = 48) -> Modality:
    cells = mx.arange(tokens, dtype=mx.float32)
    starts = mx.stack([cells // 16, (cells // 4) % 4, cells % 4])
    return Modality(
        latent=mx.random.normal((1, tokens, 3), key=mx.random.key(10)),
        context=mx.random.normal((1, 2, 256), key=mx.random.key(11)) * 0.1,
        timesteps=mx.full((1,), 0.5),
        sigma=mx.array([0.5]),
        positions=mx.stack((starts, starts + 1.0), axis=-1)[None],
    )


def test_blocks_give_every_attention_call_the_switch() -> None:
    for enabled in (True, False):
        for block in _model(enabled).transformer_blocks:
            for attention in (
                block.attn1,
                block.attn2,
                block.audio_attn1,
                block.audio_attn2,
                block.audio_to_video_attn,
                block.video_to_audio_attn,
            ):
                assert attention.dense_kernel is enabled


def test_the_model_routes_video_self_and_text_attention_to_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_dense_calls(monkeypatch)
    with_kernel, _ = _model(True)(_video())
    mx.eval(with_kernel)
    assert len(calls) == 4
    without_kernel, _ = _model(False)(_video())
    mx.eval(without_kernel)
    assert len(calls) == 4
    assert _relative_error(with_kernel, without_kernel) < 1e-2
