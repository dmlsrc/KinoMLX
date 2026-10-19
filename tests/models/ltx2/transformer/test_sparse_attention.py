"""Sparse video self-attention inside the LTX transformer."""

import math
from dataclasses import replace

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.sparse_attention import SparseAttentionSchedule, stage_steps
from kinomlx.models.ltx2.transformer import LTXAVModel, Modality
from kinomlx.models.ltx2.transformer import attention as attention_module
from kinomlx.models.ltx2.transformer import model as model_module

from ._synthetic import build_shaped_attention, build_shaped_ltx_model

# Four 64-token blocks of video: enough for routing to leave blocks out.
_GRID = (4, 8, 8)
_TOKENS = math.prod(_GRID)
_LAYERS = 3


def _sparse_step(tau: float, dense_layers: tuple[int, ...] = (0,)):
    (step,) = stage_steps(SparseAttentionSchedule((tau,), dense_layers), grid=_GRID, tokens=_TOKENS)
    assert step is not None
    return step


def _relative_error(actual: mx.array, expected: mx.array) -> float:
    difference = actual.astype(mx.float32) - expected.astype(mx.float32)
    reference = expected.astype(mx.float32)
    return math.sqrt((difference * difference).sum().item() / (reference * reference).sum().item())


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_attention_routing_every_block_matches_dense_attention(dtype: mx.Dtype) -> None:
    attention = build_shaped_attention(256, heads=2, dim_head=128, apply_gated_attention=True)
    attention.set_dtype(dtype)
    value = (mx.random.normal((1, _TOKENS, 256), key=mx.random.key(3)) * 0.5).astype(dtype)
    dense = attention(value)
    every_block = attention(value, sparse=_sparse_step(-1e6))
    sparse = attention(value, sparse=_sparse_step(1.5))
    mx.eval(dense, every_block, sparse)
    assert every_block.dtype == dense.dtype == dtype
    assert _relative_error(every_block, dense) < 1e-2
    assert not mx.array_equal(sparse, dense).item()


def test_sparse_attention_refuses_cross_attention_and_masks() -> None:
    attention = build_shaped_attention(256, heads=2, dim_head=128)
    attention.set_dtype(mx.float16)
    value = mx.zeros((1, _TOKENS, 256), dtype=mx.float16)
    with pytest.raises(ValueError, match="only to unmasked self-attention"):
        attention(value, context=value, sparse=_sparse_step(1.0))
    with pytest.raises(ValueError, match="only to unmasked self-attention"):
        attention(value, mask=mx.ones((1, _TOKENS, _TOKENS)), sparse=_sparse_step(1.0))


def _model(dtype: mx.Dtype) -> LTXAVModel:
    def factory() -> LTXAVModel:
        return LTXAVModel(
            num_layers=_LAYERS,
            video_heads=2,
            video_head_dim=128,
            audio_heads=2,
            audio_head_dim=2,
            video_in_channels=3,
            video_out_channels=3,
            audio_in_channels=2,
            audio_out_channels=2,
            video_context_dim=256,
            audio_context_dim=4,
            video_max_pos=(8, 8, 8),
            audio_max_pos=(8,),
            compute_dtype=dtype,
            double_precision_rope=True,
        )

    return build_shaped_ltx_model(factory, seed=5)


def _video(sparse_attention=None) -> Modality:
    frames, height, width = _GRID
    cells = [(f, h, w) for f in range(frames) for h in range(height) for w in range(width)]
    starts = mx.array(cells, dtype=mx.float32).T
    positions = mx.stack((starts, starts + 1.0), axis=-1)[None]
    return Modality(
        latent=mx.random.normal((1, _TOKENS, 3), key=mx.random.key(7)),
        context=mx.random.normal((1, 2, 256), key=mx.random.key(8)) * 0.1,
        timesteps=mx.full((1,), 0.5),
        sigma=mx.array([0.5]),
        positions=positions,
        sparse_attention=sparse_attention,
    )


def _count_sparse_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = attention_module.sparse_self_attention

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(attention_module, "sparse_self_attention", counted)
    return calls


def test_dense_layers_keep_dense_attention_and_the_rest_go_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(mx.bfloat16)
    calls = _count_sparse_calls(monkeypatch)
    dense, _ = model(_video())
    every_layer_dense, _ = model(_video(_sparse_step(1.0, dense_layers=(0, 1, 2))))
    mx.eval(dense, every_layer_dense)
    assert calls == []
    assert mx.array_equal(every_layer_dense, dense).item()

    routed_everywhere, _ = model(_video(_sparse_step(-1e6, dense_layers=(0,))))
    mx.eval(routed_everywhere)
    assert len(calls) == _LAYERS - 1
    assert _relative_error(routed_everywhere, dense) < 2e-2

    calls.clear()
    no_dense_layers, _ = model(_video(_sparse_step(1.5, dense_layers=())))
    mx.eval(no_dense_layers)
    assert len(calls) == _LAYERS
    assert not mx.array_equal(no_dense_layers, dense).item()


def test_sparse_forwards_bypass_compiled_block_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(mx.float16)
    eager, _ = model(_video(_sparse_step(1.0)))
    mx.eval(eager)
    model.compile_block_group_size = 1
    monkeypatch.setattr(
        model_module,
        "_compile_transformer_block_group",
        lambda blocks: pytest.fail("sparse forwards must not compile block groups"),
    )
    grouped, _ = model(_video(_sparse_step(1.0)))
    mx.eval(grouped)
    assert mx.array_equal(grouped, eager).item()


def test_modalities_carry_no_sparse_attention_by_default() -> None:
    video = _video()
    assert video.sparse_attention is None
    assert replace(video, sparse_attention=_sparse_step(1.0)).sparse_attention is not None
