from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from kinomlx.models.ltx2.text_encoder import (
    Gemma4Config,
    Gemma4Model,
    Gemma4RMSNorm,
    load_gemma4_weights,
)
from kinomlx.reporting import RecordingReporter

from ._synthetic import initialize_test_parameters


def _config() -> Gemma4Config:
    return Gemma4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        max_position_embeddings=32,
        sliding_window=4,
        full_partial_rotary_factor=0.5,
        layer_types=("sliding_attention", "full_attention"),
    )


def _model() -> Gemma4Model:
    model = Gemma4Model(_config())
    initialize_test_parameters(model)
    return model


def test_gemma4_rms_norm_uses_direct_checkpoint_scale() -> None:
    norm = Gemma4RMSNorm(2, 1e-6)
    norm.weight = mx.array([2.0, 3.0], dtype=mx.float32)
    value = mx.array([[3.0, 4.0]], dtype=mx.float32)

    result = norm(value)
    expected = value * mx.rsqrt(mx.mean(value**2, axis=-1, keepdims=True) + 1e-6)
    expected = expected * norm.weight

    assert mx.allclose(result, expected).item()


def test_gemma4_returns_exact_projection_states_and_named_boundaries() -> None:
    model = _model()
    reporter = RecordingReporter()
    final, states = model(
        mx.array([[0, 2, 3]], dtype=mx.int32),
        attention_mask=mx.array([[0, 1, 1]], dtype=mx.int32),
        reporter=reporter,
    )
    boundaries = model.forward_boundaries(
        mx.array([[0, 2, 3]], dtype=mx.int32),
        attention_mask=mx.array([[0, 1, 1]], dtype=mx.int32),
    )

    mx.eval(final, *states, *boundaries.values())
    assert tuple(final.shape) == (1, 3, 8)
    assert len(states) == 3
    assert mx.array_equal(final, states[-1]).item()
    assert tuple(boundaries) == ("embedding", "layer.00", "layer.01", "final_norm")
    assert mx.array_equal(final, boundaries["final_norm"]).item()
    assert reporter.events[0] == (
        "start",
        "encode prompt with Gemma 4",
        {"total": 2, "unit": "layer"},
    )
    assert reporter.events[-1] == ("end", "encode prompt with Gemma 4", {})


def test_gemma4_left_padding_drives_positions_and_masks() -> None:
    model = _model()
    attention_mask = mx.array([[0, 0, 1, 1]], dtype=mx.int32)

    positions = model.position_ids(attention_mask)
    full, sliding = model.attention_masks(attention_mask)
    mx.eval(positions, full, sliding)

    assert positions.tolist() == [[0, 0, 0, 1]]
    assert tuple(full.shape) == (1, 1, 4, 4)
    assert tuple(sliding.shape) == (1, 1, 4, 4)
    assert full[0, 0].tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


def test_gemma4_loader_binds_every_consumed_target_and_ignores_baggage(tmp_path) -> None:
    source = _model()
    weights = {f"model.{key}": value for key, value in tree_flatten(source.parameters())}
    consumed = len(weights)
    weights["community.wrapper.unused"] = mx.zeros((7,))
    path = tmp_path / "gemma4.safetensors"
    mx.save_safetensors(str(path), weights)

    target = Gemma4Model(_config())
    assert load_gemma4_weights(target, path) == consumed
    input_ids = mx.array([[0, 2, 3]], dtype=mx.int32)
    attention_mask = mx.array([[0, 1, 1]], dtype=mx.int32)
    expected, _ = source(input_ids, attention_mask=attention_mask)
    actual, _ = target(input_ids, attention_mask=attention_mask)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected).item()


@pytest.mark.parametrize("failure", ["missing", "shape"])
def test_gemma4_loader_rejects_invalid_consumed_target_before_mutation(
    failure: str,
    tmp_path,
) -> None:
    source = _model()
    weights = {f"model.{key}": value for key, value in tree_flatten(source.parameters())}
    key = "model.layers.0.self_attn.q_proj.weight"
    if failure == "missing":
        weights.pop(key)
    else:
        weights[key] = mx.zeros((1, 1))
    path = tmp_path / "gemma4.safetensors"
    mx.save_safetensors(str(path), weights)

    target = Gemma4Model(_config())
    with pytest.raises(ValueError, match="LTX-2.5 Gemma 4.*consumed"):
        load_gemma4_weights(target, path)
    assert target.embed_tokens.weight.size == 0
