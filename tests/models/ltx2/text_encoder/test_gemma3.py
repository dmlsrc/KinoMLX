from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from kinomlx.models.ltx2.text_encoder import Gemma3Config, Gemma3Model, load_gemma3_weights
from kinomlx.reporting import RecordingReporter

from ._synthetic import initialize_test_parameters


def _config() -> Gemma3Config:
    return Gemma3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        query_pre_attn_scalar=4,
        max_position_embeddings=32,
        sliding_window=4,
        layer_types=("sliding_attention", "full_attention"),
    )


def _model() -> Gemma3Model:
    model = Gemma3Model(_config())
    initialize_test_parameters(model)
    return model


def test_gemma_shell_construction_does_not_draw_random_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_random(*_args, **_kwargs):
        raise AssertionError("Gemma construction must not draw random values")

    monkeypatch.setattr(mx.random, "uniform", unexpected_random)
    monkeypatch.setattr(mx.random, "normal", unexpected_random)

    model = Gemma3Model(_config())

    assert tuple(model.embed_tokens.weight.shape) == (0, 0)


def test_gemma_returns_embedding_and_every_layer_state() -> None:
    model = _model()
    reporter = RecordingReporter()
    final, states = model(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        reporter=reporter,
    )
    assert states is not None
    mx.eval(final, *states)
    assert tuple(final.shape) == (1, 3, 8)
    assert len(states) == 3
    assert mx.allclose(final, states[-1]).item()
    assert mx.all(mx.isfinite(final)).item()
    assert reporter.events[0] == (
        "start",
        "encode prompt with Gemma 3",
        {"total": 2, "unit": "layer"},
    )
    assert reporter.events[-1] == ("end", "encode prompt with Gemma 3", {})


def test_gemma_rejects_mask_shape_mismatch() -> None:
    model = _model()
    with pytest.raises(ValueError, match="attention_mask"):
        model(
            mx.array([[2, 3, 4]], dtype=mx.int32),
            attention_mask=mx.ones((1, 2), dtype=mx.int32),
        )


def test_gemma3_loader_binds_every_consumed_target_and_ignores_baggage(tmp_path) -> None:
    source = _model()
    weights = {
        f"language_model.model.{key}": value for key, value in tree_flatten(source.parameters())
    }
    consumed = len(weights)
    weights["community.wrapper.unused"] = mx.zeros((7,))
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), weights)

    target = Gemma3Model(_config())
    assert load_gemma3_weights(target, path) == consumed
    input_ids = mx.array([[2, 3, 4]], dtype=mx.int32)
    expected, _ = source(input_ids)
    actual, _ = target(input_ids)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected).item()


@pytest.mark.parametrize("failure", ["missing", "shape"])
def test_gemma3_loader_rejects_invalid_consumed_target_before_mutation(
    failure: str,
    tmp_path,
) -> None:
    source = _model()
    weights = {
        f"language_model.model.{key}": value for key, value in tree_flatten(source.parameters())
    }
    key = "language_model.model.layers.0.self_attn.q_proj.weight"
    if failure == "missing":
        weights.pop(key)
    else:
        weights[key] = mx.zeros((1, 1))
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), weights)

    target = Gemma3Model(_config())
    with pytest.raises(ValueError, match="LTX-2.3 Gemma 3.*consumed"):
        load_gemma3_weights(target, path)
    assert target.embed_tokens.weight.size == 0
