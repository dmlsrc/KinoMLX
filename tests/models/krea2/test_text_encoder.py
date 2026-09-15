"""Synthetic graph, tap ordering, position, and loading contracts for Qwen text."""

import math
from pathlib import Path

import mlx.core as mx
import pytest

import kinomlx.models.krea2.text_encoder.loading as loading_module
from kinomlx.models.krea2.metadata import Krea2TextConfig
from kinomlx.models.krea2.text_encoder import load_qwen3_vl_text_weights
from kinomlx.models.krea2.text_encoder.model import Qwen3VLTextModel
from tests._reporting import RecordingReporter
from tests.models.krea2._source_bindings import fixture_bindings

_LOADER_TARGET_DIGEST = "3430c09f3f529a7f83c8"


def _config() -> Krea2TextConfig:
    return Krea2TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        mrope_section=(2, 1, 1),
        mrope_interleaved=True,
        vocab_size=64,
    )


def _populate(model: Qwen3VLTextModel) -> None:
    config = model.config
    mx.random.seed(20260831)
    model.embed_tokens.weight = mx.random.uniform(
        low=-0.1,
        high=0.1,
        shape=(config.vocab_size, config.hidden_size),
    ).astype(mx.float16)
    query_dims = config.num_attention_heads * config.head_dim
    kv_dims = config.num_key_value_heads * config.head_dim
    for layer in model.layers:
        layer.self_attn.q_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(query_dims, config.hidden_size),
        ).astype(mx.float16)
        layer.self_attn.k_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(kv_dims, config.hidden_size),
        ).astype(mx.float16)
        layer.self_attn.v_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(kv_dims, config.hidden_size),
        ).astype(mx.float16)
        layer.self_attn.o_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(config.hidden_size, query_dims),
        ).astype(mx.float16)
        layer.self_attn.q_norm.weight = mx.ones((config.head_dim,), dtype=mx.float16)
        layer.self_attn.k_norm.weight = mx.ones((config.head_dim,), dtype=mx.float16)
        layer.mlp.gate_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(config.intermediate_size, config.hidden_size),
        ).astype(mx.float16)
        layer.mlp.up_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(config.intermediate_size, config.hidden_size),
        ).astype(mx.float16)
        layer.mlp.down_proj.weight = mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(config.hidden_size, config.intermediate_size),
        ).astype(mx.float16)
        layer.input_layernorm.weight = mx.ones((config.hidden_size,), dtype=mx.float16)
        layer.post_attention_layernorm.weight = mx.ones(
            (config.hidden_size,),
            dtype=mx.float16,
        )


def _inputs() -> tuple[mx.array, mx.array, mx.array]:
    input_ids = mx.array(
        (
            (1, 2, 3, 4, 5, 6),
            (7, 8, 0, 0, 9, 10),
        ),
        dtype=mx.int32,
    )
    attention_mask = mx.array(
        (
            (1, 1, 1, 1, 1, 1),
            (1, 1, 0, 0, 1, 1),
        ),
        dtype=mx.bool_,
    )
    rows = mx.array(
        (
            (0, 1, 2, 3, 4, 5),
            (0, 1, 1, 1, 2, 3),
        ),
        dtype=mx.int32,
    )
    return input_ids, attention_mask, mx.stack((rows, rows, rows), axis=0)


def _loader_config() -> Krea2TextConfig:
    return Krea2TextConfig(
        hidden_size=8,
        intermediate_size=8,
        num_attention_heads=1,
        num_hidden_layers=36,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        mrope_section=(2, 1, 1),
        mrope_interleaved=True,
        vocab_size=16,
    )


@pytest.mark.requires_metal
def test_qwen_text_graph_uses_gqa_and_returns_declared_taps_in_order() -> None:
    model = Qwen3VLTextModel(_config(), (1, 3))
    _populate(model)
    input_ids, attention_mask, position_ids = _inputs()
    reporter = RecordingReporter()

    output = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        reporter=reporter,
    )
    assert output.selected_layers == (1, 3)
    assert output.hidden_states.shape == (2, 6, 2, 32)
    assert mx.all(mx.isfinite(output.hidden_states)).item()

    hidden = model.embed_tokens(input_ids)
    mask = model._attention_mask(attention_mask, input_ids.shape[1])
    rotary = model.rotary(hidden, position_ids)
    expected = []
    for completed, layer in enumerate(model.layers, start=1):
        hidden = layer(hidden, mask, rotary)
        mx.eval(hidden)
        if completed in model.selected_layers:
            expected.append(hidden)
    assert mx.array_equal(output.hidden_states, mx.stack(expected, axis=2)).item()
    assert reporter.events == [
        ("start", "encode prompt with Qwen3-VL", {"total": None, "unit": "it"}),
        ("end", "encode prompt with Qwen3-VL", {}),
    ]


@pytest.mark.requires_metal
def test_qwen_text_graph_rejects_raw_positions_for_middle_padding() -> None:
    model = Qwen3VLTextModel(_config(), (1,))
    _populate(model)
    input_ids, attention_mask, _position_ids = _inputs()
    raw = mx.arange(input_ids.shape[1], dtype=mx.int32)
    raw = mx.broadcast_to(raw[None, None, :], (3, input_ids.shape[0], input_ids.shape[1]))

    with pytest.raises(ValueError, match="cumulative valid positions"):
        model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=raw,
        )


@pytest.mark.requires_metal
def test_qwen_text_graph_rejects_noninteger_positions() -> None:
    model = Qwen3VLTextModel(_config(), (1,))
    _populate(model)
    input_ids, attention_mask, position_ids = _inputs()

    with pytest.raises(ValueError, match="integer dtype"):
        model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids.astype(mx.float32),
        )


@pytest.mark.requires_metal
def test_qwen_loader_binds_the_complete_default_target_graph(tmp_path: Path) -> None:
    model = Qwen3VLTextModel(_loader_config(), (35,))
    targets = loading_module._targets(model)
    assert len(targets) == 386
    cache = tmp_path / "text_encoder.safetensors"
    mx.save_safetensors(
        str(cache),
        {key: mx.zeros(target.shape, dtype=mx.float16) for key, target in targets.items()},
    )
    reporter = RecordingReporter()

    receipt = load_qwen3_vl_text_weights(
        model, source_bindings=fixture_bindings(model, cache, "text"), reporter=reporter
    )

    assert receipt.target_count == 386
    assert receipt.target_digest == _LOADER_TARGET_DIGEST
    assert receipt.source_files == (cache.name,)
    assert reporter.events == [
        ("start", "bind Qwen text weights", {"total": 1, "unit": "file"}),
        ("advance", "bind Qwen text weights", {"advance": 1.0}),
        ("end", "bind Qwen text weights", {}),
    ]


def test_qwen_text_graph_stops_after_last_selected_layer() -> None:
    model = Qwen3VLTextModel(_config(), (1, 3))
    assert len(model.layers) == 3
    assert model.executed_layers == 3
    assert not hasattr(model, "norm")
    assert not hasattr(model, "visual")
    assert not hasattr(model, "lm_head")


@pytest.mark.requires_metal
def test_qwen_loader_binds_source_bindings_like_the_prepared_cache(tmp_path: Path) -> None:
    from kinomlx.models.krea2.resource_types import Krea2TensorBinding

    direct = Qwen3VLTextModel(_loader_config(), (35,))
    cached = Qwen3VLTextModel(_loader_config(), (35,))
    targets = loading_module._targets(direct)
    source = tmp_path / "model.safetensors"
    cache = tmp_path / "text_encoder.safetensors"
    values = {
        key: (mx.arange(math.prod(target.shape), dtype=mx.float32) / 64 + index)
        .reshape(target.shape)
        .astype(mx.bfloat16)
        for index, (key, target) in enumerate(targets.items())
    }
    # The source carries original names and unrelated baggage; the cache is canonical.
    mx.save_safetensors(
        str(source),
        {f"language_model.{key}": value for key, value in values.items()}
        | {"visual.baggage": mx.zeros((2,), dtype=mx.bfloat16)},
    )
    mx.save_safetensors(str(cache), values)
    bindings = tuple(Krea2TensorBinding(source, f"language_model.{key}", key) for key in targets)

    direct_receipt = load_qwen3_vl_text_weights(direct, source_bindings=bindings)
    cached_receipt = load_qwen3_vl_text_weights(
        cached, source_bindings=fixture_bindings(cached, cache, "text")
    )

    assert direct_receipt.target_count == cached_receipt.target_count == 386
    assert direct_receipt.target_digest == cached_receipt.target_digest
    assert direct_receipt.source_files == (source.name,)
    cached_targets = loading_module._targets(cached)
    for key, target in targets.items():
        actual = getattr(target.owner, target.attribute)
        other = cached_targets[key]
        expected = getattr(other.owner, other.attribute)
        assert actual.dtype == expected.dtype == mx.bfloat16, key
        assert mx.array_equal(actual, expected).item(), key


def test_qwen_loader_refuses_both_routes_and_incomplete_source_bindings(tmp_path: Path) -> None:
    from kinomlx.models.krea2.resource_types import Krea2TensorBinding

    model = Qwen3VLTextModel(_loader_config(), (35,))
    binding = Krea2TensorBinding(tmp_path / "model.safetensors", "a", "embed_tokens.weight")
    with pytest.raises(ValueError, match="cover the target graph"):
        load_qwen3_vl_text_weights(model, source_bindings=(binding,))
