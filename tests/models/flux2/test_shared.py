"""Klein callers exercise the shared tokenizer and Qwen language layers."""

import json

import mlx.core as mx
import pytest

from kinomlx._json import load_json_object
from kinomlx.huggingface import resolve_cached_snapshot
from kinomlx.modeling.qwen import QwenDecoderLayer
from kinomlx.models.flux2.catalog import profile
from kinomlx.models.flux2.metadata import Flux2TextConfig
from kinomlx.models.flux2.text_encoder import Flux2TextEncoder
from kinomlx.settings import Settings
from kinomlx.tokenizers.byte_bpe import ByteBPETables, QwenByteBPETokenizer


def test_klein_token_ids_match_reference_byte_bpe():
    tokenizers = pytest.importorskip("tokenizers")
    try:
        root = (
            resolve_cached_snapshot(Settings.from_env().hf_home, profile("kv").repo_id).path
            / "tokenizer"
        )
    except FileNotFoundError:
        pytest.skip("Klein tokenizer is not cached")
    tables = ByteBPETables.from_source(
        load_json_object(root / "tokenizer.json"), load_json_object(root / "tokenizer_config.json")
    )
    runtime = QwenByteBPETokenizer(tables)
    reference = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))
    corpus = [
        "",
        " ",
        "  A small red sphere.\n",
        "12,345.67890",
        "I'm we're I'D",
        "caf\u00e9 cafe\u0301",
        "\u65e5\u672c\u8a9e",
        "\U0001f469\u200d\U0001f4bb",
        "<|im_start|>user\nhello<|im_end|>",
    ]
    corpus += [f" {number}  tabs\tline\n punctuation!? " for number in range(128)]
    for text in corpus:
        assert runtime.encode(text) == tuple(reference.encode(text, add_special_tokens=False).ids)
    assert ByteBPETables.from_payload(json.loads(json.dumps(tables.to_payload()))) == tables


def test_plain_qwen_taps_use_shared_layers_and_causal_padding():
    config = Flux2TextConfig(
        hidden_size=8,
        intermediate_size=12,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=16,
    )
    model = Flux2TextEncoder(config)
    mx.random.seed(11)
    model.embed_tokens.weight = mx.random.normal((16, 8))
    for layer in model.layers:
        assert isinstance(layer, QwenDecoderLayer)
        layer.input_layernorm.weight = mx.ones(8)
        layer.post_attention_layernorm.weight = mx.ones(8)
        layer.self_attn.q_norm.weight = mx.ones(4)
        layer.self_attn.k_norm.weight = mx.ones(4)
        for projection, shape in (
            (layer.self_attn.q_proj, (8, 8)),
            (layer.self_attn.k_proj, (4, 8)),
            (layer.self_attn.v_proj, (4, 8)),
            (layer.self_attn.o_proj, (8, 8)),
            (layer.mlp.gate_proj, (12, 8)),
            (layer.mlp.up_proj, (12, 8)),
            (layer.mlp.down_proj, (8, 12)),
        ):
            projection.weight = mx.random.normal(shape) * 0.05
    original = model(mx.array([[1, 2, 3, 4]]), mx.array([[1, 1, 1, 1]]))
    changed = model(mx.array([[1, 2, 8, 9]]), mx.array([[1, 1, 1, 1]]))
    assert original.shape == (1, 4, 24)
    assert mx.allclose(original[:, :2], changed[:, :2]).item()
    assert not mx.allclose(original[:, 2:], changed[:, 2:]).item()
    mask = mx.array([[1, 0, 1, 1]])
    masked = model(mx.array([[1, 2, 3, 4]]), mask)
    masked_change = model(mx.array([[1, 9, 3, 4]]), mask)
    assert mx.allclose(masked[:, 2:], masked_change[:, 2:]).item()
    assert not mx.allclose(original[..., :8], original[..., 16:]).item()
