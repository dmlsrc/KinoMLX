"""Exact Krea transformer consumed-target and source-loading contracts."""

from dataclasses import replace
from math import prod
from pathlib import Path

import mlx.core as mx
import pytest

import kinomlx.modeling.loading as shared_loading
import kinomlx.models.krea2.transformer.loading as loading_module
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.models.krea2.transformer import Krea2Transformer
from kinomlx.models.krea2.transformer.loading import (
    KREA2_TRANSFORMER_CONSUMED_TENSORS,
    load_krea2_transformer_weights,
)
from kinomlx.models.krea2.weight_keys import expected_transformer_keys, transformer_weight_transform
from tests._reporting import RecordingReporter
from tests.models.krea2._source_bindings import fixture_bindings

from ._fixtures import official_transformer_config, tiny_transformer_config

_TINY_TARGET_DIGEST = "3e17eae24267353b2f8b"


def _weights(model: Krea2Transformer) -> dict[str, mx.array]:
    return {
        key: mx.zeros(
            target.shape,
            dtype=mx.float16,
        )
        for key, target in loading_module.transformer_weight_targets(model).items()
    }


def test_official_raw_and_turbo_graph_has_exactly_430_canonical_targets() -> None:
    model = Krea2Transformer(official_transformer_config())
    targets = loading_module.transformer_weight_targets(model)

    assert KREA2_TRANSFORMER_CONSUMED_TENSORS == 430
    assert len(targets) == KREA2_TRANSFORMER_CONSUMED_TENSORS
    assert frozenset(targets) == expected_transformer_keys(model.config)


@pytest.mark.requires_metal
def test_transformer_loader_binds_every_tiny_graph_target_once(tmp_path: Path) -> None:
    model = Krea2Transformer(tiny_transformer_config())
    cache = tmp_path / "transformer.safetensors"
    weights = _weights(model)
    mx.save_safetensors(str(cache), weights)
    reporter = RecordingReporter()

    receipt = load_krea2_transformer_weights(
        model,
        source_bindings=fixture_bindings(model, cache, "transformer"),
        expected_dtype=mx.float16,
        reporter=reporter,
    )

    assert receipt.target_count == len(weights) == 68
    assert receipt.target_digest == _TINY_TARGET_DIGEST
    assert receipt.source_files == (cache.name,)
    assert reporter.events == [
        ("start", "bind transformer weights", {"total": 1, "unit": "file"}),
        ("advance", "bind transformer weights", {"advance": 1.0}),
        ("end", "bind transformer weights", {}),
    ]


@pytest.mark.parametrize(
    "missing",
    [
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.to_gate.weight",
        "transformer_blocks.0.ff.gate.weight",
        "transformer_blocks.0.ff.up.weight",
        "transformer_blocks.0.ff.down.weight",
    ],
)
def test_transformer_loader_rejects_missing_attention_and_mlp_targets(
    tmp_path: Path,
    missing: str,
) -> None:
    model = Krea2Transformer(tiny_transformer_config())
    weights = _weights(model)
    del weights[missing]
    cache = tmp_path / f"missing-{missing.rsplit('.', 2)[-2]}.safetensors"
    mx.save_safetensors(str(cache), weights)

    with pytest.raises(
        ValueError,
        match="has no tensor",
    ):
        load_krea2_transformer_weights(
            model,
            source_bindings=fixture_bindings(model, cache, "transformer"),
            expected_dtype=mx.float16,
        )


def test_transformer_loader_rejects_wrong_target_shape(tmp_path: Path) -> None:
    model = Krea2Transformer(tiny_transformer_config())
    weights = _weights(model)
    key = "transformer_blocks.0.attn.to_k.weight"
    weights[key] = mx.zeros((1, 1), dtype=mx.float16)
    cache = tmp_path / "wrong-shape.safetensors"
    mx.save_safetensors(str(cache), weights)

    with pytest.raises(ValueError, match="to_k.weight.*has prepared shape"):
        load_krea2_transformer_weights(
            model,
            source_bindings=fixture_bindings(model, cache, "transformer"),
            expected_dtype=mx.float16,
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_transformer_loader_accepts_each_requested_weight_dtype(tmp_path: Path, dtype) -> None:
    model = Krea2Transformer(tiny_transformer_config())
    cache = tmp_path / "transformer.safetensors"
    mx.save_safetensors(
        str(cache), {key: value.astype(dtype) for key, value in _weights(model).items()}
    )
    load_krea2_transformer_weights(
        model, source_bindings=fixture_bindings(model, cache, "transformer"), expected_dtype=dtype
    )
    assert all(
        getattr(target.owner, target.attribute).dtype == dtype
        for target in loading_module.transformer_weight_targets(model).values()
    )


@pytest.mark.parametrize("reference", [False, True])
def test_loader_binds_baked_weights_directly_and_only_restores_reference_layout(
    tmp_path, monkeypatch, reference
):
    config = replace(tiny_transformer_config(), intermediate_size=40)
    model = Krea2Transformer(config, reference_denoise=reference)
    weights = {key: value.astype(mx.float32) for key, value in _weights(model).items()}
    key = "transformer_blocks.0.ff.down.weight"
    source = mx.arange(config.hidden_dim * 40, dtype=mx.float32).reshape(config.hidden_dim, 40)
    weights[key] = source / 1000
    cache = tmp_path / "transformer.safetensors"
    mx.save_safetensors(str(cache), weights)
    monkeypatch.setattr(shared_loading, "load_weights", lambda _path: dict(weights))
    load_krea2_transformer_weights(
        model,
        source_bindings=fixture_bindings(model, cache, "transformer"),
        expected_dtype="float32",
    )

    for name, target in loading_module.transformer_weight_targets(model).items():
        baked = transformer_weight_transform(name) is not None
        transposed = not reference and baked
        assert isinstance(target.owner, PretransposedLinear) is transposed
        actual = getattr(target.owner, target.attribute)
        expected = weights[name].T if not reference and baked else weights[name]
        assert mx.array_equal(actual, expected).item(), name
    value = mx.arange(80, dtype=mx.float32).reshape(2, 40) / 100
    assert mx.allclose(model.transformer_blocks[0].ff.down(value), value @ weights[key].T).item()


def test_reference_fp16_loader_refuses_before_reading_weights(tmp_path, monkeypatch):
    model = Krea2Transformer(tiny_transformer_config(), reference_denoise=True)

    def unexpected_read(*args, **kwargs):
        pytest.fail("reference FP16 reached weight I/O")

    monkeypatch.setattr(loading_module, "bind_source_weights", unexpected_read)
    with pytest.raises(ValueError, match="padding can cause FP16 overflow"):
        load_krea2_transformer_weights(
            model,
            source_bindings=fixture_bindings(model, tmp_path / "missing", "transformer"),
            expected_dtype="float16",
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_source_binding_applies_layout_reshape_and_dtype(tmp_path, dtype):
    from kinomlx.models.krea2.resource_types import Krea2TensorBinding

    direct = Krea2Transformer(tiny_transformer_config())
    source = tmp_path / "source.safetensors"
    bindings = []
    raw = {}
    for key, target in loading_module.transformer_weight_targets(direct).items():
        value = mx.arange(prod(target.shape), dtype=mx.float32)
        value = (value.reshape(target.shape) / 1000).astype(mx.bfloat16)
        rows = target.shape[0] if key.endswith("scale_shift_table") else None
        raw[key] = value.reshape(-1) if rows is not None else value
        bindings.append(
            Krea2TensorBinding(source, key, key, rows, transformer_weight_transform(key))
        )
    mx.save_safetensors(str(source), raw)
    load_krea2_transformer_weights(direct, source_bindings=tuple(bindings), expected_dtype=dtype)
    for key, target in loading_module.transformer_weight_targets(direct).items():
        actual = getattr(target.owner, target.attribute)
        expected = raw[key].reshape(target.shape).astype(dtype)
        if key.startswith("transformer_blocks.") and key.endswith(".ff.down.weight"):
            expected = expected.T
        assert actual.dtype == dtype
        assert mx.array_equal(actual, expected).item(), key
