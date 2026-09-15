"""Exact Krea Qwen Image VAE cache-binding tests."""

import math
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.krea2.resource_types import Krea2TensorBinding
from kinomlx.models.krea2.vae.loading import _targets, load_krea2_vae_weights
from kinomlx.models.krea2.vae.model import Krea2ImageDecoder
from kinomlx.models.krea2.weight_keys import (
    KREA2_VAE_CONV2D_OHWI,
    KREA2_VAE_CURRENT_FRAME_OHWI,
    KREA2_VAE_FLATTEN,
    expected_vae_keys,
    vae_bindings,
)
from tests.models.krea2._source_bindings import fixture_bindings

from ._vae_fixtures import tiny_vae_config, write_tiny_vae_cache


def test_loader_binds_complete_graph_and_records_stable_receipt(tmp_path: Path) -> None:
    model = Krea2ImageDecoder(tiny_vae_config(), compute_dtype=mx.float32)
    cache = tmp_path / "vae.safetensors"
    targets = _targets(model)
    write_tiny_vae_cache(cache, model)

    receipt = load_krea2_vae_weights(
        model,
        source_bindings=fixture_bindings(model, cache, "vae"),
        expected_dtype=model.compute_dtype,
    )

    assert receipt.target_count == len(targets)
    assert receipt.source_files == ("vae.safetensors",)
    assert len(receipt.target_digest) == 20
    assert model.load_receipt == receipt


def test_loader_rejects_missing_target_before_first_assignment(tmp_path: Path) -> None:
    model = Krea2ImageDecoder(tiny_vae_config(), compute_dtype=mx.float32)
    cache = tmp_path / "vae.safetensors"
    weights = write_tiny_vae_cache(cache, model)
    weights.pop("decoder.conv_out.bias")
    mx.save_safetensors(str(cache), weights)

    with pytest.raises(ValueError, match="decoder.conv_out.bias"):
        load_krea2_vae_weights(
            model,
            source_bindings=fixture_bindings(model, cache, "vae"),
            expected_dtype=model.compute_dtype,
        )
    assert model.post_quant_conv.conv.weight.size == 0


def test_loader_rejects_wrong_native_conv_layout(tmp_path: Path) -> None:
    model = Krea2ImageDecoder(tiny_vae_config(), compute_dtype=mx.float32)
    cache = tmp_path / "vae.safetensors"
    weights = write_tiny_vae_cache(cache, model)
    weights["decoder.conv_in.weight"] = mx.zeros((4, 2, 3, 3), dtype=mx.float32)
    mx.save_safetensors(str(cache), weights)

    with pytest.raises(ValueError, match="has prepared shape"):
        load_krea2_vae_weights(
            model,
            source_bindings=fixture_bindings(model, cache, "vae"),
            expected_dtype=model.compute_dtype,
        )


def _source_layout(prepared: mx.array, transform: str | None) -> mx.array:
    """Invert one source transform so ``prepare_weight`` recovers ``prepared``."""
    if transform == KREA2_VAE_CONV2D_OHWI:
        return prepared.transpose(0, 3, 1, 2)
    if transform == KREA2_VAE_CURRENT_FRAME_OHWI:
        current = prepared.transpose(0, 3, 1, 2)[:, :, None]
        return mx.concatenate([mx.full(current.shape, -7.0, dtype=current.dtype), current], axis=2)
    if transform == KREA2_VAE_FLATTEN:
        return prepared.reshape(-1, 1, 1, 1)
    assert transform is None, transform
    return prepared


def test_loader_binds_source_layout_like_the_prepared_cache(tmp_path: Path) -> None:
    config = tiny_vae_config()
    direct = Krea2ImageDecoder(config, compute_dtype=mx.bfloat16)
    cached = Krea2ImageDecoder(config, compute_dtype=mx.bfloat16)
    targets = _targets(direct)
    bindings = vae_bindings(tuple(sorted(expected_vae_keys(config))), config)
    assert {binding.target_key for binding in bindings} == set(targets)
    source_file = tmp_path / "diffusion_pytorch_model.safetensors"
    cache = tmp_path / "vae.safetensors"
    prepared = {}
    source = {"encoder.baggage": mx.zeros((1,), dtype=mx.float32)}
    for index, binding in enumerate(bindings):
        shape = targets[binding.target_key].shape
        value = (mx.arange(math.prod(shape), dtype=mx.float32) / 7 + index).reshape(shape)
        prepared[binding.target_key] = value.astype(mx.bfloat16)
        source[binding.source_key] = _source_layout(value, binding.transform)
    mx.save_safetensors(str(source_file), source)
    mx.save_safetensors(str(cache), prepared)
    source_bindings = tuple(
        Krea2TensorBinding(source_file, b.source_key, b.target_key, None, b.transform)
        for b in bindings
    )

    direct_receipt = load_krea2_vae_weights(
        direct, source_bindings=source_bindings, expected_dtype="bfloat16"
    )
    cached_receipt = load_krea2_vae_weights(
        cached, source_bindings=fixture_bindings(cached, cache, "vae"), expected_dtype="bfloat16"
    )

    assert direct_receipt.target_digest == cached_receipt.target_digest
    assert direct_receipt.source_files == (source_file.name,)
    cached_targets = _targets(cached)
    for key, target in targets.items():
        actual = getattr(target.owner, target.attribute)
        other = cached_targets[key]
        expected = getattr(other.owner, other.attribute)
        assert actual.dtype == expected.dtype == mx.bfloat16, key
        assert actual.shape == expected.shape, key
        assert mx.array_equal(actual, expected).item(), key


def test_direct_loader_requires_the_component_dtype_policy(tmp_path: Path) -> None:
    model = Krea2ImageDecoder(tiny_vae_config(), compute_dtype=mx.float32)
    binding = Krea2TensorBinding(tmp_path / "source.safetensors", "a", "decoder.conv_out.bias")
    with pytest.raises(ValueError, match="dtype policy"):
        load_krea2_vae_weights(model, source_bindings=(binding,))
    with pytest.raises(ValueError, match="requires the component dtype policy"):
        load_krea2_vae_weights(model, source_bindings=(binding,))
