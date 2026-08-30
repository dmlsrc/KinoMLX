"""Expansion math, geometry handling, and the gain-map sidecar."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.io.safetensors import load_weights_with_metadata
from kinomlx.models.gmnet.catalog import GMNetVariant, variant_spec
from kinomlx.models.gmnet.expand import (
    expand_image,
    reconstruct_linear_hdr,
    write_gain_map_sidecar,
)
from kinomlx.models.gmnet.net import GMNet


def test_reconstruction_law_hits_known_values():
    image = mx.array([[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])
    gain = mx.array([[1.0, 0.0]])
    linear = np.array(reconstruct_linear_hdr(image, gain, math.log2(5.0)))
    np.testing.assert_allclose(linear[0, 0], 5.0, rtol=1e-6)
    np.testing.assert_allclose(linear[0, 1], 0.5**2.2, rtol=1e-6)


def test_reconstruction_clips_out_of_range_inputs():
    image = mx.array([[[2.0, -1.0, 1.0]]])
    gain = mx.array([[3.0]])
    linear = np.array(reconstruct_linear_hdr(image, gain, 2.0))
    np.testing.assert_allclose(linear[0, 0], [4.0, 0.0, 4.0], rtol=1e-6)


def _random_model() -> GMNet:
    model = GMNet()
    mx.eval(model.parameters())
    model.eval()
    return model


def test_expand_image_realworld_geometry_and_bounds():
    spec = variant_spec(GMNetVariant.REALWORLD)
    image = mx.random.uniform(shape=(33, 47, 3))
    result = expand_image(_random_model(), image, spec)

    assert tuple(result.linear_rgb.shape) == (33, 47, 3)
    assert tuple(result.gain_map.shape) == (33, 47)
    gain = np.array(result.gain_map)
    assert gain.min() >= 0.0
    assert gain.max() <= 1.0
    linear = np.array(result.linear_rgb)
    sdr_linear = np.clip(np.array(image), 0.0, 1.0) ** 2.2
    assert (linear >= sdr_linear - 1e-5).all()
    assert linear.max() <= spec.peak_over_sdr_white + 1e-4
    assert math.isfinite(result.qmax_normalized)
    assert result.peak_linear >= 0.0


def test_expand_image_synthetic_runs_full_resolution():
    spec = variant_spec(GMNetVariant.SYNTHETIC)
    image = mx.random.uniform(shape=(32, 40, 3))
    result = expand_image(_random_model(), image, spec)
    assert tuple(result.linear_rgb.shape) == (32, 40, 3)
    assert np.array(result.linear_rgb).max() <= spec.peak_over_sdr_white + 1e-4


def test_gain_map_sidecar_round_trips(tmp_path):
    spec = variant_spec(GMNetVariant.REALWORLD)
    image = mx.random.uniform(shape=(16, 16, 3))
    result = expand_image(_random_model(), image, spec)

    source = tmp_path / "input.png"
    source.write_bytes(b"not really a png; only hashed")
    target = write_gain_map_sidecar(
        tmp_path / "input.gain_map.safetensors", result, source_image=source
    )

    tensors, metadata = load_weights_with_metadata(target)
    np.testing.assert_array_equal(np.array(tensors["gain_map"]), np.array(result.gain_map))
    assert metadata["producer"] == "gmnet"
    assert metadata["variant"] == "realworld"
    assert metadata["peak_over_sdr_white"] == "5"
    assert float(metadata["gain_stops_max"]) == pytest.approx(spec.gain_stops, rel=1e-7)
    assert float(metadata["qmax_normalized"]) == pytest.approx(result.qmax_normalized, rel=1e-6)
    assert metadata["source_image"] == "input.png"
    assert len(metadata["source_image_sha256"]) == 64
    assert "2.2" in metadata["reconstruction"]
