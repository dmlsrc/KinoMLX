"""Still encoder shape, normalization, source-layout, and admission contracts."""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.krea2.vae.encoder import Krea2Downsample, Krea2ImageEncoder
from kinomlx.models.krea2.vae.encoder_loading import (
    encoder_source_shape,
    encoder_weight_targets,
    load_krea2_encoder_weights,
)

from ._fixtures import tiny_vae_config


def test_encoder_posterior_mode_and_normalization(tmp_path: Path) -> None:
    config = replace(
        tiny_vae_config(latent_mean=(2.0, -3.0), latent_std=(2.0, 4.0)), num_res_blocks=1
    )
    model = Krea2ImageEncoder(config, compute_dtype="float32")
    weights = {
        key: mx.full(encoder_source_shape(key, target.shape), float(key.endswith(".gamma")))
        for key, target in encoder_weight_targets(model).items()
    }
    # The variance half must never enter a deterministic reference latent.
    weights["quant_conv.bias"] = mx.array([4.0, 5.0, 90.0, -90.0])
    source = tmp_path / "source.safetensors"
    mx.save_safetensors(str(source), weights)
    receipt = load_krea2_encoder_weights(model, source)
    actual = model(mx.zeros((2, 8, 12, 3)))
    assert actual.shape == (2, 2, 4, 6)
    assert mx.array_equal(actual[:, 0], mx.ones((2, 4, 6)))
    assert mx.array_equal(actual[:, 1], mx.full((2, 4, 6), 2.0))
    assert receipt.target_count == len(weights)


def test_downsample_pads_bottom_and_right_only() -> None:
    model = Krea2Downsample(1)
    model.resample.conv.weight = mx.ones((1, 3, 3, 1))
    model.resample.conv.bias = mx.zeros((1,))
    result = model(mx.ones((1, 4, 4, 1)))
    assert mx.array_equal(result[0, :, :, 0], mx.array([[9.0, 6.0], [6.0, 4.0]]))


@pytest.mark.parametrize(
    ("pixels", "error"),
    [
        (mx.zeros((1, 7, 8, 3)), ValueError),
        (mx.zeros((1, 8, 8, 4)), ValueError),
        (mx.zeros((1, 8, 8, 3), dtype=mx.uint8), TypeError),
        (mx.full((1, 8, 8, 3), float("nan")), ValueError),
        (mx.full((1, 8, 8, 3), 1.1), ValueError),
    ],
)
def test_encoder_rejects_bad_pixels_before_using_weights(pixels: mx.array, error: type) -> None:
    model = Krea2ImageEncoder(replace(tiny_vae_config(), num_res_blocks=1))
    with pytest.raises(error):
        model(pixels)


def test_incomplete_source_does_not_bind_weights(tmp_path: Path) -> None:
    model = Krea2ImageEncoder(replace(tiny_vae_config(), num_res_blocks=1))
    source = tmp_path / "incomplete.safetensors"
    mx.save_safetensors(str(source), {"encoder.conv_in.bias": mx.zeros((2,))})
    with pytest.raises(ValueError, match="incomplete"):
        load_krea2_encoder_weights(model, source)
    assert model.encoder.conv_in.conv.weight.size == 0
