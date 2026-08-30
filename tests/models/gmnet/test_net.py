"""GMNet architecture: key map, primitive ops, forward shapes, strict load."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from kinomlx.io.safetensors import save_weights
from kinomlx.models.gmnet.net import (
    EXPECTED_CHECKPOINT_KEYS,
    GMNet,
    adaptive_average_pool,
    checkpoint_key_map,
    load_gmnet_weights,
    per_channel_conv3x3,
    pixel_shuffle,
)


def test_checkpoint_key_map_covers_the_full_generator():
    mapping = checkpoint_key_map()
    assert len(mapping) == 63
    assert len(EXPECTED_CHECKPOINT_KEYS) == 126
    assert len(set(mapping.values())) == len(mapping)
    assert mapping["HRconv"] == "hr_conv"
    assert mapping["down1.conv.4"] == "down1.convs.2"
    assert mapping["mask_est.2"] == "mask_est.1"


def test_key_map_targets_exist_as_model_parameters():
    model = GMNet()
    flat = dict(tree_flatten(model.parameters()))
    for target in checkpoint_key_map().values():
        assert f"{target}.weight" in flat
        assert f"{target}.bias" in flat
    assert len(flat) == 126


def test_pixel_shuffle_matches_torch_channel_ordering():
    rng = np.random.default_rng(7)
    batch, height, width, out_channels, factor = 2, 3, 4, 5, 2
    data = rng.standard_normal((batch, height, width, out_channels * factor * factor)).astype(
        np.float32
    )
    shuffled = np.array(pixel_shuffle(mx.array(data), factor))
    for b in range(batch):
        for c in range(out_channels):
            for y in range(height * factor):
                for x in range(width * factor):
                    channel = c * factor * factor + (y % factor) * factor + (x % factor)
                    expected = data[b, y // factor, x // factor, channel]
                    assert shuffled[b, y, x, c] == expected


@pytest.mark.parametrize(("height", "width", "size"), [(7, 7, 3), (64, 64, 3), (5, 9, 1)])
def test_adaptive_average_pool_matches_reference_bins(height, width, size):
    rng = np.random.default_rng(11)
    data = rng.standard_normal((1, height, width, 3)).astype(np.float32)
    pooled = np.array(adaptive_average_pool(mx.array(data), size))
    assert pooled.shape == (1, size, size, 3)
    for i in range(size):
        row_start, row_end = (i * height) // size, math.ceil((i + 1) * height / size)
        for j in range(size):
            column_start, column_end = (j * width) // size, math.ceil((j + 1) * width / size)
            expected = data[:, row_start:row_end, column_start:column_end, :].mean(axis=(1, 2))
            np.testing.assert_allclose(pooled[:, i, j, :], expected, atol=1e-6)


def test_per_channel_conv3x3_matches_direct_convolution():
    rng = np.random.default_rng(13)
    batch, height, width, channels = 2, 6, 5, 4
    data = rng.standard_normal((batch, height, width, channels)).astype(np.float32)
    kernels = rng.standard_normal((batch, 3, 3, channels)).astype(np.float32)

    produced = np.array(per_channel_conv3x3(mx.array(data), mx.array(kernels)))

    padded = np.pad(data, ((0, 0), (1, 1), (1, 1), (0, 0)))
    expected = np.zeros_like(data)
    for b in range(batch):
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    window = padded[b, y : y + 3, x : x + 3, c]
                    expected[b, y, x, c] = float((window * kernels[b, :, :, c]).sum())
    np.testing.assert_allclose(produced, expected, atol=1e-5)


def test_forward_shapes_and_qmax_scaling():
    model = GMNet()
    mx.eval(model.parameters())
    image = mx.random.uniform(shape=(1, 32, 48, 3))
    thumbnail = mx.random.uniform(shape=(1, 24, 24, 3))
    gain, scaled, qmax = model(image, thumbnail)
    mx.eval(gain, scaled, qmax)
    assert tuple(gain.shape) == (1, 32, 48, 1)
    assert tuple(scaled.shape) == (1, 32, 48, 1)
    assert tuple(qmax.shape) == (1, 1, 1, 1)
    np.testing.assert_allclose(
        np.array(scaled), np.array(qmax) * np.array(gain), rtol=1e-5, atol=1e-6
    )


def test_forward_rounds_odd_extents_up_to_even():
    model = GMNet()
    mx.eval(model.parameters())
    gain, _scaled, _qmax = model(
        mx.random.uniform(shape=(1, 31, 45, 3)),
        mx.random.uniform(shape=(1, 16, 16, 3)),
    )
    assert tuple(gain.shape) == (1, 32, 46, 1)


def _upstream_layout_state_dict() -> dict[str, mx.array]:
    """Random tensors under upstream names/OIHW layout, shaped from the model."""
    flat = dict(tree_flatten(GMNet().parameters()))
    inverse = {target: source for source, target in checkpoint_key_map().items()}
    state: dict[str, mx.array] = {}
    for name, value in flat.items():
        stem, _, leaf = name.rpartition(".")
        tensor = mx.random.normal(shape=value.shape).astype(mx.float32)
        if leaf == "weight":
            tensor = tensor.transpose(0, 3, 1, 2)  # NHWC weights back to OIHW
        state[f"{inverse[stem]}.{leaf}"] = tensor
    return state


def test_load_gmnet_weights_round_trips_layout(tmp_path):
    state = _upstream_layout_state_dict()
    path = tmp_path / "gmnet_synthetic_random.safetensors"
    save_weights(path, state, {"variant": "synthetic"})

    model, metadata = load_gmnet_weights(path)
    assert metadata == {"variant": "synthetic"}
    flat = dict(tree_flatten(model.parameters()))
    probe = np.array(state["down1.conv.0.weight"])  # OIHW
    loaded = np.array(flat["down1.convs.0.weight"])  # OHWI
    np.testing.assert_array_equal(loaded, probe.transpose(0, 2, 3, 1))


def test_load_gmnet_weights_rejects_wrong_key_sets(tmp_path):
    path = tmp_path / "wrong.safetensors"
    save_weights(path, {"unrelated": mx.zeros((1,))})
    with pytest.raises(ValueError, match="not a GMNet generator checkpoint"):
        load_gmnet_weights(path)


def test_load_gmnet_weights_rejects_non_floating_tensors(tmp_path):
    state = _upstream_layout_state_dict()
    state["conv_last.bias"] = state["conv_last.bias"].astype(mx.int32)
    path = tmp_path / "gmnet_integer.safetensors"
    save_weights(path, state)

    with pytest.raises(ValueError, match="generator weights must be floating-point"):
        load_gmnet_weights(path)
