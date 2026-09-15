"""Native Qwen Image VAE math, geometry, dtype, and tiling contracts."""

import math
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.krea2.vae import (
    Krea2DecodedRGB,
    Krea2ImageDecoder,
    Krea2RMSNorm,
    Krea2VAETilingConfig,
)
from kinomlx.models.krea2.vae.tiling import blend_horizontal, blend_vertical
from tests._reporting import RecordingReporter

from ._vae_fixtures import loaded_tiny_vae, tiny_vae_config


def test_latent_denormalization_precedes_frame_insertion_and_decoder_math(
    tmp_path: Path,
) -> None:
    config = tiny_vae_config(latent_mean=(10.0, 20.0), latent_std=(2.0, 3.0))
    decoder = loaded_tiny_vae(tmp_path, config=config)
    boundaries: dict[str, mx.array] = {}

    result = decoder(
        mx.ones((1, 2, 3, 5), dtype=mx.float32),
        observer=lambda name, value: boundaries.__setitem__(name, value),
    )

    assert tuple(boundaries["normalized_input"].shape) == (1, 2, 1, 3, 5)
    assert mx.array_equal(
        boundaries["normalized_input"][:, :, 0, 0, 0],
        mx.array([[12.0, 23.0]], dtype=mx.float32),
    ).item()
    assert tuple(result.pixels.shape) == (1, 6, 10, 3)


def test_decoder_preserves_unclamped_floating_rgb_until_output_station(
    tmp_path: Path,
) -> None:
    decoder = loaded_tiny_vae(
        tmp_path,
        overrides={"decoder.conv_out.bias": mx.array([2.0, 0.0, -2.0])},
    )

    result = decoder(mx.zeros((1, 2, 1, 1), dtype=mx.float32))

    assert isinstance(result, Krea2DecodedRGB)
    assert mx.array_equal(result.pixels[0, 0, 0], mx.array([2.0, 0.0, -2.0])).item()


def test_channel_order_and_odd_latent_geometry_are_preserved(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(
        tmp_path,
        overrides={"decoder.conv_out.bias": mx.array([0.125, 0.25, 0.5])},
    )

    result = decoder(mx.zeros((2, 2, 3, 5), dtype=mx.float32))

    assert tuple(result.pixels.shape) == (2, 6, 10, 3)
    assert mx.array_equal(
        result.pixels[1, 5, 9],
        mx.array([0.125, 0.25, 0.5]),
    ).item()


def test_upscale2x_decoder_pixel_shuffles_packed_rgb_channels(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(
        tmp_path,
        output_scale=2,
        overrides={"decoder.conv_out.bias": mx.arange(12, dtype=mx.float32)},
    )

    result = decoder(mx.zeros((1, 2, 1, 1), dtype=mx.float32))

    assert tuple(result.pixels.shape) == (1, 4, 4, 3)
    expected = mx.array(
        [
            [[0.0, 4.0, 8.0], [1.0, 5.0, 9.0]],
            [[2.0, 6.0, 10.0], [3.0, 7.0, 11.0]],
        ],
        dtype=mx.float32,
    )
    assert mx.array_equal(result.pixels[0, :2, :2], expected).item()


def test_qwen_rms_norm_matches_channel_l2_definition() -> None:
    norm = Krea2RMSNorm(2)
    norm.gamma = mx.ones((2,), dtype=mx.float32)
    result = norm(mx.array([[[[3.0, 4.0]]]], dtype=mx.float32))
    expected = mx.array([3.0, 4.0], dtype=mx.float32) * (math.sqrt(2.0) / 5.0)
    assert mx.allclose(result[0, 0, 0], expected, atol=1e-7).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_decoder_preserves_selected_hot_dtype(tmp_path: Path, dtype: mx.Dtype) -> None:
    decoder = loaded_tiny_vae(tmp_path, compute_dtype=dtype)
    result = decoder(mx.zeros((1, 2, 2, 2), dtype=mx.float32))
    assert result.pixels.dtype == dtype


def test_full_and_tiled_zero_graph_are_seam_free_and_pixel_exact(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(
        tmp_path,
        overrides={"decoder.conv_out.bias": mx.array([0.1, 0.2, 0.3])},
    )
    latent = mx.zeros((1, 2, 5, 7), dtype=mx.float32)

    full = decoder(latent).pixels
    tiled = decoder(
        latent,
        tiling=Krea2VAETilingConfig(tile_sample_size=4, tile_sample_stride=2),
    ).pixels

    assert tuple(tiled.shape) == tuple(full.shape)
    assert mx.array_equal(tiled, full).item()


def test_upscale2x_full_and_tiled_decode_share_final_geometry(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(tmp_path, output_scale=2)
    latent = mx.zeros((1, 2, 5, 7), dtype=mx.float32)

    full = decoder(latent).pixels
    tiled = decoder(
        latent,
        tiling=Krea2VAETilingConfig(tile_sample_size=4, tile_sample_stride=2),
    ).pixels

    assert tuple(full.shape) == (1, 20, 28, 3)
    assert tuple(tiled.shape) == tuple(full.shape)
    assert mx.array_equal(tiled, full).item()


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize(
    ("dtype", "extent", "expected_row"),
    [
        (mx.float16, 3, [0.5107421875, 0.52099609375, 0.53125]),
        (mx.bfloat16, 3, [0.5078125, 0.51953125, 0.53125]),
        (mx.float16, 48, [1.3505859375, 1.3798828125, 1.41015625]),
        (mx.bfloat16, 48, [1.3515625, 1.375, 1.40625]),
        (mx.float16, 64, [1.365234375, 1.39453125, 1.423828125]),
        (mx.bfloat16, 64, [1.359375, 1.390625, 1.421875]),
        (mx.float32, 3, [0.5104166269302368, 0.5208333730697632, 0.53125]),
        (mx.float32, 48, [1.3502603769302368, 1.3802083730697632, 1.4101563692092896]),
        (mx.float32, 64, [1.3642578125, 1.39453125, 1.4248046875]),
    ],
)
def test_nonconstant_overlap_matches_reference_scalar_products(
    dtype, extent, expected_row, horizontal
) -> None:
    previous = (mx.arange((extent + 4) * 9).reshape(1, extent + 4, 3, 3) / 32).astype(dtype)
    current = -(previous + 1).astype(dtype)
    if horizontal:
        actual = blend_horizontal(
            previous.transpose(0, 2, 1, 3), current.transpose(0, 2, 1, 3), extent
        ).transpose(0, 2, 1, 3)
    else:
        actual = blend_vertical(previous, current, extent)
    # Literal reference outputs at a nontrivial overlap position, plus exact
    # first-edge selection and preservation of the unblended remainder.
    assert actual.dtype == dtype
    assert mx.array_equal(actual[0, 1, 0], mx.array(expected_row, dtype)).item()
    assert mx.array_equal(actual[:, 0], previous[:, -extent]).item()
    assert mx.array_equal(actual[:, extent:], current[:, extent:]).item()


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("extent", [0, -1, 100])
def test_blend_clamps_overlap_to_available_pixels(horizontal, extent) -> None:
    previous = mx.arange(18, dtype=mx.float32).reshape(1, 2, 3, 3)
    current = mx.full((1, 2, 3, 3), -1.0)
    blend = blend_horizontal if horizontal else blend_vertical
    actual = blend(previous, current, extent)
    if extent <= 0:
        assert actual is current
    else:
        assert actual.shape == current.shape
        assert mx.array_equal(
            actual[:, :, 0] if horizontal else actual[:, 0],
            previous[:, :, 0] if horizontal else previous[:, 0],
        ).item()


def test_decoder_rejects_nonfinite_product(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(
        tmp_path,
        overrides={"decoder.conv_out.bias": mx.array([float("inf"), 0.0, 0.0])},
    )
    with pytest.raises(ValueError, match="nonfinite RGB"):
        decoder(mx.zeros((1, 2, 1, 1), dtype=mx.float32))


def test_untiled_decode_reports_one_indeterminate_phase(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(tmp_path)
    reporter = RecordingReporter()

    decoder(mx.zeros((1, 2, 2, 2), dtype=mx.float32), reporter=reporter)

    assert reporter.events == [
        ("start", "decode VAE", {"total": None, "unit": "it"}),
        ("end", "decode VAE", {}),
    ]


@pytest.mark.parametrize(
    ("latent", "message"),
    [
        (mx.zeros((1, 3, 2, 2)), "2 channels"),
        (mx.zeros((1, 2, 0, 2)), "positive"),
        (mx.zeros((1, 2, 2, 2), dtype=mx.int32), "must be floating"),
    ],
)
def test_decoder_rejects_invalid_latent_contract(latent: mx.array, message: str) -> None:
    decoder = Krea2ImageDecoder(tiny_vae_config(), compute_dtype=mx.float32)
    with pytest.raises((TypeError, ValueError), match=message):
        decoder(latent)


def test_tiling_requires_compression_aligned_geometry(tmp_path: Path) -> None:
    decoder = loaded_tiny_vae(tmp_path)
    with pytest.raises(ValueError, match="align to spatial compression"):
        decoder(
            mx.zeros((1, 2, 3, 3), dtype=mx.float32),
            tiling=Krea2VAETilingConfig(tile_sample_size=5, tile_sample_stride=3),
        )
