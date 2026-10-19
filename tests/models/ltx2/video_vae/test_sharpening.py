"""Context sharpening: the spatial unsharp-mask kernel, its fallback, and the settings record."""

import math

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.kernels.context_sharpening import gaussian_taps, spatial_unsharp_mask
from kinomlx.models.ltx2.video_vae.sharpening import ContextSharpening
from kinomlx.models.ltx2.video_vae.tiling import decode_streaming

_TOLERANCE = {
    mx.float32: (1e-5, 1e-5),
    mx.float16: (2e-3, 1e-3),
    mx.bfloat16: (1.6e-2, 8e-3),
}


def _expected(value: mx.array, sigma: float, amount: float) -> mx.array:
    """The definition, spelled out: a 2D Gaussian over clamped neighbors of each frame."""
    taps = gaussian_taps(sigma)
    radius = (len(taps) - 1) // 2
    source = value.astype(mx.float32)
    height, width = int(value.shape[2]), int(value.shape[3])
    blurred = mx.zeros_like(source)
    for dy in range(-radius, radius + 1):
        rows = mx.clip(mx.arange(height) + dy, 0, height - 1)
        shifted_rows = mx.take(source, rows, axis=2)
        for dx in range(-radius, radius + 1):
            columns = mx.clip(mx.arange(width) + dx, 0, width - 1)
            weight = taps[dy + radius] * taps[dx + radius]
            blurred = blurred + weight * mx.take(shifted_rows, columns, axis=3)
    return (source + amount * (source - blurred)).astype(value.dtype)


def _assert_close(got: mx.array, want: mx.array) -> None:
    rtol, atol = _TOLERANCE[got.dtype]
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert mx.allclose(got.astype(mx.float32), want.astype(mx.float32), rtol=rtol, atol=atol).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    ("shape", "sigma"),
    [
        ((1, 3, 9, 11, 8), 0.5),
        ((2, 2, 2, 3, 4), 1.0),
        ((1, 2, 13, 7, 12), 2.0),
    ],
)
def test_unsharp_mask_matches_its_definition(dtype, shape, sigma) -> None:
    mx.random.seed(11)
    value = (mx.random.normal(shape) * 0.5).astype(dtype)

    _assert_close(
        spatial_unsharp_mask(value, sigma=sigma, amount=2.0), _expected(value, sigma, 2.0)
    )


def test_channels_that_are_not_a_multiple_of_four_take_the_fallback() -> None:
    mx.random.seed(12)
    value = (mx.random.normal((1, 2, 5, 6, 6)) * 0.5).astype(mx.float16)

    _assert_close(spatial_unsharp_mask(value, sigma=0.5, amount=3.0), _expected(value, 0.5, 3.0))


def test_cpu_fallback_matches_the_kernel() -> None:
    mx.random.seed(13)
    value = (mx.random.normal((1, 3, 10, 9, 8)) * 0.5).astype(mx.float32)
    kernel = spatial_unsharp_mask(value, sigma=1.0, amount=1.5)
    mx.eval(kernel)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        fallback = spatial_unsharp_mask(value, sigma=1.0, amount=1.5)
        mx.eval(fallback)
    finally:
        mx.set_default_device(previous)

    _assert_close(fallback, kernel)


def test_a_flat_frame_is_left_unchanged() -> None:
    value = mx.full((1, 2, 6, 7, 4), 0.375, dtype=mx.float16)

    assert mx.array_equal(spatial_unsharp_mask(value, sigma=1.0, amount=4.0), value).item()


def test_gaussian_taps_are_normalized_and_reach_three_sigma() -> None:
    for sigma in (0.25, 0.5, 1.0, 2.0):
        taps = gaussian_taps(sigma)
        assert len(taps) == 2 * max(1, math.ceil(3 * sigma)) + 1
        assert math.isclose(math.fsum(taps), 1.0)
        assert taps == tuple(reversed(taps))
    with pytest.raises(ValueError, match="sigma"):
        gaussian_taps(0.0)


def test_settings_give_the_radius_in_output_pixels() -> None:
    mx.random.seed(14)
    context = (mx.random.normal((1, 2, 7, 9, 8)) * 0.5).astype(mx.float32)
    settings = ContextSharpening(amount=2, radius=2)

    assert settings.amount == 2.0
    assert settings.to_dict() == {"amount": 2.0, "radius_pixels": 2.0}
    assert mx.array_equal(
        settings.apply(context, patch_size=4),
        spatial_unsharp_mask(context, sigma=0.5, amount=2.0),
    ).item()


@pytest.mark.parametrize(
    ("amount", "radius", "message"),
    [
        (0.0, 2.0, "amount must be greater than 0"),
        (10.5, 2.0, "amount must be greater than 0"),
        (2.0, 0.0, "radius must be greater than 0"),
        (2.0, 8.5, "radius must be greater than 0"),
        (math.inf, 2.0, "amount must be a finite number"),
        (True, 2.0, "amount must be a finite number"),
    ],
)
def test_settings_reject_values_outside_their_range(amount, radius, message) -> None:
    with pytest.raises(ValueError, match=message):
        ContextSharpening(amount=amount, radius=radius)


def test_a_decoder_without_its_own_streaming_refuses_context_sharpening() -> None:
    def conv_decoder(latent, **_kwargs):
        return latent

    chunks = decode_streaming(
        mx.zeros((1, 8, 2, 3, 3)),
        conv_decoder,
        context_sharpening=ContextSharpening(amount=1.0),
    )
    with pytest.raises(ValueError, match="only to the diffusion video VAE"):
        next(chunks)
