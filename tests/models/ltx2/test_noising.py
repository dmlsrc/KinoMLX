"""Behavioral tests for LTX-2 latent noise injection."""

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.noising import GaussianNoiser
from kinomlx.models.ltx2.types import LatentState


def _state(
    latent: mx.array,
    mask: mx.array,
    *,
    clean: mx.array | None = None,
) -> LatentState:
    """Construct a LatentState with placeholder positions."""
    return LatentState(
        latent=latent,
        denoise_mask=mask,
        positions=mx.zeros_like(latent),
        clean_latent=latent if clean is None else clean,
    )


# ---------------------------------------------------------------------------
# Mask behavior - full-noise / no-noise extremes
# ---------------------------------------------------------------------------


def test_full_mask_replaces_latent_with_scaled_noise() -> None:
    """mask=1 everywhere -> output is pure scaled noise (input fully overwritten)."""
    latent = mx.ones((1, 4, 3), dtype=mx.float32) * 99.0
    mask = mx.ones((1, 4, 1), dtype=mx.float32)
    noiser = GaussianNoiser(seed=0)
    out = noiser(_state(latent, mask), scale=1.0).latent
    # The 99.0 sentinel can't survive a full-noise blend at scale=1.
    assert not mx.allclose(out, mx.array(99.0), rtol=1e-5, atol=1e-8).item()


def test_zero_mask_preserves_latent_exactly() -> None:
    """mask=0 everywhere -> output equals input (noise contribution zero)."""
    latent = mx.arange(12, dtype=mx.float32).reshape(1, 4, 3) * 0.5
    mask = mx.zeros((1, 4, 1), dtype=mx.float32)
    noiser = GaussianNoiser(seed=0)
    out = noiser(_state(latent, mask), scale=1.0).latent
    assert mx.allclose(out, latent, rtol=1e-5, atol=1e-8).item()


# ---------------------------------------------------------------------------
# Determinism - seeded noiser is reproducible, advances between calls
# ---------------------------------------------------------------------------


def test_seeded_noisers_with_same_seed_match() -> None:
    """Two fresh noisers, same seed -> same output sequence."""
    latent = mx.ones((1, 4, 3), dtype=mx.float32)
    mask = mx.ones((1, 4, 1), dtype=mx.float32)
    a = GaussianNoiser(seed=42)(_state(latent, mask)).latent
    b = GaussianNoiser(seed=42)(_state(latent, mask)).latent
    assert mx.allclose(a, b, rtol=1e-5, atol=1e-8).item()


def test_seeded_noiser_advances_between_calls() -> None:
    """Consecutive calls on one seeded noiser draw distinct samples."""
    latent = mx.ones((1, 4, 3), dtype=mx.float32)
    mask = mx.ones((1, 4, 1), dtype=mx.float32)
    noiser = GaussianNoiser(seed=42)
    first = noiser(_state(latent, mask)).latent
    second = noiser(_state(latent, mask)).latent
    assert not mx.allclose(first, second, rtol=1e-5, atol=1e-8).item()


def test_advance_matches_discarding_prior_modality_draws_exactly() -> None:
    """Stage restart can recover the same later key without materializing old noise."""
    latent = mx.ones((1, 4, 3), dtype=mx.float32)
    mask = mx.ones((1, 4, 1), dtype=mx.float32)
    state = _state(latent, mask)
    uninterrupted = GaussianNoiser(seed=42)
    uninterrupted(state)
    uninterrupted(state)
    expected = uninterrupted(state).latent

    restarted = GaussianNoiser(seed=42)
    restarted.advance(tuple(state.latent.shape))
    restarted.advance(tuple(state.latent.shape))
    actual = restarted(state).latent

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("shape", [(-1,), (True,), (1.5,)])
def test_advance_rejects_invalid_shapes(shape) -> None:
    with pytest.raises(ValueError, match="shape dimensions"):
        GaussianNoiser(seed=42).advance(shape)


# ---------------------------------------------------------------------------
# Mask shape handling - 2D expanded vs 3D direct
# ---------------------------------------------------------------------------


def test_2d_mask_is_expanded_to_broadcast() -> None:
    """A (B, T) mask is auto-expanded to (B, T, 1) so it broadcasts."""
    latent = mx.ones((1, 4, 3), dtype=mx.float32)
    mask_2d = mx.ones((1, 4), dtype=mx.float32)
    mask_3d = mx.ones((1, 4, 1), dtype=mx.float32)
    out_2d = GaussianNoiser(seed=7)(_state(latent, mask_2d)).latent
    out_3d = GaussianNoiser(seed=7)(_state(latent, mask_3d)).latent
    assert mx.allclose(out_2d, out_3d, rtol=1e-5, atol=1e-8).item()


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------


def test_dtype_preserved_through_blend() -> None:
    """bf16 input -> bf16 output despite internal float promotion."""
    latent = mx.ones((1, 4, 3), dtype=mx.bfloat16)
    mask = mx.ones((1, 4, 1), dtype=mx.bfloat16)
    out = GaussianNoiser(seed=0)(_state(latent, mask)).latent
    assert out.dtype == mx.bfloat16


# ---------------------------------------------------------------------------
# LTX-2 conditioning parity - clean_latent differs from generative latent
# ---------------------------------------------------------------------------


def test_fractional_mask_matches_two_lerp_reference() -> None:
    """Stage-2 soft conditioning follows clean -> noised-generative compositing."""
    latent = mx.array([[[1.0, -2.0], [0.0, 0.0]]], dtype=mx.float32)
    clean = mx.array([[[1.0, -2.0], [8.0, -4.0]]], dtype=mx.float32)
    mask = mx.array([[[1.0], [0.05]]], dtype=mx.float32)
    scale = 0.4

    root_key = mx.random.key(7)
    _next_key, sample_key = mx.random.split(root_key)
    noise = mx.random.normal(latent.shape, dtype=latent.dtype, key=sample_key)
    noised = latent + scale * (noise - latent)
    expected = clean + mask * (noised - clean)

    out = GaussianNoiser(seed=7)(_state(latent, mask, clean=clean), scale=scale).latent
    assert mx.allclose(out, expected, atol=1e-5, rtol=1e-7).item()


def test_zero_mask_preserves_clean_latent_not_generative_latent() -> None:
    """A fully conditioned token must come from clean_latent."""
    latent = mx.zeros((1, 2, 3), dtype=mx.float32)
    clean = mx.arange(6, dtype=mx.float32).reshape(1, 2, 3)
    mask = mx.zeros((1, 2, 1), dtype=mx.float32)

    out = GaussianNoiser(seed=3)(_state(latent, mask, clean=clean), scale=0.5).latent
    assert mx.array_equal(out, clean).item()


def test_full_mask_ignores_clean_latent() -> None:
    """The generative region follows latent -> noise regardless of clean values."""
    latent = mx.ones((1, 2, 3), dtype=mx.float32)
    clean = mx.full((1, 2, 3), 99.0, dtype=mx.float32)
    mask = mx.ones((1, 2, 1), dtype=mx.float32)
    scale = 0.3

    root_key = mx.random.key(11)
    _next_key, sample_key = mx.random.split(root_key)
    noise = mx.random.normal(latent.shape, dtype=latent.dtype, key=sample_key)
    expected = latent + scale * (noise - latent)

    out = GaussianNoiser(seed=11)(_state(latent, mask, clean=clean), scale=scale).latent
    assert mx.allclose(out, expected, atol=1e-5, rtol=1e-7).item()
