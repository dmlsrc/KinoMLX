"""Behavioral tests for ``kinomlx.samplers.steps``."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.samplers.steps import euler_step


def test_euler_step_moves_sample_toward_denoised() -> None:
    """One step at sigma -> sigma_next pulls the sample partway toward denoised."""
    sample = mx.array([10.0, 10.0, 10.0], dtype=mx.float32)
    denoised = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    out = euler_step(sample, denoised, sigma=1.0, sigma_next=0.5)
    # Closed form: denoised + (0.5/1.0) * (sample - denoised) = 5.
    assert np.allclose(np.asarray(out), 5.0)


def test_euler_step_to_zero_sigma_lands_at_denoised() -> None:
    """sigma_next=0 collapses the sample to the denoised prediction."""
    sample = mx.array([3.0, -2.0, 7.5], dtype=mx.float32)
    denoised = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)
    out = euler_step(sample, denoised, sigma=0.5, sigma_next=0.0)
    assert np.allclose(np.asarray(out), np.asarray(denoised))


def test_euler_step_preserves_input_dtype() -> None:
    """sample.dtype round-trips even though math runs in fp32."""
    sample = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
    denoised = mx.array([0.0, 0.0, 0.0], dtype=mx.bfloat16)
    out = euler_step(sample, denoised, sigma=1.0, sigma_next=0.5)
    assert out.dtype == mx.bfloat16


def test_euler_step_fp32_input_skips_cast() -> None:
    """Fp32 input -> fp32 output (no spurious cast)."""
    sample = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    denoised = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    out = euler_step(sample, denoised, sigma=1.0, sigma_next=0.5)
    assert out.dtype == mx.float32


def test_euler_step_identity_when_sigma_equals_sigma_next() -> None:
    """sigma_next/sigma == 1 -> output equals sample (no progress made)."""
    sample = mx.array([5.0, 7.0, -3.0], dtype=mx.float32)
    denoised = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
    out = euler_step(sample, denoised, sigma=0.5, sigma_next=0.5)
    assert np.allclose(np.asarray(out), np.asarray(sample))


@pytest.mark.parametrize("sigma", [0.0, -0.5])
def test_euler_step_rejects_nonpositive_current_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="current sigma must be positive"):
        euler_step(
            mx.array([1.0]),
            mx.array([0.0]),
            sigma=sigma,
            sigma_next=-1.0,
        )
