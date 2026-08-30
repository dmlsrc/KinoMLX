"""Deterministic and RF-ancestral Euler diffusion steps."""

from __future__ import annotations

import math

import mlx.core as mx


def euler_step(
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """One first-order Euler diffusion step.

    Computes ``denoised + (sigma_next / sigma) * (sample - denoised)``
    in fp32 then casts back to ``sample.dtype``.

    Algebraically equivalent to the velocity form
    ``sample + velocity * (sigma_next - sigma)`` where
    ``velocity = (sample - denoised) / sigma``, but expressed as a
    single fused lerp.  No measured speed difference under MLX lazy
    graph fusion; reads more directly.

    Args:
        sample: Current noisy sample at ``sigma``.
        denoised: Model's denoised prediction.
        sigma: Current noise level.
        sigma_next: Target (lower) noise level.

    Returns:
        Updated sample at ``sigma_next``, in ``sample.dtype``.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"current sigma must be positive and finite, got {sigma!r}")
    sample_dtype = sample.dtype
    sample_f32 = sample if sample_dtype == mx.float32 else sample.astype(mx.float32)
    denoised_f32 = denoised if denoised.dtype == mx.float32 else denoised.astype(mx.float32)
    result = denoised_f32 + (sigma_next / sigma) * (sample_f32 - denoised_f32)
    if sample_dtype == mx.float32:
        return result
    return result.astype(sample_dtype)


def euler_ancestral_rf_step(
    sample: mx.array,
    denoised: mx.array,
    noise: mx.array | None,
    *,
    sigma: float,
    sigma_next: float,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> mx.array:
    """Advance one rectified-flow sample with Euler ancestral noise.

    This is the MLX form of the Apache Diffusers LTX RF scheduler's
    explicit-sigma step. The model wrapper has already converted velocity to
    ``denoised`` (x0), so this function owns only the deterministic downstep
    and stochastic renoise terms. The terminal transition lands directly on
    x0 and must not receive a noise tensor.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"current sigma must be positive and finite, got {sigma!r}")
    if not math.isfinite(sigma_next) or sigma_next < 0.0:
        raise ValueError(f"next sigma must be non-negative and finite, got {sigma_next!r}")
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError(f"eta must be non-negative and finite, got {eta!r}")
    if not math.isfinite(s_noise) or s_noise < 0.0:
        raise ValueError(f"s_noise must be non-negative and finite, got {s_noise!r}")

    sample_dtype = sample.dtype
    sample_f32 = sample if sample_dtype == mx.float32 else sample.astype(mx.float32)
    denoised_f32 = denoised if denoised.dtype == mx.float32 else denoised.astype(mx.float32)
    if tuple(denoised.shape) != tuple(sample.shape):
        raise ValueError(
            f"denoised shape {tuple(denoised.shape)} does not match sample {tuple(sample.shape)}"
        )

    if sigma_next == 0.0:
        if noise is not None:
            raise ValueError("the terminal ancestral transition must not receive noise")
        return denoised_f32 if sample_dtype == mx.float32 else denoised_f32.astype(sample_dtype)
    if noise is None:
        raise ValueError("a non-terminal ancestral transition requires noise")
    if tuple(noise.shape) != tuple(sample.shape):
        raise ValueError(
            f"noise shape {tuple(noise.shape)} does not match sample {tuple(sample.shape)}"
        )
    if noise.dtype != sample_dtype:
        raise ValueError(f"noise dtype {noise.dtype} does not match sample dtype {sample_dtype}")
    if not mx.all(mx.isfinite(noise)).item():
        raise ValueError("ancestral noise must be finite")

    sigma_f32 = mx.array(sigma, dtype=mx.float32)
    sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
    downstep_ratio = 1.0 + (sigma_next_f32 / sigma_f32 - 1.0) * eta
    sigma_down = sigma_next_f32 * downstep_ratio
    alpha_next = 1.0 - sigma_next_f32
    alpha_down = 1.0 - sigma_down
    sigma_ratio = sigma_down / sigma_f32
    deterministic = sigma_ratio * sample_f32 + (1.0 - sigma_ratio) * denoised_f32
    renoise_squared = sigma_next_f32**2 - sigma_down**2 * alpha_next**2 / (alpha_down**2 + 1e-12)
    renoise = mx.sqrt(mx.maximum(renoise_squared, mx.array(0.0, dtype=mx.float32)))
    result = (
        alpha_next / (alpha_down + 1e-12) * deterministic
        + noise.astype(mx.float32) * renoise * s_noise
    )
    if sample_dtype == mx.float32:
        return result
    return result.astype(sample_dtype)


__all__ = ["euler_ancestral_rf_step", "euler_step"]
