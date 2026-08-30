"""Distilled joint denoise-loop contracts."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.denoise import denoise_loop, denoise_step
from kinomlx.models.ltx2.state import post_process_latent
from kinomlx.reporting import RecordingReporter
from kinomlx.samplers.steps import euler_ancestral_rf_step
from kinomlx.types import LatentState


def _state(*, mixed: bool = False) -> LatentState:
    mask = mx.array([[[0.5], [1.0]]], dtype=mx.float32) if mixed else mx.ones((1, 2, 1))
    return LatentState(
        latent=mx.ones((1, 2, 3), dtype=mx.float32),
        denoise_mask=mask,
        positions=mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        clean_latent=mx.full((1, 2, 3), 4.0),
        uniform_mask=not mixed,
    )


class _ZeroX0:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, video, audio=None):
        self.calls.append((video, audio))
        return (
            mx.zeros_like(video.latent),
            None if audio is None else mx.zeros_like(audio.latent),
        )


def test_uniform_state_uses_batch_scalar_timestep() -> None:
    model = _ZeroX0()
    result, _ = denoise_step(
        _state(),
        None,
        sigma=1.0,
        sigma_next=0.0,
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
    )
    assert model.calls[0][0].timesteps.shape == (1,)
    assert mx.array_equal(result.latent, mx.zeros_like(result.latent)).item()


def test_mixed_mask_uses_per_token_timestep_and_preserves_clean_fraction() -> None:
    model = _ZeroX0()
    result, _ = denoise_step(
        _state(mixed=True),
        None,
        sigma=1.0,
        sigma_next=0.0,
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
    )
    assert model.calls[0][0].timesteps.shape == (1, 2)
    assert mx.array_equal(
        model.calls[0][0].timesteps,
        mx.array([[0.5, 1.0]]),
    ).item()
    assert mx.array_equal(result.latent[:, 0], mx.full((1, 3), 2.0)).item()
    assert mx.array_equal(result.latent[:, 1], mx.zeros((1, 3))).item()


def test_ancestral_renoise_reapplies_the_conditioning_mask() -> None:
    state = _state(mixed=True)
    noise = mx.ones_like(state.latent)
    model = _ZeroX0()

    result, _ = denoise_step(
        state,
        None,
        sigma=1.0,
        sigma_next=0.5,
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
        step_kind="ancestral-rf",
        video_noise=noise,
    )

    prediction = post_process_latent(mx.zeros_like(state.latent), state)
    raw_step = euler_ancestral_rf_step(
        state.latent,
        prediction,
        noise,
        sigma=1.0,
        sigma_next=0.5,
    )
    expected = post_process_latent(raw_step, state)
    assert mx.allclose(result.latent, expected).item()
    assert not mx.allclose(result.latent[:, 0], raw_step[:, 0]).item()
    assert mx.allclose(result.latent[:, 1], raw_step[:, 1]).item()


def test_disabled_video_remains_cross_context_while_audio_updates() -> None:
    model = _ZeroX0()
    video = _state()
    audio = _state()
    video_result, audio_result = denoise_step(
        video,
        audio,
        sigma=1.0,
        sigma_next=0.0,
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
        audio_context=mx.zeros((1, 1, 4)),
        video_enabled=False,
    )
    assert model.calls[0][0].enabled is False
    assert model.calls[0][1].enabled is True
    assert video_result is video
    assert audio_result is not None
    assert mx.array_equal(audio_result.latent, mx.zeros_like(audio_result.latent)).item()


def test_loop_reports_every_step_and_invokes_callback() -> None:
    model = _ZeroX0()
    reporter = RecordingReporter()
    callbacks = []
    denoise_loop(
        _state(),
        None,
        (1.0, 0.5, 0.0),
        transformer=model,
        video_context=mx.zeros((1, 1, 4)),
        reporter=reporter,
        phase="stage",
        callback=lambda step, total: callbacks.append((step, total)),
    )
    assert len(model.calls) == 2
    assert callbacks == [(1, 2), (2, 2)]
    assert reporter.events == [
        ("start", "stage", {"total": 2, "unit": "step"}),
        ("advance", "stage", {"advance": 1.0}),
        ("advance", "stage", {"advance": 1.0}),
        ("end", "stage", {}),
    ]


@pytest.mark.parametrize(
    "sigmas",
    [
        (0.5, 0.0, -0.5),
        (0.5, -0.1, -0.5),
    ],
)
def test_loop_rejects_nonpositive_current_sigmas(sigmas: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="current denoise sigma must be positive"):
        denoise_loop(
            _state(),
            None,
            sigmas,
            transformer=_ZeroX0(),
            video_context=mx.zeros((1, 1, 4)),
        )
