"""Phase F injected-noise parity for the LTX-2.5 distilled sampler."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import kinomlx.samplers.steps as sampler_steps
from kinomlx.models.ltx2.denoise import denoise_loop
from kinomlx.models.ltx2.sigmas import (
    DISTILLED_STAGE_1_SIGMAS,
    DISTILLED_STAGE_2_SIGMAS,
)
from kinomlx.types import LatentState

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "ltx25_ancestral_parity.json"


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _array(value: object) -> mx.array:
    return mx.array(value, dtype=mx.float32)


def _state(value: object) -> LatentState:
    latent = _array(value)
    return LatentState(
        latent=latent,
        denoise_mask=mx.ones((*latent.shape[:-1], 1), dtype=mx.float32),
        positions=mx.zeros((1, 3, latent.shape[1], 2), dtype=mx.float32),
        clean_latent=mx.zeros_like(latent),
    )


class _FixtureTransformer:
    def __init__(self, transitions: list[dict[str, object]]) -> None:
        self._transitions = transitions
        self.calls = 0

    def __call__(self, video, audio=None):
        transition = self._transitions[self.calls]
        np.testing.assert_allclose(
            np.asarray(video.latent),
            np.asarray(_array(transition["video_input"])),
            atol=2e-6,
            rtol=0.0,
        )
        if audio is None:
            raise AssertionError("the joint fixture requires an audio state")
        np.testing.assert_allclose(
            np.asarray(audio.latent),
            np.asarray(_array(transition["audio_input"])),
            atol=2e-6,
            rtol=0.0,
        )
        self.calls += 1
        return (
            _array(transition["video_denoised"]),
            _array(transition["audio_denoised"]),
        )


class _FixtureNoise:
    def __init__(self, transitions: list[dict[str, object]]) -> None:
        self._expected = [
            (transition, modality)
            for transition in transitions[:-1]
            for modality in ("video", "audio")
        ]
        self.calls: list[tuple[int, int, str]] = []

    def __call__(
        self,
        *,
        stage: int,
        transition: int,
        modality: str,
        shape: tuple[int, ...],
        dtype: mx.Dtype,
    ) -> mx.array:
        if not self._expected:
            raise AssertionError("the sampler requested an extra noise tensor")
        expected, expected_modality = self._expected.pop(0)
        assert stage == 1
        assert transition == expected["transition"]
        assert modality == expected_modality
        value = _array(expected[f"{modality}_noise"])
        assert tuple(value.shape) == shape
        assert value.dtype == dtype
        self.calls.append((stage, transition, modality))
        return value

    def assert_exhausted(self) -> None:
        assert self._expected == []


class _RejectNoise:
    def __call__(self, **_kwargs) -> mx.array:
        raise AssertionError("deterministic stage 2 must not request noise")


def test_fixture_records_the_pinned_scheduler_and_literal_product_schedules() -> None:
    fixture = _fixture()
    assert fixture["diffusers_commit"] == "2f7e0154a9db246e95c9ede43edba7db5b130805"
    assert (
        fixture["scheduler_source_sha256"]
        == "bdb4a55614c727e490563de1872cf5cd3b079761a587ec64c02c5b5bd73e32d6"
    )
    np.testing.assert_array_equal(
        np.asarray(fixture["stage_1_sigmas"], dtype=np.float32),
        np.asarray(DISTILLED_STAGE_1_SIGMAS, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(fixture["stage_2_sigmas"], dtype=np.float32),
        np.asarray(DISTILLED_STAGE_2_SIGMAS, dtype=np.float32),
    )
    assert fixture["ancestral_seed"] == fixture["request_seed"] + 10_000
    assert fixture["draw_order"] == ["video", "audio"]
    assert fixture["eta"] == 1.0
    assert fixture["s_noise"] == 1.0


def test_ancestral_step_matches_the_pinned_apache_transition() -> None:
    fixture = _fixture()
    transition = fixture["stage_1"][0]
    step = sampler_steps.euler_ancestral_rf_step(
        _array(transition["video_input"]),
        _array(transition["video_denoised"]),
        _array(transition["video_noise"]),
        sigma=transition["sigma"],
        sigma_next=transition["sigma_next"],
        eta=fixture["eta"],
        s_noise=fixture["s_noise"],
    )
    np.testing.assert_allclose(
        np.asarray(step),
        np.asarray(_array(transition["video_output"])),
        atol=2e-6,
        rtol=0.0,
    )


def test_injected_joint_stage1_matches_all_eight_reference_transitions() -> None:
    fixture = _fixture()
    transitions = fixture["stage_1"]
    transformer = _FixtureTransformer(transitions)
    provider = _FixtureNoise(transitions)
    video, audio = denoise_loop(
        _state(transitions[0]["video_input"]),
        _state(transitions[0]["audio_input"]),
        fixture["stage_1_sigmas"],
        transformer=transformer,
        video_context=mx.zeros((1, 1, 4), dtype=mx.float32),
        audio_context=mx.zeros((1, 1, 4), dtype=mx.float32),
        step_kind="ancestral-rf",
        stage=1,
        noise_provider=provider,
    )

    assert transformer.calls == 8
    provider.assert_exhausted()
    assert provider.calls == [
        (1, transition, modality) for transition in range(7) for modality in ("video", "audio")
    ]
    np.testing.assert_allclose(
        np.asarray(video.latent),
        np.asarray(_array(transitions[-1]["video_output"])),
        atol=2e-6,
        rtol=0.0,
    )
    assert audio is not None
    np.testing.assert_allclose(
        np.asarray(audio.latent),
        np.asarray(_array(transitions[-1]["audio_output"])),
        atol=2e-6,
        rtol=0.0,
    )


def test_stage2_remains_deterministic_and_never_requests_noise() -> None:
    fixture = _fixture()
    transitions = fixture["stage_2"]
    transformer = _FixtureTransformer(transitions)
    video, audio = denoise_loop(
        _state(transitions[0]["video_input"]),
        _state(transitions[0]["audio_input"]),
        fixture["stage_2_sigmas"],
        transformer=transformer,
        video_context=mx.zeros((1, 1, 4), dtype=mx.float32),
        audio_context=mx.zeros((1, 1, 4), dtype=mx.float32),
        step_kind="deterministic-euler",
        stage=2,
        noise_provider=_RejectNoise(),
    )

    assert transformer.calls == 3
    np.testing.assert_allclose(
        np.asarray(video.latent),
        np.asarray(_array(transitions[-1]["video_output"])),
        atol=2e-6,
        rtol=0.0,
    )
    assert audio is not None
    np.testing.assert_allclose(
        np.asarray(audio.latent),
        np.asarray(_array(transitions[-1]["audio_output"])),
        atol=2e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (mx.zeros((1, 1), dtype=mx.float32), "shape"),
        (mx.zeros((1, 2, 3), dtype=mx.float16), "dtype"),
        (mx.full((1, 2, 3), float("inf"), dtype=mx.float32), "finite"),
    ],
)
def test_ancestral_provider_output_is_validated_before_the_transformer(
    replacement: mx.array,
    message: str,
) -> None:
    calls = 0

    class _NeverTransformer:
        def __call__(self, *_args):
            nonlocal calls
            calls += 1
            raise AssertionError("invalid injected noise must fail first")

    def invalid_noise(**_kwargs):
        return replacement

    with pytest.raises(ValueError, match=message):
        denoise_loop(
            _state([[[-0.75, 0.5, 1.25], [0.125, -0.25, 0.875]]]),
            None,
            (1.0, 0.5, 0.0),
            transformer=_NeverTransformer(),
            video_context=mx.zeros((1, 1, 4), dtype=mx.float32),
            step_kind="ancestral-rf",
            stage=1,
            noise_provider=invalid_noise,
        )
    assert calls == 0
