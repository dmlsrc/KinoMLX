"""Tests for the sequence-start audio onset detector and trim mitigation
(``kinomlx.audio.onset``).

Synthetic MLX-native fixtures model the signal classes the detector must
separate:

  * click signature   loud 60 ms burst, ~190 ms silence, then steady quiet
                      speech -- detector MUST fire
  * ambient onset     quiet noise throughout, no burst at t=0 -- MUST NOT fire
  * loud speech       loud sustained content from t=0 (no silent gap) -- MUST
                      NOT fire (the AV-sync-safety case: a real loud onset
                      must not get trimmed)
  * silent            all zeros -- MUST NOT fire (no div-by-zero)
  * too short         100 ms clip, can't evaluate the silence window -- MUST
                      NOT fire

Then the trim, the high-level mitigation modes, the latent-domain detector,
and the CLI parser are each checked. Pure MLX (no pyobjc), so this runs
everywhere.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinomlx.audio.onset import (
    DEFAULT_DETECT_THRESHOLD_RATIO,
    DEFAULT_DETECT_WINDOW_MS,
    DEFAULT_SILENCE_END_MS,
    DEFAULT_SILENCE_RATIO,
    DEFAULT_SILENCE_START_MS,
    DEFAULT_TRIM_MS,
    detect_onset_latent_spike,
    detect_onset_spike,
    mitigate_onset,
    parse_trim_mode,
    trim_onset,
)

SR = 48000


def _noise(shape: tuple[int, ...], seed: int) -> mx.array:
    """Deterministic N(0,1) float32 noise (seeded so fixtures are stable)."""
    return mx.random.normal(shape=shape, key=mx.random.key(seed))


# ---------------------------------------------------------------------------
# Waveform fixtures (channels, samples)
# ---------------------------------------------------------------------------


def _click_signature(duration_s: float = 2.0) -> mx.array:
    """Loud 60 ms burst, ~190 ms silence, then steady quiet speech."""
    n = int(duration_s * SR)
    burst_n = int(0.060 * SR)
    silence_n = int(0.250 * SR)
    out = mx.zeros((2, n), dtype=mx.float32)
    out[:, :burst_n] = _noise((2, burst_n), 1) * 0.5
    out[:, silence_n:] = _noise((2, n - silence_n), 2) * 0.15
    return out


def _ambient(duration_s: float = 2.0) -> mx.array:
    return _noise((2, int(duration_s * SR)), 3) * 0.02


def _loud_speech(duration_s: float = 2.0) -> mx.array:
    """Sustained loud content from t=0 with no silent gap (AV-sync case)."""
    return _noise((2, int(duration_s * SR)), 4) * 0.3


def _silent(duration_s: float = 2.0) -> mx.array:
    return mx.zeros((2, int(duration_s * SR)), dtype=mx.float32)


def _too_short() -> mx.array:
    return _noise((2, int(0.100 * SR)), 5) * 0.5


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detector_fires_on_click_signature():
    assert detect_onset_spike(_click_signature(), SR)


@pytest.mark.parametrize(
    "factory",
    [_ambient, _loud_speech, _silent, _too_short],
    ids=["ambient", "loud_speech", "silent", "too_short"],
)
def test_detector_ignores_non_click_signals(factory):
    assert not detect_onset_spike(factory(), SR)


def test_detector_accepts_shape_variants():
    ct = _click_signature()
    assert detect_onset_spike(ct, SR)  # (C, T)
    assert detect_onset_spike(ct[None], SR)  # (B, C, T)
    assert detect_onset_spike(mx.mean(ct, axis=0), SR)  # (T,) mono


def test_detector_threshold_ratio_disables_when_raised():
    # A ratio far above the click's first-window/global ratio fails condition 1.
    assert not detect_onset_spike(_click_signature(), SR, threshold_ratio=20.0)


def test_detector_silence_condition_rejects_sustained_loud():
    # Even with a trivially low first-window threshold (condition 1 trips), the
    # non-silent tail keeps the silence condition from passing.
    assert not detect_onset_spike(_loud_speech(), SR, threshold_ratio=0.1)


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


def test_trim_preserves_sample_count_and_zeros_leading():
    click = _click_signature()
    n_zero = int(120.0 / 1000.0 * SR)
    trimmed = trim_onset(click, SR, trim_ms=120.0)

    assert trimmed.shape[1] == click.shape[1]  # mute, never drop
    assert bool(mx.all(trimmed[:, :n_zero] == 0.0).item())
    assert mx.array_equal(trimmed[:, n_zero:], click[:, n_zero:])


@pytest.mark.parametrize("trim_ms", [0.0, -5.0])
def test_trim_nonpositive_is_passthrough(trim_ms):
    click = _click_signature()
    out = trim_onset(click, SR, trim_ms=trim_ms)
    assert isinstance(out, mx.array)
    assert mx.array_equal(out, click)


def test_trim_clamps_to_clip_length():
    click = _click_signature(duration_s=0.5)
    trimmed = trim_onset(click, SR, trim_ms=10_000.0)
    assert trimmed.shape == click.shape
    assert bool(mx.all(trimmed == 0.0).item())


# ---------------------------------------------------------------------------
# mitigate_onset modes
# ---------------------------------------------------------------------------


def test_mitigate_auto_fires_on_click():
    click = _click_signature()
    n_zero = int(120.0 / 1000.0 * SR)
    r = mitigate_onset(click, SR, mode="auto", trim_ms=120.0)

    assert r.applied
    assert r.detected
    assert r.mode == "auto"
    assert r.samples.shape[1] == click.shape[1]
    assert bool(mx.all(r.samples[:, :n_zero] == 0.0).item())


def test_mitigate_auto_passes_ambient_through():
    amb = _ambient()
    r = mitigate_onset(amb, SR, mode="auto")
    assert not r.applied
    assert not r.detected
    assert mx.array_equal(r.samples, amb)


def test_mitigate_off_never_trims():
    click = _click_signature()
    r = mitigate_onset(click, SR, mode="off")
    assert not r.applied
    assert not r.detected
    assert mx.array_equal(r.samples, click)


def test_mitigate_force_trims_even_without_spike():
    amb = _ambient()
    n_zero = int(80.0 / 1000.0 * SR)
    r = mitigate_onset(amb, SR, mode="force", trim_ms=80.0)

    assert r.applied
    assert not r.detected  # diagnostic still ran; ambient has no spike
    assert bool(mx.all(r.samples[:, :n_zero] == 0.0).item())


def test_mitigate_does_not_mutate_input():
    click = _click_signature()
    snapshot = mx.array(click)
    mitigate_onset(click, SR, mode="force", trim_ms=80.0)
    assert mx.array_equal(click, snapshot)


def test_mitigate_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown onset mitigation mode"):
        mitigate_onset(_click_signature(), SR, mode="bogus")


# ---------------------------------------------------------------------------
# Latent-domain detector
# ---------------------------------------------------------------------------


def _audio_latent(frame0_scale: list[float], frames: int = 64, seed: int = 0) -> mx.array:
    """(1, 8, frames, 16) N(0,1), with frame 0 scaled per channel."""
    a = mx.random.normal(shape=(1, 8, frames, 16), key=mx.random.key(seed))
    scale = mx.array(frame0_scale, dtype=mx.float32).reshape(1, 8, 1, 1)
    first = a[:, :, :1, :] * scale
    return mx.concatenate([first, a[:, :, 1:, :]], axis=2)


def test_latent_detector_fires_on_concentrated_spike():
    scale = [1.0] * 8
    scale[4] = 6.0
    assert detect_onset_latent_spike(_audio_latent(scale))


def test_latent_detector_ignores_broad_elevation():
    # Every channel elevated at frame 0 -> not concentrated -> decodes quiet.
    assert not detect_onset_latent_spike(_audio_latent([3.0] * 8))


def test_latent_detector_ignores_normal_first_frame():
    assert not detect_onset_latent_spike(_audio_latent([1.0] * 8))


# ---------------------------------------------------------------------------
# CLI parser + constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("auto", ("auto", DEFAULT_TRIM_MS)),
        ("AUTO", ("auto", DEFAULT_TRIM_MS)),
        ("off", ("off", 0.0)),
        ("none", ("off", 0.0)),
        ("0", ("off", 0.0)),
        ("150", ("force", 150.0)),
        ("80.5", ("force", 80.5)),
    ],
)
def test_parse_trim_mode(spec, expected):
    assert parse_trim_mode(spec) == expected


@pytest.mark.parametrize("spec", ["yes", "abc", "1.2.3", "-5", "nan", "inf"])
def test_parse_trim_mode_rejects_invalid(spec):
    with pytest.raises(ValueError, match="--audio-onset-trim"):
        parse_trim_mode(spec)


def test_default_constants():
    assert DEFAULT_DETECT_WINDOW_MS == 50.0
    assert DEFAULT_DETECT_THRESHOLD_RATIO == 2.0
    assert DEFAULT_SILENCE_START_MS == 100.0
    assert DEFAULT_SILENCE_END_MS == 250.0
    assert DEFAULT_SILENCE_RATIO == 0.1
    assert DEFAULT_TRIM_MS == 120.0
