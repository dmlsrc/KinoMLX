"""Sequence-start onset-spike detection and mitigation.

Some distilled LTX-2.3 AV generations produce a short (~65 ms) loud
transient at t=0 followed by clean silence until the first spoken word:

  * Dialog-heavy prompts: the model encodes a broadband loud onset that
    the vocoder's bandwidth extension synthesizes into an audible click.
  * Ambient-onset prompts: the elevated latent at t=0 is still there
    (a universal attention-sink behavior) but decodes to quiet.

So the mitigation is detect-then-trim, not unconditional surgery, and the
trim zero-fills the leading region rather than dropping samples - dropping
would shift audio relative to video and break lip sync.

Detection
---------
The signature is *loud burst followed by silence*, not just *loud burst*.
A real speech onset at t=0 would also be loud at t=0 but would NOT be
followed by silence - it would sustain. So we use a two-window check:

  first 50 ms RMS > 2.0x global RMS
  AND
  mean RMS over 100-250 ms < 0.1x global RMS

A legitimate loud-speech onset trips condition 1 but not 2 (sustained
content keeps the 100-250 ms RMS near global RMS), so the trim does not
fire and lip sync at t=0 is preserved.

Trim
----
The trim zeros out the leading 120 ms by default - long enough to clear
the diagnosed click's ~95 ms decay tail with margin, short enough to sit
comfortably inside the intentional silence the model places before the
first spoken word (95-250 ms on the diagnosed clip). Sample count is
preserved.

Public API
----------
- ``detect_onset_spike(samples, sample_rate, ...)`` -> bool
- ``trim_onset(samples, sample_rate, *, trim_ms)`` -> trimmed mx.array
- ``mitigate_onset(samples, sample_rate, *, mode, trim_ms)`` -> result
- ``detect_onset_latent_spike(latent, ...)`` -> bool
- ``parse_trim_mode(s)`` -> (mode, trim_ms) for the CLI flag
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .trim_mode import DEFAULT_TRIM_MS
from .trim_mode import parse_trim_mode as parse_trim_mode

# ---------------------------------------------------------------------------
# Threshold constants - the single source of truth for what counts as an
# onset spike, shared by the detector and the mitigation entry point below.
# ---------------------------------------------------------------------------

DEFAULT_DETECT_WINDOW_MS = 50.0
"""First-window size for the detector."""

DEFAULT_DETECT_THRESHOLD_RATIO = 2.0
"""Multiplier on global RMS that the first window must exceed."""

DEFAULT_SILENCE_START_MS = 100.0
"""Lower bound of the 'must be quiet' trail-off window."""

DEFAULT_SILENCE_END_MS = 250.0
"""Upper bound of the 'must be quiet' trail-off window."""

DEFAULT_SILENCE_RATIO = 0.1
"""Multiplier on global RMS that the trail-off window must NOT exceed."""

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _to_audio_ct_mlx(samples: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Coerce (B,C,T) / (C,T) / (T,) to one MLX (C,T) audio track.

    A three-dimensional input selects batch item zero. Public callers that
    accept batched audio must validate cardinality before using this low-level
    single-track onset API.
    """
    if not isinstance(samples, mx.array):
        raise TypeError(f"samples must be an MLX array, got {type(samples).__name__}")
    arr = samples
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"audio_waveform must be (B,C,T), (C,T), or (T,); got shape {arr.shape}")
    return arr


def _rms(values: mx.array) -> mx.array:
    """Root-mean-square of an MLX vector."""
    return mx.sqrt(mx.mean(mx.square(values)))


def detect_onset_spike(
    samples: mx.array,
    sample_rate: int,
    *,
    window_ms: float = DEFAULT_DETECT_WINDOW_MS,
    threshold_ratio: float = DEFAULT_DETECT_THRESHOLD_RATIO,
    silence_start_ms: float = DEFAULT_SILENCE_START_MS,
    silence_end_ms: float = DEFAULT_SILENCE_END_MS,
    silence_ratio: float = DEFAULT_SILENCE_RATIO,
) -> bool:
    """Return True iff the click signature is present.

    The signature is *loud first window* AND *near-silence in the
    [silence_start_ms, silence_end_ms] trail-off window*. Operates on a
    mono mix of the input channels, which matches the perception case
    where the click is heard "at the start," not "in the left ear only."

    Degrades gracefully on inputs too short or too quiet to evaluate: if
    the clip is shorter than the silence-window endpoint, or global RMS is
    essentially zero, returns False.
    """
    # float64 reductions need the CPU device (Metal has no double support);
    # the windows are tiny, so the host round-trip is negligible.
    old_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        arr = _to_audio_ct_mlx(samples, dtype=mx.float64)
        mono = mx.mean(arr, axis=0)
        n_total = mono.shape[0]

        win = max(1, int(round(window_ms / 1000.0 * sample_rate)))
        silence_start = int(round(silence_start_ms / 1000.0 * sample_rate))
        silence_end = int(round(silence_end_ms / 1000.0 * sample_rate))

        # Need the full silence window present to evaluate the second condition.
        if n_total < silence_end or win <= 0:
            return False

        global_rms = _rms(mono)
        mx.eval(global_rms)
        global_rms_value = float(global_rms)
        if global_rms_value <= 1e-9:
            return False

        first_rms = _rms(mono[:win])
        tail_rms = _rms(mono[silence_start:silence_end])
        mx.eval(first_rms, tail_rms)
        if float(first_rms) <= threshold_ratio * global_rms_value:
            return False

        return float(tail_rms) < silence_ratio * global_rms_value
    finally:
        mx.set_default_device(old_device)


# ---------------------------------------------------------------------------
# Trim - zero-fill leading region, preserve sample count
# ---------------------------------------------------------------------------


def trim_onset(
    samples: mx.array,
    sample_rate: int,
    *,
    trim_ms: float = DEFAULT_TRIM_MS,
) -> mx.array:
    """Zero out the leading ``trim_ms`` milliseconds of every channel.

    Returns a float32 MLX array normalized to (C,T). A (B,C,T) input selects
    batch item zero for this low-level single-track API; the public encoder
    validates that the batch contains exactly one item before calling it.
    Sample count is preserved so video and audio stay in sync.

    ``trim_ms <= 0`` is a pass-through. ``trim_ms`` is clamped to the clip
    duration: anything longer than the clip silences the entire track,
    which is the right behavior if a caller deliberately asks for it.
    """
    arr = _to_audio_ct_mlx(samples, dtype=mx.float32)
    if trim_ms <= 0:
        return arr
    n_zero = int(round(trim_ms / 1000.0 * sample_rate))
    n_zero = min(n_zero, arr.shape[1])
    if n_zero <= 0:
        return arr
    if n_zero == arr.shape[1]:
        return mx.zeros_like(arr)
    leading = mx.zeros((arr.shape[0], n_zero), dtype=arr.dtype)
    return mx.concatenate([leading, arr[:, n_zero:]], axis=1)


# ---------------------------------------------------------------------------
# High-level mitigation entry - what the encoders call
# ---------------------------------------------------------------------------


@dataclass
class OnsetTrimResult:
    """Outcome of an onset-mitigation call.

    ``samples`` is a float32 MLX array with channel layout normalized to
    (C, T). When ``applied=True`` the leading ``trim_ms`` is zero-filled.

    ``detected`` is the diagnostic verdict from the two-window check - it
    can be True even when ``applied=False`` (e.g. mode="off"), and False
    when ``applied=True`` (e.g. mode="force" / explicit N_ms).
    """

    samples: mx.array
    applied: bool
    detected: bool
    trim_ms: float
    mode: str
    detail: str


def mitigate_onset(
    samples: mx.array,
    sample_rate: int,
    *,
    mode: str = "auto",
    trim_ms: float = DEFAULT_TRIM_MS,
) -> OnsetTrimResult:
    """Apply the sequence-start onset mitigation according to ``mode``.

    Modes:
      auto   detect-then-trim. Trim is applied only when the two-window
             detector fires. Quiet-onset clips pass through unchanged.
      off    no detection, no trim. Returned ``samples`` is a (C,T) MLX
             array normalized from the input, untouched.
      force  always trim by ``trim_ms``, regardless of detection. Used by
             the CLI when the user specifies an explicit N_ms.

    ``trim_ms`` is the duration of the leading zero-fill applied when a
    trim occurs. Detector parameters are not exposed here - callers that
    want non-default thresholds should call ``detect_onset_spike`` +
    ``trim_onset`` directly.
    """
    arr = _to_audio_ct_mlx(samples, dtype=mx.float32)
    mode = mode.lower()
    if mode == "off":
        return OnsetTrimResult(
            samples=arr,
            applied=False,
            detected=False,
            trim_ms=0.0,
            mode="off",
            detail="onset trim disabled",
        )

    if mode == "auto":
        is_spike = detect_onset_spike(arr, sample_rate)
        if not is_spike:
            return OnsetTrimResult(
                samples=arr,
                applied=False,
                detected=False,
                trim_ms=0.0,
                mode="auto",
                detail="no onset spike detected",
            )
        trimmed = trim_onset(arr, sample_rate, trim_ms=trim_ms)
        return OnsetTrimResult(
            samples=trimmed,
            applied=True,
            detected=True,
            trim_ms=float(trim_ms),
            mode="auto",
            detail=f"onset spike detected; zero-filled leading {trim_ms:g} ms",
        )

    if mode == "force":
        trimmed = trim_onset(arr, sample_rate, trim_ms=trim_ms)
        # Still report the diagnostic verdict so the run-log captures
        # whether the forced trim was actually needed.
        is_spike = detect_onset_spike(arr, sample_rate)
        return OnsetTrimResult(
            samples=trimmed,
            applied=True,
            detected=is_spike,
            trim_ms=float(trim_ms),
            mode="force",
            detail=f"forced zero-fill of leading {trim_ms:g} ms",
        )

    raise ValueError(f"Unknown onset mitigation mode {mode!r}. Expected one of: auto, off, force.")


# ---------------------------------------------------------------------------
# Latent-domain onset DETECTION (causal-VAE sequence-start boundary spike)
# ---------------------------------------------------------------------------
# The audio VAE is causal in time, so its first latent frame is encoded against
# zero-padding (a cold-start boundary). In the 8-channel audio latent this lands
# as a concentrated single-channel spike at frame 0 that the vocoder renders as a
# broadband t=0 click. Total frame-0 energy is normal -- it is the per-channel
# concentration that clicks -- so the waveform RMS detector above misses it.
#
# This detector is the reliable trigger for that click class. The mitigation is
# the waveform `trim_onset` zero-fill applied to the decoded audio: the spike
# decodes to a transient confined to the leading ~120 ms (well before the first
# spoken word at ~175 ms), so zeroing that region clears it with no content loss.
# Flattening the latent first was tried and dropped: it only dents the spike (the
# decoder is itself causal and leaves an audible residual), and the waveform
# zero-fill overwrites that region anyway. Decode-only -- saved sidecars and
# stage-2 latents are upstream and untouched.

DEFAULT_LATENT_SPIKE_RATIO = 2.0
DEFAULT_LATENT_CONCENTRATION = 1.8


def _audio_latent_ct(latent: mx.array) -> mx.array:
    """(B, C, T, F) or (C, T, F) -> (C, T, F)."""
    return latent[0] if latent.ndim == 4 else latent


def detect_onset_latent_spike(
    latent: mx.array,
    *,
    ratio: float = DEFAULT_LATENT_SPIKE_RATIO,
    concentration: float = DEFAULT_LATENT_CONCENTRATION,
) -> bool:
    """True iff the first audio-latent frame has a concentrated single-channel spike.

    Per-channel-per-frame RMS; a channel's frame-0 energy must exceed ``ratio`` x
    that channel's mean over all frames (anomalous start), AND the frame-0
    per-channel profile must be concentrated -- peak channel > ``concentration`` x
    the mean channel -- so a broad all-channel elevation (which decodes quietly)
    does not trip it.
    """
    a = _audio_latent_ct(latent).astype(mx.float32)
    if a.shape[1] < 4:
        return False
    perch = mx.sqrt(mx.mean(a * a, axis=2))  # (C, T)
    base = mx.mean(perch, axis=1) + 1e-9  # (C,)
    f0 = perch[:, 0] / base  # (C,)
    peak = float(mx.max(f0))
    mean = float(mx.mean(f0))
    return peak > ratio and peak > concentration * mean
