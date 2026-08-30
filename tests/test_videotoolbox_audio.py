"""Round-trip and sample-buffer tests for ``kinomlx.videotoolbox.audio``.

The VideoToolbox audio path was lifted whole into KinoMLX with no tests, and
commit 9de34ab's memoryview copy-elisions - ``make_sample_buffer``'s block slice
and ``_write_wav``'s ``cast("B")`` - were verified only by ``py_compile`` plus
ad-hoc API probes. These tests pin the actual behavior:

- ``AudioTrack.make_sample_buffer`` builds a CMSampleBuffer that spans the
  requested frame range, carries the right presentation timestamp, and whose
  block bytes equal the source slice exactly (the regression net for the
  zero-copy slice).
- ``write_wav_int16`` / ``write_wav_float32`` -> ``read_wav`` round-trips a known
  waveform (exact for float32, within int16 quantization for int16) through the
  native AVFoundation ``AVAudioFile`` writer/reader.

Every test here drives a real AVFoundation / CoreMedia stack, so the module is
tagged ``requires_avfoundation`` for explicit native-test selection.
"""

from __future__ import annotations

import CoreMedia
import mlx.core as mx
import pytest

from kinomlx.videotoolbox.audio import (
    AudioTrack,
    read_wav,
    write_wav_float32,
    write_wav_int16,
)

pytestmark = pytest.mark.requires_avfoundation

SAMPLE_RATE = 48000


def _ramp_waveform(channels: int, frames: int) -> mx.array:
    """Deterministic (channels, frames) float32 ramp spanning [-1, 1] exactly."""
    n = channels * frames
    return (mx.arange(n, dtype=mx.float32) / (n - 1) * 2 - 1).reshape(channels, frames)


def _sample_buffer_block_bytes(sample_buf) -> bytes:
    """Copy the contiguous interleaved PCM bytes back out of a CMSampleBuffer."""
    block = CoreMedia.CMSampleBufferGetDataBuffer(sample_buf)
    length = CoreMedia.CMBlockBufferGetDataLength(block)
    status, data = CoreMedia.CMBlockBufferCopyDataBytes(block, 0, length, None)
    assert status == 0, f"CMBlockBufferCopyDataBytes status={status}"
    return bytes(data)


# --------------------------------------------------------------------------- #
# AudioTrack.make_sample_buffer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 50), (10, 30), (0, 1), (49, 50)],
    ids=["full", "subrange", "single", "last"],
)
def test_make_sample_buffer_spans_frame_count(start, end):
    track = AudioTrack(_ramp_waveform(2, 50), SAMPLE_RATE)
    sb = track.make_sample_buffer(start, end)

    assert sb is not None
    assert CoreMedia.CMSampleBufferIsValid(sb)
    assert CoreMedia.CMSampleBufferGetNumSamples(sb) == end - start
    # One interleaved float32 frame is 4 bytes per channel.
    block = CoreMedia.CMSampleBufferGetDataBuffer(sb)
    assert CoreMedia.CMBlockBufferGetDataLength(block) == (end - start) * 4 * track.channels
    # The format description rides along so the AVAssetWriterInput can encode it.
    assert CoreMedia.CMSampleBufferGetFormatDescription(sb) is not None


@pytest.mark.parametrize("start", [0, 10, 49])
def test_make_sample_buffer_presentation_timestamp(start):
    track = AudioTrack(_ramp_waveform(2, 50), SAMPLE_RATE)
    sb = track.make_sample_buffer(start, 50)

    pts = CoreMedia.CMSampleBufferGetPresentationTimeStamp(sb)
    # PTS is start_frame / sample_rate: value == start_frame, timescale == rate.
    assert pts.value == start
    assert pts.timescale == SAMPLE_RATE


def test_make_sample_buffer_bytes_match_source():
    """The block bytes equal the interleaved source slice exactly.

    Direct regression net for commit 9de34ab: make_sample_buffer slices a
    memoryview of self._bytes instead of a bytes copy. An off-by-one in the
    slice offset math would surface here even though the smoke checks pass.
    """
    channels, frames = 2, 50
    w = _ramp_waveform(channels, frames)
    track = AudioTrack(w, SAMPLE_RATE)
    start, end = 10, 30
    sb = track.make_sample_buffer(start, end)

    bpf = 4 * channels
    assert _sample_buffer_block_bytes(sb) == track._bytes[start * bpf : end * bpf]

    # Decoded back to floats it matches the interleaved (frames, channels)
    # waveform slice with no loss (float32 in, float32 out).
    got = mx.array(memoryview(_sample_buffer_block_bytes(sb)).cast("f")).reshape(
        end - start,
        channels,
    )
    expected = mx.transpose(w)[start:end]
    assert mx.max(mx.abs(got - expected)).item() == 0.0


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 0), (30, 30), (30, 10)],
    ids=["empty-at-zero", "empty-midway", "reversed"],
)
def test_make_sample_buffer_empty_range_returns_none(start, end):
    track = AudioTrack(_ramp_waveform(2, 50), SAMPLE_RATE)
    assert track.make_sample_buffer(start, end) is None


# --------------------------------------------------------------------------- #
# write_wav_int16 / write_wav_float32 -> read_wav round-trips
# --------------------------------------------------------------------------- #

# int16 tol: the worst-case quantization step is ~3.05e-5 (1/32768); 1e-4 sits
# comfortably above that yet far below anything that would mean broken bytes.
_ROUNDTRIP_WRITERS = [
    pytest.param(write_wav_float32, 1e-6, id="float32-exact"),
    pytest.param(write_wav_int16, 1e-4, id="int16-quantized"),
]


@pytest.mark.parametrize(("writer", "tol"), _ROUNDTRIP_WRITERS)
def test_write_read_roundtrip_stereo(writer, tol, tmp_path):
    aud = _ramp_waveform(2, 100)
    path = tmp_path / "stereo.wav"

    writer(aud, path, SAMPLE_RATE)
    sr, samples = read_wav(path)

    assert sr == SAMPLE_RATE
    assert tuple(int(x) for x in samples.shape) == (2, 100)
    assert mx.max(mx.abs(samples - aud)).item() < tol


@pytest.mark.parametrize(("writer", "tol"), _ROUNDTRIP_WRITERS)
def test_write_read_roundtrip_mono(writer, tol, tmp_path):
    """Exercises read_wav's channels == 1 branch (chans[0][None, :])."""
    aud = _ramp_waveform(1, 64)
    path = tmp_path / "mono.wav"

    writer(aud, path, 16000)
    sr, samples = read_wav(path)

    assert sr == 16000
    assert tuple(int(x) for x in samples.shape) == (1, 64)
    assert mx.max(mx.abs(samples - aud)).item() < tol


def test_write_accepts_batched_input(tmp_path):
    """_write_wav drops a leading batch dim: (B, C, T) -> (C, T)."""
    aud = _ramp_waveform(2, 80)
    path = tmp_path / "batched.wav"

    write_wav_float32(aud[None], path, SAMPLE_RATE)  # (1, 2, 80) in
    sr, samples = read_wav(path)

    assert sr == SAMPLE_RATE
    assert tuple(int(x) for x in samples.shape) == (2, 80)
    assert mx.max(mx.abs(samples - aud)).item() == 0.0


# --------------------------------------------------------------------------- #
# AudioTrack.save_wav and input-container contract
# --------------------------------------------------------------------------- #


def test_save_wav_roundtrip(tmp_path):
    """save_wav writes the in-memory PCM as a float32 WAV; it round-trips exactly."""
    aud = _ramp_waveform(2, 100)
    track = AudioTrack(aud, SAMPLE_RATE)
    path = tmp_path / "track.wav"

    track.save_wav(path)
    sr, samples = read_wav(path)

    assert sr == SAMPLE_RATE
    assert tuple(int(x) for x in samples.shape) == (2, 100)
    assert mx.max(mx.abs(samples - aud)).item() == 0.0


def test_audiotrack_rejects_numpy_input():
    numpy = pytest.importorskip("numpy")
    frames = 50
    n = 2 * frames
    np_w = (numpy.arange(n, dtype=numpy.float32) / (n - 1) * 2 - 1).reshape(2, frames)

    with pytest.raises(TypeError, match="waveform must be an MLX array"):
        AudioTrack(np_w, SAMPLE_RATE)


def test_audiotrack_rejects_non_2d_waveform():
    """The constructor requires a (channels, samples) array."""
    with pytest.raises(ValueError, match="channels, samples"):
        AudioTrack(mx.zeros((50,)), SAMPLE_RATE)
