"""In-memory PCM audio wrapped as CMSampleBuffers for AVAssetWriter."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import AVFoundation as av
import CoreAudio
import CoreMedia
import Foundation
import mlx.core as mx

from kinomlx.io.buffer import mlx_array_from_buffer

# CoreAudio FormatID constants (avoid importing the whole module just for these)
AUDIO_FORMAT_LPCM = 1819304813  # 'lpcm' kAudioFormatLinearPCM
AUDIO_FORMAT_AAC = 1633772320  # 'aac ' kAudioFormatMPEG4AAC
AUDIO_FORMAT_ALAC = 1634492771  # 'alac' kAudioFormatAppleLossless


class AudioTrack:
    """In-memory audio decoded from a latent. No disk WAV unless save_wav()
    is called explicitly.

    Constructed from a ``(channels, samples)`` MLX float32 array. Builds
    CMSampleBuffers on demand via `make_sample_buffer(start_frame, end_frame)`
    - the AVWriter's GCD audio pump pulls these in chunks as the encoder
    drains.

    Format: interleaved 32-bit float PCM in the source sample rate. The
    writer's audio output settings (ALAC / AAC) handle the encode-time
    conversion.
    """

    def __init__(self, waveform: mx.array, sample_rate: int):
        # Normalize the public MLX (channels, samples) boundary to float32.
        if not isinstance(waveform, mx.array):
            raise TypeError(f"waveform must be an MLX array, got {type(waveform).__name__}")
        w = waveform
        if w.dtype != mx.float32:
            w = w.astype(mx.float32)
        if w.ndim != 2:
            raise ValueError(f"AudioTrack expects (channels, samples); got {w.shape}")
        self.sample_rate = int(sample_rate)
        self.channels = int(w.shape[0])
        self.n_samples = int(w.shape[1])
        # Interleave: (channels, samples) -> (samples, channels) row-major bytes,
        # straight from the MLX buffer.
        self._bytes = bytes(memoryview(mx.contiguous(mx.transpose(w))))
        bytes_per_frame = 4 * self.channels

        asbd = CoreAudio.AudioStreamBasicDescription(
            float(self.sample_rate),
            AUDIO_FORMAT_LPCM,
            CoreAudio.kAudioFormatFlagIsFloat | CoreAudio.kAudioFormatFlagIsPacked,
            bytes_per_frame,  # mBytesPerPacket
            1,  # mFramesPerPacket
            bytes_per_frame,  # mBytesPerFrame
            self.channels,
            32,  # mBitsPerChannel
            0,
        )
        err, fmt = CoreMedia.CMAudioFormatDescriptionCreate(
            None,
            asbd,
            0,
            None,
            0,
            None,
            None,
            None,
        )
        if err != 0 or fmt is None:
            raise RuntimeError(f"CMAudioFormatDescriptionCreate failed: status={err}")
        self.format_desc = cast(object, fmt)

    def save_wav(self, path: Path) -> None:
        """Write the in-memory PCM out as a float32 WAV (for --save-audio-sidecar)."""
        samples = mlx_array_from_buffer(memoryview(self._bytes).cast("f")).reshape(
            self.n_samples,
            self.channels,
        )
        write_wav_float32(mx.transpose(samples), path, self.sample_rate)

    def make_sample_buffer(self, start_frame: int, end_frame: int) -> object | None:
        """Build a CMSampleBuffer for audio frames [start_frame, end_frame).

        Returns None if the range is empty. Caller is responsible for
        appendSampleBuffer-ing it to an AVAssetWriterInput.
        """
        n = end_frame - start_frame
        if n <= 0:
            return None
        bytes_per_frame = 4 * self.channels
        # Zero-copy view into self._bytes; CMBlockBufferReplaceDataBytes copies it
        # into the block below, so the intermediate bytes slice is avoidable.
        chunk_bytes = memoryview(self._bytes)[
            start_frame * bytes_per_frame : end_frame * bytes_per_frame
        ]
        data_len = len(chunk_bytes)

        err, block = CoreMedia.CMBlockBufferCreateWithMemoryBlock(
            None,
            None,
            data_len,
            None,
            None,
            0,
            data_len,
            1,
            None,
        )
        if err != 0 or block is None:
            raise RuntimeError(f"CMBlockBufferCreateWithMemoryBlock failed: {err}")
        err = CoreMedia.CMBlockBufferReplaceDataBytes(chunk_bytes, block, 0, data_len)
        if err != 0:
            raise RuntimeError(f"CMBlockBufferReplaceDataBytes failed: {err}")

        pts = CoreMedia.CMTimeMake(start_frame, self.sample_rate)
        err, sample_buf = CoreMedia.CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            None,
            block,
            self.format_desc,
            n,
            pts,
            None,
            None,
        )
        if err != 0 or sample_buf is None:
            raise RuntimeError(
                f"CMAudioSampleBufferCreateReadyWithPacketDescriptions failed: {err}"
            )
        return cast(object, sample_buf)


def read_wav(path: Path | str) -> tuple[int, mx.array]:
    """Read a WAV (or any AVFoundation-supported audio file) into
    ``(sample_rate, (channels, frames) float32 mlx array in [-1, 1])``.

    Uses AVFoundation's AVAudioFile, which reads PCM int16/24/32 AND IEEE float32
    - including the float32 sidecars that stdlib ``wave`` rejects - with no numpy /
    scipy / soundfile. Samples come straight from the AVAudioPCMBuffer's
    deinterleaved float channels via the buffer protocol.
    """
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    audio_file, err = av.AVAudioFile.alloc().initForReading_error_(url, None)
    if audio_file is None:
        raise RuntimeError(f"AVAudioFile could not open {path}: {err}")
    fmt = audio_file.processingFormat()
    buf = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
        fmt,
        int(audio_file.length()),
    )
    ok, err = audio_file.readIntoBuffer_error_(buf, None)
    if not ok:
        raise RuntimeError(f"AVAudioFile could not read {path}: {err}")
    channels = int(fmt.channelCount())
    frames = int(buf.frameLength())
    fcd = buf.floatChannelData()  # deinterleaved float32: channels x frames
    # as_buffer(n) exposes n elements (4 bytes each) as a uint8 view; cast to f32.
    chans = [
        mlx_array_from_buffer(memoryview(fcd[c].as_buffer(frames)).cast("f"))
        for c in range(channels)
    ]
    samples = chans[0][None, :] if channels == 1 else mx.stack(chans, axis=0)
    return int(fmt.sampleRate()), samples


def _write_wav(
    samples: mx.array,
    path: Path | str,
    sample_rate: int,
    *,
    float32: bool,
) -> None:
    """Write ``(B,C,T)``/``(C,T)`` MLX samples to WAV via AVFoundation's
    AVAudioFile - native macOS, no struct/wave hand-rolling.

    float32=True writes an IEEE float32 WAV; otherwise int16 PCM. The samples are
    written into a float32 AVAudioPCMBuffer and AVAudioFile converts to the file
    format and writes the container/header.
    """
    if not isinstance(samples, mx.array):
        raise TypeError(f"samples must be an MLX array, got {type(samples).__name__}")
    w = samples
    if w.dtype != mx.float32:
        w = w.astype(mx.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim != 2:
        raise ValueError(f"audio must be (B,C,T) or (C,T); got shape {w.shape}")
    channels, frames = int(w.shape[0]), int(w.shape[1])
    settings: dict[object, object] = {
        av.AVFormatIDKey: AUDIO_FORMAT_LPCM,
        av.AVSampleRateKey: float(sample_rate),
        av.AVNumberOfChannelsKey: channels,
        av.AVLinearPCMBitDepthKey: 32 if float32 else 16,
        av.AVLinearPCMIsFloatKey: float32,
        av.AVLinearPCMIsBigEndianKey: False,
        av.AVLinearPCMIsNonInterleaved: False,
    }
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    out, err = av.AVAudioFile.alloc().initForWriting_settings_error_(url, settings, None)
    if out is None:
        raise RuntimeError(f"AVAudioFile could not open {path} for writing: {err}")
    buf = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
        out.processingFormat(),
        frames,
    )
    buf.setFrameLength_(frames)
    fcd = buf.floatChannelData()  # deinterleaved float32, channels x frames
    for c in range(channels):
        # cast("B") is a zero-copy byte view of the float32 samples (byte-identical
        # to bytes(...)); the slice-assign copies it into the AVAudio channel buffer.
        memoryview(fcd[c].as_buffer(frames))[:] = memoryview(mx.contiguous(w[c])).cast("B")
    ok, err = out.writeFromBuffer_error_(buf, None)
    if not ok:
        raise RuntimeError(f"AVAudioFile could not write {path}: {err}")


def write_wav_int16(
    audio_waveform: mx.array,
    path: Path | str,
    sample_rate: int,
) -> None:
    """Write a stereo int16 PCM WAV from MLX ``(B,C,T)`` or ``(C,T)``."""
    _write_wav(audio_waveform, path, sample_rate, float32=False)


def write_wav_float32(
    audio_waveform: mx.array,
    path: Path | str,
    sample_rate: int,
) -> None:
    """Write an IEEE float32 WAV from MLX samples without int16 quantization."""
    _write_wav(audio_waveform, path, sample_rate, float32=True)


def audio_writer_settings(
    codec: str,
    sample_rate: int,
    channels: int,
) -> dict[object, object]:
    """AVAssetWriterInput output settings for the configured audio codec."""
    if codec == "alac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_ALAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitDepthHintKey: 24,
        }
    if codec == "aac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_AAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitRateKey: 256000,
        }
    raise ValueError(f"Unknown audio codec {codec!r}")
