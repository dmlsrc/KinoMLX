"""Checkpoint-backed causal STFT used by LTX bandwidth extension."""

from __future__ import annotations

import mlx.core as mx

import kinomlx._mlx_nn as nn


class STFT1d(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        frequencies = filter_length // 2 + 1
        shape = (frequencies * 2, 1, filter_length)
        self.forward_basis = mx.zeros(shape)
        self.inverse_basis = mx.zeros(shape)

    @property
    def basis_shape(self) -> tuple[int, ...]:
        frequencies = self.filter_length // 2 + 1
        return (frequencies * 2, 1, self.filter_length)

    def __call__(self, waveform: mx.array) -> tuple[mx.array, mx.array]:
        if waveform.ndim == 2:
            waveform = waveform[:, None, :]
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("STFT input must have shape (batch, samples) or (batch, 1, samples)")
        left_padding = max(0, self.win_length - self.hop_length)
        waveform = mx.pad(waveform, [(0, 0), (0, 0), (left_padding, 0)])
        weight = self.forward_basis.astype(waveform.dtype).transpose(0, 2, 1)
        spectrum = mx.conv1d(
            waveform.transpose(0, 2, 1),
            weight,
            stride=self.hop_length,
        ).transpose(0, 2, 1)
        mx.eval(spectrum)
        frequencies = spectrum.shape[1] // 2
        real = spectrum[:, :frequencies]
        imaginary = spectrum[:, frequencies:]
        magnitude = mx.sqrt(real**2 + imaginary**2)
        phase = mx.arctan2(
            imaginary.astype(mx.float32),
            real.astype(mx.float32),
        ).astype(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = STFT1d(filter_length, hop_length, win_length)
        self.mel_channels = mel_channels
        self.mel_basis = mx.zeros((mel_channels, filter_length // 2 + 1))

    @property
    def mel_basis_shape(self) -> tuple[int, ...]:
        return (self.mel_channels, self.stft_fn.filter_length // 2 + 1)

    def mel_spectrogram(
        self,
        waveform: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        magnitude, phase = self.stft_fn(waveform)
        energy = mx.sqrt(mx.sum(magnitude**2, axis=1))
        mel = mx.einsum(
            "mf,bft->bmt",
            self.mel_basis.astype(magnitude.dtype),
            magnitude,
        )
        return mx.log(mx.maximum(mel, 1e-5)), magnitude, phase, energy


__all__ = ["MelSTFT", "STFT1d"]
