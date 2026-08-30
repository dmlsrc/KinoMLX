"""Shared native-MLX blocks for the LTX audio VAE."""

from __future__ import annotations

from enum import Enum

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.kernels import pixel_norm, silu


class CausalityAxis(Enum):
    NONE = "none"
    WIDTH = "width"
    HEIGHT = "height"


class PixelNorm(nn.Module):
    """Reference-stepwise per-location RMS normalization."""

    def __init__(self, axis: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        self.axis = axis
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return pixel_norm(x, axis=self.axis, eps=self.eps)


class PerChannelStatistics(nn.Module):
    """Dataset statistics applied in patchified audio-latent space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mean_of_means = mx.zeros((channels,))
        self.std_of_means = mx.ones((channels,))

    def normalize(self, x: mx.array) -> mx.array:
        return (x - self.mean_of_means[None, None, :]) / self.std_of_means[
            None,
            None,
            :,
        ]

    def denormalize(self, x: mx.array) -> mx.array:
        return (
            x * self.std_of_means[None, None, :]
            + self.mean_of_means[
                None,
                None,
                :,
            ]
        )


def _conv_values() -> tuple[mx.array, mx.array]:
    return mx.zeros((0, 0, 0, 0)), mx.zeros((0,))


class CausalConv2d(nn.Module):
    """Channels-first 2D convolution with one causal spatial axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.weight, self.bias = _conv_values()
        self.causality_axis = causality_axis
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if downsample:
            table = {
                CausalityAxis.NONE: ((0, 1), (0, 1)),
                CausalityAxis.WIDTH: ((0, 1), (2, 0)),
                CausalityAxis.HEIGHT: ((2, 0), (0, 1)),
            }
            height, width = table[causality_axis]
        else:
            pad = kernel_size - 1
            symmetric = (pad // 2, pad - pad // 2)
            height = (pad, 0) if causality_axis is CausalityAxis.HEIGHT else symmetric
            width = (pad, 0) if causality_axis is CausalityAxis.WIDTH else symmetric
        self.padding = ((0, 0), (0, 0), height, width)

    @property
    def native_weight_shape(self) -> tuple[int, ...]:
        return (
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.in_channels,
        )

    @property
    def checkpoint_weight_shape(self) -> tuple[int, ...]:
        return (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if tuple(self.weight.shape) != self.native_weight_shape:
            raise RuntimeError("audio VAE convolution weights have not been loaded")
        x = mx.pad(x, list(self.padding)).transpose(0, 2, 3, 1)
        x = mx.conv2d(x, self.weight, stride=self.stride)
        return x.transpose(0, 3, 1, 2) + self.bias[None, :, None, None]


class ResBlock2d(nn.Module):
    """PixelNorm/SiLU residual block used throughout the audio VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        causality_axis: CausalityAxis,
    ) -> None:
        super().__init__()
        self.norm1 = PixelNorm()
        self.norm2 = PixelNorm()
        self.conv1 = CausalConv2d(
            in_channels,
            out_channels,
            causality_axis=causality_axis,
        )
        self.conv2 = CausalConv2d(
            out_channels,
            out_channels,
            causality_axis=causality_axis,
        )
        self.skip = (
            CausalConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                causality_axis=causality_axis,
            )
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.skip(x) if self.skip is not None else x
        x = self.conv1(silu(self.norm1(x)))
        x = self.conv2(silu(self.norm2(x)))
        return residual + x


class Downsample2d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        causality_axis: CausalityAxis,
    ) -> None:
        super().__init__()
        self.conv = CausalConv2d(
            channels,
            channels,
            stride=2,
            causality_axis=causality_axis,
            downsample=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        causality_axis: CausalityAxis,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        self.conv = CausalConv2d(
            channels,
            channels,
            causality_axis=causality_axis,
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch, channels, height, width = x.shape
        x = x[:, :, :, None, :, None]
        x = mx.broadcast_to(x, (batch, channels, height, 2, width, 2))
        x = self.conv(x.reshape(batch, channels, height * 2, width * 2))
        if self.causality_axis is CausalityAxis.HEIGHT:
            return x[:, :, 1:, :]
        if self.causality_axis is CausalityAxis.WIDTH:
            return x[:, :, :, 1:]
        return x


__all__ = [
    "CausalConv2d",
    "CausalityAxis",
    "Downsample2d",
    "PerChannelStatistics",
    "PixelNorm",
    "ResBlock2d",
    "Upsample2d",
]
