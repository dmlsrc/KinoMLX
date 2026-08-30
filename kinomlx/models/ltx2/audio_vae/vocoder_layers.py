"""Convolution, activation, and resampling layers for the LTX vocoder."""

from __future__ import annotations

import math

import mlx.core as mx

import kinomlx._mlx_nn as nn


def _conv1d_values(*, bias: bool) -> tuple[mx.array, mx.array | None]:
    return mx.zeros((0, 0, 0)), mx.zeros((0,)) if bias else None


class Conv1d(nn.Module):
    """Channels-first wrapper around native MLX Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight, self.bias = _conv1d_values(bias=bias)
        self.use_bias = bias
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation

    @property
    def native_weight_shape(self) -> tuple[int, ...]:
        return (self.out_channels, self.kernel_size, self.in_channels)

    @property
    def checkpoint_weight_shape(self) -> tuple[int, ...]:
        return (self.out_channels, self.in_channels, self.kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        if tuple(self.weight.shape) != self.native_weight_shape:
            raise RuntimeError("vocoder Conv1d weights have not been loaded")
        samples = x.transpose(0, 2, 1)
        if self.padding:
            samples = mx.pad(
                samples,
                [(0, 0), (self.padding, self.padding), (0, 0)],
            )
        output = mx.conv1d(
            samples,
            self.weight,
            stride=self.stride,
            dilation=self.dilation,
        ).transpose(0, 2, 1)
        if self.bias is not None:
            output = output + self.bias[None, :, None]
        return output


class ConvTranspose1d(nn.Module):
    """Channels-first wrapper around native MLX transposed Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.weight, self.bias = _conv1d_values(bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @property
    def native_weight_shape(self) -> tuple[int, ...]:
        return (self.out_channels, self.kernel_size, self.in_channels)

    @property
    def checkpoint_weight_shape(self) -> tuple[int, ...]:
        return (self.in_channels, self.out_channels, self.kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        if tuple(self.weight.shape) != self.native_weight_shape:
            raise RuntimeError("vocoder ConvTranspose1d weights have not been loaded")
        x = mx.conv_transpose1d(
            x.transpose(0, 2, 1),
            self.weight,
            stride=self.stride,
            padding=self.padding,
        ).transpose(0, 2, 1)
        assert self.bias is not None
        return x + self.bias[None, :, None]


class SnakeBeta(nn.Module):
    """Log-parameterized periodic BigVGAN activation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        shape = (1, -1, 1)
        alpha = mx.exp(self.alpha).reshape(shape)
        beta = mx.exp(self.beta).reshape(shape)
        return x + mx.sin(alpha * x) ** 2 / (beta + 1e-9)


def _replicate_pad(x: mx.array, left: int, right: int) -> mx.array:
    parts = []
    if left:
        parts.append(mx.repeat(x[:, :, :1], left, axis=2))
    parts.append(x)
    if right:
        parts.append(mx.repeat(x[:, :, -1:], right, axis=2))
    return mx.concatenate(parts, axis=2)


def _depthwise_conv1d(
    x: mx.array,
    filter_value: mx.array,
    *,
    stride: int,
) -> mx.array:
    batch, channels, length = x.shape
    weight = filter_value.reshape(1, filter_value.shape[2], 1).astype(x.dtype)
    output = mx.conv1d(x.reshape(batch * channels, length, 1), weight, stride=stride)
    return output.reshape(batch, channels, -1)


def _depthwise_conv_transpose1d(
    x: mx.array,
    filter_value: mx.array,
    *,
    stride: int,
) -> mx.array:
    batch, channels, length = x.shape
    weight = filter_value.reshape(1, filter_value.shape[2], 1).astype(x.dtype)
    output = mx.conv_transpose1d(
        x.reshape(batch * channels, length, 1),
        weight,
        stride=stride,
    )
    return output.reshape(batch, channels, -1)


class LowPassFilter1d(nn.Module):
    def __init__(self, *, stride: int = 1, kernel_size: int = 12) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.filter = mx.zeros((1, 1, kernel_size), dtype=mx.float32)
        self.checkpoint_filter_loaded = False

    def __call__(self, x: mx.array) -> mx.array:
        if not self.checkpoint_filter_loaded:
            raise RuntimeError("vocoder low-pass filter has not been loaded")
        return _depthwise_conv1d(
            _replicate_pad(x, self.pad_left, self.pad_right),
            self.filter,
            stride=self.stride,
        )


class UpSample1d(nn.Module):
    """Checkpoint-backed Kaiser or generated Hann sinc upsampler."""

    def __init__(
        self,
        ratio: int,
        *,
        kernel_size: int | None = None,
        window_type: str = "checkpoint",
    ) -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError("upsample ratio must be positive")
        self.ratio = ratio
        if window_type == "hann":
            rolloff = 0.99
            width = math.ceil(6 / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            taps = []
            for index in range(self.kernel_size):
                position = (index / ratio - width) * rolloff
                bounded = max(-6.0, min(6.0, position))
                window = math.cos(bounded * math.pi / 12) ** 2
                sinc = 1.0 if position == 0 else math.sin(math.pi * position) / (math.pi * position)
                taps.append(sinc * window * rolloff / ratio)
            self.filter = mx.array(taps, dtype=mx.float32).reshape(1, 1, -1)
            self.checkpoint_filter_loaded = True
        elif window_type == "checkpoint":
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * ratio + (self.kernel_size - ratio) // 2
            self.pad_right = self.pad * ratio + (self.kernel_size - ratio + 1) // 2
            self.filter = mx.zeros((1, 1, self.kernel_size), dtype=mx.float32)
            self.checkpoint_filter_loaded = False
        else:
            raise ValueError(f"unsupported resampling window: {window_type}")

    def __call__(self, x: mx.array) -> mx.array:
        if not self.checkpoint_filter_loaded:
            raise RuntimeError("vocoder upsample filter has not been loaded")
        x = _replicate_pad(x, self.pad, self.pad)
        x = self.ratio * _depthwise_conv_transpose1d(
            x,
            self.filter,
            stride=self.ratio,
        )
        stop = x.shape[2] - self.pad_right
        return x[:, :, self.pad_left : stop]


class DownSample1d(nn.Module):
    """Anti-aliased decimator: low-pass filter, then stride."""

    def __init__(self, ratio: int, *, kernel_size: int = 12) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(kernel_size=kernel_size, stride=ratio)

    def __call__(self, x: mx.array) -> mx.array:
        """Decimate by the configured ratio."""
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Alias-free activation: upsample, SnakeBeta, downsample."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(2, kernel_size=12)
        self.downsample = DownSample1d(2, kernel_size=12)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock1(nn.Module):
    """BigVGAN AMP residual block with checkpoint-backed anti-alias filters."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.convs1 = [
            Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation,
            )
            for dilation in dilations
        ]
        self.convs2 = [
            Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            for _ in dilations
        ]
        self.acts1 = [Activation1d(channels) for _ in dilations]
        self.acts2 = [Activation1d(channels) for _ in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2, act1, act2 in zip(
            self.convs1,
            self.convs2,
            self.acts1,
            self.acts2,
            strict=True,
        ):
            x = x + conv2(act2(conv1(act1(x))))
            mx.eval(x)
        return x


__all__ = [
    "AMPBlock1",
    "Activation1d",
    "Conv1d",
    "ConvTranspose1d",
    "DownSample1d",
    "LowPassFilter1d",
    "SnakeBeta",
    "UpSample1d",
]
