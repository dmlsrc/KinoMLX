"""MLX GMNet architecture and checkpoint-exact weight loading.

The module structure follows the published GMNet generator (Liao et al.,
ICLR 2025): a global branch that squeezes a 256x256 SDR thumbnail into
per-channel 3x3 kernels, a channel gate, and a Qmax scalar; and a local
branch that runs the (optionally downscaled) SDR image at half
resolution through three residual stages modulated by those global
predictions, then PixelShuffles back up to a one-channel gain map.

Layout is MLX-native NHWC. Converted safetensors keep the upstream
key names and OIHW conv layout; :func:`load_gmnet_weights` remaps keys
and transposes conv weights at load time (remappings live as data in
:func:`checkpoint_key_map`).
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

import kinomlx._mlx_nn as nn
from kinomlx.io.safetensors import load_weights_with_metadata

THUMBNAIL_SIZE = 256
FEATURES = 64
_RESIDUAL_BLOCKS_PER_STAGE = 5
_KERNEL_GRID = 3


class _DownConv(nn.Module):
    """Stride-2 entry conv plus two refining convs, ReLU after each."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convs = [
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.convs:
            x = nn.relu(conv(x))
        return x


class _ResidualBlock(nn.Module):
    """Conv-ReLU-Conv with an identity skip and no normalization."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.conv2(nn.relu(self.conv1(x)))


class _SqueezeHead(nn.Module):
    """Adaptive average pool to a fixed grid, then a 1x1 bottleneck stack."""

    def __init__(self, size: int, out_channels: int, features: int) -> None:
        super().__init__()
        self.size = size
        self.convs = [
            nn.Conv2d(features, features * 2, 1),
            nn.Conv2d(features * 2, features, 1),
            nn.Conv2d(features, out_channels, 1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = adaptive_average_pool(x, self.size)
        x = nn.relu(self.convs[0](x))
        x = nn.relu(self.convs[1](x))
        return self.convs[2](x)


def adaptive_average_pool(x: mx.array, size: int) -> mx.array:
    """Average NHWC input into a ``size x size`` grid with reference bin edges.

    Bin ``i`` covers rows ``floor(i * H / size)`` through
    ``ceil((i + 1) * H / size)``, matching the adaptive pooling the upstream
    squeeze heads were trained with (bins may overlap when ``size`` does not
    divide the extent).
    """
    _, height, width, _ = x.shape
    rows = []
    for i in range(size):
        row_start = (i * height) // size
        row_end = math.ceil((i + 1) * height / size)
        columns = []
        for j in range(size):
            column_start = (j * width) // size
            column_end = math.ceil((j + 1) * width / size)
            cell = x[:, row_start:row_end, column_start:column_end, :]
            columns.append(mx.mean(cell, axis=(1, 2)))
        rows.append(mx.stack(columns, axis=1))
    return mx.stack(rows, axis=1)


def pixel_shuffle(x: mx.array, factor: int) -> mx.array:
    """Rearrange NHWC channels into space with PyTorch PixelShuffle ordering."""
    batch, height, width, channels = x.shape
    if channels % (factor * factor):
        raise ValueError(f"cannot shuffle {channels} channels by factor {factor}")
    out_channels = channels // (factor * factor)
    x = x.reshape(batch, height, width, out_channels, factor, factor)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(batch, height * factor, width * factor, out_channels)


def per_channel_conv3x3(x: mx.array, kernels: mx.array) -> mx.array:
    """Depthwise 3x3 cross-correlation with data-dependent kernels.

    ``kernels`` is the ``(B, 3, 3, C)`` squeeze-head output; entry
    ``(b, dy, dx, c)`` weights the tap at spatial offset ``(dy - 1, dx - 1)``
    of channel ``c``, zero-padded at the border. Expressed as nine shifted
    multiplies because a grouped ``mx.conv2d`` with ``groups == C`` is far
    off the memory-bandwidth floor on this class of hardware.
    """
    _, height, width, _ = x.shape
    padded = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    taps = [
        padded[:, dy : dy + height, dx : dx + width, :] * kernels[:, dy : dy + 1, dx : dx + 1, :]
        for dy in range(_KERNEL_GRID)
        for dx in range(_KERNEL_GRID)
    ]
    total = taps[0]
    for tap in taps[1:]:
        total = total + tap
    return total


class GMNet(nn.Module):
    """The GMNet generator: SDR image plus thumbnail to gain map and Qmax."""

    def __init__(self, features: int = FEATURES, gain_channels: int = 1) -> None:
        super().__init__()
        blocks = _RESIDUAL_BLOCKS_PER_STAGE

        # Global branch over the 256x256 thumbnail.
        self.down1 = _DownConv(3, features // 4)
        self.down2 = _DownConv(features // 4, features)
        self.res_y = [_ResidualBlock(features) for _ in range(blocks)]
        self.sq_ker = _SqueezeHead(_KERNEL_GRID, features, features)
        self.sq_chn = _SqueezeHead(1, features, features)
        self.sq_qmax = _SqueezeHead(1, 1, features)
        self.mask_est = [
            nn.Conv2d(features, features, 3, stride=1, padding=1),
            nn.Conv2d(features, 1, 3, stride=1, padding=1),
        ]

        # Local branch over the full input at half resolution.
        self.down_x = _DownConv(3, features)
        self.res1 = [_ResidualBlock(features) for _ in range(blocks)]
        self.res2 = [_ResidualBlock(features) for _ in range(blocks)]
        self.res3 = [_ResidualBlock(features) for _ in range(blocks)]

        # Tail: PixelShuffle x2 back to input resolution.
        self.upconv = nn.Conv2d(features, 4 * features, 3, stride=1, padding=1)
        self.hr_conv = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(features, gain_channels, 3, stride=1, padding=1)

    def __call__(
        self,
        image: mx.array,
        thumbnail: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return ``(gain_map, qmax * gain_map, qmax)`` for NHWC ``[0, 1]`` inputs.

        ``gain_map`` is the normalized gain-map prediction and the second
        output is it scaled by the predicted normalized Qmax, both shaped
        ``(B, H', W', 1)`` where ``H'``/``W'`` round the input extents up to
        even values; ``qmax`` is the ``(B, 1, 1, 1)`` scalar head. The
        upstream forward returns only the first two - the scalar is exposed
        here so callers can record it without re-deriving it.
        """
        y = self.down2(self.down1(thumbnail))
        for block in self.res_y:
            y = block(y)
        kernels = self.sq_ker(y)
        channel_gate = self.sq_chn(y)
        qmax = self.sq_qmax(y)

        x = self.down_x(image)
        for block in self.res1:
            x = block(x)
        mask = per_channel_conv3x3(x, kernels)
        mask = mx.sigmoid(self.mask_est[1](nn.relu(self.mask_est[0](mask))))
        x = x * mask
        for block in self.res2:
            x = block(x)
        x = x * mx.sigmoid(channel_gate)
        for block in self.res3:
            x = block(x)
        out = nn.relu(pixel_shuffle(self.upconv(x), 2))
        out = self.conv_last(nn.relu(self.hr_conv(out)))
        return out, qmax * out, qmax


def checkpoint_key_map() -> dict[str, str]:
    """Upstream checkpoint key stems mapped to MLX module paths.

    Stems omit the ``.weight`` / ``.bias`` leaf. Upstream ``nn.Sequential``
    indices skip activation slots (0, 2, 4), which collapse to dense list
    indices here; ``HRconv`` renames to the convention-conforming
    ``hr_conv``.
    """
    mapping: dict[str, str] = {}
    sequential = ((0, 0), (2, 1), (4, 2))
    for stack in ("down1", "down2", "down_x"):
        for torch_index, mlx_index in sequential:
            mapping[f"{stack}.conv.{torch_index}"] = f"{stack}.convs.{mlx_index}"
    for head in ("sq_ker", "sq_chn", "sq_qmax"):
        for torch_index, mlx_index in sequential:
            mapping[f"{head}.conv.{torch_index}"] = f"{head}.convs.{mlx_index}"
    for stage in ("res_y", "res1", "res2", "res3"):
        for block in range(_RESIDUAL_BLOCKS_PER_STAGE):
            for conv in ("conv1", "conv2"):
                mapping[f"{stage}.{block}.{conv}"] = f"{stage}.{block}.{conv}"
    mapping["mask_est.0"] = "mask_est.0"
    mapping["mask_est.2"] = "mask_est.1"
    mapping["upconv"] = "upconv"
    mapping["HRconv"] = "hr_conv"
    mapping["conv_last"] = "conv_last"
    return mapping


EXPECTED_CHECKPOINT_KEYS = frozenset(
    f"{stem}.{leaf}" for stem in checkpoint_key_map() for leaf in ("weight", "bias")
)


def load_gmnet_weights(path: Path | str) -> tuple[GMNet, dict[str, str]]:
    """Build a GMNet and load one converted safetensors checkpoint into it.

    The file must carry exactly the upstream generator state dict (126
    tensors, upstream names, OIHW conv layout); anything missing or extra
    fails before construction. Returns the evaluated model and the file's
    string metadata.
    """
    weights, metadata = load_weights_with_metadata(path)
    provided = frozenset(weights)
    missing = sorted(EXPECTED_CHECKPOINT_KEYS - provided)
    unexpected = sorted(provided - EXPECTED_CHECKPOINT_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"{path} is not a GMNet generator checkpoint: "
            f"missing {missing[:4]}{'...' if len(missing) > 4 else ''}, "
            f"unexpected {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}"
        )

    mapping = checkpoint_key_map()
    converted: list[tuple[str, mx.array]] = []
    for key, value in weights.items():
        if value.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
            raise ValueError(
                f"{path}: GMNet weight {key} has unsupported dtype {value.dtype}; "
                "generator weights must be floating-point"
            )
        stem, _, leaf = key.rpartition(".")
        tensor = value.astype(mx.float32)
        if leaf == "weight":
            if tensor.ndim != 4:
                raise ValueError(f"{path}: conv weight {key} must be 4-D, got {tensor.ndim}-D")
            tensor = tensor.transpose(0, 2, 3, 1)
        converted.append((f"{mapping[stem]}.{leaf}", tensor))

    model = GMNet()
    model.load_weights(converted, strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, metadata


__all__ = [
    "EXPECTED_CHECKPOINT_KEYS",
    "FEATURES",
    "GMNet",
    "THUMBNAIL_SIZE",
    "adaptive_average_pool",
    "checkpoint_key_map",
    "load_gmnet_weights",
    "per_channel_conv3x3",
    "pixel_shuffle",
]
