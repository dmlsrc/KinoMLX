"""Complete consumed-target loading for the LTX-2 BWE vocoder family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx

from kinomlx.io.safetensors import load_weights
from kinomlx.reporting import NullReporter, Reporter

from .vocoder import Vocoder, VocoderWithBWE
from .vocoder_layers import (
    Activation1d,
    AMPBlock1,
    Conv1d,
    ConvTranspose1d,
    LowPassFilter1d,
    UpSample1d,
)


@dataclass(frozen=True)
class _Target:
    destination: object
    attribute: str
    shape: tuple[int, ...]
    layout: Literal["identity", "conv", "transpose"] = "identity"
    marks_filter: bool = False


def _conv1d_targets(
    targets: dict[str, _Target],
    prefix: str,
    conv: Conv1d,
) -> None:
    targets[f"{prefix}.weight"] = _Target(
        conv,
        "weight",
        conv.checkpoint_weight_shape,
        "conv",
    )
    if conv.use_bias:
        targets[f"{prefix}.bias"] = _Target(conv, "bias", (conv.out_channels,))


def _transpose_targets(
    targets: dict[str, _Target],
    prefix: str,
    conv: ConvTranspose1d,
) -> None:
    targets[f"{prefix}.weight"] = _Target(
        conv,
        "weight",
        conv.checkpoint_weight_shape,
        "transpose",
    )
    targets[f"{prefix}.bias"] = _Target(conv, "bias", (conv.out_channels,))


def _activation_targets(
    targets: dict[str, _Target],
    prefix: str,
    activation: Activation1d,
) -> None:
    channels = activation.act.alpha.shape[0]
    targets[f"{prefix}.act.alpha"] = _Target(activation.act, "alpha", (channels,))
    targets[f"{prefix}.act.beta"] = _Target(activation.act, "beta", (channels,))
    targets[f"{prefix}.upsample.filter"] = _Target(
        activation.upsample,
        "filter",
        (1, 1, activation.upsample.kernel_size),
        marks_filter=True,
    )
    targets[f"{prefix}.downsample.lowpass.filter"] = _Target(
        activation.downsample.lowpass,
        "filter",
        (1, 1, activation.downsample.lowpass.kernel_size),
        marks_filter=True,
    )


def _block_targets(
    targets: dict[str, _Target],
    prefix: str,
    block: AMPBlock1,
) -> None:
    for index, conv in enumerate(block.convs1):
        _conv1d_targets(targets, f"{prefix}.convs1.{index}", conv)
    for index, conv in enumerate(block.convs2):
        _conv1d_targets(targets, f"{prefix}.convs2.{index}", conv)
    for index, activation in enumerate(block.acts1):
        _activation_targets(targets, f"{prefix}.acts1.{index}", activation)
    for index, activation in enumerate(block.acts2):
        _activation_targets(targets, f"{prefix}.acts2.{index}", activation)


def _vocoder_targets(model: Vocoder, prefix: str) -> dict[str, _Target]:
    targets: dict[str, _Target] = {}
    _conv1d_targets(targets, f"{prefix}.conv_pre", model.conv_pre)
    for index, upsample in enumerate(model.ups):
        _transpose_targets(targets, f"{prefix}.ups.{index}", upsample)
    for index, block in enumerate(model.resblocks):
        _block_targets(targets, f"{prefix}.resblocks.{index}", block)
    _activation_targets(targets, f"{prefix}.act_post", model.act_post)
    _conv1d_targets(targets, f"{prefix}.conv_post", model.conv_post)
    return targets


def _targets(model: VocoderWithBWE) -> dict[str, _Target]:
    targets = _vocoder_targets(model.vocoder, "vocoder.vocoder")
    targets.update(_vocoder_targets(model.bwe_generator, "vocoder.bwe_generator"))
    stft = model.mel_stft.stft_fn
    targets["vocoder.mel_stft.stft_fn.forward_basis"] = _Target(
        stft,
        "forward_basis",
        stft.basis_shape,
    )
    targets["vocoder.mel_stft.stft_fn.inverse_basis"] = _Target(
        stft,
        "inverse_basis",
        stft.basis_shape,
    )
    targets["vocoder.mel_stft.mel_basis"] = _Target(
        model.mel_stft,
        "mel_basis",
        model.mel_stft.mel_basis_shape,
    )
    return targets


def load_vocoder_weights(
    model: VocoderWithBWE,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Load every consumed vocoder tensor while ignoring unrelated baggage."""
    targets = _targets(model)
    sink = reporter if reporter is not None else NullReporter()
    phase = "load audio vocoder"
    sink.phase_start(phase, total=len(targets), unit="tensor")
    weights: dict[str, mx.array] = {}
    prepared: dict[str, mx.array] = {}
    try:
        weights = load_weights(path)
        bindings: dict[str, str] = {}
        for logical_key in targets:
            aliases = [logical_key]
            if logical_key.startswith("vocoder."):
                aliases.append(logical_key.removeprefix("vocoder."))
            matches = [alias for alias in aliases if alias in weights]
            if not matches:
                for alias in aliases:
                    matches.extend(sorted(name for name in weights if name.endswith(f".{alias}")))
                    if matches:
                        break
            if matches:
                bindings[logical_key] = matches[0]
        missing = sorted(targets.keys() - bindings.keys())
        if missing:
            raise ValueError(
                "unsupported BWE vocoder checkpoint: "
                f"missing {len(missing)} consumed tensors (first: {missing[0]})"
            )
        for key, target in targets.items():
            source_key = bindings[key]
            value = weights[source_key]
            if tuple(value.shape) != target.shape:
                raise ValueError(
                    f"vocoder tensor {source_key!r} has shape {tuple(value.shape)}, "
                    f"expected {target.shape}"
                )
            if target.layout == "conv":
                value = mx.contiguous(value.transpose(0, 2, 1))
            elif target.layout == "transpose":
                value = mx.contiguous(value.transpose(1, 2, 0))
            prepared[key] = value.astype(mx.float32)
        for key, target in targets.items():
            setattr(target.destination, target.attribute, prepared[key])
            if target.marks_filter:
                filter_destination = cast(
                    UpSample1d | LowPassFilter1d,
                    target.destination,
                )
                filter_destination.checkpoint_filter_loaded = True
            sink.phase_advance(phase)
        return len(targets)
    finally:
        prepared.clear()
        weights.clear()
        sink.phase_end(phase)


__all__ = ["load_vocoder_weights"]
