"""Strict audio VAE family loading with idempotent Conv2d layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from kinomlx.io.safetensors import load_weights
from kinomlx.reporting import NullReporter, Reporter

from .blocks import CausalConv2d, ResBlock2d
from .decoder import AudioDecoder
from .encoder import AudioEncoder


@dataclass
class _Target:
    destinations: list[tuple[object, str]]
    shape: tuple[int, ...]
    native_shape: tuple[int, ...] | None = None


def _add(targets: dict[str, _Target], key: str, target: _Target) -> None:
    existing = targets.get(key)
    if existing is None:
        targets[key] = target
        return
    if existing.shape != target.shape or existing.native_shape != target.native_shape:
        raise RuntimeError(f"conflicting audio VAE target declaration: {key}")
    existing.destinations.extend(target.destinations)


def _conv_targets(
    targets: dict[str, _Target],
    prefix: str,
    conv: CausalConv2d,
) -> None:
    _add(
        targets,
        f"{prefix}.weight",
        _Target(
            [(conv, "weight")],
            conv.checkpoint_weight_shape,
            conv.native_weight_shape,
        ),
    )
    _add(
        targets,
        f"{prefix}.bias",
        _Target([(conv, "bias")], (conv.out_channels,)),
    )


def _resblock_targets(
    targets: dict[str, _Target],
    prefix: str,
    block: ResBlock2d,
) -> None:
    _conv_targets(targets, f"{prefix}.conv1.conv", block.conv1)
    _conv_targets(targets, f"{prefix}.conv2.conv", block.conv2)
    if block.skip is not None:
        _conv_targets(targets, f"{prefix}.nin_shortcut.conv", block.skip)


def _statistics_targets(
    targets: dict[str, _Target],
    model: AudioDecoder | AudioEncoder,
) -> None:
    stats = model.per_channel_statistics
    channels = model.config.channels
    _add(
        targets,
        "audio_vae.per_channel_statistics.mean-of-means",
        _Target([(stats, "mean_of_means")], (channels,)),
    )
    _add(
        targets,
        "audio_vae.per_channel_statistics.std-of-means",
        _Target([(stats, "std_of_means")], (channels,)),
    )


def _decoder_targets(model: AudioDecoder) -> dict[str, _Target]:
    targets: dict[str, _Target] = {}
    _conv_targets(targets, "audio_vae.decoder.conv_in.conv", model.conv_in)
    _resblock_targets(targets, "audio_vae.decoder.mid.block_1", model.mid_block_1)
    _resblock_targets(targets, "audio_vae.decoder.mid.block_2", model.mid_block_2)
    for index, stage in enumerate(model.up_blocks):
        checkpoint_level = len(model.up_blocks) - 1 - index
        for block_index, block in enumerate(stage["res_blocks"]):
            _resblock_targets(
                targets,
                f"audio_vae.decoder.up.{checkpoint_level}.block.{block_index}",
                block,
            )
        upsample = stage["upsample"]
        if upsample is not None:
            _conv_targets(
                targets,
                f"audio_vae.decoder.up.{checkpoint_level}.upsample.conv.conv",
                upsample.conv,
            )
    _conv_targets(targets, "audio_vae.decoder.conv_out.conv", model.conv_out)
    _statistics_targets(targets, model)
    return targets


def _encoder_targets(model: AudioEncoder) -> dict[str, _Target]:
    targets: dict[str, _Target] = {}
    _conv_targets(targets, "audio_vae.encoder.conv_in.conv", model.conv_in)
    for level, stage in enumerate(model.down_blocks):
        for block_index, block in enumerate(stage["res_blocks"]):
            _resblock_targets(
                targets,
                f"audio_vae.encoder.down.{level}.block.{block_index}",
                block,
            )
        downsample = stage["downsample"]
        if downsample is not None:
            _conv_targets(
                targets,
                f"audio_vae.encoder.down.{level}.downsample.conv",
                downsample.conv,
            )
    _resblock_targets(targets, "audio_vae.encoder.mid.block_1", model.mid_block_1)
    _resblock_targets(targets, "audio_vae.encoder.mid.block_2", model.mid_block_2)
    _conv_targets(targets, "audio_vae.encoder.conv_out.conv", model.conv_out)
    _statistics_targets(targets, model)
    return targets


def _merge(*groups: dict[str, _Target]) -> dict[str, _Target]:
    merged: dict[str, _Target] = {}
    for group in groups:
        for key, target in group.items():
            _add(merged, key, target)
    return merged


def _load(
    path: Path | str,
    targets: dict[str, _Target],
    *,
    reporter: Reporter | None,
    phase: str,
) -> int:
    sink = reporter if reporter is not None else NullReporter()
    sink.phase_start(phase, total=len(targets), unit="tensor")
    weights: dict[str, mx.array] = {}
    prepared: dict[str, mx.array] = {}
    try:
        weights = load_weights(path)
        bindings: dict[str, str] = {}
        for logical_key in targets:
            aliases = [logical_key]
            if logical_key.startswith("audio_vae."):
                aliases.append(logical_key.removeprefix("audio_vae."))
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
                "unsupported audio VAE checkpoint: "
                f"missing {len(missing)} consumed tensors (first: {missing[0]})"
            )
        for key, target in targets.items():
            source_key = bindings[key]
            value = weights[source_key]
            shape = tuple(value.shape)
            if shape == target.shape and target.native_shape is not None:
                value = mx.contiguous(value.transpose(0, 2, 3, 1))
            elif shape != target.shape and shape != target.native_shape:
                options = [target.shape]
                if target.native_shape is not None:
                    options.append(target.native_shape)
                raise ValueError(
                    f"audio VAE tensor {source_key!r} has shape {shape}, expected one of {options}"
                )
            prepared[key] = value
        for key, target in targets.items():
            value = prepared[key]
            for destination, attribute in target.destinations:
                setattr(destination, attribute, value)
            sink.phase_advance(phase)
        return len(targets)
    finally:
        prepared.clear()
        weights.clear()
        sink.phase_end(phase)


def load_audio_decoder_weights(
    model: AudioDecoder,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Load decoder and statistics from a full audio-family cache."""
    return _load(
        path,
        _decoder_targets(model),
        reporter=reporter,
        phase="load audio VAE decoder",
    )


def load_audio_encoder_weights(
    model: AudioEncoder,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Load encoder and statistics from a full audio-family cache."""
    return _load(
        path,
        _encoder_targets(model),
        reporter=reporter,
        phase="load audio VAE encoder",
    )


def load_audio_vae_weights(
    encoder: AudioEncoder,
    decoder: AudioDecoder,
    path: Path | str,
    *,
    reporter: Reporter | None = None,
) -> int:
    """Load every consumed encoder/decoder/statistics target and ignore extras."""
    if encoder.config != decoder.config:
        raise ValueError("audio encoder and decoder configurations must match")
    return _load(
        path,
        _merge(_encoder_targets(encoder), _decoder_targets(decoder)),
        reporter=reporter,
        phase="load audio VAE",
    )


__all__ = [
    "load_audio_decoder_weights",
    "load_audio_encoder_weights",
    "load_audio_vae_weights",
]
