"""Native Conv3d video VAE model and loading tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.video_vae.loading as video_vae_loading
from kinomlx.io.safetensors import save_weights
from kinomlx.models.ltx2.video_vae.decoder import (
    NativeConv3dVideoDecoder,
    NativeDepthToSpaceUpsample3d,
    load_native_vae_decoder_weights,
)
from kinomlx.models.ltx2.video_vae.encoder import (
    NativeConv3dVideoEncoder,
    NativeSpaceToDepthDownsample3d,
    load_native_vae_encoder_weights,
)
from kinomlx.models.ltx2.video_vae.loading import load_native_video_vae
from kinomlx.reporting import RecordingReporter

from ._fixtures import mini_vae_config, mini_vae_mapping


def test_space_to_depth_downsample_shape() -> None:
    block = NativeSpaceToDepthDownsample3d(
        in_channels=2,
        out_channels=8,
        stride=(2, 2, 2),
    )
    source = mx.zeros((1, 3, 4, 4, 2))
    output = block(source)
    mx.eval(output)
    assert tuple(output.shape) == (1, 2, 2, 2, 8)


def test_depth_to_space_upsample_shape() -> None:
    block = NativeDepthToSpaceUpsample3d(
        in_channels=8,
        stride=(2, 2, 2),
        out_channels_reduction_factor=2,
    )
    source = mx.zeros((1, 2, 2, 2, 8))
    output = block(source)
    mx.eval(output)
    assert tuple(output.shape) == (1, 3, 4, 4, 4)


def test_miniature_encoder_decoder_shape_dtype_and_reporting() -> None:
    config = mini_vae_config()
    encoder = NativeConv3dVideoEncoder(config, compute_dtype=mx.float32)
    decoder = NativeConv3dVideoDecoder(config, compute_dtype=mx.float32)
    reporter = RecordingReporter()
    video = mx.random.normal((1, 3, 5, 8, 8))

    latent = encoder(video, reporter=reporter)
    decoded = decoder(latent, reporter=reporter)
    mx.eval(latent, decoded)

    assert tuple(latent.shape) == (1, 4, 2, 1, 1)
    assert latent.dtype == mx.float32
    assert tuple(decoded.shape) == tuple(video.shape)
    assert decoded.dtype == mx.float32
    assert mx.all(mx.isfinite(latent)).item()
    assert mx.all(mx.isfinite(decoded)).item()
    assert reporter.events[0] == (
        "start",
        "VAE encode",
        {"total": 10, "unit": "block"},
    )
    assert ("end", "VAE encode", {}) in reporter.events
    assert reporter.events[-1] == ("end", "VAE decode", {})


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 4, 5, 8, 8), "expected BCFHW video"),
        ((1, 3, 4, 8, 8), "frame count must be"),
        ((1, 3, 5, 7, 8), "must be divisible"),
    ],
)
def test_encoder_rejects_incompatible_video_geometry(
    shape: tuple[int, ...],
    message: str,
) -> None:
    encoder = NativeConv3dVideoEncoder(mini_vae_config(), compute_dtype=mx.float32)
    with pytest.raises(ValueError, match=message):
        encoder(mx.zeros(shape))


def test_decoder_accepts_unbatched_latent() -> None:
    decoder = NativeConv3dVideoDecoder(mini_vae_config(), compute_dtype=mx.float32)
    output = decoder(mx.zeros((4, 2, 1, 1)))
    mx.eval(output)
    assert tuple(output.shape) == (1, 3, 5, 8, 8)


def _raw_weight_map(model: Any, family: str) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for prefix, block in model._iter_convs():
        native_weight = block.conv.weight
        raw_weight = mx.full(native_weight.shape, 0.125).transpose(0, 4, 1, 2, 3)
        weights[f"vae.{family}.{prefix}.weight"] = raw_weight
        weights[f"vae.{family}.{prefix}.bias"] = mx.zeros_like(block.conv.bias)
    return weights


def _complete_checkpoint_weights() -> dict[str, mx.array]:
    config = mini_vae_config()
    encoder = NativeConv3dVideoEncoder(config, compute_dtype=mx.float32)
    decoder = NativeConv3dVideoDecoder(config, compute_dtype=mx.float32)
    weights = {
        "vae.per_channel_statistics.mean-of-means": mx.zeros((config.latent_channels,)),
        "vae.per_channel_statistics.std-of-means": mx.ones((config.latent_channels,)),
    }
    weights.update(_raw_weight_map(encoder, "encoder"))
    weights.update(_raw_weight_map(decoder, "decoder"))
    return weights


def _replace_statistics(
    weights: dict[str, mx.array],
    *,
    prefix: str,
    mean_name: str,
    std_name: str,
) -> None:
    mean = weights.pop("vae.per_channel_statistics.mean-of-means")
    std = weights.pop("vae.per_channel_statistics.std-of-means")
    weights[f"{prefix}.{mean_name}"] = mean
    weights[f"{prefix}.{std_name}"] = std


def test_weight_loaders_require_and_assign_every_tensor() -> None:
    config = mini_vae_config()
    weights = _complete_checkpoint_weights()
    encoder = NativeConv3dVideoEncoder(config, compute_dtype=mx.float32)
    decoder = NativeConv3dVideoDecoder(config, compute_dtype=mx.float32)

    encoder_count = load_native_vae_encoder_weights(encoder, weights)
    decoder_count = load_native_vae_decoder_weights(decoder, weights)
    expected_encoder = 2 + 2 * sum(1 for _item in encoder._iter_convs())
    expected_decoder = 2 + 2 * sum(1 for _item in decoder._iter_convs())
    assert encoder_count == expected_encoder
    assert decoder_count == expected_decoder
    assert decoder.load_receipt is not None
    assert decoder.load_receipt.loaded_tensors == expected_decoder
    assert decoder.load_receipt.ignored_decoder_tensors == ()
    assert mx.all(encoder.conv_in.conv.weight == 0.125).item()
    assert mx.all(decoder.conv_out.conv.weight == 0.125).item()


def test_weight_loaders_accept_component_local_video_prefixes() -> None:
    config = mini_vae_config()
    weights = {
        key.removeprefix("vae."): value for key, value in _complete_checkpoint_weights().items()
    }
    encoder = NativeConv3dVideoEncoder(config, compute_dtype=mx.float32)
    decoder = NativeConv3dVideoDecoder(config, compute_dtype=mx.float32)

    encoder_count = load_native_vae_encoder_weights(encoder, weights)
    decoder_count = load_native_vae_decoder_weights(decoder, weights)

    assert encoder_count == 2 + 2 * sum(1 for _item in encoder._iter_convs())
    assert decoder_count == 2 + 2 * sum(1 for _item in decoder._iter_convs())
    assert mx.all(encoder.conv_in.conv.weight == 0.125).item()
    assert mx.all(decoder.conv_out.conv.weight == 0.125).item()


@pytest.mark.parametrize(
    "prefix",
    [
        "vae.per_channel_statistics",
        "vae_encoder.per_channel_statistics",
        "per_channel_statistics",
    ],
)
@pytest.mark.parametrize(
    ("mean_name", "std_name"),
    [
        ("mean-of-means", "std-of-means"),
        ("mean", "std"),
    ],
)
def test_encoder_accepts_all_statistic_key_spellings(
    prefix: str,
    mean_name: str,
    std_name: str,
) -> None:
    config = mini_vae_config()
    weights = _complete_checkpoint_weights()
    _replace_statistics(
        weights,
        prefix=prefix,
        mean_name=mean_name,
        std_name=std_name,
    )
    encoder = NativeConv3dVideoEncoder(config, compute_dtype=mx.float32)
    load_native_vae_encoder_weights(encoder, weights)
    assert mx.array_equal(
        encoder.per_channel_statistics.mean_of_means,
        mx.zeros((config.latent_channels,)),
    ).item()


@pytest.mark.parametrize(
    "prefix",
    [
        "vae.per_channel_statistics",
        "vae_decoder.per_channel_statistics",
        "per_channel_statistics",
    ],
)
@pytest.mark.parametrize(
    ("mean_name", "std_name"),
    [
        ("mean-of-means", "std-of-means"),
        ("mean", "std"),
    ],
)
def test_decoder_accepts_all_statistic_key_spellings(
    prefix: str,
    mean_name: str,
    std_name: str,
) -> None:
    config = mini_vae_config()
    weights = _complete_checkpoint_weights()
    _replace_statistics(
        weights,
        prefix=prefix,
        mean_name=mean_name,
        std_name=std_name,
    )
    decoder = NativeConv3dVideoDecoder(config, compute_dtype=mx.float32)
    load_native_vae_decoder_weights(decoder, weights)
    assert mx.array_equal(
        decoder.per_channel_statistics.std_of_means,
        mx.ones((config.latent_channels,)),
    ).item()


def test_incomplete_weight_load_does_not_mutate_model() -> None:
    encoder = NativeConv3dVideoEncoder(
        mini_vae_config(),
        compute_dtype=mx.float32,
    )
    original_bias = encoder.conv_in.conv.bias
    with pytest.raises(ValueError, match="weights are incomplete"):
        load_native_vae_encoder_weights(encoder, {})
    assert encoder.conv_in.conv.bias is original_bias


def test_checkpoint_loader_builds_matched_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "mini.safetensors"
    save_weights(
        checkpoint,
        _complete_checkpoint_weights(),
        metadata={"config": json.dumps({"vae": mini_vae_mapping()})},
    )
    reporter = RecordingReporter()
    cache_clears: list[bool] = []
    collections: list[bool] = []
    monkeypatch.setattr(
        video_vae_loading.mx,
        "clear_cache",
        lambda: cache_clears.append(True),
    )
    monkeypatch.setattr(
        video_vae_loading.gc,
        "collect",
        lambda: collections.append(True),
    )
    bundle = load_native_video_vae(
        checkpoint,
        compute_dtype=mx.float32,
        reporter=reporter,
    )
    assert bundle.encoder.config is bundle.config
    assert bundle.decoder.config is bundle.config
    assert bundle.config.latent_channels == 4
    assert cache_clears == [True]
    assert collections == [True]
    assert reporter.events == [
        ("start", "load video VAE", {"total": 4, "unit": "step"}),
        ("advance", "load video VAE", {"advance": 1.0}),
        ("advance", "load video VAE", {"advance": 1.0}),
        ("advance", "load video VAE", {"advance": 1.0}),
        ("advance", "load video VAE", {"advance": 1.0}),
        ("end", "load video VAE", {}),
    ]


def test_checkpoint_loader_accepts_explicit_config_for_split_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "video_vae.safetensors"
    save_weights(checkpoint, _complete_checkpoint_weights())
    config = mini_vae_config()
    bundle = load_native_video_vae(
        checkpoint,
        config=config,
        compute_dtype=mx.float32,
    )
    assert bundle.config is config


def test_checkpoint_loader_does_not_take_constructor_authority_from_cache_sidecar(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "video_vae.safetensors"
    save_weights(checkpoint, _complete_checkpoint_weights())
    checkpoint.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "kind": "video_vae",
                "schema_version": 999,
                "config": {"vae": mini_vae_mapping()},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="family caches do not carry constructor authority"):
        load_native_video_vae(checkpoint, compute_dtype=mx.float32)


def test_checkpoint_loader_does_not_mask_invalid_embedded_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "video_vae.safetensors"
    save_weights(
        checkpoint,
        _complete_checkpoint_weights(),
        metadata={"config": "{}"},
    )
    checkpoint.with_suffix(".metadata.json").write_text(
        json.dumps({"kind": "video_vae", "schema_version": 3}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="encoder_blocks must be a sequence"):
        load_native_video_vae(checkpoint, compute_dtype=mx.float32)


def test_checkpoint_loader_releases_weight_map_after_assignment_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "incomplete.safetensors"
    weights = _complete_checkpoint_weights()
    weights.pop("vae.encoder.conv_in.conv.weight")
    save_weights(
        checkpoint,
        weights,
        metadata={"config": json.dumps({"vae": mini_vae_mapping()})},
    )
    collections: list[bool] = []
    cache_clears: list[bool] = []
    monkeypatch.setattr(
        video_vae_loading.gc,
        "collect",
        lambda: collections.append(True),
    )
    monkeypatch.setattr(
        video_vae_loading.mx,
        "clear_cache",
        lambda: cache_clears.append(True),
    )

    with pytest.raises(ValueError, match="encoder weights are incomplete"):
        load_native_video_vae(checkpoint, compute_dtype=mx.float32)
    assert collections == [True]
    assert cache_clears == [True]
