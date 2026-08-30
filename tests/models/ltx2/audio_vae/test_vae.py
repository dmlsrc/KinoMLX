from __future__ import annotations

import json

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.audio_vae.config as config_module
from kinomlx.models.ltx2.audio_vae import (
    AudioDecoder,
    AudioEncoder,
    AudioVAEConfig,
    load_audio_vae_weights,
)
from kinomlx.models.ltx2.audio_vae.loading import (
    _decoder_targets,
    _encoder_targets,
    _merge,
)
from kinomlx.models.ltx2.patchifier import AudioPatchifier
from kinomlx.models.ltx2.types import AudioLatentShape
from kinomlx.reporting import RecordingReporter


def _config() -> AudioVAEConfig:
    return AudioVAEConfig(
        channels=8,
        input_channels=2,
        output_channels=2,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        latent_channels=2,
        mel_bins=8,
        sample_rate=16000,
        hop_length=160,
        n_fft=16,
    )


def _weights(encoder: AudioEncoder, decoder: AudioDecoder) -> dict[str, mx.array]:
    targets = _merge(_encoder_targets(encoder), _decoder_targets(decoder))
    result = {}
    for key, target in targets.items():
        if key.endswith("std-of-means"):
            value = mx.ones(target.shape)
        elif key.endswith(("mean-of-means", ".bias")):
            value = mx.zeros(target.shape)
        else:
            value = mx.random.normal(target.shape) * 0.02
        result[key] = value
    return result


def test_audio_patchifier_round_trip_and_causal_timing() -> None:
    patchifier = AudioPatchifier()
    shape = AudioLatentShape(1, 2, 3, 4)
    latent = mx.arange(24, dtype=mx.float32).reshape(shape.to_tuple())
    tokens = patchifier.patchify(latent)
    restored = patchifier.unpatchify(tokens, shape)
    bounds = patchifier.get_patch_grid_bounds(shape)
    mx.eval(restored, bounds)
    assert tuple(tokens.shape) == (1, 3, 8)
    assert mx.array_equal(restored, latent).item()
    assert tuple(bounds.shape) == (1, 1, 3, 2)
    assert bounds[0, 0, 0].tolist() == pytest.approx([0.0, 0.01])


def test_audio_vae_loads_consumed_targets_and_ignores_baggage(tmp_path) -> None:
    config = _config()
    source_encoder = AudioEncoder(config)
    source_decoder = AudioDecoder(config)
    weights = _weights(source_encoder, source_decoder)
    consumed = len(weights)
    weights["community.wrapper.unused"] = mx.zeros((1,))
    path = tmp_path / "audio_vae.safetensors"
    mx.save_safetensors(str(path), weights)

    encoder = AudioEncoder(config, compute_dtype=mx.float32)
    decoder = AudioDecoder(config, compute_dtype=mx.float32)
    reporter = RecordingReporter()
    assert load_audio_vae_weights(encoder, decoder, path, reporter=reporter) == consumed
    spectrogram = mx.random.normal((1, 2, 7, 8))
    latent = encoder(spectrogram, reporter=reporter)
    decoded = decoder(latent, reporter=reporter)
    mx.eval(latent, decoded)
    assert tuple(latent.shape) == (1, 2, 4, 4)
    assert latent.dtype == mx.float32
    assert tuple(decoded.shape) == (1, 2, 7, 8)
    assert mx.all(mx.isfinite(decoded)).item()
    assert [event[:2] for event in reporter.events if event[0] in {"start", "end"}] == [
        ("start", "load audio VAE"),
        ("end", "load audio VAE"),
        ("start", "audio VAE encode"),
        ("end", "audio VAE encode"),
        ("start", "audio VAE decode"),
        ("end", "audio VAE decode"),
    ]


def test_audio_vae_loader_rejects_partial_family(tmp_path) -> None:
    path = tmp_path / "audio_vae.safetensors"
    mx.save_safetensors(
        str(path),
        {"audio_vae.per_channel_statistics.std-of-means": mx.ones((8,))},
    )
    reporter = RecordingReporter()
    with pytest.raises(ValueError, match="missing"):
        load_audio_vae_weights(
            AudioEncoder(_config()),
            AudioDecoder(_config()),
            path,
            reporter=reporter,
        )
    assert reporter.events[-1] == ("end", "load audio VAE", {})


def test_audio_vae_loader_preflights_every_shape_before_mutation(tmp_path) -> None:
    config = _config()
    source_encoder = AudioEncoder(config)
    source_decoder = AudioDecoder(config)
    weights = _weights(source_encoder, source_decoder)
    weights["audio_vae.decoder.conv_out.conv.weight"] = mx.zeros((1,))
    path = tmp_path / "wrong-shape.safetensors"
    mx.save_safetensors(str(path), weights)

    encoder = AudioEncoder(config)
    decoder = AudioDecoder(config)
    original = encoder.conv_in.weight
    with pytest.raises(ValueError, match="decoder.conv_out.conv.weight.*shape"):
        load_audio_vae_weights(encoder, decoder, path)
    assert encoder.conv_in.weight is original


@pytest.mark.parametrize("flag", ["give_pre_end", "tanh_out"])
def test_audio_vae_config_rejects_unsupported_decoder_modes(monkeypatch, flag: str) -> None:
    ddconfig = {
        "ch": 8,
        "in_channels": 2,
        "out_ch": 2,
        "ch_mult": [1, 2],
        "num_res_blocks": 1,
        "z_channels": 2,
        "double_z": True,
        "mel_bins": 8,
        "causality_axis": "height",
        "norm_type": "pixel",
        flag: True,
    }
    metadata = {
        "audio_vae": {
            "model": {"params": {"ddconfig": ddconfig}},
            "preprocessing": {
                "stft": {"hop_length": 160, "filter_length": 1024, "causal": True},
                "mel": {"n_mel_channels": 8},
            },
        }
    }
    monkeypatch.setattr(
        config_module,
        "read_metadata",
        lambda _path: {"config": json.dumps(metadata)},
    )
    with pytest.raises(ValueError, match=flag):
        AudioVAEConfig.from_checkpoint("checkpoint.safetensors")


def test_audio_vae_config_ignores_training_only_metadata(monkeypatch) -> None:
    metadata = {
        "audio_vae": {
            "model": {
                "params": {
                    "ddconfig": {
                        "ch": 8,
                        "in_channels": 2,
                        "out_ch": 2,
                        "ch_mult": [1, 2],
                        "num_res_blocks": 1,
                        "z_channels": 2,
                        "double_z": True,
                        "mel_bins": 8,
                        "causality_axis": "height",
                        "norm_type": "pixel",
                        "dropout": 0.5,
                        "future_training_note": "ignored",
                    }
                }
            },
            "preprocessing": {
                "stft": {"hop_length": 160, "filter_length": 1024, "causal": True},
                "mel": {"n_mel_channels": 8},
            },
        }
    }
    monkeypatch.setattr(
        config_module,
        "read_metadata",
        lambda _path: {"config": json.dumps(metadata)},
    )

    assert AudioVAEConfig.from_checkpoint("checkpoint.safetensors").channels == 8


@pytest.mark.parametrize(
    ("is_causal", "causality_axis"),
    [(True, "none"), (False, "height")],
)
def test_audio_vae_config_rejects_inconsistent_causality(
    is_causal: bool,
    causality_axis: str,
) -> None:
    with pytest.raises(ValueError, match="STFT and convolution causality disagree"):
        AudioVAEConfig(is_causal=is_causal, causality_axis=causality_axis)
