from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.audio_vae import (
    BWEVocoderConfig,
    VocoderConfig,
    VocoderWithBWE,
    load_vocoder_weights,
)
from kinomlx.models.ltx2.audio_vae.vocoder_loading import _targets
from kinomlx.reporting import RecordingReporter


def _generator(
    *,
    rate: int,
    kernel: int,
    sample_rate: int,
    final_activation: bool,
) -> VocoderConfig:
    return VocoderConfig(
        resblock_kernel_sizes=(3,),
        upsample_rates=(rate,),
        upsample_kernel_sizes=(kernel,),
        resblock_dilation_sizes=((1,),),
        upsample_initial_channels=8,
        output_sample_rate=sample_rate,
        mel_bins=4,
        apply_final_activation=final_activation,
    )


def _config() -> BWEVocoderConfig:
    return BWEVocoderConfig(
        vocoder=_generator(
            rate=2,
            kernel=4,
            sample_rate=8000,
            final_activation=True,
        ),
        generator=_generator(
            rate=4,
            kernel=8,
            sample_rate=16000,
            final_activation=False,
        ),
        input_sample_rate=8000,
        output_sample_rate=16000,
        n_fft=8,
        hop_length=2,
        mel_bins=4,
    )


def test_vocoder_direct_defaults_match_reference_constructor() -> None:
    config = _generator(
        rate=2,
        kernel=4,
        sample_rate=8000,
        final_activation=True,
    )
    assert config.use_tanh_at_final is True
    assert config.use_bias_at_final is True


def _weights(model: VocoderWithBWE) -> dict[str, mx.array]:
    result = {}
    for key, target in _targets(model).items():
        if key.endswith(".filter"):
            value = mx.ones(target.shape) / target.shape[-1]
        elif key.endswith("mel_basis"):
            value = mx.ones(target.shape) * 0.1
        elif key.endswith(("forward_basis", "inverse_basis")):
            value = mx.random.normal(target.shape) * 0.05
        elif key.endswith((".alpha", ".beta", ".bias")):
            value = mx.zeros(target.shape)
        else:
            value = mx.random.normal(target.shape) * 0.02
        result[key] = value
    return result


def test_bwe_vocoder_loads_consumed_targets_and_ignores_baggage(tmp_path) -> None:
    model = VocoderWithBWE(_config())
    weights = _weights(model)
    consumed = len(weights)
    weights["community.wrapper.unused"] = mx.zeros((1,))
    path = tmp_path / "vocoder.safetensors"
    mx.save_safetensors(str(path), weights)
    reporter = RecordingReporter()
    assert load_vocoder_weights(model, path, reporter=reporter) == consumed

    mel = mx.random.normal((1, 2, 2, 4)).astype(mx.bfloat16)
    waveform = model(mel, reporter=reporter)
    mx.eval(waveform)
    assert tuple(waveform.shape) == (1, 2, 8)
    assert waveform.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(waveform)).item()
    assert mx.any(waveform != 0).item()
    assert [event[:2] for event in reporter.events if event[0] in {"start", "end"}] == [
        ("start", "load audio vocoder"),
        ("end", "load audio vocoder"),
        ("start", "vocode audio with bandwidth extension"),
        ("start", "synthesize base audio"),
        ("end", "synthesize base audio"),
        ("start", "synthesize bandwidth extension"),
        ("end", "synthesize bandwidth extension"),
        ("end", "vocode audio with bandwidth extension"),
    ]


def test_bwe_vocoder_loader_requires_every_filter(tmp_path) -> None:
    model = VocoderWithBWE(_config())
    weights = _weights(model)
    weights.pop(next(key for key in weights if key.endswith(".filter")))
    path = tmp_path / "vocoder.safetensors"
    mx.save_safetensors(str(path), weights)
    reporter = RecordingReporter()
    with pytest.raises(ValueError, match="missing"):
        load_vocoder_weights(model, path, reporter=reporter)
    assert reporter.events[-1] == ("end", "load audio vocoder", {})


def test_bwe_vocoder_loader_preflights_every_shape_before_mutation(tmp_path) -> None:
    source = VocoderWithBWE(_config())
    weights = _weights(source)
    weights["vocoder.bwe_generator.conv_post.weight"] = mx.zeros((1,))
    path = tmp_path / "wrong-shape.safetensors"
    mx.save_safetensors(str(path), weights)

    target = VocoderWithBWE(_config())
    original = target.vocoder.conv_pre.weight
    with pytest.raises(ValueError, match="bwe_generator.conv_post.weight.*shape"):
        load_vocoder_weights(target, path)
    assert target.vocoder.conv_pre.weight is original


def test_bwe_config_rejects_cross_component_rate_or_mel_mismatch() -> None:
    config = _config()
    with pytest.raises(ValueError, match="base vocoder output sample rate"):
        BWEVocoderConfig(
            vocoder=_generator(
                rate=2,
                kernel=4,
                sample_rate=16000,
                final_activation=True,
            ),
            generator=config.generator,
            input_sample_rate=config.input_sample_rate,
            output_sample_rate=config.output_sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            mel_bins=config.mel_bins,
        )
    with pytest.raises(ValueError, match="mel bins"):
        BWEVocoderConfig(
            vocoder=config.vocoder,
            generator=replace(config.generator, mel_bins=config.mel_bins * 2),
            input_sample_rate=config.input_sample_rate,
            output_sample_rate=config.output_sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            mel_bins=config.mel_bins,
        )
