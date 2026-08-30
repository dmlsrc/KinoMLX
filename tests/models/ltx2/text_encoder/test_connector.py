from __future__ import annotations

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import kinomlx.models.ltx2.text_encoder.encoder as encoder_module
from kinomlx.models.ltx2.text_encoder import (
    AudioVideoGemmaTextEncoderModel,
    AVTextEncoderConfig,
    create_av_text_encoder_v2,
    load_av_text_encoder_v2_weights,
    norm_and_concat_per_token_rms,
)
from kinomlx.reporting import RecordingReporter

from ._synthetic import initialize_test_parameters


def _config() -> AVTextEncoderConfig:
    return AVTextEncoderConfig(
        hidden_dim=4,
        num_gemma_states=2,
        video_inner_dim=8,
        audio_inner_dim=4,
        video_heads=2,
        video_head_dim=4,
        audio_heads=1,
        audio_head_dim=4,
        num_layers=1,
        num_registers=2,
        positional_max=(64,),
        double_precision_rope=False,
    )


def _synthetic_model(
    config: AVTextEncoderConfig | None = None,
) -> AudioVideoGemmaTextEncoderModel:
    model = create_av_text_encoder_v2(_config() if config is None else config)
    initialize_test_parameters(model)
    return model


def _checkpoint_key(key: str) -> str:
    if key.startswith("feature_extractor."):
        return key.replace("feature_extractor.", "text_embedding_projection.", 1)
    if key.startswith("embeddings_connector."):
        key = key.replace(
            "embeddings_connector.",
            "model.diffusion_model.video_embeddings_connector.",
            1,
        )
    elif key.startswith("audio_embeddings_connector."):
        key = key.replace(
            "audio_embeddings_connector.",
            "model.diffusion_model.audio_embeddings_connector.",
            1,
        )
    key = key.replace(".attn1.to_out.", ".attn1.to_out.0.")
    key = key.replace(".ff.project_in.proj.", ".ff.net.0.proj.")
    return key.replace(".ff.project_out.", ".ff.net.2.")


def test_text_model_construction_does_not_draw_random_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_random(*_args, **_kwargs):
        raise AssertionError("text-model construction must not draw random values")

    monkeypatch.setattr(mx.random, "uniform", unexpected_random)
    monkeypatch.setattr(mx.random, "normal", unexpected_random)

    shell = create_av_text_encoder_v2(_config())

    assert tuple(shell.embeddings_connector.learnable_registers.shape) == (0, 0)


def test_per_token_rms_normalizes_each_layer_and_zeros_padding() -> None:
    encoded = mx.array([[[[3.0, 0.0], [4.0, 2.0]], [[6.0, 1.0], [8.0, 1.0]]]])
    mask = mx.array([[1, 0]], dtype=mx.int32)
    result = norm_and_concat_per_token_rms(encoded, mask)
    mx.eval(result)
    assert tuple(result.shape) == (1, 2, 4)
    assert mx.allclose(result[0, 0], mx.array([0.848528, 0.0, 1.131371, 1.414213])).item()
    assert mx.all(result[0, 1] == 0).item()


def test_connector_extends_real_tokens_to_full_context() -> None:
    model = _synthetic_model()
    states = (
        mx.random.normal((1, 3, 4)),
        mx.random.normal((1, 3, 4)),
    )
    reporter = RecordingReporter()
    output = model(
        states,
        mx.ones((1, 3), dtype=mx.int32),
        reporter=reporter,
    )
    mx.eval(output.video_encoding, output.audio_encoding, output.attention_mask)
    assert tuple(output.video_encoding.shape) == (1, 1024, 8)
    assert tuple(output.audio_encoding.shape) == (1, 1024, 4)
    assert tuple(output.attention_mask.shape) == (1, 1024)
    assert mx.all(output.attention_mask == 1).item()
    assert mx.all(mx.isfinite(output.video_encoding)).item()
    assert reporter.events[0] == (
        "start",
        "project audio/video text context",
        {"total": 3, "unit": "stage"},
    )
    assert reporter.events[-1] == (
        "end",
        "project audio/video text context",
        {},
    )


def test_public_connector_rejects_padded_attention_masks_explicitly() -> None:
    model = _synthetic_model()
    states = (
        mx.random.normal((1, 3, 4)),
        mx.random.normal((1, 3, 4)),
    )

    with pytest.raises(ValueError, match="padded attention masks.*trim padding"):
        model(states, mx.array([[0, 1, 1]], dtype=mx.int32))


def test_audio_connector_can_use_an_independent_layer_count() -> None:
    config = AVTextEncoderConfig(
        hidden_dim=4,
        num_gemma_states=2,
        video_inner_dim=8,
        audio_inner_dim=4,
        video_heads=2,
        video_head_dim=4,
        audio_heads=1,
        audio_head_dim=4,
        num_layers=1,
        audio_num_layers=2,
        num_registers=2,
        positional_max=(64,),
        double_precision_rope=False,
    )
    model = _synthetic_model(config)
    assert len(model.embeddings_connector.transformer_1d_blocks) == 1
    assert len(model.audio_embeddings_connector.transformer_1d_blocks) == 2


def test_checkpoint_config_honors_independent_audio_layer_count(monkeypatch) -> None:
    transformer = {
        "caption_channels": 3840,
        "cross_attention_dim": 4096,
        "audio_cross_attention_dim": 2048,
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 64,
        "connector_num_layers": 8,
        "audio_connector_num_layers": 3,
    }
    monkeypatch.setattr(
        encoder_module,
        "read_metadata",
        lambda _path: {"config": json.dumps({"transformer": transformer})},
    )
    config = AVTextEncoderConfig.from_checkpoint("checkpoint.safetensors")
    assert config.num_layers == 8
    assert config.audio_num_layers == 3


def test_checkpoint_config_uses_video_head_dim_as_legacy_audio_fallback(monkeypatch) -> None:
    transformer = {
        "connector_attention_head_dim": 96,
        "cross_attention_dim": 3072,
        "audio_cross_attention_dim": 3072,
    }
    monkeypatch.setattr(
        encoder_module,
        "read_metadata",
        lambda _path: {"config": json.dumps({"transformer": transformer})},
    )

    config = AVTextEncoderConfig.from_checkpoint("checkpoint.safetensors")

    assert config.video_head_dim == 96
    assert config.audio_head_dim == 96


def test_checkpoint_config_fallbacks_match_ltx23_v2_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        encoder_module,
        "read_metadata",
        lambda _path: {"config": json.dumps({"transformer": {}})},
    )
    config = AVTextEncoderConfig.from_checkpoint("checkpoint.safetensors")
    assert config == AVTextEncoderConfig(audio_num_layers=8)


@pytest.mark.parametrize("spelling", ["float64", "FLOAT64", "Float64"])
def test_checkpoint_config_normalizes_frequency_precision_case(
    monkeypatch,
    spelling: str,
) -> None:
    monkeypatch.setattr(
        encoder_module,
        "read_metadata",
        lambda _path: {"config": json.dumps({"transformer": {"frequencies_precision": spelling}})},
    )

    config = AVTextEncoderConfig.from_checkpoint("checkpoint.safetensors")

    assert config.double_precision_rope is True


def test_connector_loader_round_trips_small_model_and_ignores_baggage(tmp_path) -> None:
    source = _synthetic_model()
    weights = {_checkpoint_key(key): value for key, value in tree_flatten(source.parameters())}
    consumed = len(weights)
    weights["community.wrapper.unused"] = mx.zeros((1,))
    path = tmp_path / "connector.safetensors"
    mx.save_safetensors(str(path), weights)

    target = create_av_text_encoder_v2(_config())
    reporter = RecordingReporter()
    assert load_av_text_encoder_v2_weights(target, path, reporter=reporter) == consumed
    states = (mx.random.normal((1, 2, 4)), mx.random.normal((1, 2, 4)))
    mask = mx.ones((1, 2), dtype=mx.int32)
    expected = source(states, mask)
    actual = target(states, mask)
    mx.eval(expected.video_encoding, actual.video_encoding)
    assert mx.allclose(actual.video_encoding, expected.video_encoding).item()
    assert mx.allclose(actual.audio_encoding, expected.audio_encoding).item()
    assert reporter.events[-1] == ("end", "load AV text connectors", {})


def test_connector_loader_rejects_missing_consumed_tensors(tmp_path) -> None:
    path = tmp_path / "connector.safetensors"
    mx.save_safetensors(str(path), {"wrong.weight": mx.zeros((1,))})
    with pytest.raises(ValueError, match="missing.*consumed"):
        load_av_text_encoder_v2_weights(create_av_text_encoder_v2(_config()), path)


def test_connector_loader_binds_split_projection_and_connector_sources(tmp_path) -> None:
    source = _synthetic_model()
    checkpoint_weights = {
        _checkpoint_key(key): value for key, value in tree_flatten(source.parameters())
    }
    projection_weights = {
        key: value
        for key, value in checkpoint_weights.items()
        if key.startswith("text_embedding_projection.")
    }
    connector_weights = {
        key: value
        for key, value in checkpoint_weights.items()
        if key.startswith("model.diffusion_model.")
    }
    projection_weights["community.text_baggage"] = mx.zeros((3,))
    connector_weights["community.transformer_baggage"] = mx.zeros((5,))
    projection_path = tmp_path / "text.safetensors"
    connector_path = tmp_path / "transformer.safetensors"
    mx.save_safetensors(str(projection_path), projection_weights)
    mx.save_safetensors(str(connector_path), connector_weights)

    target = create_av_text_encoder_v2(_config())
    assert (
        load_av_text_encoder_v2_weights(
            target,
            connector_path,
            projection_path=projection_path,
        )
        == len(projection_weights) + len(connector_weights) - 2
    )

    states = (mx.random.normal((1, 2, 4)), mx.random.normal((1, 2, 4)))
    mask = mx.ones((1, 2), dtype=mx.int32)
    expected = source(states, mask)
    actual = target(states, mask)
    mx.eval(expected.video_encoding, actual.video_encoding)
    assert mx.allclose(actual.video_encoding, expected.video_encoding).item()
    assert mx.allclose(actual.audio_encoding, expected.audio_encoding).item()


def test_split_loader_preflights_both_sources_before_mutation(tmp_path) -> None:
    source = _synthetic_model()
    checkpoint_weights = {
        _checkpoint_key(key): value for key, value in tree_flatten(source.parameters())
    }
    projection_weights = {
        key: value
        for key, value in checkpoint_weights.items()
        if key.startswith("text_embedding_projection.")
    }
    connector_weights = {
        key: value
        for key, value in checkpoint_weights.items()
        if key.startswith("model.diffusion_model.")
    }
    connector_weights.pop(next(iter(connector_weights)))
    projection_path = tmp_path / "text.safetensors"
    connector_path = tmp_path / "transformer.safetensors"
    mx.save_safetensors(str(projection_path), projection_weights)
    mx.save_safetensors(str(connector_path), connector_weights)

    target = create_av_text_encoder_v2(_config())
    with pytest.raises(ValueError, match="LTX AV text connectors.*consumed"):
        load_av_text_encoder_v2_weights(
            target,
            connector_path,
            projection_path=projection_path,
        )
    assert target.feature_extractor.video_aggregate_embed.weight.size == 0
