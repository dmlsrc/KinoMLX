"""Prompt station releases each concrete text model before its next boundary."""

from __future__ import annotations

import weakref

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.text_encoder.encoder as encoder_module
from kinomlx.models.ltx2.text_encoder.encoder import AudioVideoGemmaEncoderOutput


@pytest.mark.parametrize("model_generation", ["2.3", "2.5"])
def test_prompt_encoding_releases_gemma_then_connectors_before_return(
    model_generation: str,
    monkeypatch,
) -> None:
    gemma_ref = None
    connector_ref = None

    class _Tokenizer:
        def __init__(self, _path) -> None:
            pass

        def encode(self, _prompt, *, max_length, pad_to_max):
            del max_length, pad_to_max
            return mx.zeros((1, 2), dtype=mx.int32), mx.ones((1, 2), dtype=mx.int32)

    class _Gemma:
        def __init__(self, _config) -> None:
            nonlocal gemma_ref
            gemma_ref = weakref.ref(self)

        def __call__(self, *_args, **_kwargs):
            state = mx.zeros((1, 2, 4), dtype=mx.bfloat16)
            return state, (state, state)

    class _Connector:
        def __init__(self) -> None:
            nonlocal connector_ref
            connector_ref = weakref.ref(self)

        def __call__(self, hidden_states, attention_mask, *, reporter=None):
            del hidden_states, reporter
            return AudioVideoGemmaEncoderOutput(
                video_encoding=mx.zeros((1, 2, 4096), dtype=mx.bfloat16),
                audio_encoding=mx.zeros((1, 2, 2048), dtype=mx.bfloat16),
                attention_mask=attention_mask,
            )

    def create_connector(_path):
        assert gemma_ref is not None
        assert gemma_ref() is None
        return _Connector()

    monkeypatch.setattr(encoder_module, "GemmaTokenizer", _Tokenizer)
    monkeypatch.setattr(encoder_module, "Gemma3Model", _Gemma)
    monkeypatch.setattr(encoder_module, "Gemma4Model", _Gemma)
    monkeypatch.setattr(encoder_module, "load_gemma3_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(encoder_module, "load_gemma4_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        encoder_module,
        "create_av_text_encoder_v2_from_checkpoint",
        create_connector,
    )
    monkeypatch.setattr(
        "kinomlx.models.ltx2.text_encoder.loading.load_av_text_encoder_v2_weights",
        lambda *_args, **_kwargs: None,
    )

    output = encoder_module.encode_prompt(
        "prompt",
        gemma_path="gemma",
        connector_path="connector.safetensors",
        projection_path="projection.safetensors",
        config_path="checkpoint.safetensors",
        model_generation=model_generation,
        pad_prompt_to_max=False,
    )

    assert output.video_encoding.shape == (1, 2, 4096)
    assert connector_ref is not None
    assert connector_ref() is None


@pytest.mark.parametrize(
    "failure_point",
    ["gemma load", "gemma forward", "connector load", "connector forward"],
)
@pytest.mark.parametrize("model_generation", ["2.3", "2.5"])
def test_prompt_encoding_releases_each_model_when_its_station_fails(
    model_generation: str,
    failure_point: str,
    monkeypatch,
) -> None:
    refs = {}

    class _Tokenizer:
        def __init__(self, _path) -> None:
            pass

        def encode(self, _prompt, *, max_length, pad_to_max):
            del max_length, pad_to_max
            return mx.zeros((1, 2), dtype=mx.int32), mx.ones((1, 2), dtype=mx.int32)

    class _Gemma:
        def __init__(self, _config) -> None:
            refs["gemma"] = weakref.ref(self)

        def __call__(self, *_args, **_kwargs):
            if failure_point == "gemma forward":
                raise RuntimeError(failure_point)
            state = mx.zeros((1, 2, 4), dtype=mx.bfloat16)
            return state, (state, state)

    class _Connector:
        def __init__(self) -> None:
            refs["connector"] = weakref.ref(self)

        def __call__(self, hidden_states, attention_mask, *, reporter=None):
            del hidden_states, attention_mask, reporter
            raise RuntimeError("connector forward")

    def load_gemma(*_args, **_kwargs) -> None:
        if failure_point == "gemma load":
            raise RuntimeError(failure_point)

    def create_connector(_path):
        assert refs["gemma"]() is None
        return _Connector()

    def load_connector(*_args, **_kwargs) -> None:
        if failure_point == "connector load":
            raise RuntimeError(failure_point)

    monkeypatch.setattr(encoder_module, "GemmaTokenizer", _Tokenizer)
    monkeypatch.setattr(encoder_module, "Gemma3Model", _Gemma)
    monkeypatch.setattr(encoder_module, "Gemma4Model", _Gemma)
    monkeypatch.setattr(encoder_module, "load_gemma3_weights", load_gemma)
    monkeypatch.setattr(encoder_module, "load_gemma4_weights", load_gemma)
    monkeypatch.setattr(
        encoder_module,
        "create_av_text_encoder_v2_from_checkpoint",
        create_connector,
    )
    monkeypatch.setattr(
        "kinomlx.models.ltx2.text_encoder.loading.load_av_text_encoder_v2_weights",
        load_connector,
    )

    with pytest.raises(RuntimeError, match=failure_point):
        encoder_module.encode_prompt(
            "prompt",
            gemma_path="gemma",
            connector_path="connector.safetensors",
            projection_path="projection.safetensors",
            config_path="checkpoint.safetensors",
            model_generation=model_generation,
            pad_prompt_to_max=False,
        )

    assert refs["gemma"]() is None
    if failure_point.startswith("connector"):
        assert refs["connector"]() is None
