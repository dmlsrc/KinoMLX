"""Encoder-neutral text-conditioning station and provenance contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.text_conditioning as text_module
from kinomlx.models.ltx2.resources import ComponentKind
from kinomlx.models.ltx2.text_conditioning import (
    NativeTextConditioner,
    TextConditioningProvenance,
    text_conditioning_provenance,
)
from kinomlx.models.ltx2.text_encoder.encoder import AudioVideoGemmaEncoderOutput
from kinomlx.models.ltx2.text_encoder.tokenizer_cache import TokenizerCache
from kinomlx.models.ltx2.types import DistilledRequest


class _Metadata:
    def __init__(self, family: str) -> None:
        self.family = family

    def get(self, name: str, default: str | None = None) -> str | None:
        return {"family": self.family}.get(name, default)


class _Resources:
    def __init__(
        self,
        *,
        gemma_path: Path | None = Path("gemma"),
        generation: str = "ltx-2.3",
        tokenizer_cache: TokenizerCache | None = None,
    ) -> None:
        self.tokenizer_cache = tokenizer_cache
        self.capabilities = SimpleNamespace(
            model_generation=generation,
            text_encoder_family=(
                "gemma4-12b-ltx" if generation in {"2.5", "ltx-2.5"} else "gemma-3-12b-it"
            ),
        )
        connector_cache = (
            None if generation in {"2.5", "ltx-2.5"} else Path("connector.safetensors")
        )
        self.components = {
            ComponentKind.TEXT_ENCODER: (
                None
                if gemma_path is None
                else SimpleNamespace(
                    kind=ComponentKind.TEXT_ENCODER,
                    source_path=gemma_path,
                    cache_path=None,
                    source_fingerprint="text-fingerprint",
                    metadata=_Metadata("gemma4-12b-ltx"),
                )
            ),
            ComponentKind.TEXT_PROJECTION: SimpleNamespace(
                kind=ComponentKind.TEXT_PROJECTION,
                source_path=Path("text.safetensors"),
                cache_path=connector_cache,
                source_fingerprint="projection-fingerprint",
                metadata=_Metadata("ltx2-text-projection"),
            ),
            ComponentKind.CONNECTOR: SimpleNamespace(
                kind=ComponentKind.CONNECTOR,
                source_path=Path("transformer.safetensors"),
                cache_path=connector_cache,
                source_fingerprint="connector-fingerprint",
                metadata=_Metadata("ltx-2.3-av-connectors"),
            ),
            ComponentKind.TRANSFORMER: SimpleNamespace(
                kind=ComponentKind.TRANSFORMER,
                source_path=Path("transformer.safetensors"),
                cache_path=None,
                source_fingerprint="transformer-fingerprint",
                metadata=_Metadata("ltx2-av-transformer"),
            ),
        }

    @property
    def gemma_path(self) -> Path | None:
        text = self.components[ComponentKind.TEXT_ENCODER]
        return None if text is None else text.source_path

    @property
    def transformer_path(self) -> Path:
        return self.components[ComponentKind.TRANSFORMER].source_path

    def optional(self, kind: ComponentKind):
        return self.components.get(kind)

    def require(self, kind: ComponentKind):
        component = self.optional(kind)
        if component is None:
            raise LookupError(kind.value)
        return component


def _tokenizer_cache(tmp_path: Path) -> TokenizerCache:
    model = tmp_path / "tokenizer.model"
    metadata = tmp_path / "tokenizer.metadata.json"
    model.write_bytes(b"model")
    metadata.write_bytes(b"metadata")
    return TokenizerCache(
        model_path=model,
        metadata_path=metadata,
        source_json_sha256="source-json-digest",
        model_sha256="model-digest",
    )


def _encoded(tokens: int = 2) -> AudioVideoGemmaEncoderOutput:
    return AudioVideoGemmaEncoderOutput(
        video_encoding=mx.zeros((1, tokens, 4096)),
        audio_encoding=mx.zeros((1, tokens, 2048)),
        attention_mask=mx.ones((1, tokens)),
    )


def _provenance() -> TextConditioningProvenance:
    return TextConditioningProvenance(
        model_generation="ltx-2.3",
        text_encoder_identity="gemma-3-12b-it",
        projection_identity="ltx-2.3-av-connectors:connector-fingerprint",
    )


def test_provenance_identifies_model_encoder_and_projection_layout() -> None:
    assert text_conditioning_provenance(_Resources()) == _provenance()


def test_provenance_records_tokenizer_and_both_packaged_sources(tmp_path: Path) -> None:
    cache = _tokenizer_cache(tmp_path)
    provenance = text_conditioning_provenance(_Resources(generation="2.5", tokenizer_cache=cache))

    assert provenance.tokenizer_source_sha256 == "source-json-digest"
    assert provenance.tokenizer_model_sha256 == "model-digest"
    assert provenance.tokenizer_metadata_sha256 == hashlib.sha256(b"metadata").hexdigest()
    assert provenance.tokenization_policy is not None
    assert provenance.text_artifact_identity == "gemma4-12b-ltx:text-fingerprint"
    assert provenance.projection_source_identity == ("ltx2-text-projection:projection-fingerprint")
    assert provenance.connector_source_identity == ("ltx-2.3-av-connectors:connector-fingerprint")


def test_legacy_sidecar_identity_is_accepted_only_for_ltx23(tmp_path: Path) -> None:
    cache = _tokenizer_cache(tmp_path)
    expected_23 = text_conditioning_provenance(
        _Resources(generation="ltx-2.3", tokenizer_cache=cache)
    )
    assert _provenance().is_compatible_with(expected_23)

    expected_25 = text_conditioning_provenance(_Resources(generation="2.5", tokenizer_cache=cache))
    legacy_25 = TextConditioningProvenance(
        model_generation="2.5",
        text_encoder_identity=expected_25.text_encoder_identity,
        projection_identity=expected_25.projection_identity,
    )
    assert not legacy_25.is_compatible_with(expected_25)


def test_ltx25_incompatible_schema3_sidecar_fails_before_prompt_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    resources = _Resources(
        generation="2.5",
        tokenizer_cache=_tokenizer_cache(tmp_path),
    )
    metadata = {
        "schema_version": "3",
        **text_conditioning_provenance(resources).to_metadata(),
        "projection_source_identity": "projection:different",
    }
    monkeypatch.setattr(
        text_module,
        "load_text_conditioning",
        lambda path, *, reporter=None, metadata_policy="require": (_encoded(), metadata),
    )
    monkeypatch.setattr(
        text_module,
        "encode_text",
        lambda *_args, **_kwargs: pytest.fail("mismatched replay must not load text weights"),
    )

    with pytest.raises(ValueError, match="provenance does not match"):
        NativeTextConditioner()(
            DistilledRequest(text_conditioning=sidecar),
            resources,
        )


def test_saved_conditioning_bypasses_prompt_models(monkeypatch, tmp_path: Path) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    metadata = {
        "prompt": "saved prompt",
        **_provenance().to_metadata(),
    }
    monkeypatch.setattr(
        text_module,
        "load_text_conditioning",
        lambda path, *, reporter=None, metadata_policy="require": (_encoded(4), metadata),
    )
    monkeypatch.setattr(
        text_module,
        "encode_text",
        lambda *_args, **_kwargs: pytest.fail("saved conditioning must not load Gemma"),
    )

    result = NativeTextConditioner()(
        DistilledRequest(text_conditioning=sidecar),
        _Resources(gemma_path=None),
    )

    assert result.prompt == "saved prompt"
    assert result.video_encoding.shape == (1, 4, 4096)
    assert result.provenance == _provenance()


def test_incompatible_sidecar_fails_before_prompt_encoding(monkeypatch, tmp_path: Path) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    metadata = {
        **_provenance().to_metadata(),
        "text_encoder_identity": "ltx-specific-gemma-4",
    }
    monkeypatch.setattr(
        text_module,
        "load_text_conditioning",
        lambda path, *, reporter=None, metadata_policy="require": (_encoded(), metadata),
    )
    monkeypatch.setattr(
        text_module,
        "encode_text",
        lambda *_args, **_kwargs: pytest.fail("mismatched replay must fail before encoding"),
    )

    with pytest.raises(ValueError, match="provenance does not match"):
        NativeTextConditioner()(
            DistilledRequest(text_conditioning=sidecar),
            _Resources(),
        )


def test_restart_replay_observes_identity_mismatch_without_veto(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    metadata = {
        "schema_version": "3",
        **_provenance().to_metadata(),
        "projection_identity": "community:connector",
    }
    policies = []
    monkeypatch.setattr(
        text_module,
        "load_text_conditioning",
        lambda path, *, reporter=None, metadata_policy="require": (
            policies.append(metadata_policy) or _encoded(),
            metadata,
        ),
    )
    monkeypatch.setattr(
        text_module,
        "encode_text",
        lambda *_args, **_kwargs: pytest.fail("replay must not load prompt weights"),
    )

    with caplog.at_level("WARNING"):
        result = NativeTextConditioner(replay_identity_policy="observe")(
            DistilledRequest(text_conditioning=sidecar),
            _Resources(),
        )

    assert policies == ["observe"]
    assert result.provenance.projection_identity == "community:connector"
    assert result.replay_receipt is not None
    assert result.replay_receipt["identity_match"] is False
    assert result.replay_receipt["policy"] == "observe"
    assert "identity is advisory during restart" in caplog.text


def test_prompt_station_delegates_concrete_encoder_selection_to_resources(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        text_module,
        "encode_text",
        lambda prompt, **kwargs: calls.append((prompt, kwargs)) or _encoded(),
    )
    request = DistilledRequest(prompt="a lighthouse", pad_prompt_to_max=False)

    result = NativeTextConditioner()(request, _Resources())

    assert result.prompt == "a lighthouse"
    assert result.provenance == _provenance()
    assert calls[0][0] == "a lighthouse"
    assert calls[0][1]["gemma_path"] == Path("gemma")
    assert calls[0][1]["connector_path"] == Path("connector.safetensors")
    assert calls[0][1]["projection_path"] == Path("connector.safetensors")
    assert calls[0][1]["config_path"] == Path("transformer.safetensors")
    assert calls[0][1]["model_generation"] == "ltx-2.3"
    assert calls[0][1]["pad_prompt_to_max"] is False


def test_ltx25_prompt_station_selects_packaged_text_projection_and_connectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    monkeypatch.setattr(
        text_module,
        "encode_text",
        lambda prompt, **kwargs: calls.append((prompt, kwargs)) or _encoded(),
    )
    resources = _Resources(
        gemma_path=Path("text.safetensors"),
        generation="2.5",
        tokenizer_cache=_tokenizer_cache(tmp_path),
    )

    NativeTextConditioner()(DistilledRequest(prompt="a lighthouse"), resources)

    assert calls[0][1]["gemma_path"] == Path("text.safetensors")
    assert calls[0][1]["projection_path"] == Path("text.safetensors")
    assert calls[0][1]["connector_path"] == Path("transformer.safetensors")
    assert calls[0][1]["config_path"] == Path("transformer.safetensors")
    assert calls[0][1]["model_generation"] == "2.5"


def test_prompt_station_requires_the_selected_encoder_assets() -> None:
    with pytest.raises(FileNotFoundError, match="no local gemma-3-12b-it"):
        NativeTextConditioner()(
            DistilledRequest(prompt="a lighthouse"),
            _Resources(gemma_path=None),
        )
