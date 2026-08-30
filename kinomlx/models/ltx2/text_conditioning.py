"""Public text-conditioning station and reusable encoded-text product."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import mlx.core as mx

from kinomlx.reporting import Reporter

from .encode import encode_text, load_text_conditioning
from .resources import ComponentKind, ComponentResource, LTX2Resources
from .text_encoder.tokenizer_cache import (
    TOKENIZER_CACHE_SCHEMA_VERSION,
    TOKENIZER_SERIALIZATION_POLICY,
)
from .types import DistilledRequest

_CORE_PROVENANCE_FIELDS = (
    "model_generation",
    "text_encoder_identity",
    "projection_identity",
)
_EXTENDED_PROVENANCE_FIELDS = (
    "tokenizer_source_sha256",
    "tokenizer_model_sha256",
    "tokenizer_metadata_sha256",
    "tokenization_policy",
    "text_artifact_identity",
    "projection_source_identity",
    "connector_source_identity",
)
_PROVENANCE_FIELDS = _CORE_PROVENANCE_FIELDS + _EXTENDED_PROVENANCE_FIELDS
_LOG = logging.getLogger(__name__)

ReplayIdentityPolicy = Literal["require", "observe"]


@dataclass(frozen=True)
class TextConditioningProvenance:
    """Compatibility identity required to replay encoded text safely."""

    model_generation: str
    text_encoder_identity: str
    projection_identity: str
    tokenizer_source_sha256: str | None = None
    tokenizer_model_sha256: str | None = None
    tokenizer_metadata_sha256: str | None = None
    tokenization_policy: str | None = None
    text_artifact_identity: str | None = None
    projection_source_identity: str | None = None
    connector_source_identity: str | None = None

    def to_metadata(self) -> dict[str, str]:
        metadata = {
            "model_generation": self.model_generation,
            "text_encoder_identity": self.text_encoder_identity,
            "projection_identity": self.projection_identity,
        }
        extended = (
            ("tokenizer_source_sha256", self.tokenizer_source_sha256),
            ("tokenizer_model_sha256", self.tokenizer_model_sha256),
            ("tokenizer_metadata_sha256", self.tokenizer_metadata_sha256),
            ("tokenization_policy", self.tokenization_policy),
            ("text_artifact_identity", self.text_artifact_identity),
            ("projection_source_identity", self.projection_source_identity),
            ("connector_source_identity", self.connector_source_identity),
        )
        metadata.update((name, value) for name, value in extended if value is not None)
        return metadata

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, str],
        *,
        source: Path,
        require_extended: bool = False,
    ) -> TextConditioningProvenance:
        values: dict[str, str] = {}
        for name in _CORE_PROVENANCE_FIELDS:
            value = metadata.get(name)
            if not value:
                raise ValueError(f"{source}: text conditioning is missing {name}")
            values[name] = value
        extended: dict[str, str | None] = {}
        for name in _EXTENDED_PROVENANCE_FIELDS:
            value = metadata.get(name)
            if require_extended and not value:
                raise ValueError(f"{source}: text conditioning is missing {name}")
            extended[name] = value or None
        return cls(
            model_generation=values["model_generation"],
            text_encoder_identity=values["text_encoder_identity"],
            projection_identity=values["projection_identity"],
            **extended,
        )

    def is_compatible_with(self, expected: TextConditioningProvenance) -> bool:
        """Accept exact schema-3 identity or a valid legacy LTX-2.3 identity."""
        if self == expected:
            return True
        legacy_generation = self.model_generation.lower() in {"2.3", "ltx-2.3"}
        extended_values = (
            self.tokenizer_source_sha256,
            self.tokenizer_model_sha256,
            self.tokenizer_metadata_sha256,
            self.tokenization_policy,
            self.text_artifact_identity,
            self.projection_source_identity,
            self.connector_source_identity,
        )
        return (
            legacy_generation
            and not any(extended_values)
            and self.model_generation == expected.model_generation
            and self.text_encoder_identity == expected.text_encoder_identity
            and self.projection_identity == expected.projection_identity
        )


@dataclass(frozen=True)
class EncodedTextConditioning:
    """Materialized text context with no live text-model ownership."""

    video_encoding: mx.array
    audio_encoding: mx.array
    attention_mask: mx.array
    prompt: str
    provenance: TextConditioningProvenance
    replay_receipt: Mapping[str, object] | None = None


class TextConditioner(Protocol):
    """Stateless host port for prompt encoding or sidecar replay."""

    def __call__(
        self,
        request: DistilledRequest,
        resources: LTX2Resources,
        *,
        reporter: Reporter | None = None,
    ) -> EncodedTextConditioning: ...


def _component_identity(component: ComponentResource) -> str:
    family = component.metadata.get("family", component.kind.value)
    return f"{family}:{component.source_fingerprint}"


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_conditioning_provenance(resources: LTX2Resources) -> TextConditioningProvenance:
    """Resolve the checkpoint, encoder-family, and projection compatibility identity."""
    connector = resources.require(ComponentKind.CONNECTOR)
    projection_family = connector.metadata.get("family", "connector")
    base = TextConditioningProvenance(
        model_generation=resources.capabilities.model_generation,
        text_encoder_identity=resources.capabilities.text_encoder_family,
        projection_identity=f"{projection_family}:{connector.source_fingerprint}",
    )
    tokenizer = resources.tokenizer_cache
    if tokenizer is None:
        if resources.capabilities.model_generation.lower() in {"2.5", "ltx-2.5"}:
            raise LookupError("LTX-2.5 text conditioning requires a prepared tokenizer cache")
        return base
    text = resources.require(ComponentKind.TEXT_ENCODER)
    projection = resources.require(ComponentKind.TEXT_PROJECTION)
    return TextConditioningProvenance(
        model_generation=base.model_generation,
        text_encoder_identity=base.text_encoder_identity,
        projection_identity=base.projection_identity,
        tokenizer_source_sha256=tokenizer.source_json_sha256,
        tokenizer_model_sha256=tokenizer.model_sha256,
        tokenizer_metadata_sha256=_sha256_file(tokenizer.metadata_path),
        tokenization_policy=(
            f"{TOKENIZER_SERIALIZATION_POLICY}:cache-v{TOKENIZER_CACHE_SCHEMA_VERSION}:"
            "left-padding-v1"
        ),
        text_artifact_identity=_component_identity(text),
        projection_source_identity=_component_identity(projection),
        connector_source_identity=_component_identity(connector),
    )


class NativeTextConditioner:
    """Native sequential Gemma/connector station with a sidecar fast path."""

    def __init__(
        self,
        *,
        replay_identity_policy: ReplayIdentityPolicy = "require",
    ) -> None:
        if replay_identity_policy not in {"require", "observe"}:
            raise ValueError("replay_identity_policy must be require or observe")
        self.replay_identity_policy = replay_identity_policy

    def __call__(
        self,
        request: DistilledRequest,
        resources: LTX2Resources,
        *,
        reporter: Reporter | None = None,
    ) -> EncodedTextConditioning:
        expected = text_conditioning_provenance(resources)
        if request.text_conditioning is not None:
            encoded, metadata = load_text_conditioning(
                request.text_conditioning,
                reporter=reporter,
                metadata_policy=self.replay_identity_policy,
            )
            parse_error = None
            try:
                actual = TextConditioningProvenance.from_metadata(
                    metadata,
                    source=request.text_conditioning,
                    require_extended=(
                        metadata.get("schema_version") == "3"
                        and self.replay_identity_policy == "require"
                    ),
                )
            except ValueError as exc:
                if self.replay_identity_policy == "require":
                    raise
                actual = None
                parse_error = str(exc)
            identity_match = None if actual is None else actual.is_compatible_with(expected)
            if identity_match is False and self.replay_identity_policy == "require":
                raise ValueError(
                    f"{request.text_conditioning}: text-conditioning provenance does not match "
                    "the prepared resources"
                )
            if self.replay_identity_policy == "observe" and identity_match is not True:
                difference = (
                    parse_error
                    if parse_error is not None
                    else "declared identity differs from the prepared resources"
                )
                _LOG.warning(
                    "%s: %s text-conditioning identity is advisory during restart (%s); "
                    "continuing because consumed tensors fit",
                    request.text_conditioning,
                    expected.model_generation,
                    difference,
                )
            declared_identity = {
                name: metadata[name] for name in _PROVENANCE_FIELDS if metadata.get(name)
            }
            replay_receipt: dict[str, object] = {
                "source": str(request.text_conditioning),
                "policy": self.replay_identity_policy,
                "declared_identity": declared_identity,
                "prepared_identity": expected.to_metadata(),
                "identity_match": identity_match,
                "identity_parse_error": parse_error,
                "provenance_fallback": (None if actual is not None else "prepared_resources"),
            }
            return EncodedTextConditioning(
                video_encoding=encoded.video_encoding,
                audio_encoding=encoded.audio_encoding,
                attention_mask=encoded.attention_mask,
                prompt=metadata.get("prompt", request.prompt),
                provenance=actual if actual is not None else expected,
                replay_receipt=replay_receipt,
            )

        text = resources.optional(ComponentKind.TEXT_ENCODER)
        if text is None:
            family = resources.capabilities.text_encoder_family
            raise FileNotFoundError(
                f"no local {family} text artifact was prepared; set the selected "
                "text_encoder_path or gemma_path"
            )
        connector = resources.require(ComponentKind.CONNECTOR)
        projection = resources.require(ComponentKind.TEXT_PROJECTION)
        connector_path = connector.cache_path or connector.source_path
        projection_path = projection.cache_path or projection.source_path
        encoded = encode_text(
            request.prompt,
            gemma_path=text.source_path,
            connector_path=connector_path,
            projection_path=projection_path,
            config_path=resources.transformer_path,
            model_generation=resources.capabilities.model_generation,
            tokenizer_cache=resources.tokenizer_cache,
            pad_prompt_to_max=request.pad_prompt_to_max,
            reporter=reporter,
        )
        return EncodedTextConditioning(
            video_encoding=encoded.video_encoding,
            audio_encoding=encoded.audio_encoding,
            attention_mask=encoded.attention_mask,
            prompt=request.prompt,
            provenance=expected,
        )


__all__ = [
    "EncodedTextConditioning",
    "NativeTextConditioner",
    "ReplayIdentityPolicy",
    "TextConditioner",
    "TextConditioningProvenance",
    "text_conditioning_provenance",
]
