"""Structural Krea text-conditioning stage and reusable sidecar product."""

import gc
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

import mlx.core as mx

from kinomlx._json import JsonObject
from kinomlx.artifacts import TensorArtifact
from kinomlx.components import ComponentLease
from kinomlx.io.identity import payload_digest
from kinomlx.io.safetensors import SafetensorsTensor, inspect_safetensors, load_weights
from kinomlx.reporting import NullReporter, Reporter, reported_phase

from .resource_types import Krea2Resources
from .source_loading import load_component_tokenizer
from .text_encoder import (
    Qwen3VLLoadReceipt,
    Qwen3VLTextModel,
    load_qwen3_vl_text_weights,
    qwen3_vl_target_identity,
)
from .tokenizer import (
    KREA_PREFIX_TOKENS,
    QWEN_TOKENIZER_TABLE_POLICY,
    Krea2TokenBatch,
    prepare_krea2_token_batch,
)
from .tokenizer.template import KREA_MAX_QWEN_TOKENS, regular_text_prefix

type PromptInput = str | Sequence[str]

TEXT_CONDITIONING = "text_conditioning"
TEXT_CONDITIONING_ARTIFACT = "krea2_text_conditioning"
TEXT_CONDITIONING_SCHEMA_VERSION = "1"
ReplayIdentityPolicy = Literal["require", "observe"]
_FLOAT_DTYPES = frozenset({mx.bfloat16, mx.float16, mx.float32})
_FLOAT_STORAGE_DTYPES = frozenset({"BF16", "F16", "F32"})
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Krea2PromptConditioning:
    """One materialized Krea text context with no live Qwen ownership."""

    hidden_states: mx.array
    attention_mask: mx.array
    position_ids: mx.array
    prompts: tuple[str, ...]
    token_identity: str


@dataclass(frozen=True)
class Krea2TextConditioningProvenance:
    """Source receipt and reusable text-graph identity for one product."""

    declared_variant: str
    is_distilled: bool
    recipe_identity: str
    tokenizer_source_identity: str
    tokenizer_graph_identity: str
    # Persisted schema spelling for the derived tokenizer identity; no disk cache is read.
    tokenizer_cache_identity: str
    text_source_identity: str
    text_graph_identity: str
    text_dtype_policy: str
    selected_layers: tuple[int, ...]
    qwen_target_count: int
    qwen_target_digest: str
    compatibility_identity: str
    system_prompt: str | None = None
    prefix_tokens: int = KREA_PREFIX_TOKENS

    def to_metadata(self) -> dict[str, object]:
        """Return the JSON-ready replay identity."""
        result: dict[str, object] = {
            "declared_variant": self.declared_variant,
            "is_distilled": self.is_distilled,
            "recipe_identity": self.recipe_identity,
            "tokenizer_source_identity": self.tokenizer_source_identity,
            "tokenizer_graph_identity": self.tokenizer_graph_identity,
            "tokenizer_cache_identity": self.tokenizer_cache_identity,
            "text_source_identity": self.text_source_identity,
            "text_graph_identity": self.text_graph_identity,
            "text_dtype_policy": self.text_dtype_policy,
            "selected_layers": list(self.selected_layers),
            "qwen_target_count": self.qwen_target_count,
            "qwen_target_digest": self.qwen_target_digest,
            "compatibility_identity": self.compatibility_identity,
        }
        if self.system_prompt is not None:
            result.update(system_prompt=self.system_prompt, prefix_tokens=self.prefix_tokens)
        return result

    @classmethod
    def from_metadata(
        cls,
        value: object,
        *,
        source: Path,
    ) -> Krea2TextConditioningProvenance:
        """Parse and self-validate one serialized replay identity."""
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"{source}: text-conditioning provenance must be an object")
        raw = dict(cast(Mapping[str, object], value))
        strings = (
            "declared_variant",
            "recipe_identity",
            "tokenizer_source_identity",
            "tokenizer_graph_identity",
            "tokenizer_cache_identity",
            "text_source_identity",
            "text_graph_identity",
            "text_dtype_policy",
            "qwen_target_digest",
            "compatibility_identity",
        )
        for name in strings:
            field = raw.get(name)
            if not isinstance(field, str) or not field:
                raise ValueError(f"{source}: text-conditioning provenance is missing {name}")
        is_distilled = raw.get("is_distilled")
        if not isinstance(is_distilled, bool):
            raise ValueError(f"{source}: text-conditioning provenance is_distilled must be boolean")
        selected_layers_raw = raw.get("selected_layers")
        if (
            not isinstance(selected_layers_raw, list)
            or not selected_layers_raw
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer <= 0
                for layer in selected_layers_raw
            )
        ):
            raise ValueError(
                f"{source}: text-conditioning provenance selected_layers must be positive integers"
            )
        selected_layers = tuple(selected_layers_raw)
        if tuple(sorted(set(selected_layers))) != selected_layers:
            raise ValueError(
                f"{source}: text-conditioning provenance selected_layers must be sorted and unique"
            )
        target_count = raw.get("qwen_target_count")
        if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count <= 0:
            raise ValueError(
                f"{source}: text-conditioning provenance qwen_target_count must be positive"
            )
        system_prompt = raw.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError(f"{source}: system_prompt must be a string or null")
        regular_text_prefix(system_prompt)
        prefix_tokens = raw.get("prefix_tokens", KREA_PREFIX_TOKENS)
        if (
            isinstance(prefix_tokens, bool)
            or not isinstance(prefix_tokens, int)
            or not 0 < prefix_tokens <= KREA_MAX_QWEN_TOKENS - 512
        ):
            raise ValueError(f"{source}: invalid system prefix token count")
        if system_prompt is None and prefix_tokens != KREA_PREFIX_TOKENS:
            raise ValueError(f"{source}: default system prefix token count must be 34")
        result = cls(
            declared_variant=cast(str, raw["declared_variant"]),
            is_distilled=is_distilled,
            recipe_identity=cast(str, raw["recipe_identity"]),
            tokenizer_source_identity=cast(str, raw["tokenizer_source_identity"]),
            tokenizer_graph_identity=cast(str, raw["tokenizer_graph_identity"]),
            tokenizer_cache_identity=cast(str, raw["tokenizer_cache_identity"]),
            text_source_identity=cast(str, raw["text_source_identity"]),
            text_graph_identity=cast(str, raw["text_graph_identity"]),
            text_dtype_policy=cast(str, raw["text_dtype_policy"]),
            selected_layers=selected_layers,
            qwen_target_count=target_count,
            qwen_target_digest=cast(str, raw["qwen_target_digest"]),
            compatibility_identity=cast(str, raw["compatibility_identity"]),
            system_prompt=system_prompt,
            prefix_tokens=prefix_tokens,
        )
        expected_identity = payload_digest(result._identity_fields())
        if result.compatibility_identity != expected_identity:
            raise ValueError(
                f"{source}: text-conditioning provenance compatibility identity is invalid"
            )
        return result

    def _identity_fields(self) -> JsonObject:
        result: JsonObject = {
            "schema_version": 1,
            "tokenizer_source_identity": self.tokenizer_source_identity,
            "tokenizer_graph_identity": self.tokenizer_graph_identity,
            "tokenizer_cache_identity": self.tokenizer_cache_identity,
            "text_source_identity": self.text_source_identity,
            "text_graph_identity": self.text_graph_identity,
            "text_dtype_policy": self.text_dtype_policy,
            "selected_layers": list(self.selected_layers),
            "qwen_target_count": self.qwen_target_count,
            "qwen_target_digest": self.qwen_target_digest,
        }
        if self.system_prompt is not None:
            result.update(system_prompt=self.system_prompt, prefix_tokens=self.prefix_tokens)
        return result

    def is_compatible_with(self, expected: Krea2TextConditioningProvenance) -> bool:
        """Compare only resources that participate in Qwen conditioning."""
        return (
            self.tokenizer_source_identity == expected.tokenizer_source_identity
            and self.tokenizer_graph_identity == expected.tokenizer_graph_identity
            and self.tokenizer_cache_identity == expected.tokenizer_cache_identity
            and self.text_source_identity == expected.text_source_identity
            and self.text_graph_identity == expected.text_graph_identity
            and self.text_dtype_policy == expected.text_dtype_policy
            and self.selected_layers == expected.selected_layers
            and self.qwen_target_count == expected.qwen_target_count
            and self.qwen_target_digest == expected.qwen_target_digest
            and self.system_prompt == expected.system_prompt
            and self.prefix_tokens == expected.prefix_tokens
        )


@dataclass(frozen=True)
class Krea2TextConditioning:
    """Positive and optional negative text products from one Qwen lifetime."""

    positive: Krea2PromptConditioning
    negative: Krea2PromptConditioning | None
    provenance: Krea2TextConditioningProvenance
    replay_receipt: Mapping[str, object] | None = None


class Krea2TextConditionerPort(Protocol):
    """Structural host port for native or caller-provided Krea conditioning."""

    def __call__(
        self,
        prompts: PromptInput,
        resources: Krea2Resources,
        *,
        negative_prompts: PromptInput | None = None,
        max_sequence_length: int = 512,
        system_prompt: str | None = None,
        reporter: Reporter | None = None,
    ) -> Krea2TextConditioning: ...


def _prompts(value: PromptInput, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or not value:
        raise TypeError(f"{name} must be a string or nonempty sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain only strings")
    return tuple(value)


def _cleanup_qwen() -> None:
    gc.collect()
    mx.clear_cache()


def _encode_batch(
    model: Qwen3VLTextModel,
    token_batch: Krea2TokenBatch,
    prompts: tuple[str, ...],
    *,
    reporter: Reporter | None,
) -> Krea2PromptConditioning:
    input_ids = mx.array(token_batch.input_ids, dtype=mx.int32)
    attention_mask = mx.array(token_batch.attention_mask, dtype=mx.bool_)
    position_ids = mx.array(token_batch.position_ids, dtype=mx.int32)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        reporter=reporter,
    )
    hidden_states = output.hidden_states[:, token_batch.prefix_tokens :, :, :]
    conditioning_mask = attention_mask[:, token_batch.prefix_tokens :]
    conditioning_positions = position_ids[:, :, token_batch.prefix_tokens :]
    mx.eval(hidden_states, conditioning_mask, conditioning_positions)
    return Krea2PromptConditioning(
        hidden_states=hidden_states,
        attention_mask=conditioning_mask,
        position_ids=conditioning_positions,
        prompts=prompts,
        token_identity=token_batch.identity,
    )


def text_conditioning_provenance(
    resources: Krea2Resources,
    receipt: Qwen3VLLoadReceipt | None = None,
    *,
    system_prompt: str | None = None,
    prefix_tokens: int | None = None,
    tokenizer_identity: str | None = None,
) -> Krea2TextConditioningProvenance:
    """Resolve the exact resource and Qwen target identity without loading weights.

    ``tokenizer_identity`` lets a caller that already loaded the tokenizer pass
    its object identity instead of deriving the source tables a second time.
    """
    tokenizer_cache_identity = (
        resources.tokenizer.tokenizer_identity if tokenizer_identity is None else tokenizer_identity
    )
    target_count, target_digest = qwen3_vl_target_identity(
        resources.text_config,
        resources.capabilities.selected_text_layers,
    )
    if receipt is not None and (receipt.target_count, receipt.target_digest) != (
        target_count,
        target_digest,
    ):
        raise RuntimeError("loaded Krea Qwen target identity changed after preparation")
    if prefix_tokens is None:
        prefix_tokens = (
            KREA_PREFIX_TOKENS
            if system_prompt is None
            else len(
                load_component_tokenizer(resources.tokenizer).encode(
                    regular_text_prefix(system_prompt)
                )
            )
        )
    result = Krea2TextConditioningProvenance(
        declared_variant=resources.variant.value,
        is_distilled=resources.capabilities.is_distilled,
        recipe_identity=resources.recipe_identity,
        tokenizer_source_identity=resources.tokenizer.source_fingerprint,
        tokenizer_graph_identity=resources.tokenizer.graph_fingerprint,
        tokenizer_cache_identity=tokenizer_cache_identity,
        text_source_identity=resources.text_encoder.source_fingerprint,
        text_graph_identity=resources.text_encoder.graph_fingerprint,
        text_dtype_policy=resources.text_encoder.dtype_policy,
        selected_layers=resources.capabilities.selected_text_layers,
        qwen_target_count=target_count,
        qwen_target_digest=target_digest,
        compatibility_identity="",
        system_prompt=system_prompt,
        prefix_tokens=prefix_tokens,
    )
    return replace(result, compatibility_identity=payload_digest(result._identity_fields()))


def _serialized_prompts(
    metadata: Mapping[str, str],
    *,
    source: Path,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    raw_positive = metadata.get("positive_prompts")
    raw_negative = metadata.get("negative_prompts")
    if raw_positive is None or raw_negative is None:
        raise ValueError(f"{source}: text conditioning is missing prompt metadata")
    try:
        positive_value: object = json.loads(raw_positive)
        negative_value: object = json.loads(raw_negative)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{source}: text-conditioning prompt metadata is invalid") from exc
    if (
        not isinstance(positive_value, list)
        or not positive_value
        or any(not isinstance(prompt, str) for prompt in positive_value)
    ):
        raise ValueError(
            f"{source}: positive text-conditioning prompts must be strings in a nonempty batch"
        )
    positive = tuple(positive_value)
    if negative_value is None:
        negative = None
    elif (
        isinstance(negative_value, list)
        and len(negative_value) == len(positive)
        and all(isinstance(prompt, str) for prompt in negative_value)
    ):
        negative = tuple(negative_value)
    else:
        raise ValueError(
            f"{source}: negative text-conditioning prompts must be null or match the positive batch"
        )
    return positive, negative


def read_text_conditioning_prompts(
    path: Path | str,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Read the authoritative prompt batch from a Krea sidecar header only."""
    source = Path(path).expanduser().absolute()
    inventory = inspect_safetensors(source)
    metadata = dict(inventory.metadata)
    if metadata.get("schema_version") != TEXT_CONDITIONING_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: unsupported Krea text-conditioning schema "
            f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("artifact") != TEXT_CONDITIONING_ARTIFACT:
        raise ValueError(f"{source}: expected a Krea 2 text-conditioning artifact")
    return _serialized_prompts(metadata, source=source)


def text_conditioning_artifact(conditioning: Krea2TextConditioning) -> TensorArtifact:
    """Build the public Krea encoded-prompt artifact."""
    tensors: tuple[tuple[str, mx.array], ...] = (
        ("positive_hidden_states", conditioning.positive.hidden_states),
        ("positive_attention_mask", conditioning.positive.attention_mask),
        ("positive_position_ids", conditioning.positive.position_ids),
    )
    metadata: tuple[tuple[str, str], ...] = (
        ("schema_version", TEXT_CONDITIONING_SCHEMA_VERSION),
        ("artifact", TEXT_CONDITIONING_ARTIFACT),
        (
            "positive_prompts",
            json.dumps(list(conditioning.positive.prompts), separators=(",", ":")),
        ),
        ("positive_token_identity", conditioning.positive.token_identity),
        (
            "negative_prompts",
            json.dumps(
                None if conditioning.negative is None else list(conditioning.negative.prompts),
                separators=(",", ":"),
            ),
        ),
        (
            "provenance",
            json.dumps(
                conditioning.provenance.to_metadata(),
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )
    if conditioning.negative is not None:
        tensors += (
            ("negative_hidden_states", conditioning.negative.hidden_states),
            ("negative_attention_mask", conditioning.negative.attention_mask),
            ("negative_position_ids", conditioning.negative.position_ids),
        )
        metadata += (("negative_token_identity", conditioning.negative.token_identity),)
    return TensorArtifact(
        name=TEXT_CONDITIONING,
        tensors=tensors,
        metadata=metadata,
        reporting_phase="save text conditioning",
    )


def _validate_branch_descriptors(
    descriptors: Mapping[str, SafetensorsTensor],
    *,
    prefix: str,
    batch: int,
    resources: Krea2Resources,
    source: Path,
) -> None:
    expected_shapes = {
        f"{prefix}_hidden_states": (
            batch,
            512,
            len(resources.capabilities.selected_text_layers),
            resources.text_config.hidden_size,
        ),
        f"{prefix}_attention_mask": (batch, 512),
        f"{prefix}_position_ids": (3, batch, 512),
    }
    expected_dtypes = {
        f"{prefix}_hidden_states": _FLOAT_STORAGE_DTYPES,
        f"{prefix}_attention_mask": frozenset({"BOOL"}),
        f"{prefix}_position_ids": frozenset({"I32"}),
    }
    for name, expected_shape in expected_shapes.items():
        descriptor = descriptors[name]
        if descriptor.shape != expected_shape:
            raise ValueError(
                f"{source}: {name} must have shape {expected_shape}, got {descriptor.shape}"
            )
        if descriptor.dtype not in expected_dtypes[name]:
            expected = ", ".join(sorted(expected_dtypes[name]))
            raise ValueError(
                f"{source}: {name} must use {expected} storage, got {descriptor.dtype}"
            )


def _materialized_branch(
    arrays: Mapping[str, mx.array],
    *,
    prefix: str,
    prompts: tuple[str, ...],
    token_identity: str,
    source: Path,
    prefix_tokens: int = KREA_PREFIX_TOKENS,
) -> Krea2PromptConditioning:
    hidden_states = arrays[f"{prefix}_hidden_states"]
    attention_mask = arrays[f"{prefix}_attention_mask"]
    position_ids = arrays[f"{prefix}_position_ids"]
    if hidden_states.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{source}: {prefix}_hidden_states has unsupported loaded dtype")
    if attention_mask.dtype != mx.bool_:
        raise ValueError(f"{source}: {prefix}_attention_mask must load as boolean")
    if position_ids.dtype != mx.int32:
        raise ValueError(f"{source}: {prefix}_position_ids must load as int32")
    if not bool(mx.all(mx.isfinite(hidden_states)).item()):
        raise ValueError(f"{source}: {prefix}_hidden_states must contain only finite values")
    if not bool(mx.all(position_ids >= 0).item()) or not bool(
        mx.all(position_ids < 512 + prefix_tokens).item()
    ):
        raise ValueError(f"{source}: {prefix}_position_ids are outside the encoded sequence")
    mx.eval(hidden_states, attention_mask, position_ids)
    return Krea2PromptConditioning(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prompts=prompts,
        token_identity=token_identity,
    )


def load_text_conditioning(
    path: Path | str,
    resources: Krea2Resources,
    *,
    expected_prompts: tuple[str, ...] | None = None,
    expected_negative_prompts: tuple[str, ...] | None = None,
    system_prompt: str | None = None,
    reporter: Reporter | None = None,
    metadata_policy: ReplayIdentityPolicy = "require",
) -> Krea2TextConditioning:
    """Load a structurally exact saved Krea prompt product without Qwen."""
    if metadata_policy not in {"require", "observe"}:
        raise ValueError("metadata_policy must be require or observe")
    source = Path(path).expanduser().absolute()
    sink = reporter if reporter is not None else NullReporter()
    with reported_phase(sink, "load text conditioning"):
        inventory = inspect_safetensors(source)
        metadata = dict(inventory.metadata)
        schema_matches = metadata.get("schema_version") == TEXT_CONDITIONING_SCHEMA_VERSION
        artifact_matches = metadata.get("artifact") == TEXT_CONDITIONING_ARTIFACT
        if metadata_policy == "require" and not schema_matches:
            raise ValueError(
                f"{source}: unsupported Krea text-conditioning schema "
                f"{metadata.get('schema_version')!r}"
            )
        if metadata_policy == "require" and not artifact_matches:
            raise ValueError(f"{source}: expected a Krea 2 text-conditioning artifact")
        if metadata_policy == "observe" and (not schema_matches or not artifact_matches):
            _LOG.warning(
                "%s: advisory Krea text-conditioning envelope differs; continuing to "
                "consumed-tensor validation",
                source,
            )

        prompts, negative_prompts = _serialized_prompts(metadata, source=source)
        batch = len(prompts)
        positive_names = {
            "positive_hidden_states",
            "positive_attention_mask",
            "positive_position_ids",
        }
        negative_names = {
            "negative_hidden_states",
            "negative_attention_mask",
            "negative_position_ids",
        }
        names = set(inventory.tensor_names)
        has_negative = bool(names & negative_names)
        required = positive_names | (negative_names if has_negative else set())
        if names != required:
            missing = sorted(required - names)
            extra = sorted(names - required)
            details = []
            if missing:
                details.append(f"missing tensors {missing}")
            if extra:
                details.append(f"unexpected tensors {extra}")
            raise ValueError(f"{source}: " + "; ".join(details))
        if has_negative != (negative_prompts is not None):
            raise ValueError(f"{source}: negative prompt metadata does not match its tensors")
        if expected_prompts is not None and batch != len(expected_prompts):
            saved_rows = "row" if batch == 1 else "rows"
            requested_rows = "row" if len(expected_prompts) == 1 else "rows"
            raise ValueError(
                f"{source}: saved text conditioning has {batch} {saved_rows}, but the request "
                f"expands to {len(expected_prompts)} {requested_rows}; saved rows are already the "
                "expanded prompt batch, so use images_per_prompt=1 when replaying them"
            )
        if expected_prompts is not None and prompts != expected_prompts:
            if metadata_policy == "require":
                raise ValueError(f"{source}: saved positive prompts do not match the request")
            _LOG.warning(
                "%s: saved positive prompts replace the parent prompt during restart",
                source,
            )
        if (
            expected_prompts is not None
            and expected_negative_prompts is not None
            and expected_negative_prompts != negative_prompts
        ):
            if metadata_policy == "require":
                raise ValueError(f"{source}: saved negative prompts do not match the request")
            _LOG.warning(
                "%s: saved negative prompts replace the parent prompt during restart",
                source,
            )
        descriptors = {tensor.name: tensor for tensor in inventory.tensors}
        _validate_branch_descriptors(
            descriptors,
            prefix="positive",
            batch=batch,
            resources=resources,
            source=source,
        )
        if has_negative:
            _validate_branch_descriptors(
                descriptors,
                prefix="negative",
                batch=batch,
                resources=resources,
                source=source,
            )

        prepared_provenance = text_conditioning_provenance(resources, system_prompt=system_prompt)
        parse_error = None
        try:
            raw_provenance: object = json.loads(metadata["provenance"])
            declared_provenance = Krea2TextConditioningProvenance.from_metadata(
                raw_provenance,
                source=source,
            )
        except (KeyError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
            if metadata_policy == "require":
                raise ValueError(f"{source}: invalid text-conditioning provenance: {exc}") from exc
            declared_provenance = None
            parse_error = str(exc)
        if (declared_provenance is None and system_prompt is not None) or (
            declared_provenance is not None
            and (
                declared_provenance.system_prompt != system_prompt
                or declared_provenance.prefix_tokens != prepared_provenance.prefix_tokens
            )
        ):
            raise ValueError(
                f"{source}: saved system prompt does not match the request; re-encode text conditioning"
            )
        identity_match = (
            None
            if declared_provenance is None
            else declared_provenance.is_compatible_with(prepared_provenance)
        )
        if identity_match is False and metadata_policy == "require":
            raise ValueError(
                f"{source}: text-conditioning provenance does not match the prepared resources"
            )
        if metadata_policy == "observe" and identity_match is not True:
            difference = (
                parse_error
                if parse_error is not None
                else "declared identity differs from the prepared resources"
            )
            _LOG.warning(
                "%s: Krea text-conditioning identity is advisory during restart (%s); "
                "continuing because consumed tensors fit",
                source,
                difference,
            )

        arrays = load_weights(source)
        positive_identity = metadata.get("positive_token_identity")
        if not positive_identity:
            raise ValueError(f"{source}: text conditioning is missing positive token identity")
        positive = _materialized_branch(
            arrays,
            prefix="positive",
            prompts=prompts,
            token_identity=positive_identity,
            source=source,
            prefix_tokens=prepared_provenance.prefix_tokens,
        )
        negative = None
        if negative_prompts is not None:
            negative_identity = metadata.get("negative_token_identity")
            if not negative_identity:
                raise ValueError(f"{source}: text conditioning is missing negative token identity")
            negative = _materialized_branch(
                arrays,
                prefix="negative",
                prompts=negative_prompts,
                token_identity=negative_identity,
                source=source,
                prefix_tokens=prepared_provenance.prefix_tokens,
            )
        receipt: dict[str, object] = {
            "source": str(source),
            "policy": metadata_policy,
            "declared_identity": (
                None if declared_provenance is None else declared_provenance.to_metadata()
            ),
            "prepared_identity": prepared_provenance.to_metadata(),
            "identity_match": identity_match,
            "identity_parse_error": parse_error,
        }
        return Krea2TextConditioning(
            positive=positive,
            negative=negative,
            provenance=(
                prepared_provenance if declared_provenance is None else declared_provenance
            ),
            replay_receipt=receipt,
        )


class Krea2TextConditioner:
    """Load Qwen once, encode requested branches sequentially, then release it."""

    def __call__(
        self,
        prompts: PromptInput,
        resources: Krea2Resources,
        *,
        negative_prompts: PromptInput | None = None,
        max_sequence_length: int = 512,
        system_prompt: str | None = None,
        reporter: Reporter | None = None,
    ) -> Krea2TextConditioning:
        positive_prompts = _prompts(prompts, name="prompts")
        negative = (
            None
            if negative_prompts is None
            else _prompts(negative_prompts, name="negative_prompts")
        )
        if negative is not None and len(negative) != len(positive_prompts):
            raise ValueError("negative_prompts must match the positive prompt batch size")
        if resources.tokenizer.graph_fingerprint != QWEN_TOKENIZER_TABLE_POLICY:
            raise ValueError("prepared Krea tokenizer graph is not the native table policy")
        tokenizer = load_component_tokenizer(resources.tokenizer)
        tokenizer_identity = resources.tokenizer.tokenizer_identity
        positive_tokens = prepare_krea2_token_batch(
            tokenizer,
            positive_prompts,
            max_sequence_length=max_sequence_length,
            system_prompt=system_prompt,
        )
        negative_tokens = (
            None
            if negative is None
            else prepare_krea2_token_batch(
                tokenizer,
                negative,
                max_sequence_length=max_sequence_length,
                system_prompt=system_prompt,
            )
        )
        del tokenizer
        lease = ComponentLease(
            Qwen3VLTextModel(
                resources.text_config,
                resources.capabilities.selected_text_layers,
            ),
            cleanup=_cleanup_qwen,
        )
        qwen = lease.value
        try:
            receipt = load_qwen3_vl_text_weights(
                qwen,
                source_bindings=resources.text_encoder.bindings,
                reporter=reporter,
            )
            positive = _encode_batch(
                qwen,
                positive_tokens,
                positive_prompts,
                reporter=reporter,
            )
            encoded_negative = (
                None
                if negative_tokens is None or negative is None
                else _encode_batch(
                    qwen,
                    negative_tokens,
                    negative,
                    reporter=reporter,
                )
            )
        finally:
            del qwen
            lease.close()
        return Krea2TextConditioning(
            positive=positive,
            negative=encoded_negative,
            provenance=text_conditioning_provenance(
                resources,
                receipt,
                system_prompt=system_prompt,
                prefix_tokens=positive_tokens.prefix_tokens,
                tokenizer_identity=tokenizer_identity,
            ),
        )


__all__ = [
    "TEXT_CONDITIONING",
    "Krea2PromptConditioning",
    "Krea2TextConditioner",
    "Krea2TextConditionerPort",
    "Krea2TextConditioning",
    "Krea2TextConditioningProvenance",
    "PromptInput",
    "ReplayIdentityPolicy",
    "load_text_conditioning",
    "read_text_conditioning_prompts",
    "text_conditioning_artifact",
    "text_conditioning_provenance",
]
