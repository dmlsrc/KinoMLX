"""Stable cache schemas, modes, payloads, and artifact path derivation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NotRequired, Required, TypedDict, Unpack

import mlx.core as mx

from kinomlx._typing import JsonObject, JsonValue

from .policy import (
    DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    LayoutSpecs,
    normalize_transformer_layout_policy,
)

# Schema values explicitly invalidate incompatible on-disk cache contracts.
TRANSFORMER_CACHE_SCHEMA_VERSION = 3
FAMILY_CACHE_SCHEMA_VERSION = 3
COMPONENT_CACHE_SCHEMA_VERSION = 2
FP8_DEQUANT_POLICY_VERSION = 2

LAYOUT_KEY_PREFIX = "__layout__."
QUANT_KEY_PREFIX = "__quant__."

TRANSFORMER_CACHE_QUANTIZE_OFF = "off"
TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS = "mxfp8-blocks"
TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE = "mxfp8-blocks-pretranspose"
TRANSFORMER_CACHE_QUANTIZE_MODES = (
    TRANSFORMER_CACHE_QUANTIZE_OFF,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS,
    TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE,
)
WEIGHT_FAMILIES = ("connector", "video_vae", "audio_vae", "vocoder")
WEIGHT_FAMILY_FILENAMES = {
    "connector": "connector.safetensors",
    "video_vae": "video_vae.safetensors",
    "audio_vae": "audio_vae.safetensors",
    "vocoder": "vocoder.safetensors",
}
WEIGHT_FAMILY_LABELS = {
    "connector": "Connector",
    "video_vae": "Video VAE",
    "audio_vae": "Audio VAE",
    "vocoder": "Vocoder",
}

DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES = 4 * 1024**3


class TransformerCacheOptions(TypedDict, total=False):
    """Keyword contract shared by transformer cache identity helpers."""

    include_audio: Required[bool]
    transformer_dtype: str | mx.Dtype | None
    video_ff_layout_specs: LayoutSpecs
    video_ff_layout_layers: tuple[int, ...]
    video_attn_layout_specs: LayoutSpecs
    video_attn_layout_layers: tuple[int, ...]
    audio_ff_layout_specs: LayoutSpecs | None
    audio_ff_layout_layers: tuple[int, ...] | None
    audio_attn_layout_specs: LayoutSpecs | None
    audio_attn_layout_layers: tuple[int, ...] | None
    adaln_pretranspose: bool
    transformer_cache_quantize: str
    video_ff_quantize_specs: tuple[tuple[str, str], ...]
    video_ff_quantize_layers: tuple[int, ...]
    video_ff_quantize_group_size: int | None
    video_ff_quantize_bits: int | None
    video_ff_dtype: mx.Dtype | None
    audio_ff_dtype: mx.Dtype | None
    constructor_identity: tuple[tuple[str, str], ...] | None


class ComponentCacheIdentity(TypedDict):
    """Keyword identity required for one component cache."""

    source_component: str
    source_fingerprint: str
    model_generation: str
    constructor_identity: tuple[tuple[str, str], ...]
    dtype_policy: NotRequired[tuple[tuple[str, str], ...]]
    layout_policy: NotRequired[tuple[tuple[str, str], ...]]


def default_cache_root() -> Path:
    """Return the library fallback used when no Settings cache root is passed."""
    return Path("~/.cache/kinomlx").expanduser()


def file_signature(path: Path | str) -> JsonObject:
    """Return the source identity folded into cache invalidation payloads."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dtype_payload_name(dtype: str | mx.Dtype | None) -> str | None:
    """Map an MLX dtype to the spelling used by cache metadata."""
    if dtype is None:
        return None
    if dtype == mx.bfloat16:
        return "bfloat16"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.float32:
        return "float32"
    return str(dtype)


def _canonical_layout_specs(
    specs: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    return [{"target": target, "layout": layout} for target, layout in specs]


def _canonical_quant_specs(
    specs: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    return [{"target": target, "mode": mode} for target, mode in specs]


def transformer_cache_payload(
    weights_path: Path | str,
    *,
    transformer_dtype: str | mx.Dtype | None = None,
    include_audio: bool,
    video_ff_layout_specs: LayoutSpecs = DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    video_ff_layout_layers: tuple[int, ...] = (),
    video_attn_layout_specs: LayoutSpecs = DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    video_attn_layout_layers: tuple[int, ...] = (),
    audio_ff_layout_specs: LayoutSpecs | None = None,
    audio_ff_layout_layers: tuple[int, ...] | None = None,
    audio_attn_layout_specs: LayoutSpecs | None = None,
    audio_attn_layout_layers: tuple[int, ...] | None = None,
    adaln_pretranspose: bool = False,
    transformer_cache_quantize: str = TRANSFORMER_CACHE_QUANTIZE_OFF,
    video_ff_quantize_specs: tuple[tuple[str, str], ...] = (),
    video_ff_quantize_layers: tuple[int, ...] = (),
    video_ff_quantize_group_size: int | None = None,
    video_ff_quantize_bits: int | None = None,
    video_ff_dtype: mx.Dtype | None = None,
    audio_ff_dtype: mx.Dtype | None = None,
    constructor_identity: tuple[tuple[str, str], ...] | None = None,
) -> JsonObject:
    """Build the canonical transformer cache identity payload."""
    from .weights import normalize_transformer_cache_dtypes

    transformer_dtype, video_ff_dtype, audio_ff_dtype = normalize_transformer_cache_dtypes(
        transformer_dtype,
        video_ff_dtype,
        audio_ff_dtype,
    )
    if transformer_cache_quantize not in TRANSFORMER_CACHE_QUANTIZE_MODES:
        raise ValueError(
            f"Unsupported transformer cache quantization mode: {transformer_cache_quantize}"
        )
    layout_policy = normalize_transformer_layout_policy(
        include_audio=include_audio,
        transformer_dtype=transformer_dtype,
        video_ff_dtype=video_ff_dtype,
        audio_ff_dtype=audio_ff_dtype,
        video_ff_layout_specs=video_ff_layout_specs,
        video_ff_layout_layers=video_ff_layout_layers,
        video_attn_layout_specs=video_attn_layout_specs,
        video_attn_layout_layers=video_attn_layout_layers,
        audio_ff_layout_specs=audio_ff_layout_specs,
        audio_ff_layout_layers=audio_ff_layout_layers,
        audio_attn_layout_specs=audio_attn_layout_specs,
        audio_attn_layout_layers=audio_attn_layout_layers,
        video_ff_quantize_specs=video_ff_quantize_specs,
        video_ff_quantize_layers=video_ff_quantize_layers,
        layouts_enabled=(transformer_cache_quantize == TRANSFORMER_CACHE_QUANTIZE_OFF),
    )
    video_ff_layout_specs = layout_policy.video_ff_specs
    video_ff_layout_layers = layout_policy.video_ff_layers
    video_attn_layout_specs = layout_policy.video_attn_specs
    video_attn_layout_layers = layout_policy.video_attn_layers
    audio_ff_layout_specs = layout_policy.audio_ff_specs
    audio_ff_layout_layers = layout_policy.audio_ff_layers
    audio_attn_layout_specs = layout_policy.audio_attn_specs
    audio_attn_layout_layers = layout_policy.audio_attn_layers
    video_ff_quantize_layers = layout_policy.video_ff_quantize_layers
    payload: JsonObject = {
        "schema_version": TRANSFORMER_CACHE_SCHEMA_VERSION,
        "source": file_signature(weights_path),
        "include_audio": include_audio,
        "video_ff_layout_specs": _canonical_layout_specs(video_ff_layout_specs),
        "video_ff_layout_layers": list(video_ff_layout_layers),
        "video_attn_layout_specs": _canonical_layout_specs(video_attn_layout_specs),
        "video_attn_layout_layers": list(video_attn_layout_layers),
    }
    if constructor_identity is not None:
        payload["constructor"] = dict(constructor_identity)
    if audio_ff_layout_specs:
        payload["audio_ff_layout_specs"] = _canonical_layout_specs(audio_ff_layout_specs)
        payload["audio_ff_layout_layers"] = list(audio_ff_layout_layers)
    if audio_attn_layout_specs:
        payload["audio_attn_layout_specs"] = _canonical_layout_specs(audio_attn_layout_specs)
        payload["audio_attn_layout_layers"] = list(audio_attn_layout_layers)
    if adaln_pretranspose:
        payload["adaln_pretranspose"] = True
    if transformer_cache_quantize != TRANSFORMER_CACHE_QUANTIZE_OFF:
        payload["transformer_cache_quantize"] = transformer_cache_quantize
        if video_ff_quantize_group_size is not None:
            payload["transformer_cache_quantize_group_size"] = video_ff_quantize_group_size
        if video_ff_quantize_bits is not None:
            payload["transformer_cache_quantize_bits"] = video_ff_quantize_bits

    # Import here to keep schema/path derivation independent from tensor loading.
    from .weights import checkpoint_has_fp8_tensors

    if checkpoint_has_fp8_tensors(weights_path):
        payload["fp8_dequant_policy"] = FP8_DEQUANT_POLICY_VERSION
    if video_ff_quantize_specs:
        payload.update(
            {
                "video_ff_quantize_specs": _canonical_quant_specs(video_ff_quantize_specs),
                "video_ff_quantize_layers": list(video_ff_quantize_layers),
                "video_ff_quantize_group_size": video_ff_quantize_group_size,
                "video_ff_quantize_bits": video_ff_quantize_bits,
            }
        )
    if video_ff_dtype is not None and video_ff_dtype != mx.bfloat16:
        payload["video_ff_dtype"] = dtype_payload_name(video_ff_dtype)
    if audio_ff_dtype is not None and audio_ff_dtype != mx.bfloat16:
        payload["audio_ff_dtype"] = dtype_payload_name(audio_ff_dtype)
    if transformer_dtype is not None and transformer_dtype != mx.bfloat16:
        payload["transformer_dtype"] = dtype_payload_name(transformer_dtype)
    return payload


def family_cache_payload(
    weights_path: Path | str,
    *,
    kind: str,
    source_component: str | None = None,
) -> JsonObject:
    """Build one family's source-level cache identity payload."""
    if kind not in WEIGHT_FAMILIES:
        raise ValueError(f"Unsupported weight family: {kind}")
    payload: JsonObject = {
        "schema_version": FAMILY_CACHE_SCHEMA_VERSION,
        "source": file_signature(weights_path),
        "kind": kind,
    }
    if source_component is not None:
        payload["source_component"] = source_component
    return payload


def family_directory_payload(
    weights_path: Path | str,
    *,
    source_component: str | None = None,
) -> JsonObject:
    """Build the shared directory identity for all source-level families."""
    payload: JsonObject = {
        "schema_version": FAMILY_CACHE_SCHEMA_VERSION,
        "source": file_signature(weights_path),
    }
    if source_component is not None:
        payload["source_component"] = source_component
    return payload


def component_cache_payload(
    source_path: Path | str,
    *,
    source_component: str,
    source_fingerprint: str,
    model_generation: str,
    constructor_identity: tuple[tuple[str, str], ...],
    dtype_policy: tuple[tuple[str, str], ...] = (),
    layout_policy: tuple[tuple[str, str], ...] = (),
) -> JsonObject:
    """Build a cache identity for one logical component from any packaging.

    The schema version represents KinoMLX's binding/conversion contract. A
    source-wide tensor-schema hash is intentionally absent: required consumed
    targets are validated by the binder, and unrelated source baggage does not
    define compatibility. Training labels such as ``distilled`` and ``dev``
    have no cache authority.
    """
    if not source_component:
        raise ValueError("source_component must not be empty")
    if not source_fingerprint:
        raise ValueError("source_fingerprint must not be empty")
    if model_generation not in {"2.3", "2.5"}:
        raise ValueError("model_generation must be 2.3 or 2.5")
    return {
        "schema_version": COMPONENT_CACHE_SCHEMA_VERSION,
        "source": file_signature(source_path),
        "source_component": source_component,
        "source_fingerprint": source_fingerprint,
        "model_generation": model_generation,
        "constructor": dict(constructor_identity),
        "dtype_policy": dict(dtype_policy),
        "layout_policy": dict(layout_policy),
    }


def payload_digest(payload: dict[str, JsonValue]) -> str:
    """Return the stable 20-hex cache identity."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def safe_stem(path: Path | str) -> str:
    """Return a bounded filesystem-safe checkpoint stem."""
    stem = Path(path).stem
    safe = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in stem
    )
    return safe[:80] or "transformer"


def transformer_cache_paths(
    weights_path: Path | str,
    cache_root: Path | str | None,
    **options: Unpack[TransformerCacheOptions],
) -> tuple[Path, Path, JsonObject]:
    """Resolve transformer artifact, sidecar, and expected metadata payload."""
    payload = transformer_cache_payload(weights_path, **options)
    root = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    cache_dir = root / f"{safe_stem(weights_path)}-{payload_digest(payload)}"
    return cache_dir / "transformer.safetensors", cache_dir / "metadata.json", payload


def transformer_fp16_ranges_path(cache_file: Path | str) -> Path:
    """Return the per-cache FP16 tensor-peak sidecar path."""
    return Path(cache_file).with_name("fp16-ranges.json")


def weight_family_cache_paths(
    weights_path: Path | str,
    cache_root: Path | str | None,
    family: str,
    *,
    source_component: str | None = None,
) -> tuple[Path, Path, JsonObject]:
    """Resolve one family artifact, sidecar, and expected metadata payload."""
    payload = family_cache_payload(
        weights_path,
        kind=family,
        source_component=source_component,
    )
    directory_payload = family_directory_payload(
        weights_path,
        source_component=source_component,
    )
    root = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    cache_dir = root / (f"{safe_stem(weights_path)}-{payload_digest(directory_payload)}")
    cache_file = cache_dir / WEIGHT_FAMILY_FILENAMES[family]
    metadata_file = cache_dir / f"{cache_file.stem}.metadata.json"
    return cache_file, metadata_file, payload


def component_cache_paths(
    source_path: Path | str,
    cache_root: Path | str | None,
    **identity: Unpack[ComponentCacheIdentity],
) -> tuple[Path, Path, JsonObject]:
    """Resolve the artifact, sidecar, and identity for one logical component."""
    payload = component_cache_payload(source_path, **identity)
    root = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    cache_dir = root / f"{safe_stem(source_path)}-component-{payload_digest(payload)}"
    return cache_dir / "component.safetensors", cache_dir / "metadata.json", payload


__all__ = [
    "DEFAULT_TRANSFORMER_SHARD_LIMIT_BYTES",
    "FAMILY_CACHE_SCHEMA_VERSION",
    "FP8_DEQUANT_POLICY_VERSION",
    "LAYOUT_KEY_PREFIX",
    "QUANT_KEY_PREFIX",
    "COMPONENT_CACHE_SCHEMA_VERSION",
    "TRANSFORMER_CACHE_QUANTIZE_MODES",
    "TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS",
    "TRANSFORMER_CACHE_QUANTIZE_MXFP8_BLOCKS_PRETRANSPOSE",
    "TRANSFORMER_CACHE_QUANTIZE_OFF",
    "TRANSFORMER_CACHE_SCHEMA_VERSION",
    "WEIGHT_FAMILIES",
    "WEIGHT_FAMILY_FILENAMES",
    "WEIGHT_FAMILY_LABELS",
    "default_cache_root",
    "dtype_payload_name",
    "family_cache_payload",
    "family_directory_payload",
    "file_signature",
    "payload_digest",
    "safe_stem",
    "component_cache_paths",
    "component_cache_payload",
    "transformer_cache_paths",
    "transformer_fp16_ranges_path",
    "transformer_cache_payload",
    "weight_family_cache_paths",
]
