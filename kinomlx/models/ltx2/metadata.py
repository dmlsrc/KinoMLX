"""Header-only LTX-2 component configuration and compatibility checks.

This module implements first-party parsing from neutral safetensors metadata
facts. It never loads tensor payloads and never uses filenames or fingerprints
as compatibility gates. Source-wide tensor schemas are deliberately not load
gates: each loader owns complete coverage of the logical targets it consumes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kinomlx._typing import JsonObject, JsonValue
from kinomlx.io.safetensors import read_header, read_metadata

from .audio_vae.config import AudioVAEConfig
from .audio_vae.vocoder import BWEVocoderConfig
from .transformer.graph import convert_checkpoint_key, transformer_parameter_shapes
from .video_vae.config import VideoVAEConfig

_MISSING = object()

_SUPPORTED_GENERATIONS = frozenset({"2.3", "2.5"})
_SUPPORTED_TRANSFORMER_SOURCE_DTYPES = frozenset({"BF16", "F16", "F32", "F8_E4M3"})


def generation_label(model_generation: str) -> str:
    """Return the explicit diagnostic label for one implemented generation."""
    if model_generation not in _SUPPORTED_GENERATIONS:
        raise ValueError(f"unsupported LTX model generation {model_generation!r}")
    return f"LTX-{model_generation}"


_TRANSFORMER_CONSTANTS: dict[str, object] = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_type": "default",
    "double_self_attention": False,
    "norm_elementwise_affine": False,
    "norm_num_groups": 32,
    "num_embeds_ada_norm": 1000,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "cross_attention_norm": True,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "causal_temporal_positioning": True,
    "use_audio_video_cross_attention": True,
    "share_ff": False,
    "av_cross_ada_norm": True,
    "use_embeddings_connector": True,
    "connector_attention_head_dim": 128,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "connector_norm_output": True,
    "use_middle_indices_grid": True,
    "apply_gated_attention": True,
    "connector_apply_gated_attention": True,
    "caption_projection_first_linear": False,
    "caption_projection_second_linear": False,
    "caption_proj_input_norm": False,
    "caption_proj_before_connector": True,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "cross_attention_adaln": True,
    "rope_type": "split",
    "frequencies_precision": "float64",
    "text_encoder_norm_type": "PER_TOKEN_RMS",
}

_TRANSFORMER_DIMENSION_FIELDS = frozenset(
    {
        "num_layers",
        "in_channels",
        "out_channels",
        "num_attention_heads",
        "attention_head_dim",
        "audio_num_attention_heads",
        "audio_attention_head_dim",
        "audio_out_channels",
        "cross_attention_dim",
        "audio_cross_attention_dim",
    }
)
_TRANSFORMER_DIMENSION_DEFAULTS: dict[str, int] = {
    "num_layers": 48,
    "in_channels": 128,
    "out_channels": 128,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_out_channels": 128,
    "cross_attention_dim": 4096,
    "audio_cross_attention_dim": 2048,
}
_TRANSFORMER_FLOAT_FIELDS = frozenset(
    {
        "positional_embedding_theta",
        "timestep_scale_multiplier",
        "av_ca_timestep_scale_multiplier",
        "norm_eps",
    }
)
_TRANSFORMER_FLOAT_DEFAULTS: dict[str, float] = {
    "positional_embedding_theta": 10000.0,
    "timestep_scale_multiplier": 1000.0,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "norm_eps": 1e-6,
}

# Known-generation presets make sparse community metadata usable without
# pretending inferred facts were declared. Effective generation is selected
# first from declared graph fields and consumed tensor structure; unsupported
# graph signatures never inherit either preset.
_COMMON_TRANSFORMER_PRESET: dict[str, object] = {
    **_TRANSFORMER_CONSTANTS,
    **_TRANSFORMER_DIMENSION_DEFAULTS,
    **_TRANSFORMER_FLOAT_DEFAULTS,
    "caption_channels": 3840,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "audio_positional_embedding_max_pos": [20],
}
_LTX23_TRANSFORMER_PRESET: dict[str, object] = {
    **_COMMON_TRANSFORMER_PRESET,
    "ff_bias": True,
    "audio_ff_bias": True,
    "use_keyframes_abs_pos_embedding": False,
    "use_prompt_adaln_single": True,
}
_LTX25_TRANSFORMER_PRESET: dict[str, object] = {
    **_COMMON_TRANSFORMER_PRESET,
    "ff_bias": False,
    "audio_ff_bias": True,
    "use_keyframes_abs_pos_embedding": True,
    "use_prompt_adaln_single": True,
}
_TRANSFORMER_PRESETS = {
    "2.3": _LTX23_TRANSFORMER_PRESET,
    "2.5": _LTX25_TRANSFORMER_PRESET,
}


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(raw: str | None, *, field: str, path: Path) -> JsonObject:
    if raw is None:
        raise ValueError(f"{path.name}: missing {field} metadata")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: {field} metadata is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path.name}: {field} metadata must be an object")
    return cast(JsonObject, parsed)


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _strict_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_int_tuple(value: object, *, field: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    parsed = tuple(_strict_int(item, field=field) for item in value)
    if len(parsed) != length:
        raise ValueError(f"{field} must contain {length} values")
    return parsed


def _expect_if_present(
    mapping: Mapping[str, object],
    key: str,
    expected: object,
    *,
    path: Path,
    component: str,
) -> None:
    if key not in mapping:
        return
    actual = mapping.get(key, _MISSING)
    equivalent = (
        actual.casefold() == expected.casefold()
        if isinstance(actual, str) and isinstance(expected, str)
        else actual == expected and type(actual) is type(expected)
    )
    if not equivalent:
        raise ValueError(
            f"{path.name}: {component} has unsupported {key}={actual!r}; expected {expected!r}"
        )


def _effective_transformer_variant(
    raw: Mapping[str, object],
    *,
    header: Mapping[str, JsonValue],
) -> tuple[bool, bool, bool, bool, bool]:
    """Select the binder from consumed tensor structure when it is available."""
    configured_ff_bias = _strict_bool(raw.get("ff_bias", True), field="transformer.ff_bias")
    configured_audio_ff_bias = _strict_bool(
        raw.get("audio_ff_bias", True),
        field="transformer.audio_ff_bias",
    )
    configured_keyframes = _strict_bool(
        raw.get("use_keyframes_abs_pos_embedding", False),
        field="transformer.use_keyframes_abs_pos_embedding",
    )
    configured_prompt_adaln = _strict_bool(
        raw.get("use_prompt_adaln_single", True),
        field="transformer.use_prompt_adaln_single",
    )
    keys = tuple(
        converted.target_key
        for name in header
        if name != "__metadata__"
        and (converted := convert_checkpoint_key(name, include_audio=True)) is not None
    )
    has_transformer_tensors = any(
        name.startswith("transformer_blocks.")
        or name in {"patchify_proj.weight", "proj_out.weight"}
        for name in keys
    )
    if not has_transformer_tensors:
        return (
            configured_ff_bias,
            configured_audio_ff_bias,
            configured_keyframes,
            configured_prompt_adaln,
            False,
        )
    inferred_ff_bias = any(
        name.startswith("transformer_blocks.")
        and name.endswith((".ff.project_in.proj.bias", ".ff.project_out.bias"))
        for name in keys
    )
    inferred_keyframes = "keyframes_abs_pos_embedding" in keys
    has_audio_ff = any(".audio_ff." in name and name.endswith(".weight") for name in keys)
    inferred_audio_ff_bias = (
        any(
            ".audio_ff." in name
            and name.endswith((".audio_ff.project_in.proj.bias", ".audio_ff.project_out.bias"))
            for name in keys
        )
        if has_audio_ff
        else configured_audio_ff_bias
    )
    video_prompt = "prompt_adaln_single.linear.weight" in keys
    audio_prompt = "audio_prompt_adaln_single.linear.weight" in keys
    has_audio_transformer = any(
        name.startswith(("audio_patchify_proj.", "transformer_blocks.0.audio_")) for name in keys
    )
    mixed_prompt_adaln = (
        "use_prompt_adaln_single" not in raw
        and has_audio_transformer
        and video_prompt != audio_prompt
    )
    has_prompt_tables = any(
        name.startswith("transformer_blocks.") and name.endswith("prompt_scale_shift_table")
        for name in keys
    )
    inferred_prompt_adaln = (
        video_prompt if video_prompt or audio_prompt or has_prompt_tables else None
    )
    return (
        configured_ff_bias if "ff_bias" in raw else inferred_ff_bias,
        (configured_audio_ff_bias if "audio_ff_bias" in raw else inferred_audio_ff_bias),
        (configured_keyframes if "use_keyframes_abs_pos_embedding" in raw else inferred_keyframes),
        (
            configured_prompt_adaln
            if "use_prompt_adaln_single" in raw
            else (
                configured_prompt_adaln if inferred_prompt_adaln is None else inferred_prompt_adaln
            )
        ),
        mixed_prompt_adaln,
    )


@dataclass(frozen=True)
class TransformerConstructorConfig:
    """Immutable constructor facts shared by resource and cache preparation."""

    model_generation: str
    declared_model_version: str | None
    num_layers: int
    video_in_channels: int
    video_out_channels: int
    video_heads: int
    video_head_dim: int
    audio_heads: int
    audio_head_dim: int
    audio_out_channels: int
    video_context_dim: int
    audio_context_dim: int
    caption_channels: int
    video_max_pos: tuple[int, int, int]
    audio_max_pos: tuple[int]
    positional_embedding_theta: float
    timestep_scale_multiplier: float
    av_ca_timestep_scale_multiplier: float
    norm_eps: float
    ff_bias: bool
    audio_ff_bias: bool
    use_keyframes_abs_pos_embedding: bool
    use_prompt_adaln_single: bool
    config_digest: str
    inferred_fields: tuple[str, ...] = ()

    @property
    def video_hidden_dim(self) -> int:
        return self.video_heads * self.video_head_dim

    @property
    def audio_hidden_dim(self) -> int:
        return self.audio_heads * self.audio_head_dim

    def cache_identity(self) -> tuple[tuple[str, str], ...]:
        """Return graph-changing facts in a stable string-only representation."""
        return (
            ("audio_ff_bias", str(self.audio_ff_bias).lower()),
            ("config_digest", self.config_digest),
            ("ff_bias", str(self.ff_bias).lower()),
            ("generation", self.model_generation),
            ("keyframes_abs_pos", str(self.use_keyframes_abs_pos_embedding).lower()),
            ("prompt_adaln", str(self.use_prompt_adaln_single).lower()),
        )


@dataclass(frozen=True)
class LTX2CheckpointConfig:
    """Typed metadata from either a monolith or a split transformer artifact."""

    transformer: TransformerConstructorConfig
    video_vae: VideoVAEConfig | None

    @property
    def model_generation(self) -> str:
        return self.transformer.model_generation

    @property
    def declared_model_version(self) -> str | None:
        return self.transformer.declared_model_version


def _transformer_config(
    raw: Mapping[str, object],
    *,
    header: Mapping[str, JsonValue],
    declared_model_version: str | None,
    path: Path,
) -> TransformerConstructorConfig:
    # Unknown metadata is receipt material, not authority over a graph KinoMLX
    # does not consume. Recognized graph/math fields still reject values that
    # would change execution. Tensor structure selects allocation-bearing
    # options, including stale wrapper metadata.
    (
        ff_bias,
        audio_ff_bias,
        keyframe_embedding,
        prompt_adaln,
        mixed_prompt_adaln,
    ) = _effective_transformer_variant(raw, header=header)
    signature = (ff_bias, keyframe_embedding)
    if signature == (True, False):
        generation = "2.3"
    elif signature == (False, True):
        generation = "2.5"
    else:
        raise ValueError(
            f"{path.name}: LTX transformer constructor combination ff_bias={ff_bias!r}, "
            f"use_keyframes_abs_pos_embedding={keyframe_embedding!r} is not implemented"
        )
    label = generation_label(generation)
    component = f"{label} transformer compatibility"
    if mixed_prompt_adaln:
        raise ValueError(
            f"{path.name}: {component} does not implement mixed video/audio prompt AdaLN tensors"
        )

    preset = _TRANSFORMER_PRESETS[generation]
    inferred_fields = tuple(sorted(preset.keys() - raw.keys()))
    constructor_raw = {**preset, **raw}
    constructor_raw.update(
        {
            "ff_bias": ff_bias,
            "audio_ff_bias": audio_ff_bias,
            "use_keyframes_abs_pos_embedding": keyframe_embedding,
            "use_prompt_adaln_single": prompt_adaln,
        }
    )

    for key, expected in _TRANSFORMER_CONSTANTS.items():
        _expect_if_present(
            constructor_raw,
            key,
            expected,
            path=path,
            component=component,
        )

    dimensions = {
        key: _strict_int(
            constructor_raw[key],
            field=f"{component} {key}",
        )
        for key in _TRANSFORMER_DIMENSION_FIELDS
    }
    if dimensions["num_layers"] != 48:
        raise ValueError(
            f"{path.name}: {component} has unsupported num_layers={dimensions['num_layers']}; "
            "expected 48"
        )
    floats = {
        key: _strict_float(
            constructor_raw[key],
            field=f"{component} {key}",
        )
        for key in _TRANSFORMER_FLOAT_FIELDS
    }
    for key, expected in _TRANSFORMER_FLOAT_DEFAULTS.items():
        if floats[key] != expected:
            raise ValueError(
                f"{path.name}: {component} has unsupported {key}={floats[key]!r}; "
                f"expected {expected!r}"
            )

    video_max_pos = _strict_int_tuple(
        constructor_raw["positional_embedding_max_pos"],
        field=f"{component} positional_embedding_max_pos",
        length=3,
    )
    audio_max_pos = _strict_int_tuple(
        constructor_raw["audio_positional_embedding_max_pos"],
        field=f"{component} audio_positional_embedding_max_pos",
        length=1,
    )
    if video_max_pos != (20, 2048, 2048):
        raise ValueError(
            f"{path.name}: {component} has unsupported positional_embedding_max_pos="
            f"{list(video_max_pos)!r}; expected {[20, 2048, 2048]!r}"
        )
    if audio_max_pos != (20,):
        raise ValueError(
            f"{path.name}: {component} has unsupported audio_positional_embedding_max_pos="
            f"{list(audio_max_pos)!r}; expected {[20]!r}"
        )

    caption_channels = _strict_int(
        constructor_raw["caption_channels"],
        field=f"{component} caption_channels",
    )
    normalized_constructor = {
        **dimensions,
        **floats,
        "positional_embedding_max_pos": list(video_max_pos),
        "audio_positional_embedding_max_pos": list(audio_max_pos),
        "caption_channels": caption_channels,
        "ff_bias": ff_bias,
        "audio_ff_bias": audio_ff_bias,
        "use_keyframes_abs_pos_embedding": keyframe_embedding,
        "use_prompt_adaln_single": prompt_adaln,
    }

    result = TransformerConstructorConfig(
        model_generation=generation,
        declared_model_version=declared_model_version,
        num_layers=dimensions["num_layers"],
        video_in_channels=dimensions["in_channels"],
        video_out_channels=dimensions["out_channels"],
        video_heads=dimensions["num_attention_heads"],
        video_head_dim=dimensions["attention_head_dim"],
        audio_heads=dimensions["audio_num_attention_heads"],
        audio_head_dim=dimensions["audio_attention_head_dim"],
        audio_out_channels=dimensions["audio_out_channels"],
        video_context_dim=dimensions["cross_attention_dim"],
        audio_context_dim=dimensions["audio_cross_attention_dim"],
        caption_channels=caption_channels,
        video_max_pos=(video_max_pos[0], video_max_pos[1], video_max_pos[2]),
        audio_max_pos=(audio_max_pos[0],),
        positional_embedding_theta=floats["positional_embedding_theta"],
        timestep_scale_multiplier=floats["timestep_scale_multiplier"],
        av_ca_timestep_scale_multiplier=floats["av_ca_timestep_scale_multiplier"],
        norm_eps=floats["norm_eps"],
        ff_bias=ff_bias,
        audio_ff_bias=audio_ff_bias,
        use_keyframes_abs_pos_embedding=keyframe_embedding,
        use_prompt_adaln_single=prompt_adaln,
        config_digest=_canonical_digest(normalized_constructor),
        inferred_fields=inferred_fields,
    )
    if result.video_hidden_dim != result.video_context_dim:
        raise ValueError(
            f"{path.name}: {component} video hidden/context dimensions differ: "
            f"{result.video_hidden_dim} != {result.video_context_dim}"
        )
    if result.audio_hidden_dim != result.audio_context_dim:
        raise ValueError(
            f"{path.name}: {component} audio hidden/context dimensions differ: "
            f"{result.audio_hidden_dim} != {result.audio_context_dim}"
        )
    return result


def checkpoint_config(weights_path: Path) -> LTX2CheckpointConfig:
    """Parse a supported LTX-2 transformer graph without loading weights."""
    metadata = read_metadata(weights_path)
    header = read_header(weights_path)
    config = _json_object(metadata.get("config"), field="config", path=weights_path)
    raw_transformer = config.get("transformer")
    if not isinstance(raw_transformer, Mapping):
        raise ValueError(f"{weights_path.name}: missing transformer config")
    declared_version = metadata.get("model_version")
    transformer = _transformer_config(
        raw_transformer,
        header=header,
        declared_model_version=declared_version,
        path=weights_path,
    )
    raw_vae = config.get("vae")
    if raw_vae is None:
        video_vae = None
    elif isinstance(raw_vae, Mapping):
        video_vae = VideoVAEConfig.from_mapping(raw_vae)
    else:
        raise ValueError(f"{weights_path.name}: video VAE config must be an object")
    return LTX2CheckpointConfig(transformer=transformer, video_vae=video_vae)


def _tensor_entry(
    header: Mapping[str, JsonValue],
    name: str,
    *,
    path: Path,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
    component: str | None = None,
) -> Mapping[str, JsonValue]:
    context = "" if component is None else f"{component}: "
    entry = header.get(name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"{path.name}: {context}missing consumed tensor {name}")
    raw_shape = entry.get("shape")
    if not isinstance(raw_shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_shape
    ):
        raise ValueError(f"{path.name}: {context}tensor {name} has an invalid shape")
    actual_shape = tuple(raw_shape)
    if shape is not None and actual_shape != shape:
        raise ValueError(
            f"{path.name}: {context}tensor {name} has shape {actual_shape}, expected {shape}"
        )
    actual_dtype = entry.get("dtype")
    if dtype is not None and actual_dtype != dtype:
        raise ValueError(
            f"{path.name}: {context}tensor {name} has dtype {actual_dtype!r}, expected {dtype!r}"
        )
    return entry


def _tensor_entry_from_aliases(
    header: Mapping[str, JsonValue],
    aliases: Sequence[str],
    *,
    path: Path,
    component: str,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
) -> tuple[str, Mapping[str, JsonValue]]:
    """Resolve one consumed tensor through explicit packaging aliases."""
    matches = [name for name in aliases if name in header]
    if not matches:
        raise ValueError(
            f"{path.name}: {component}: missing consumed tensor {aliases[0]} "
            f"(accepted aliases: {', '.join(aliases)})"
        )
    # Alias order is binding precedence. Community packs sometimes retain both
    # an original wrapper key and a normalized copy; the lower-priority copy is
    # unused baggage, not a reason to reject the selected target.
    name = matches[0]
    return name, _tensor_entry(
        header,
        name,
        path=path,
        shape=shape,
        dtype=dtype,
        component=component,
    )


def _suffix_aliases(header: Mapping[str, JsonValue], suffix: str) -> tuple[str, ...]:
    """Return exact and wrapper-prefixed aliases for one logical suffix."""
    exact = (suffix,) if suffix in header else ()
    wrapped = tuple(
        sorted(
            name
            for name in header
            if name != "__metadata__" and name != suffix and name.endswith("." + suffix)
        )
    )
    return exact + wrapped


_TRANSFORMER_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")


def _transformer_aliases(logical_name: str) -> tuple[str, ...]:
    return tuple(prefix + logical_name for prefix in _TRANSFORMER_PREFIXES)


def _has_transformer_prefix(header: Mapping[str, JsonValue], logical_prefix: str) -> bool:
    return any(
        key.startswith(prefix + logical_prefix)
        for prefix in _TRANSFORMER_PREFIXES
        for key in header
        if key != "__metadata__"
    )


@dataclass(frozen=True)
class TransformerTensorBinding:
    """One selected source tensor bound to one consumed MLX parameter target."""

    source_key: str
    target_key: str
    shape: tuple[int, ...]
    dtype: str


def resolve_transformer_bindings(
    path: Path,
    config: TransformerConstructorConfig,
    *,
    include_audio: bool = True,
) -> tuple[TransformerTensorBinding, ...]:
    """Resolve and validate every consumed transformer target exactly once.

    Canonical aliases win over wrapper-prefixed copies. Additional source
    tensors, including lower-priority aliases, remain non-consumed baggage.
    """
    header = read_header(path)
    component = f"{generation_label(config.model_generation)} transformer compatibility"
    expected = transformer_parameter_shapes(config, include_audio=include_audio)
    candidates: dict[str, list[tuple[int, str]]] = {}
    for source_key in header:
        if source_key == "__metadata__":
            continue
        converted = convert_checkpoint_key(source_key, include_audio=include_audio)
        if converted is None or converted.target_key not in expected:
            continue
        candidates.setdefault(converted.target_key, []).append((converted.priority, source_key))

    bindings = []
    for target_key, expected_shape in expected.items():
        aliases = candidates.get(target_key)
        if not aliases:
            raise ValueError(
                f"{path.name}: {component}: missing consumed transformer target {target_key}"
            )
        _priority, source_key = min(aliases)
        entry = _tensor_entry(
            header,
            source_key,
            path=path,
            component=component,
            shape=expected_shape,
        )
        dtype = entry.get("dtype")
        if not isinstance(dtype, str) or dtype not in _SUPPORTED_TRANSFORMER_SOURCE_DTYPES:
            raise ValueError(
                f"{path.name}: {component}: consumed tensor {source_key} has unsupported "
                f"dtype {dtype!r}"
            )
        bindings.append(
            TransformerTensorBinding(
                source_key=source_key,
                target_key=target_key,
                shape=expected_shape,
                dtype=dtype,
            )
        )
    return tuple(bindings)


def validate_transformer_header(path: Path, config: TransformerConstructorConfig) -> None:
    """Require the complete generation-selected transformer target graph."""
    resolve_transformer_bindings(path, config)


@dataclass(frozen=True)
class TextEncoderHeaderConfig:
    """The implemented Gemma 4 and packaged projection boundary."""

    family: str
    hidden_size: int
    num_hidden_layers: int
    video_projection_dim: int
    audio_projection_dim: int
    tokenizer_json_bytes: int
    config_digest: str


_TEXT_CONFIG_CONSTANTS: dict[str, object] = {
    "attention_bias": False,
    "attention_k_eq_v": True,
    "enable_moe_block": False,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 3840,
    "hidden_size_per_layer_input": 0,
    "intermediate_size": 15360,
    "moe_intermediate_size": None,
    "num_attention_heads": 16,
    "num_experts": None,
    "num_global_key_value_heads": 1,
    "num_hidden_layers": 48,
    "num_key_value_heads": 8,
    "num_kv_shared_layers": 0,
    "rms_norm_eps": 1e-6,
    "sliding_window": 1024,
    "top_k_experts": None,
    "use_bidirectional_attention": "vision",
    "use_double_wide_mlp": False,
    "vocab_size": 262144,
    "vocab_size_per_layer_input": 262144,
}

_GEMMA4_EXPECTED_LAYERS = tuple(
    "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(48)
)
_GEMMA4_EXPECTED_ROPE = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
    },
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
}


def _inspect_gemma4_text_encoder(path: Path) -> TextEncoderHeaderConfig:
    component = "LTX-2.5 text encoder compatibility"
    header = read_header(path)
    raw_metadata = header.get("__metadata__")
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"{path.name}: {component}: missing safetensors metadata")
    gemma = _json_object(
        str(raw_metadata.get("gemma_config")) if "gemma_config" in raw_metadata else None,
        field="gemma_config",
        path=path,
    )
    text = gemma.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError(f"{path.name}: {component}: gemma_config has no text_config object")
    for key, expected in _TEXT_CONFIG_CONSTANTS.items():
        _expect_if_present(text, key, expected, path=path, component=component)
    layer_types = text.get("layer_types", _GEMMA4_EXPECTED_LAYERS)
    if (
        not isinstance(layer_types, Sequence)
        or isinstance(layer_types, (str, bytes))
        or tuple(layer_types) != _GEMMA4_EXPECTED_LAYERS
    ):
        raise ValueError(f"{path.name}: {component}: unsupported layer_types")
    if text.get("rope_parameters", _GEMMA4_EXPECTED_ROPE) != _GEMMA4_EXPECTED_ROPE:
        raise ValueError(f"{path.name}: {component}: unsupported rope_parameters")

    hidden_size = _strict_int(text.get("hidden_size", 3840), field=f"{component} hidden_size")
    num_layers = _strict_int(
        text.get("num_hidden_layers", 48),
        field=f"{component} num_hidden_layers",
    )
    model_anchors = {
        "model.embed_tokens.weight": (262144, hidden_size),
        "model.layers.0.input_layernorm.weight": (hidden_size,),
        f"model.layers.{num_layers - 1}.post_feedforward_layernorm.weight": (hidden_size,),
        "model.norm.weight": (hidden_size,),
    }
    for suffix, shape in model_anchors.items():
        aliases = _suffix_aliases(header, suffix)
        _tensor_entry_from_aliases(
            header,
            aliases or (suffix,),
            path=path,
            component=component,
            shape=shape,
        )
    tokenizer_entry = _tensor_entry(
        header,
        "tokenizer_json",
        path=path,
        dtype="U8",
        component=component,
    )
    tokenizer_shape = tuple(cast(list[int], tokenizer_entry["shape"]))
    if len(tokenizer_shape) != 1 or tokenizer_shape[0] <= 0:
        raise ValueError(f"{path.name}: {component}: tokenizer_json must be nonempty")
    normalized = {key: text.get(key, expected) for key, expected in _TEXT_CONFIG_CONSTANTS.items()}
    normalized["layer_types"] = list(layer_types)
    normalized["rope_parameters"] = _GEMMA4_EXPECTED_ROPE
    return TextEncoderHeaderConfig(
        family="gemma4-12b-ltx",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        video_projection_dim=4096,
        audio_projection_dim=2048,
        tokenizer_json_bytes=tokenizer_shape[0],
        config_digest=_canonical_digest(normalized),
    )


def _inspect_gemma3_text_encoder(path: Path) -> TextEncoderHeaderConfig:
    component = "LTX-2.3 text encoder compatibility"
    if not path.is_dir():
        raise ValueError(f"{path.name}: {component}: expected a Gemma 3 directory")
    config_path = path / "config.json"
    tokenizer_path = path / "tokenizer.json"
    try:
        root = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path.name}: {component}: missing config.json") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: {component}: invalid config.json") from exc
    if not isinstance(root, Mapping) or not isinstance(root.get("text_config"), Mapping):
        raise ValueError(f"{path.name}: {component}: missing text_config")
    text = root["text_config"]
    assert isinstance(text, Mapping)
    expected = {
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "sliding_window": 1024,
    }
    for key, value in expected.items():
        _expect_if_present(text, key, value, path=path, component=component)
    weight_files = tuple(path.glob("model*.safetensors"))
    if not weight_files:
        raise ValueError(f"{path.name}: {component}: no Gemma safetensors found")
    try:
        tokenizer_bytes = tokenizer_path.stat().st_size
    except FileNotFoundError as exc:
        raise ValueError(f"{path.name}: {component}: missing tokenizer.json") from exc
    normalized = {key: text.get(key, value) for key, value in expected.items()}
    return TextEncoderHeaderConfig(
        family="gemma3-12b-it",
        hidden_size=int(normalized["hidden_size"]),
        num_hidden_layers=int(normalized["num_hidden_layers"]),
        video_projection_dim=4096,
        audio_projection_dim=2048,
        tokenizer_json_bytes=tokenizer_bytes,
        config_digest=_canonical_digest(normalized),
    )


def inspect_text_encoder(path: Path, *, model_generation: str) -> TextEncoderHeaderConfig:
    """Inspect the selected generation's text backbone without policing baggage."""
    generation_label(model_generation)
    if model_generation == "2.3":
        return _inspect_gemma3_text_encoder(path)
    return _inspect_gemma4_text_encoder(path)


@dataclass(frozen=True)
class TextProjectionHeaderConfig:
    video_projection_dim: int
    audio_projection_dim: int
    config_digest: str


def inspect_text_projection(
    path: Path,
    *,
    model_generation: str,
    hidden_size: int,
    num_hidden_layers: int,
) -> TextProjectionHeaderConfig:
    """Require only the four packaged projection tensors KinoMLX consumes."""
    component = f"{generation_label(model_generation)} text projection compatibility"
    header = read_header(path)
    projection_input = hidden_size * (num_hidden_layers + 1)
    specs = {
        "text_embedding_projection.video_aggregate_embed.weight": (4096, projection_input),
        "text_embedding_projection.video_aggregate_embed.bias": (4096,),
        "text_embedding_projection.audio_aggregate_embed.weight": (2048, projection_input),
        "text_embedding_projection.audio_aggregate_embed.bias": (2048,),
    }
    resolved = []
    for suffix, shape in specs.items():
        aliases = _suffix_aliases(header, suffix)
        name, _entry = _tensor_entry_from_aliases(
            header,
            aliases or (suffix,),
            path=path,
            component=component,
            shape=shape,
        )
        resolved.append(name)
    return TextProjectionHeaderConfig(
        video_projection_dim=4096,
        audio_projection_dim=2048,
        config_digest=_canonical_digest({"targets": resolved, "shapes": specs}),
    )


@dataclass(frozen=True)
class ConnectorHeaderConfig:
    video_context_dim: int
    audio_context_dim: int
    config_digest: str


def inspect_connectors(
    path: Path,
    *,
    config: TransformerConstructorConfig,
) -> ConnectorHeaderConfig:
    """Check connector anchors independently of transformer packaging."""
    component = f"{generation_label(config.model_generation)} connector compatibility"
    header = read_header(path)
    specs = {
        "video_embeddings_connector.learnable_registers": (128, config.video_context_dim),
        "audio_embeddings_connector.learnable_registers": (128, config.audio_context_dim),
    }
    resolved = []
    for logical_name, shape in specs.items():
        name, _entry = _tensor_entry_from_aliases(
            header,
            _transformer_aliases(logical_name),
            path=path,
            component=component,
            shape=shape,
        )
        resolved.append(name)
    for connector in ("video_embeddings_connector", "audio_embeddings_connector"):
        for index in range(8):
            logical_prefix = f"{connector}.transformer_1d_blocks.{index}."
            if not _has_transformer_prefix(header, logical_prefix):
                raise ValueError(
                    f"{path.name}: {component}: missing consumed {connector} block {index}"
                )
    return ConnectorHeaderConfig(
        video_context_dim=config.video_context_dim,
        audio_context_dim=config.audio_context_dim,
        config_digest=_canonical_digest({"targets": resolved, "shapes": specs}),
    )


@dataclass(frozen=True)
class LatentUpscalerConfig:
    """Header-derived spatial or temporal latent-upscaler graph."""

    kind: str
    in_channels: int
    mid_channels: int
    num_blocks_per_stage: int
    scale: float
    rational_resampler: bool
    config_digest: str


def inspect_latent_upscaler(
    path: Path,
    *,
    expected_kind: str,
    model_generation: str,
) -> LatentUpscalerConfig:
    """Validate consumed upscaler configuration while tolerating baggage."""
    component = (
        f"{generation_label(model_generation)} {expected_kind} latent upscaler compatibility"
    )
    metadata = read_metadata(path)
    raw = _json_object(metadata.get("config"), field="config", path=path)
    _expect_if_present(raw, "dims", 3, path=path, component=component)
    spatial = _strict_bool(
        raw.get("spatial_upsample", expected_kind == "spatial"),
        field=f"{component} spatial_upsample",
    )
    temporal = _strict_bool(
        raw.get("temporal_upsample", expected_kind == "temporal"),
        field=f"{component} temporal_upsample",
    )
    kind = "spatial" if spatial and not temporal else "temporal" if temporal and not spatial else ""
    if kind != expected_kind:
        raise ValueError(f"{path.name}: {component}: got {kind or 'mixed'} configuration")
    expected_scale = 2.0 if expected_kind == "spatial" else 1.0
    scale = _strict_float(
        raw.get("spatial_scale", expected_scale),
        field=f"{component} spatial_scale",
    )
    if expected_kind == "spatial" and scale != 2.0:
        raise ValueError(f"{path.name}: {component}: scale must be 2")
    if expected_kind == "temporal" and scale != 1.0:
        raise ValueError(f"{path.name}: {component}: spatial scale must be 1")
    in_channels = _strict_int(
        raw.get("in_channels", 128),
        field=f"{component} in_channels",
    )
    mid_channels = _strict_int(
        raw.get("mid_channels", 1024),
        field=f"{component} mid_channels",
    )
    num_blocks_per_stage = _strict_int(
        raw.get("num_blocks_per_stage", 4),
        field=f"{component} num_blocks_per_stage",
    )
    rational_resampler = _strict_bool(
        raw.get("rational_resampler", raw.get("use_rational_resampler", False)),
        field=f"{component} rational_resampler",
    )
    if expected_kind == "spatial" and rational_resampler:
        raise ValueError(
            f"{path.name}: {component}: rational spatial resampling is not implemented"
        )
    normalized = {
        "in_channels": in_channels,
        "mid_channels": mid_channels,
        "num_blocks_per_stage": num_blocks_per_stage,
        "spatial_scale": scale,
        "spatial_upsample": spatial,
        "temporal_upsample": temporal,
        "rational_resampler": rational_resampler,
    }
    config = LatentUpscalerConfig(
        kind=kind,
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_blocks_per_stage=num_blocks_per_stage,
        scale=scale,
        rational_resampler=rational_resampler,
        config_digest=_canonical_digest(normalized),
    )
    header = read_header(path)
    for suffix in ("initial_conv.weight", "final_conv.weight"):
        aliases = _suffix_aliases(header, suffix)
        _tensor_entry_from_aliases(
            header,
            aliases or (suffix,),
            path=path,
            component=component,
        )
    return config


@dataclass(frozen=True)
class DurationHeadConfig:
    video_context_dim: int
    audio_context_dim: int
    config_digest: str


def inspect_duration_head(path: Path, *, model_generation: str) -> DurationHeadConfig:
    """Validate the optional duration head's text-context boundary."""
    component = f"{generation_label(model_generation)} duration head compatibility"
    metadata = read_metadata(path)
    raw = _json_object(metadata.get("config"), field="config", path=path)
    transformer = raw.get("transformer")
    duration = raw.get("duration_head")
    if not isinstance(transformer, Mapping) or not isinstance(duration, Mapping):
        raise ValueError(f"{path.name}: {component}: config is incomplete")
    header = read_header(path)
    keys = tuple(key for key in header if key != "__metadata__")
    if not any(key.startswith("duration_head.") or ".duration_head." in key for key in keys):
        raise ValueError(f"{path.name}: {component}: no duration_head tensors found")
    video_context_dim = _strict_int(
        transformer.get("cross_attention_dim", 4096),
        field=f"{component} cross_attention_dim",
    )
    audio_context_dim = _strict_int(
        transformer.get("audio_cross_attention_dim", 2048),
        field=f"{component} audio_cross_attention_dim",
    )
    return DurationHeadConfig(
        video_context_dim=video_context_dim,
        audio_context_dim=audio_context_dim,
        config_digest=_canonical_digest(
            {
                "video_context_dim": video_context_dim,
                "audio_context_dim": audio_context_dim,
            }
        ),
    )


def inspect_audio_vae(path: Path, *, model_generation: str) -> AudioVAEConfig:
    """Identify the consumed audio/Vocoder namespaces without policing extras."""
    component = f"{generation_label(model_generation)} audio VAE compatibility"
    try:
        config = AudioVAEConfig.from_checkpoint(path)
        vocoder_config = BWEVocoderConfig.from_checkpoint(path)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {component}: {exc}") from exc
    header = read_header(path)
    keys = tuple(key for key in header if key != "__metadata__")
    audio_prefixes = ("audio_vae.", "encoder.", "decoder.", "per_channel_statistics.")
    if not any(
        key.startswith(audio_prefixes) or any(f".{prefix}" in key for prefix in audio_prefixes)
        for key in keys
    ):
        raise ValueError(f"{path.name}: {component}: no audio_vae tensors found")
    vocoder_prefixes = ("vocoder.", "bwe_generator.", "mel_stft.")
    if not any(
        key.startswith(vocoder_prefixes) or any(f".{prefix}" in key for prefix in vocoder_prefixes)
        for key in keys
    ):
        raise ValueError(f"{path.name}: {component}: no vocoder tensors found")
    input_rate = vocoder_config.input_sample_rate
    output_rate = vocoder_config.output_sample_rate
    mel_bins = vocoder_config.mel_bins
    if input_rate != config.sample_rate:
        raise ValueError(
            f"{path.name}: {component}: vocoder input sample rate {input_rate} "
            f"does not match audio VAE {config.sample_rate}"
        )
    if output_rate % input_rate:
        raise ValueError(
            f"{path.name}: {component}: vocoder output sample rate {output_rate} "
            f"is not an integer multiple of {input_rate}"
        )
    if mel_bins != config.mel_bins:
        raise ValueError(
            f"{path.name}: {component}: vocoder mel bins {mel_bins} "
            f"do not match audio VAE {config.mel_bins}"
        )
    return config


def inspect_video_vae(path: Path, *, model_generation: str) -> VideoVAEConfig:
    """Validate the consumed video-VAE topology for either packaging layout."""
    component = f"{generation_label(model_generation)} video VAE compatibility"
    try:
        config = VideoVAEConfig.from_checkpoint(path)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {component}: {exc}") from exc
    scale = config.encoder_scale
    if (scale.time, scale.height, scale.width) != (8, 32, 32):
        raise ValueError(
            f"{path.name}: {component}: compression scale "
            f"{(scale.time, scale.height, scale.width)} is not the implemented (8, 32, 32)"
        )
    header = read_header(path)
    keys = tuple(key for key in header if key != "__metadata__")
    namespace_groups = {
        "encoder": ("encoder.", "vae.encoder.", "vae_encoder."),
        "decoder": ("decoder.", "vae.decoder.", "vae_decoder."),
        "per_channel_statistics": (
            "per_channel_statistics.",
            "vae.per_channel_statistics.",
            "vae_encoder.per_channel_statistics.",
            "vae_decoder.per_channel_statistics.",
        ),
    }
    for name, prefixes in namespace_groups.items():
        if not any(
            key.startswith(prefixes) or any(f".{prefix}" in key for prefix in prefixes)
            for key in keys
        ):
            raise ValueError(f"{path.name}: {component}: no {name} tensors found")
    return config


__all__ = [
    "ConnectorHeaderConfig",
    "DurationHeadConfig",
    "LTX2CheckpointConfig",
    "LatentUpscalerConfig",
    "TextEncoderHeaderConfig",
    "TextProjectionHeaderConfig",
    "TransformerConstructorConfig",
    "checkpoint_config",
    "generation_label",
    "inspect_audio_vae",
    "inspect_connectors",
    "inspect_duration_head",
    "inspect_latent_upscaler",
    "inspect_text_encoder",
    "inspect_text_projection",
    "inspect_video_vae",
    "validate_transformer_header",
]
