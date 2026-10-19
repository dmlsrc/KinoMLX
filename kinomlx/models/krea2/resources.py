"""Immutable Krea 2 source inventory and graph receipts."""

import re
from dataclasses import dataclass, replace
from pathlib import Path

from kinomlx._json import JsonObject
from kinomlx.config_rules import validate_capabilities
from kinomlx.huggingface import resolve_cached_snapshot
from kinomlx.io.identity import payload_digest
from kinomlx.memory_policy import MLXMemoryPolicy
from kinomlx.modeling.artifacts import (
    SourceLayout,
    TokenizerSource,
    WeightSource,
    inspect_tokenizer_source,
    inspect_weight_source,
)
from kinomlx.modeling.dtypes import DTypeDirective, DTypePolicy
from kinomlx.settings import Settings
from kinomlx.tokenizers.byte_bpe import QWEN_TOKENIZER_TABLE_POLICY

from .catalog import (
    DEFAULT_VAE_DECODER,
    DEFAULT_VARIANT,
    UPSCALE2X_VAE_FILENAME,
    UPSCALE2X_VAE_REPO_ID,
    Krea2VAEDecoder,
    Krea2Variant,
    download_command,
    upscale2x_vae_download_command,
    variant_from_distilled,
    variant_spec,
)
from .metadata import (
    Krea2ModelIndex,
    Krea2SchedulerConfig,
    default_model_index,
    default_scheduler_config,
    parse_model_index,
    parse_scheduler_config,
    parse_text_config,
    parse_transformer_config,
    parse_vae_config,
    validate_cross_component_config,
)
from .resource_types import (
    Krea2Capabilities,
    Krea2ComponentKind,
    Krea2ComponentResource,
    Krea2Resources,
    Krea2TensorBinding,
)
from .settings import Krea2Settings, validate_variant_source
from .weight_keys import (
    KREA2_TRANSFORMER_WEIGHT_LAYOUT,
    KREA2_VAE_WEIGHT_LAYOUT,
    KeyBinding,
    text_encoder_bindings,
    transformer_bindings,
    vae_bindings,
)

_REVISION = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class _Baseline:
    root: Path | None
    revision: str
    variant: Krea2Variant
    model_index: Krea2ModelIndex
    scheduler: Krea2SchedulerConfig
    declared_variant: Krea2Variant | None
    variant_requires_explicit: bool


def _declared_variant(
    settings: Krea2Settings,
    *,
    model_index: Krea2ModelIndex | None,
) -> Krea2Variant:
    selected = None if settings.variant is None else Krea2Variant(settings.variant)
    if model_index is None:
        return DEFAULT_VARIANT if selected is None else selected
    return variant_from_distilled(model_index.is_distilled)


def _inspect_baseline_root(
    root: Path,
    *,
    settings: Krea2Settings,
    revision: str,
    require_model_index: bool,
) -> _Baseline:
    if not root.is_dir():
        raise FileNotFoundError(f"Krea 2 source path is not a directory: {root}")
    index_path = root / "model_index.json"
    if index_path.is_file():
        model_index = parse_model_index(index_path)
    elif require_model_index:
        raise FileNotFoundError(f"Krea 2 snapshot is missing model_index.json: {root}")
    else:
        model_index = None
    variant = _declared_variant(settings, model_index=model_index)
    if model_index is None:
        model_index = default_model_index(is_distilled=variant_spec(variant).is_distilled)
    scheduler_path = root / "scheduler" / "scheduler_config.json"
    if scheduler_path.is_file():
        scheduler = parse_scheduler_config(scheduler_path)
    elif require_model_index:
        raise FileNotFoundError(f"Krea 2 snapshot is missing scheduler config: {scheduler_path}")
    else:
        scheduler = default_scheduler_config()
    return _Baseline(
        root=root,
        revision=revision,
        variant=variant,
        model_index=model_index,
        scheduler=scheduler,
        declared_variant=(
            None if not index_path.is_file() else variant_from_distilled(model_index.is_distilled)
        ),
        variant_requires_explicit=not index_path.is_file(),
    )


def _resolve_baseline(settings: Krea2Settings, infrastructure: Settings) -> _Baseline:
    overrides = settings.component_overrides
    if settings.source_path is not None:
        root = settings.source_path.expanduser().resolve()
        revision = root.name if _REVISION.fullmatch(root.name) else "local"
        return _inspect_baseline_root(
            root,
            settings=settings,
            revision=revision,
            require_model_index=False,
        )
    if set(overrides) == {kind.value for kind in Krea2ComponentKind}:
        variant = _declared_variant(settings, model_index=None)
        return _Baseline(
            root=None,
            revision="local-components",
            variant=variant,
            model_index=default_model_index(is_distilled=variant_spec(variant).is_distilled),
            scheduler=default_scheduler_config(),
            declared_variant=None,
            variant_requires_explicit=True,
        )

    variant = DEFAULT_VARIANT if settings.variant is None else Krea2Variant(settings.variant)
    spec = variant_spec(variant)
    try:
        snapshot = resolve_cached_snapshot(infrastructure.hf_home, spec.repo_id)
    except (FileNotFoundError, ValueError) as exc:
        raise FileNotFoundError(
            f"no usable cached main snapshot for {spec.repo_id}; "
            f"run {download_command(variant)} outside KinoMLX"
        ) from exc
    return _inspect_baseline_root(
        snapshot.path,
        settings=replace(settings, variant=variant.value),
        revision=snapshot.revision,
        require_model_index=True,
    )


def _fallback(root: Path | None, *parts: str) -> Path | None:
    if root is None:
        return None
    candidate = root.joinpath(*parts)
    return candidate if candidate.is_file() else None


def _cached_upscale2x_vae(infrastructure: Settings) -> Path:
    try:
        snapshot = resolve_cached_snapshot(infrastructure.hf_home, UPSCALE2X_VAE_REPO_ID)
    except (FileNotFoundError, ValueError) as exc:
        raise FileNotFoundError(
            f"no usable cached {UPSCALE2X_VAE_REPO_ID} snapshot; "
            f"run {upscale2x_vae_download_command()} outside KinoMLX"
        ) from exc
    candidate = snapshot.path / UPSCALE2X_VAE_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(
            f"cached {UPSCALE2X_VAE_REPO_ID} snapshot is missing {UPSCALE2X_VAE_FILENAME}; "
            f"run {upscale2x_vae_download_command()} outside KinoMLX"
        )
    return candidate


def _vae_source_path(
    settings: Krea2Settings,
    infrastructure: Settings,
    root: Path | None,
) -> Path | None:
    requested = (
        DEFAULT_VAE_DECODER
        if settings.vae_decoder is None
        else Krea2VAEDecoder(settings.vae_decoder)
    )
    if settings.vae_path is None and requested is Krea2VAEDecoder.UPSCALE2X:
        return _cached_upscale2x_vae(infrastructure)
    return settings.vae_path or (None if root is None else root / "vae")


def _detect_vae_decoder(source: WeightSource) -> Krea2VAEDecoder:
    shapes = source.tensor_shapes
    heads = tuple(
        (key, shapes[key])
        for key in ("decoder.conv_out.weight", "decoder.head.2.weight")
        if key in shapes
    )
    if len(heads) != 1:
        raise ValueError(
            "Krea 2 VAE must contain exactly one recognized Diffusers or original Wan "
            "decoder output weight"
        )
    key, shape = heads[0]
    if len(shape) != 5:
        raise ValueError(f"Krea 2 VAE tensor {key!r} must be a 5D Conv3d weight")
    output_channels = shape[0]
    if output_channels == Krea2VAEDecoder.QWEN.output_channels:
        detected = Krea2VAEDecoder.QWEN
    elif output_channels == Krea2VAEDecoder.UPSCALE2X.output_channels:
        detected = Krea2VAEDecoder.UPSCALE2X
    else:
        raise ValueError(
            f"Krea 2 VAE tensor {key!r} has unsupported output width {output_channels}; "
            "expected 3 or 12"
        )
    bias_key = key.removesuffix("weight") + "bias"
    bias_shape = shapes.get(bias_key)
    if bias_shape != (output_channels,):
        raise ValueError(
            f"Krea 2 VAE tensor {bias_key!r} must have shape ({output_channels},), got {bias_shape}"
        )
    return detected


def _source_file_by_key(source: WeightSource) -> dict[str, Path]:
    return {
        key: tensor_file.path for tensor_file in source.tensor_files for key in tensor_file.keys
    }


def _attach_files(
    source: WeightSource,
    bindings: tuple[KeyBinding, ...],
) -> tuple[Krea2TensorBinding, ...]:
    files = _source_file_by_key(source)
    return tuple(
        Krea2TensorBinding(
            source_file=files[binding.source_key],
            source_key=binding.source_key,
            target_key=binding.target_key,
            reshape_rows=binding.reshape_rows,
            transform=binding.transform,
        )
        for binding in bindings
    )


def _graph_digest(values: JsonObject) -> str:
    return payload_digest(values)


def _component_resource(
    *,
    kind: Krea2ComponentKind,
    source: WeightSource,
    bindings: tuple[KeyBinding, ...],
    graph_fingerprint: str,
    dtype_policy: DTypePolicy,
    layout: str | None = None,
) -> Krea2ComponentResource:
    return Krea2ComponentResource(
        kind=kind,
        source_path=source.source_path,
        config_path=source.config_path,
        layout=source.layout.value if layout is None else layout,
        source_fingerprint=source.source_fingerprint,
        graph_fingerprint=graph_fingerprint,
        dtype_policy=dtype_policy,
        tensor_count=source.tensor_count,
        bindings=_attach_files(source, bindings),
    )


def _tokenizer_resource(
    source: TokenizerSource,
) -> Krea2ComponentResource:
    return Krea2ComponentResource(
        kind=Krea2ComponentKind.TOKENIZER,
        source_path=source.source_path,
        config_path=source.tokenizer_config,
        layout=SourceLayout.TOKENIZER.value,
        source_fingerprint=source.source_fingerprint,
        graph_fingerprint=QWEN_TOKENIZER_TABLE_POLICY,
        dtype_policy=DTypeDirective.JSON,
        tensor_count=0,
        tokenizer_json=source.tokenizer_json,
        tokenizer_config=source.tokenizer_config,
    )


def prepare_resources(
    settings: Krea2Settings,
    *,
    infrastructure: Settings,
) -> Krea2Resources:
    """Resolve and structurally inspect Krea 2 sources without loading arrays."""
    settings.validate()
    infrastructure.validate()
    baseline = _resolve_baseline(settings, infrastructure)
    validate_variant_source(
        settings,
        declared_variant=(
            None if baseline.declared_variant is None else baseline.declared_variant.value
        ),
        variant_requires_explicit=baseline.variant_requires_explicit,
    )
    root = baseline.root

    tokenizer_path = settings.tokenizer_path or (None if root is None else root / "tokenizer")
    text_path = settings.text_encoder_path or (None if root is None else root / "text_encoder")
    transformer_path = settings.transformer_path or (None if root is None else root / "transformer")
    vae_path = _vae_source_path(settings, infrastructure, root)
    missing = [
        name
        for name, path in (
            ("tokenizer", tokenizer_path),
            ("text_encoder", text_path),
            ("transformer", transformer_path),
            ("vae", vae_path),
        )
        if path is None
    ]
    if missing:
        raise ValueError(f"Krea 2 component selection is incomplete: {', '.join(missing)}")
    assert tokenizer_path is not None
    assert text_path is not None
    assert transformer_path is not None
    assert vae_path is not None

    tokenizer_source = inspect_tokenizer_source(tokenizer_path)
    text_source = inspect_weight_source(
        text_path,
        filenames=("model.safetensors",),
        config_fallback=_fallback(root, "text_encoder", "config.json"),
    )
    transformer_source = inspect_weight_source(
        transformer_path,
        filenames=("diffusion_pytorch_model.safetensors", "model.safetensors"),
        config_fallback=_fallback(root, "transformer", "config.json"),
    )
    vae_source = inspect_weight_source(
        vae_path,
        filenames=("diffusion_pytorch_model.safetensors", "model.safetensors"),
        config_fallback=_fallback(root, "vae", "config.json"),
    )
    vae_decoder = _detect_vae_decoder(vae_source)
    validate_capabilities(
        settings,
        {
            "declared_variant": (
                None if baseline.declared_variant is None else baseline.declared_variant.value
            ),
            "variant_requires_explicit": baseline.variant_requires_explicit,
            "detected_vae_decoder": vae_decoder.value,
        },
        location="[model_settings]",
    )

    text_config = parse_text_config(text_source.config_path)
    if tokenizer_source.maximum_token_id >= text_config.vocab_size:
        raise ValueError(
            "Krea 2 tokenizer ID exceeds the text embedding vocabulary: "
            f"maximum={tokenizer_source.maximum_token_id}, vocab_size={text_config.vocab_size}"
        )
    transformer_config = parse_transformer_config(transformer_source.config_path)
    vae_config = parse_vae_config(vae_source.config_path)
    validate_cross_component_config(
        baseline.model_index,
        transformer_config,
        text_config,
        vae_config,
    )
    text_keys = text_encoder_bindings(
        text_source.tensor_keys,
        text_config,
        baseline.model_index,
    )
    transformer_keys = transformer_bindings(
        transformer_source.tensor_keys,
        transformer_config,
    )
    vae_keys = vae_bindings(vae_source.tensor_keys, vae_config)

    variant = baseline.variant
    transformer_native = any(
        key.startswith(("blocks.", "txtfusion.", "txtmlp."))
        for key in transformer_source.tensor_keys
    )
    transformer_layout = (
        "diffusers-sharded"
        if transformer_source.layout is SourceLayout.SHARDED
        else ("native-monolith" if transformer_native else "diffusers-single")
    )
    text_graph = _graph_digest(
        {
            "config": text_config.config_digest,
            "selected_layers": list(baseline.model_index.selected_text_layers),
            "executed_layers": max(baseline.model_index.selected_text_layers),
        }
    )
    transformer_graph = _graph_digest(
        {
            "config": transformer_config.config_digest,
            "patch_size": baseline.model_index.patch_size,
            "weight_layout": KREA2_TRANSFORMER_WEIGHT_LAYOUT,
        }
    )
    vae_graph_payload: JsonObject = {
        "config": vae_config.config_digest,
        "weight_layout": KREA2_VAE_WEIGHT_LAYOUT,
    }
    if vae_decoder is not DEFAULT_VAE_DECODER:
        vae_graph_payload["decoder"] = vae_decoder.value
        vae_graph_payload["output_scale"] = vae_decoder.output_scale
    vae_graph = _graph_digest(vae_graph_payload)
    spec = variant_spec(variant)
    recipe_identity = _graph_digest(
        {
            "variant": variant.value,
            "is_distilled": spec.is_distilled,
            "scheduler": baseline.scheduler.config_digest,
            "transformer_source": transformer_source.source_fingerprint,
        }
    )
    tokenizer = _tokenizer_resource(tokenizer_source)
    text_encoder = _component_resource(
        kind=Krea2ComponentKind.TEXT_ENCODER,
        source=text_source,
        bindings=text_keys,
        graph_fingerprint=text_graph,
        dtype_policy=DTypeDirective.SOURCE,
    )
    transformer = _component_resource(
        kind=Krea2ComponentKind.TRANSFORMER,
        source=transformer_source,
        bindings=transformer_keys,
        graph_fingerprint=transformer_graph,
        dtype_policy=settings.transformer_dtype,
        layout=transformer_layout,
    )
    vae = _component_resource(
        kind=Krea2ComponentKind.VAE,
        source=vae_source,
        bindings=vae_keys,
        graph_fingerprint=vae_graph,
        dtype_policy=settings.vae_dtype,
    )
    capabilities = Krea2Capabilities(
        variant=variant,
        is_distilled=spec.is_distilled,
        recipe_families=spec.recipe_families,
        negative_prompt=not spec.is_distilled,
        latent_channels=vae_config.z_dim,
        spatial_compression=vae_config.spatial_compression,
        transformer_patch_size=baseline.model_index.patch_size,
        selected_text_layers=baseline.model_index.selected_text_layers,
    )
    return Krea2Resources(
        memory_policy=MLXMemoryPolicy.from_settings(infrastructure),
        variant=variant,
        revision=baseline.revision,
        source_root=root,
        recipe_identity=recipe_identity,
        model_index=baseline.model_index,
        scheduler_config=baseline.scheduler,
        transformer_config=transformer_config,
        text_config=text_config,
        vae_config=vae_config,
        vae_decoder=vae_decoder,
        capabilities=capabilities,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        check_adapter_weights=infrastructure.check_adapter_weights,
        reference_denoise=settings.reference_denoise,
        hf_home=infrastructure.hf_home,
    )


__all__ = [
    "prepare_resources",
]
