"""Immutable checkpoint inventory and prepared cache policy for LTX-2."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import mlx.core as mx

from kinomlx.reporting import Reporter
from kinomlx.settings import Settings
from kinomlx.types import SpatioTemporalScaleFactors

from .cache import (
    ensure_transformer_cache,
    ensure_weight_family_caches,
)
from .cache.schema import file_signature, payload_digest
from .cache.weights import resolve_transformer_dtype
from .compatibility import (
    LTX2CompatibilityReport,
    LTX2ComponentSources,
    inspect_ltx2_compatibility,
)
from .metadata import (
    LTX2CheckpointConfig,
    TransformerConstructorConfig,
    checkpoint_config,
)
from .precision import LTX2DTypePolicy
from .settings import (
    LTX2Settings,
    parse_attention_layout_specs,
    parse_ff_layout_specs,
    parse_ff_quantize_specs,
)
from .text_encoder.tokenizer_cache import TokenizerCache, ensure_tokenizer_cache
from .video_vae.config import VideoVAEConfig

_LTX23_REPO_ID = "Lightricks/LTX-2.3"
_LTX25_REPO_ID = "Lightricks/LTX-2.5"
# The official LTX-2.3 release pairs the QAT-unquantized Gemma-3 encoder;
# "plain" keeps the vanilla instruction-tuned release available as a switch.
_GEMMA_REPO_IDS = {
    "qat": "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
    "plain": "google/gemma-3-12b-it",
}
_LTX23_CHECKPOINT_FILENAMES = (
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-22b-distilled-1.0.safetensors",
)
_SPATIAL_UPSCALER_FILENAMES = (
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
)
_LTX25_VIDEO_VAE_PATHS = {
    "conv": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "diffusion": "vae/ltx-2.5-video-vae-bf16.safetensors",
}
_LTX25_COMPONENT_PATHS = {
    "transformer_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae_path": _LTX25_VIDEO_VAE_PATHS["conv"],
    "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upscaler_path": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    ),
    "temporal_latent_upscaler_path": (
        "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    ),
    "duration_head_path": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
}


class ComponentKind(StrEnum):
    """Recipe-neutral LTX component identities."""

    TRANSFORMER = "transformer"
    TEXT_ENCODER = "text_encoder"
    TEXT_PROJECTION = "text_projection"
    CONNECTOR = "connector"
    VIDEO_VAE = "video_vae"
    SPATIAL_UPSCALER = "spatial_upscaler"
    LATENT_TEMPORAL_UPSCALER = "latent_temporal_upscaler"
    AUDIO_VAE = "audio_vae"
    VOCODER = "vocoder"
    DURATION_HEAD = "duration_head"


class CheckpointLayout(StrEnum):
    """Physical checkpoint packaging normalized by the inventory."""

    MONOLITHIC = "monolithic"
    SPLIT = "split"
    MIXED = "mixed"


@dataclass(frozen=True)
class ComponentMetadata:
    """Small, deeply immutable component annotations."""

    entries: tuple[tuple[str, str], ...] = ()

    @classmethod
    def of(cls, **values: str) -> ComponentMetadata:
        return cls(tuple(sorted(values.items())))

    def get(self, name: str, default: str | None = None) -> str | None:
        for key, value in self.entries:
            if key == name:
                return value
        return default


@dataclass(frozen=True)
class ComponentLocator:
    """Unresolved inventory entry used for monolithic and split layouts."""

    kind: ComponentKind
    source_path: Path
    cache_path: Path | None = None
    metadata: ComponentMetadata = ComponentMetadata()
    source_fingerprint: str | None = None


@dataclass(frozen=True)
class ComponentResource:
    """One immutable source/cache location in a prepared inventory."""

    kind: ComponentKind
    source_path: Path
    source_fingerprint: str
    cache_path: Path | None
    metadata: ComponentMetadata


@dataclass(frozen=True)
class CheckpointIdentity:
    """Stable source identity independent of recipe selection."""

    source_path: Path
    source_fingerprint: str
    model_generation: str
    model_version: str | None
    layout: CheckpointLayout


@dataclass(frozen=True)
class LTX2Capabilities:
    """Checkpoint-derived behavior used for fail-fast recipe validation."""

    model_generation: str
    recipe_families: tuple[str, ...]
    condition_families: tuple[str, ...]
    video_compression: SpatioTemporalScaleFactors
    video_vae_kind: str
    text_encoder_family: str
    native_hdr: bool
    generates_audio: bool
    text_projection_family: str | None = None
    connector_family: str | None = None
    sampler_policy: str = "deterministic-euler-two-stage"
    generated_keyframes: bool = False
    duration_available: bool = False
    temporal_latent_upscaler_available: bool = False
    native_signal_domains: tuple[str, ...] = ("normalized-sdr",)


@dataclass(frozen=True)
class TransformerCachePolicy:
    """Resolved policy needed to bind a fresh model to a prepared cache."""

    include_audio: bool
    video_ff_layout_specs: tuple[tuple[str, str], ...]
    video_ff_layout_layers: tuple[int, ...]
    video_attn_layout_specs: tuple[tuple[str, str], ...]
    video_attn_layout_layers: tuple[int, ...]
    audio_ff_layout_specs: tuple[tuple[str, str], ...] | None
    audio_ff_layout_layers: tuple[int, ...] | None
    audio_attn_layout_specs: tuple[tuple[str, str], ...] | None
    audio_attn_layout_layers: tuple[int, ...] | None
    adaln_pretranspose: bool
    transformer_cache_quantize: str
    video_ff_quantize_specs: tuple[tuple[str, str], ...]
    video_ff_quantize_layers: tuple[int, ...]
    video_ff_quantize_group_size: int | None
    video_ff_quantize_bits: int | None
    resident_blocks: int | None


class _EnsureTransformerCacheOptions(TypedDict):
    transformer_dtype: str | mx.Dtype | None
    cache_mode: str
    cache_root: Path | str | None
    include_audio: bool
    video_ff_layout_specs: tuple[tuple[str, str], ...]
    video_ff_layout_layers: tuple[int, ...]
    video_attn_layout_specs: tuple[tuple[str, str], ...]
    video_attn_layout_layers: tuple[int, ...]
    audio_ff_layout_specs: tuple[tuple[str, str], ...] | None
    audio_ff_layout_layers: tuple[int, ...] | None
    audio_attn_layout_specs: tuple[tuple[str, str], ...] | None
    audio_attn_layout_layers: tuple[int, ...] | None
    adaln_pretranspose: bool
    transformer_cache_quantize: str
    video_ff_quantize_specs: tuple[tuple[str, str], ...]
    video_ff_quantize_layers: tuple[int, ...]
    video_ff_quantize_group_size: int | None
    video_ff_quantize_bits: int | None


@dataclass(frozen=True)
class TransformerExecutionPolicy:
    """Immutable constructor and allocator settings for component leases."""

    use_steel_attention: bool
    compile_attention: bool
    steel_attention_d64: bool
    steel_attention_probe: bool
    fast_mode: bool
    compile_block_groups: int | None
    transformer_compile_group_size: int | None
    mlx_cache_limit_bytes: int | None


@dataclass(frozen=True)
class LTX2Resources:
    """Prepared on-disk assets and immutable policies, never live weights."""

    checkpoint: CheckpointIdentity
    components: tuple[ComponentResource, ...]
    capabilities: LTX2Capabilities
    dtype_policy: LTX2DTypePolicy
    cache_policy: TransformerCachePolicy
    execution_policy: TransformerExecutionPolicy
    video_vae_config: VideoVAEConfig
    transformer_config: TransformerConstructorConfig | None = None
    tokenizer_cache: TokenizerCache | None = None

    def optional(self, kind: ComponentKind) -> ComponentResource | None:
        """Return an inventory item when the checkpoint/discovery provides it."""
        for component in self.components:
            if component.kind is kind:
                return component
        return None

    def require(self, kind: ComponentKind) -> ComponentResource:
        """Return a required inventory item or fail with its stable identity."""
        component = self.optional(kind)
        if component is None:
            raise LookupError(f"prepared resources do not provide {kind.value}")
        return component

    @property
    def weights_path(self) -> Path:
        if self.checkpoint.layout is not CheckpointLayout.MONOLITHIC:
            raise LookupError(
                f"{self.checkpoint.layout.value} resources do not have a monolithic weights_path"
            )
        return self.checkpoint.source_path

    @property
    def transformer_path(self) -> Path:
        return self.require(ComponentKind.TRANSFORMER).source_path

    @property
    def gemma_path(self) -> Path | None:
        component = self.optional(ComponentKind.TEXT_ENCODER)
        return None if component is None else component.source_path

    @property
    def spatial_upscaler_path(self) -> Path:
        return self.require(ComponentKind.SPATIAL_UPSCALER).source_path

    @property
    def temporal_upscaler_path(self) -> Path:
        return self.require(ComponentKind.LATENT_TEMPORAL_UPSCALER).source_path

    @property
    def duration_head_path(self) -> Path:
        return self.require(ComponentKind.DURATION_HEAD).source_path

    @property
    def transformer_cache_path(self) -> Path:
        cache_path = self.require(ComponentKind.TRANSFORMER).cache_path
        if cache_path is None:
            raise LookupError("prepared transformer resource has no cache path")
        return cache_path

    @property
    def transformer_cache_hash(self) -> str:
        return self.transformer_cache_path.parent.name.rsplit("-", 1)[-1]


def _repo_cache_dir(hf_home: Path, repo_id: str) -> Path:
    return hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"


def _cached_snapshots(hf_home: Path, repo_id: str) -> tuple[Path, ...]:
    root = _repo_cache_dir(hf_home, repo_id) / "snapshots"
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )


def _ltx25_component_relative_path(name: str, *, video_vae: str) -> str:
    if name == "video_vae_path":
        return _LTX25_VIDEO_VAE_PATHS[video_vae]
    return _LTX25_COMPONENT_PATHS[name]


def _cached_ltx25_component(
    snapshot: Path,
    name: str,
    *,
    video_vae: str = "conv",
) -> Path | None:
    candidate = snapshot / _ltx25_component_relative_path(name, video_vae=video_vae)
    return candidate if candidate.is_file() else None


def _local_ltx25_component(
    root: Path,
    name: str,
    *,
    video_vae: str = "conv",
) -> Path | None:
    """Find one official LTX-2.5 component in a local pack directory."""
    relative = Path(_ltx25_component_relative_path(name, video_vae=video_vae))
    for candidate in (root / relative, root / relative.name):
        if candidate.is_file():
            return candidate
    return None


def _discover_ltx25_pack_directory(
    settings: LTX2Settings,
    root: Path,
) -> LTX2Settings:
    """Expand one local LTX-2.5 pack root into its component paths."""
    if settings.model_generation not in (None, "2.5"):
        raise ValueError("an LTX-2.5 pack directory cannot be used with model_generation=2.3")
    pack = root.expanduser().absolute()
    transformer_path = settings.transformer_path or _local_ltx25_component(
        pack,
        "transformer_path",
    )
    text_encoder_path = settings.text_encoder_path
    if text_encoder_path is None and settings.gemma_path is None:
        text_encoder_path = _local_ltx25_component(pack, "text_encoder_path")
    video_vae_path = settings.video_vae_path or _local_ltx25_component(
        pack,
        "video_vae_path",
        video_vae=settings.video_vae,
    )

    missing: list[str] = []
    if transformer_path is None:
        missing.append(_ltx25_component_relative_path("transformer_path", video_vae="conv"))
    if text_encoder_path is None and settings.gemma_path is None:
        missing.append(_ltx25_component_relative_path("text_encoder_path", video_vae="conv"))
    if video_vae_path is None:
        missing.append(
            _ltx25_component_relative_path(
                "video_vae_path",
                video_vae=settings.video_vae,
            )
        )
    if missing:
        expected = ", ".join(missing)
        raise ValueError(
            f"LTX-2.5 pack directory {pack} is missing required components: {expected}; "
            "preserve the official subdirectories or place the canonical filenames "
            "directly in the pack root"
        )

    def optional(name: str, explicit: Path | None) -> Path | None:
        return explicit if explicit is not None else _local_ltx25_component(pack, name)

    return replace(
        settings,
        model_generation="2.5",
        weights_path=None,
        transformer_path=transformer_path,
        text_encoder_path=text_encoder_path,
        video_vae_path=video_vae_path,
        audio_vae_path=optional("audio_vae_path", settings.audio_vae_path),
        spatial_upscaler_path=optional(
            "spatial_upscaler_path",
            settings.spatial_upscaler_path,
        ),
        temporal_latent_upscaler_path=optional(
            "temporal_latent_upscaler_path",
            settings.temporal_latent_upscaler_path,
        ),
        duration_head_path=optional("duration_head_path", settings.duration_head_path),
    )


def _discover_selected_diffusion_video_vae(
    settings: LTX2Settings,
    hf_home: Path,
) -> LTX2Settings:
    """Resolve the named shared diffusion decoder independently of the transformer."""
    if settings.video_vae != "diffusion" or settings.video_vae_path is not None:
        return settings
    for snapshot in _cached_snapshots(hf_home, _LTX25_REPO_ID):
        candidate = _cached_ltx25_component(
            snapshot,
            "video_vae_path",
            video_vae="diffusion",
        )
        if candidate is not None:
            return replace(settings, video_vae_path=candidate)
    raise ValueError(
        "no cached LTX-2.5 diffusion video VAE was found; download that artifact or set "
        "--video-vae-path"
    )


def _discover_generation_selection(
    settings: LTX2Settings,
    infrastructure: Settings,
) -> LTX2Settings:
    """Resolve a cached default pack without making filenames compatibility gates."""
    if settings.weights_path is not None and settings.weights_path.expanduser().is_dir():
        return _discover_ltx25_pack_directory(settings, settings.weights_path)
    generation = settings.model_generation or "2.3"
    hf_home = infrastructure.hf_home.expanduser()
    if settings.weights_path is not None or settings.transformer_path is not None:
        return _discover_selected_diffusion_video_vae(settings, hf_home)
    if generation == "2.3":
        for snapshot in _cached_snapshots(hf_home, _LTX23_REPO_ID):
            for filename in _LTX23_CHECKPOINT_FILENAMES:
                candidate = snapshot / filename
                if candidate.is_file():
                    return _discover_selected_diffusion_video_vae(
                        replace(settings, weights_path=candidate),
                        hf_home,
                    )
        raise ValueError(
            "no cached default LTX-2.3 checkpoint was found; set --weights-path or "
            "KINO_WEIGHTS_PATH"
        )

    snapshots = _cached_snapshots(hf_home, _LTX25_REPO_ID)
    for snapshot in snapshots:
        required_names = {"transformer_path"}
        if settings.text_encoder_path is None and settings.gemma_path is None:
            required_names.add("text_encoder_path")
        if settings.video_vae_path is None:
            required_names.add("video_vae_path")
        required = {
            name: snapshot / _ltx25_component_relative_path(name, video_vae=settings.video_vae)
            for name in required_names
        }
        if not all(path.is_file() for path in required.values()):
            continue

        return replace(
            settings,
            transformer_path=_cached_ltx25_component(snapshot, "transformer_path"),
            text_encoder_path=(
                settings.text_encoder_path
                if settings.text_encoder_path is not None or settings.gemma_path is not None
                else _cached_ltx25_component(snapshot, "text_encoder_path")
            ),
            video_vae_path=(
                settings.video_vae_path
                if settings.video_vae_path is not None
                else _cached_ltx25_component(
                    snapshot,
                    "video_vae_path",
                    video_vae=settings.video_vae,
                )
            ),
            audio_vae_path=(
                settings.audio_vae_path
                if settings.audio_vae_path is not None
                else _cached_ltx25_component(snapshot, "audio_vae_path")
            ),
            spatial_upscaler_path=(
                settings.spatial_upscaler_path
                if settings.spatial_upscaler_path is not None
                else _cached_ltx25_component(snapshot, "spatial_upscaler_path")
            ),
            temporal_latent_upscaler_path=(
                settings.temporal_latent_upscaler_path
                if settings.temporal_latent_upscaler_path is not None
                else _cached_ltx25_component(snapshot, "temporal_latent_upscaler_path")
            ),
            duration_head_path=(
                settings.duration_head_path
                if settings.duration_head_path is not None
                else _cached_ltx25_component(snapshot, "duration_head_path")
            ),
        )
    raise ValueError(
        f"no complete cached LTX-2.5 pack with video_vae={settings.video_vae} was found; "
        "set --transformer-path and the component overrides"
    )


def _discover_gemma_path(
    settings: LTX2Settings,
    infrastructure: Settings,
) -> Path | None:
    if settings.gemma_path is not None:
        path = settings.gemma_path.expanduser().resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"Gemma directory has no config.json: {path}")
        return path
    # No cross-variant fallback: a missing snapshot stays unresolved rather
    # than silently swapping encoders, and downstream compatibility reports
    # the text encoder as unavailable.
    repo_id = _GEMMA_REPO_IDS[settings.gemma_variant]
    for snapshot in _cached_snapshots(infrastructure.hf_home.expanduser(), repo_id):
        if (snapshot / "config.json").is_file():
            return snapshot.resolve()
    return None


def _discover_spatial_upscaler_path(
    settings: LTX2Settings,
    infrastructure: Settings,
    weights_path: Path,
) -> Path | None:
    if settings.spatial_upscaler_path is not None:
        path = settings.spatial_upscaler_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"spatial upscaler does not exist: {path}")
        return path
    for filename in _SPATIAL_UPSCALER_FILENAMES:
        sibling = weights_path.parent / filename
        if sibling.is_file():
            return sibling.resolve()
    for snapshot in _cached_snapshots(infrastructure.hf_home.expanduser(), _LTX23_REPO_ID):
        for filename in _SPATIAL_UPSCALER_FILENAMES:
            candidate = snapshot / filename
            if candidate.is_file():
                return candidate.resolve()
    return None


def _source_fingerprint(path: Path) -> str:
    signature_path = path / "config.json" if path.is_dir() else path
    return payload_digest({"source": file_signature(signature_path)})


def _artifact_fingerprint(path: Path) -> str:
    """Return a cheap cache/receipt identity that never gates compatibility."""
    if path.is_dir():
        return f"source:{_source_fingerprint(path)}"
    if path.is_symlink():
        blob_name = path.resolve().name
        if re.fullmatch(r"[0-9a-f]{64}", blob_name):
            return f"sha256:{blob_name}"
    return f"source:{_source_fingerprint(path)}"


def resolve_component_inventory(
    locators: Iterable[ComponentLocator],
) -> tuple[ComponentResource, ...]:
    """Normalize monolithic or split locators into one immutable inventory."""
    resources = []
    seen: set[ComponentKind] = set()
    for locator in locators:
        if locator.kind in seen:
            raise ValueError(f"component inventory repeats {locator.kind.value}")
        seen.add(locator.kind)
        # Preserve a Hugging Face snapshot's logical filename. Its symlink
        # target is an extensionless blob, which MLX cannot format-dispatch.
        source_path = locator.source_path.expanduser().absolute()
        cache_path = (
            None if locator.cache_path is None else locator.cache_path.expanduser().resolve()
        )
        resources.append(
            ComponentResource(
                kind=locator.kind,
                source_path=source_path,
                source_fingerprint=(
                    locator.source_fingerprint
                    if locator.source_fingerprint is not None
                    else _source_fingerprint(source_path)
                ),
                cache_path=cache_path,
                metadata=locator.metadata,
            )
        )
    return tuple(sorted(resources, key=lambda item: item.kind.value))


def _cache_policy(settings: LTX2Settings) -> TransformerCachePolicy:
    mirrored = settings.audio_layout_mirror
    return TransformerCachePolicy(
        include_audio=True,
        video_ff_layout_specs=parse_ff_layout_specs(settings.video_ff_layout_specs),
        video_ff_layout_layers=tuple(settings.video_ff_layout_layers),
        video_attn_layout_specs=parse_attention_layout_specs(settings.video_attn_layout_specs),
        video_attn_layout_layers=tuple(settings.video_attn_layout_layers),
        audio_ff_layout_specs=(
            None if mirrored else parse_ff_layout_specs(settings.audio_ff_layout_specs)
        ),
        audio_ff_layout_layers=(None if mirrored else tuple(settings.audio_ff_layout_layers)),
        audio_attn_layout_specs=(
            None if mirrored else parse_attention_layout_specs(settings.audio_attn_layout_specs)
        ),
        audio_attn_layout_layers=(None if mirrored else tuple(settings.audio_attn_layout_layers)),
        adaln_pretranspose=settings.adaln_pretranspose,
        transformer_cache_quantize=settings.transformer_cache_quantize,
        video_ff_quantize_specs=parse_ff_quantize_specs(settings.video_ff_quantize_specs),
        video_ff_quantize_layers=tuple(settings.video_ff_quantize_layers),
        video_ff_quantize_group_size=settings.video_ff_quantize_group_size,
        video_ff_quantize_bits=settings.video_ff_quantize_bits,
        resident_blocks=settings.transformer_resident_blocks,
    )


def _transformer_cache_options(
    policy: TransformerCachePolicy,
    settings: LTX2Settings,
    infrastructure: Settings,
) -> _EnsureTransformerCacheOptions:
    return {
        "transformer_dtype": settings.transformer_dtype,
        "cache_mode": infrastructure.cache_mode,
        "cache_root": infrastructure.cache_dir,
        "include_audio": policy.include_audio,
        "video_ff_layout_specs": policy.video_ff_layout_specs,
        "video_ff_layout_layers": policy.video_ff_layout_layers,
        "video_attn_layout_specs": policy.video_attn_layout_specs,
        "video_attn_layout_layers": policy.video_attn_layout_layers,
        "audio_ff_layout_specs": policy.audio_ff_layout_specs,
        "audio_ff_layout_layers": policy.audio_ff_layout_layers,
        "audio_attn_layout_specs": policy.audio_attn_layout_specs,
        "audio_attn_layout_layers": policy.audio_attn_layout_layers,
        "adaln_pretranspose": policy.adaln_pretranspose,
        "transformer_cache_quantize": policy.transformer_cache_quantize,
        "video_ff_quantize_specs": policy.video_ff_quantize_specs,
        "video_ff_quantize_layers": policy.video_ff_quantize_layers,
        "video_ff_quantize_group_size": policy.video_ff_quantize_group_size,
        "video_ff_quantize_bits": policy.video_ff_quantize_bits,
    }


def _execution_policy(
    settings: LTX2Settings,
    infrastructure: Settings,
) -> TransformerExecutionPolicy:
    return TransformerExecutionPolicy(
        use_steel_attention=settings.steel_attention,
        compile_attention=settings.compile_attention,
        steel_attention_d64=settings.steel_attention_d64,
        steel_attention_probe=settings.steel_attention_probe,
        fast_mode=settings.fast_mode,
        compile_block_groups=settings.compile_block_groups,
        transformer_compile_group_size=settings.transformer_compile_group_size,
        mlx_cache_limit_bytes=(
            None
            if infrastructure.mlx_cache_limit_gb is None
            else int(infrastructure.mlx_cache_limit_gb * 1024**3)
        ),
    )


def _required_component_path(
    raw: Path | None,
    name: str,
    *,
    allow_directory: bool = False,
) -> Path:
    if raw is None:
        raise ValueError(f"{name} is required for componentized LTX-2 packaging")
    path = raw.expanduser().absolute()
    if not path.is_file() and not (allow_directory and path.is_dir()):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _optional_component_path(
    raw: Path | None,
    name: str,
    *,
    allow_directory: bool = False,
) -> Path | None:
    if raw is None:
        return None
    path = raw.expanduser().absolute()
    if not path.is_file() and not (allow_directory and path.is_dir()):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _component_metadata(
    *,
    family: str,
    config_digest: str,
    layout: CheckpointLayout,
    generation: str,
    declared_model_version: str | None = None,
) -> ComponentMetadata:
    return ComponentMetadata.of(
        family=family,
        layout=layout.value,
        generation=generation,
        config_digest=config_digest,
        declared_model_version=(
            "absent" if declared_model_version is None else declared_model_version
        ),
    )


def _prepare_inspected_resources(
    settings: LTX2Settings,
    infrastructure: Settings,
    *,
    layout: CheckpointLayout,
    sources: LTX2ComponentSources,
    report: LTX2CompatibilityReport,
    reporter: Reporter | None,
) -> LTX2Resources:
    """Build an inventory and generation-exact transformer cache."""
    transformer_config = report.checkpoint.transformer
    text_config = report.text_encoder
    video_config = report.video_vae
    if (text_config is None) != (report.text_encoder_source is None):
        raise ValueError(f"{report.label} compatibility has an incomplete text encoder")
    if text_config is None:
        if report.text_projection is not None or report.text_projection_source is not None:
            raise ValueError(f"{report.label} compatibility has an orphan text projection")
    elif report.text_projection is None or report.text_projection_source is None:
        raise ValueError(f"{report.label} compatibility requires a text projection")
    if report.connectors is None or report.connector_source is None:
        raise ValueError(f"{report.label} compatibility requires audio/video connectors")
    if video_config is None or report.video_vae_source is None:
        raise ValueError(f"{report.label} compatibility requires a video VAE")

    tokenizer_cache = (
        None
        if report.text_encoder_source is None
        else ensure_tokenizer_cache(
            report.text_encoder_source,
            cache_root=infrastructure.cache_dir,
            cache_mode=infrastructure.cache_mode,
        )
    )
    cache_policy = _cache_policy(settings)
    transformer_cache = ensure_transformer_cache(
        sources.transformer,
        constructor_config=transformer_config,
        reporter=reporter,
        **_transformer_cache_options(cache_policy, settings, infrastructure),
    )
    video_families = ensure_weight_family_caches(
        report.video_vae_source,
        families=("video_vae",),
        source_component="video_vae",
        cache_mode=infrastructure.cache_mode,
        cache_root=infrastructure.cache_dir,
        reporter=reporter,
    )
    audio_families = None
    if report.audio_vae_source is not None and report.audio_vae is not None:
        audio_families = ensure_weight_family_caches(
            report.audio_vae_source,
            families=("audio_vae", "vocoder"),
            source_component="audio_vae_vocoder",
            cache_mode=infrastructure.cache_mode,
            cache_root=infrastructure.cache_dir,
            reporter=reporter,
        )

    fingerprints: dict[Path, str] = {}

    def fingerprint(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in fingerprints:
            fingerprints[resolved] = _artifact_fingerprint(path)
        return fingerprints[resolved]

    generation = report.model_generation
    transformer_metadata = _component_metadata(
        family="ltx2-av-transformer",
        config_digest=transformer_config.config_digest,
        layout=layout,
        generation=generation,
        declared_model_version=transformer_config.declared_model_version,
    )
    transformer_metadata = ComponentMetadata(
        tuple(
            sorted(
                (
                    *transformer_metadata.entries,
                    (
                        "inferred_constructor_fields",
                        ",".join(transformer_config.inferred_fields) or "none",
                    ),
                )
            )
        )
    )
    connector_metadata = _component_metadata(
        family="ltx2-av-connectors",
        config_digest=report.connectors.config_digest,
        layout=layout,
        generation=generation,
    )
    video_digest = payload_digest({"constructor": repr(video_config)})
    locators = [
        ComponentLocator(
            ComponentKind.TRANSFORMER,
            sources.transformer,
            transformer_cache.cache_path,
            metadata=transformer_metadata,
            source_fingerprint=fingerprint(sources.transformer),
        ),
        ComponentLocator(
            ComponentKind.CONNECTOR,
            report.connector_source,
            metadata=connector_metadata,
            source_fingerprint=fingerprint(report.connector_source),
        ),
        ComponentLocator(
            ComponentKind.VIDEO_VAE,
            report.video_vae_source,
            video_families.cache_paths["video_vae"],
            metadata=_component_metadata(
                family=f"video-vae-{video_config.decoder_kind}",
                config_digest=video_digest,
                layout=layout,
                generation=generation,
            ),
            source_fingerprint=fingerprint(report.video_vae_source),
        ),
    ]
    if text_config is not None and report.text_encoder_source is not None:
        if report.text_projection is None or report.text_projection_source is None:
            raise RuntimeError("text projection disappeared after compatibility inspection")
        locators.extend(
            (
                ComponentLocator(
                    ComponentKind.TEXT_ENCODER,
                    report.text_encoder_source,
                    metadata=_component_metadata(
                        family=text_config.family,
                        config_digest=text_config.config_digest,
                        layout=layout,
                        generation=generation,
                    ),
                    source_fingerprint=fingerprint(report.text_encoder_source),
                ),
                ComponentLocator(
                    ComponentKind.TEXT_PROJECTION,
                    report.text_projection_source,
                    metadata=_component_metadata(
                        family="ltx2-text-projection",
                        config_digest=report.text_projection.config_digest,
                        layout=layout,
                        generation=generation,
                    ),
                    source_fingerprint=fingerprint(report.text_projection_source),
                ),
            )
        )
    if report.audio_vae_source is not None and report.audio_vae is not None:
        if audio_families is None:
            raise RuntimeError("audio family caches were not prepared")
        audio_metadata = _component_metadata(
            family="audio-vae-vocoder",
            config_digest=payload_digest({"constructor": repr(report.audio_vae)}),
            layout=layout,
            generation=generation,
        )
        for kind in (ComponentKind.AUDIO_VAE, ComponentKind.VOCODER):
            locators.append(
                ComponentLocator(
                    kind,
                    report.audio_vae_source,
                    audio_families.cache_paths[kind.value],
                    metadata=audio_metadata,
                    source_fingerprint=fingerprint(report.audio_vae_source),
                )
            )
    if sources.spatial_upscaler is not None and report.spatial_upscaler is not None:
        locators.append(
            ComponentLocator(
                ComponentKind.SPATIAL_UPSCALER,
                sources.spatial_upscaler,
                metadata=_component_metadata(
                    family="latent-spatial-x2",
                    config_digest=report.spatial_upscaler.config_digest,
                    layout=layout,
                    generation=generation,
                ),
                source_fingerprint=fingerprint(sources.spatial_upscaler),
            )
        )
    if sources.temporal_upscaler is not None and report.temporal_upscaler is not None:
        locators.append(
            ComponentLocator(
                ComponentKind.LATENT_TEMPORAL_UPSCALER,
                sources.temporal_upscaler,
                metadata=_component_metadata(
                    family="latent-temporal-x2",
                    config_digest=report.temporal_upscaler.config_digest,
                    layout=layout,
                    generation=generation,
                ),
                source_fingerprint=fingerprint(sources.temporal_upscaler),
            )
        )
    if sources.duration_head is not None and report.duration_head is not None:
        locators.append(
            ComponentLocator(
                ComponentKind.DURATION_HEAD,
                sources.duration_head,
                metadata=_component_metadata(
                    family="duration-head",
                    config_digest=report.duration_head.config_digest,
                    layout=layout,
                    generation=generation,
                ),
                source_fingerprint=fingerprint(sources.duration_head),
            )
        )

    dtype = resolve_transformer_dtype(settings.transformer_dtype)
    if dtype is None:
        dtype = mx.bfloat16
    checkpoint_fingerprint = fingerprint(sources.transformer)
    sampler_policy = (
        "deterministic-euler-two-stage"
        if generation == "2.3"
        else "ancestral-stage1-deterministic-stage2"
    )
    return LTX2Resources(
        checkpoint=CheckpointIdentity(
            source_path=sources.transformer.absolute(),
            source_fingerprint=checkpoint_fingerprint,
            model_generation=generation,
            model_version=transformer_config.declared_model_version,
            layout=layout,
        ),
        components=resolve_component_inventory(locators),
        capabilities=LTX2Capabilities(
            model_generation=generation,
            recipe_families=("distilled",),
            condition_families=("text", "image", "keyframe"),
            video_compression=video_config.encoder_scale,
            video_vae_kind=video_config.decoder_kind,
            text_encoder_family=("unavailable" if text_config is None else text_config.family),
            native_hdr=generation == "2.5",
            generates_audio=report.audio_vae is not None,
            text_projection_family="packaged-49-state",
            connector_family="ltx2-8-layer",
            sampler_policy=sampler_policy,
            generated_keyframes=(
                generation == "2.5" and transformer_config.use_keyframes_abs_pos_embedding
            ),
            duration_available=report.duration_head is not None,
            temporal_latent_upscaler_available=report.temporal_upscaler is not None,
            native_signal_domains=(
                ("normalized-sdr", "acescct-working-codes")
                if generation == "2.5"
                else ("normalized-sdr",)
            ),
        ),
        dtype_policy=LTX2DTypePolicy.reference(transformer=dtype),
        cache_policy=cache_policy,
        execution_policy=_execution_policy(settings, infrastructure),
        video_vae_config=video_config,
        transformer_config=transformer_config,
        tokenizer_cache=tokenizer_cache,
    )


def _prepare_split_resources(
    settings: LTX2Settings,
    infrastructure: Settings,
    reporter: Reporter | None,
) -> LTX2Resources:
    """Inspect a componentized pack of either generation."""
    transformer_path = _required_component_path(settings.transformer_path, "transformer_path")
    raw_text_path = (
        settings.text_encoder_path
        if settings.text_encoder_path is not None
        else settings.gemma_path
    )
    text_path = _required_component_path(
        raw_text_path,
        "text_encoder_path/gemma_path",
        allow_directory=True,
    )
    parsed = checkpoint_config(transformer_path)
    video_path = (
        transformer_path
        if settings.video_vae_path is None and parsed.video_vae is not None
        else _required_component_path(settings.video_vae_path, "video_vae_path")
    )
    audio_path = _optional_component_path(settings.audio_vae_path, "audio_vae_path")
    spatial_path = _optional_component_path(
        settings.spatial_upscaler_path,
        "spatial_upscaler_path",
    )
    temporal_path = _optional_component_path(
        settings.temporal_latent_upscaler_path,
        "temporal_latent_upscaler_path",
    )
    duration_path = _optional_component_path(settings.duration_head_path, "duration_head_path")
    sources = LTX2ComponentSources(
        transformer=transformer_path,
        text_encoder=text_path,
        video_vae=video_path,
        audio_vae=audio_path,
        spatial_upscaler=spatial_path,
        temporal_upscaler=temporal_path,
        duration_head=duration_path,
        text_projection_candidates=(text_path, transformer_path),
        connector_candidates=(transformer_path, text_path),
    )
    report = inspect_ltx2_compatibility(
        sources,
        parsed_checkpoint=parsed,
        expected_generation=settings.model_generation,
    )
    return _prepare_inspected_resources(
        settings,
        infrastructure,
        layout=CheckpointLayout.SPLIT,
        sources=sources,
        report=report,
        reporter=reporter,
    )


def prepare_resources(
    settings: LTX2Settings,
    *,
    infrastructure: Settings | None = None,
    reporter: Reporter | None = None,
) -> LTX2Resources:
    """Validate sources, prepare caches, and return no live model state."""
    host = infrastructure if infrastructure is not None else Settings.from_env()
    host.validate()
    resolved = _discover_generation_selection(settings.resolve_presets(), host)
    resolved.validate()
    if resolved.weights_path is None:
        if resolved.uses_split_checkpoint:
            return _prepare_split_resources(resolved, host, reporter)
        raise ValueError("weights_path or transformer_path is required")
    # Preserve the logical ``.safetensors`` snapshot name while preparing
    # caches. Hugging Face snapshots are symlinks to extensionless blob names;
    # resolving here both defeats MLX format inference and changes cache-dir
    # identity. File signatures and the frozen inventory canonicalize the
    # target independently.
    weights_path = resolved.weights_path.expanduser().absolute()
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {weights_path}")

    baseline = checkpoint_config(weights_path)
    transformer_path = (
        weights_path
        if resolved.transformer_path is None
        else _required_component_path(resolved.transformer_path, "transformer_path")
    )
    parsed = baseline if resolved.transformer_path is None else checkpoint_config(transformer_path)
    explicit_text = (
        resolved.text_encoder_path
        if resolved.text_encoder_path is not None
        else resolved.gemma_path
    )
    text_path: Path | None
    if explicit_text is not None:
        text_path = _required_component_path(
            explicit_text,
            "text_encoder_path/gemma_path",
            allow_directory=True,
        )
    elif parsed.model_generation == "2.3":
        text_path = _discover_gemma_path(resolved, host)
    else:
        text_path = weights_path
    video_path = (
        _required_component_path(resolved.video_vae_path, "video_vae_path")
        if resolved.video_vae_path is not None
        else weights_path
        if baseline.video_vae is not None
        else transformer_path
        if parsed.video_vae is not None
        else None
    )
    if video_path is None:
        raise ValueError(
            f"{weights_path.name}: LTX-{parsed.model_generation} monolithic packaging has no "
            "video VAE config; supply video_vae_path"
        )
    audio_path = (
        _required_component_path(resolved.audio_vae_path, "audio_vae_path")
        if resolved.audio_vae_path is not None
        else weights_path
    )
    upscaler_path = _discover_spatial_upscaler_path(resolved, host, weights_path)
    temporal_path = _optional_component_path(
        resolved.temporal_latent_upscaler_path,
        "temporal_latent_upscaler_path",
    )
    duration_path = _optional_component_path(resolved.duration_head_path, "duration_head_path")
    projection_candidates = tuple(
        path for path in (text_path, transformer_path, weights_path) if path is not None
    )
    connector_candidates = tuple(
        path for path in (transformer_path, weights_path, text_path) if path is not None
    )
    sources = LTX2ComponentSources(
        transformer=transformer_path,
        text_encoder=text_path,
        video_vae=video_path,
        audio_vae=audio_path,
        spatial_upscaler=upscaler_path,
        temporal_upscaler=temporal_path,
        duration_head=duration_path,
        text_projection_candidates=projection_candidates,
        connector_candidates=connector_candidates,
    )
    report = inspect_ltx2_compatibility(
        sources,
        parsed_checkpoint=parsed,
        expected_generation=resolved.model_generation,
    )
    if report.video_vae is None:
        raise ValueError(f"{report.label} compatibility requires a video VAE")

    return _prepare_inspected_resources(
        resolved,
        host,
        layout=(
            CheckpointLayout.MONOLITHIC
            if resolved.transformer_path is None
            else CheckpointLayout.MIXED
        ),
        sources=sources,
        report=report,
        reporter=reporter,
    )


__all__ = [
    "CheckpointIdentity",
    "CheckpointLayout",
    "ComponentKind",
    "ComponentLocator",
    "ComponentMetadata",
    "ComponentResource",
    "LTX2Capabilities",
    "LTX2CheckpointConfig",
    "LTX2Resources",
    "TransformerConstructorConfig",
    "TransformerCachePolicy",
    "TransformerExecutionPolicy",
    "TokenizerCache",
    "checkpoint_config",
    "prepare_resources",
    "resolve_component_inventory",
]
