"""Immutable LTX-2 resource preparation and inventory contracts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import kinomlx.models.ltx2.resources as resources
from kinomlx.models.ltx2.audio_vae.config import AudioVAEConfig
from kinomlx.models.ltx2.compatibility import LTX2CompatibilityReport
from kinomlx.models.ltx2.metadata import (
    ConnectorHeaderConfig,
    DurationHeadConfig,
    LatentUpscalerConfig,
    LTX2CheckpointConfig,
    TextEncoderHeaderConfig,
    TextProjectionHeaderConfig,
    TransformerConstructorConfig,
)
from kinomlx.models.ltx2.resources import (
    CheckpointLayout,
    ComponentKind,
    ComponentLocator,
    ComponentMetadata,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.text_encoder.tokenizer_cache import TokenizerCache
from kinomlx.models.ltx2.video_vae.config import LTX23_VIDEO_VAE_CONFIG
from kinomlx.settings import Settings


def _transformer(
    generation: str,
    *,
    version: str | None = None,
) -> TransformerConstructorConfig:
    is_25 = generation == "2.5"
    return TransformerConstructorConfig(
        model_generation=generation,
        declared_model_version=version or f"{generation}.0",
        num_layers=48,
        video_in_channels=128,
        video_out_channels=128,
        video_heads=32,
        video_head_dim=128,
        audio_heads=32,
        audio_head_dim=64,
        audio_out_channels=128,
        video_context_dim=4096,
        audio_context_dim=2048,
        caption_channels=3840,
        video_max_pos=(20, 2048, 2048),
        audio_max_pos=(20,),
        positional_embedding_theta=10000.0,
        timestep_scale_multiplier=1000.0,
        av_ca_timestep_scale_multiplier=1000.0,
        norm_eps=1e-6,
        ff_bias=not is_25,
        audio_ff_bias=True,
        use_keyframes_abs_pos_embedding=is_25,
        use_prompt_adaln_single=True,
        config_digest=f"transformer-{generation}-config",
    )


def _transformer_25() -> TransformerConstructorConfig:
    return _transformer("2.5")


def _parsed(
    generation: str,
    *,
    version: str | None = None,
    embedded_video_vae: bool = True,
) -> LTX2CheckpointConfig:
    return LTX2CheckpointConfig(
        transformer=_transformer(generation, version=version),
        video_vae=LTX23_VIDEO_VAE_CONFIG if embedded_video_vae else None,
    )


def _split_paths(tmp_path: Path, *names: str) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.safetensors" for name in names}
    for path in paths.values():
        path.touch()
    return paths


def _tokenizer_cache(root: Path) -> TokenizerCache:
    return TokenizerCache(
        model_path=root / "tokenizer.model",
        metadata_path=root / "tokenizer.metadata.json",
        source_json_sha256="source-json",
        model_sha256="model",
    )


def _stub_compatibility(
    monkeypatch,
    *,
    generation: str = "2.5",
    version: str | None = None,
    embedded_video_vae: bool = False,
    failure: str | None = None,
) -> list[tuple[str, Path]]:
    parsed = _parsed(
        generation,
        version=version,
        embedded_video_vae=embedded_video_vae,
    )
    calls: list[tuple[str, Path]] = []

    def parse(path: Path) -> LTX2CheckpointConfig:
        calls.append(("config", path))
        return parsed

    monkeypatch.setattr(resources, "checkpoint_config", parse)

    def inspect(sources, *, parsed_checkpoint=None, expected_generation=None):
        calls.append(("compatibility", sources.transformer))
        assert parsed_checkpoint is parsed
        if expected_generation is not None and expected_generation != generation:
            raise ValueError(
                f"LTX-{expected_generation} compatibility was requested, but consumed "
                f"transformer structure selects LTX-{generation}"
            )
        if failure is not None:
            raise ValueError(f"LTX-{generation} compatibility: {failure}")
        text = (
            None
            if sources.text_encoder is None
            else TextEncoderHeaderConfig(
                family="gemma4-12b-ltx" if generation == "2.5" else "gemma3-12b-it",
                hidden_size=3840,
                num_hidden_layers=48,
                video_projection_dim=4096,
                audio_projection_dim=2048,
                tokenizer_json_bytes=123,
                config_digest=f"text-{generation}-config",
            )
        )
        projection_source = None
        projection = None
        if text is not None:
            projection_source = sources.text_encoder if generation == "2.5" else sources.transformer
            projection = TextProjectionHeaderConfig(
                video_projection_dim=4096,
                audio_projection_dim=2048,
                config_digest="projection-config",
            )
        connector_source = sources.connector_candidates[0] if sources.connector_candidates else None
        connectors = (
            None
            if connector_source is None
            else ConnectorHeaderConfig(
                video_context_dim=4096,
                audio_context_dim=2048,
                config_digest="connector-config",
            )
        )
        spatial = (
            None
            if sources.spatial_upscaler is None
            else LatentUpscalerConfig(
                kind="spatial",
                in_channels=128,
                mid_channels=1024,
                num_blocks_per_stage=4,
                scale=2.0,
                rational_resampler=True,
                config_digest="spatial-config",
            )
        )
        temporal = (
            None
            if sources.temporal_upscaler is None
            else LatentUpscalerConfig(
                kind="temporal",
                in_channels=128,
                mid_channels=1024,
                num_blocks_per_stage=4,
                scale=1.0,
                rational_resampler=True,
                config_digest="temporal-config",
            )
        )
        duration = (
            None
            if sources.duration_head is None
            else DurationHeadConfig(
                video_context_dim=4096,
                audio_context_dim=2048,
                config_digest="duration-config",
            )
        )
        return LTX2CompatibilityReport(
            checkpoint=parsed,
            label=f"LTX-{generation}",
            declared_generation=generation,
            metadata_notes=(),
            text_encoder=text,
            text_encoder_source=sources.text_encoder,
            text_projection=projection,
            text_projection_source=projection_source,
            connectors=connectors,
            connector_source=connector_source,
            video_vae=(None if sources.video_vae is None else LTX23_VIDEO_VAE_CONFIG),
            video_vae_source=sources.video_vae,
            audio_vae=(None if sources.audio_vae is None else AudioVAEConfig()),
            audio_vae_source=sources.audio_vae,
            spatial_upscaler=spatial,
            temporal_upscaler=temporal,
            duration_head=duration,
        )

    monkeypatch.setattr(resources, "inspect_ltx2_compatibility", inspect)
    monkeypatch.setattr(
        resources,
        "_artifact_fingerprint",
        lambda path: f"sha256:{path.name}",
    )
    monkeypatch.setattr(
        resources,
        "ensure_tokenizer_cache",
        lambda path, **kwargs: _tokenizer_cache(Path(kwargs["cache_root"])),
    )

    def transformer_cache(path, **kwargs):
        assert kwargs["constructor_config"] is parsed.transformer
        cache_root = Path(kwargs["cache_root"])
        return SimpleNamespace(cache_path=cache_root / f"transformer-{generation}.safetensors")

    def component_families(path, *, families, cache_root, **kwargs):
        root = Path(cache_root)
        return SimpleNamespace(
            cache_paths={
                family: root / f"{Path(path).stem}-{family}.safetensors" for family in families
            }
        )

    monkeypatch.setattr(resources, "ensure_weight_family_caches", component_families)
    monkeypatch.setattr(resources, "ensure_transformer_cache", transformer_cache)
    return calls


def _walk(value: object):
    yield value
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _walk(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _walk(item)


def test_prepare_resources_returns_deeply_immutable_values_without_live_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    upscaler = tmp_path / "upscaler.safetensors"
    gemma = tmp_path / "gemma"
    checkpoint.touch()
    upscaler.touch()
    gemma.mkdir()
    (gemma / "config.json").touch()
    family_paths = {
        family: tmp_path / "cache" / f"{family}.safetensors"
        for family in ("connector", "video_vae", "audio_vae", "vocoder")
    }
    transformer_cache = tmp_path / "cache" / "identity" / "transformer.safetensors"
    calls: dict[str, object] = {}

    _stub_compatibility(
        monkeypatch,
        generation="2.3",
        embedded_video_vae=True,
    )

    def ensure_families(*args, **kwargs):
        calls["families"] = kwargs
        return SimpleNamespace(cache_paths=family_paths)

    monkeypatch.setattr(resources, "ensure_weight_family_caches", ensure_families)

    def ensure_transformer(*args, **kwargs):
        calls["transformer"] = kwargs
        return SimpleNamespace(cache_path=transformer_cache)

    monkeypatch.setattr(resources, "ensure_transformer_cache", ensure_transformer)
    monkeypatch.setattr(
        resources,
        "ensure_tokenizer_cache",
        lambda *args, **kwargs: _tokenizer_cache(tmp_path / "cache"),
    )
    settings = LTX2Settings(
        weights_path=checkpoint,
        gemma_path=gemma,
        spatial_upscaler_path=upscaler,
        transformer_dtype="float16",
        video_ff_layout_specs=("project_in:pretranspose",),
        video_ff_layout_layers=(1,),
        audio_layout_mirror=False,
        audio_ff_layout_specs=("project_out:pretranspose",),
        audio_ff_layout_layers=(2,),
        video_ff_quantize_specs=("project_out:mxfp8",),
        video_ff_quantize_layers=(3,),
        video_ff_quantize_group_size=32,
        video_ff_quantize_bits=8,
    )

    plan = resources.prepare_resources(
        settings,
        infrastructure=Settings(cache_dir=tmp_path / "cache", cache_mode="rebuild"),
    )

    assert plan.checkpoint.layout is CheckpointLayout.MONOLITHIC
    assert plan.checkpoint.model_version == "2.3.0"
    assert plan.dtype_policy.transformer == mx.float16
    assert plan.transformer_cache_path == transformer_cache.resolve()
    assert plan.require(ComponentKind.TEXT_ENCODER).source_path == gemma.resolve()
    assert plan.require(ComponentKind.SPATIAL_UPSCALER).source_path == upscaler.resolve()
    assert plan.tokenizer_cache == _tokenizer_cache(tmp_path / "cache")
    assert calls["transformer"]["cache_mode"] == "rebuild"
    assert calls["transformer"]["video_ff_layout_specs"] == (("project_in", "pretranspose"),)
    assert calls["transformer"]["audio_ff_layout_specs"] == (("project_out", "pretranspose"),)
    assert calls["transformer"]["video_ff_quantize_specs"] == (("project_out", "mxfp8"),)

    owned = tuple(_walk(plan))
    assert not any(isinstance(value, nn.Module) for value in owned)
    assert not any(isinstance(value, mx.array) for value in owned)
    assert not any(isinstance(value, Settings) for value in owned)
    assert not any(isinstance(value, LTX2Settings) for value in owned)
    assert not any(isinstance(value, (dict, list, set)) for value in owned)


def test_optional_recipe_components_do_not_block_resource_preparation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    cache_paths = {
        family: tmp_path / f"{family}.safetensors"
        for family in ("connector", "video_vae", "audio_vae", "vocoder")
    }
    _stub_compatibility(
        monkeypatch,
        generation="2.3",
        version="2.3.1",
        embedded_video_vae=True,
    )
    monkeypatch.setattr(
        resources,
        "ensure_weight_family_caches",
        lambda *args, **kwargs: SimpleNamespace(cache_paths=cache_paths),
    )
    monkeypatch.setattr(
        resources,
        "ensure_transformer_cache",
        lambda *args, **kwargs: SimpleNamespace(cache_path=tmp_path / "transformer.safetensors"),
    )
    monkeypatch.setattr(resources, "_cached_snapshots", lambda *args: ())

    plan = resources.prepare_resources(LTX2Settings(weights_path=checkpoint))

    assert plan.optional(ComponentKind.TEXT_ENCODER) is None
    assert plan.optional(ComponentKind.SPATIAL_UPSCALER) is None


def test_huggingface_snapshot_symlink_keeps_logical_cache_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    blob = tmp_path / "b33b7fe4bbfe084f"
    checkpoint = tmp_path / "model.safetensors"
    blob.touch()
    checkpoint.symlink_to(blob)
    seen: list[tuple[str, Path]] = []
    family_paths = {
        family: tmp_path / f"{family}.safetensors"
        for family in ("connector", "video_vae", "audio_vae", "vocoder")
    }
    inspection_calls = _stub_compatibility(
        monkeypatch,
        generation="2.3",
        version="2.3.1",
        embedded_video_vae=True,
    )
    monkeypatch.setattr(
        resources,
        "ensure_weight_family_caches",
        lambda path, **kwargs: (
            seen.append(("families", path)) or SimpleNamespace(cache_paths=family_paths)
        ),
    )
    monkeypatch.setattr(
        resources,
        "ensure_transformer_cache",
        lambda path, **kwargs: (
            seen.append(("transformer", path))
            or SimpleNamespace(cache_path=tmp_path / "transformer.safetensors")
        ),
    )
    monkeypatch.setattr(resources, "_cached_snapshots", lambda *args: ())

    plan = resources.prepare_resources(LTX2Settings(weights_path=checkpoint))

    assert inspection_calls == [
        ("config", checkpoint),
        ("compatibility", checkpoint),
    ]
    assert seen == [
        ("transformer", checkpoint),
        ("families", checkpoint),
        ("families", checkpoint),
    ]
    assert plan.weights_path == checkpoint.absolute()
    assert plan.weights_path.is_symlink()


def test_monolithic_and_future_split_sources_share_the_inventory_protocol(
    tmp_path: Path,
) -> None:
    monolith = tmp_path / "monolith.safetensors"
    split_transformer = tmp_path / "transformer.safetensors"
    split_video = tmp_path / "video_vae.safetensors"
    for path in (monolith, split_transformer, split_video):
        path.touch()
    metadata = ComponentMetadata.of(layout="test")

    monolithic = resources.resolve_component_inventory(
        (
            ComponentLocator(ComponentKind.TRANSFORMER, monolith, metadata=metadata),
            ComponentLocator(ComponentKind.VIDEO_VAE, monolith, metadata=metadata),
        )
    )
    split = resources.resolve_component_inventory(
        (
            ComponentLocator(
                ComponentKind.TRANSFORMER,
                split_transformer,
                metadata=metadata,
            ),
            ComponentLocator(ComponentKind.VIDEO_VAE, split_video, metadata=metadata),
        )
    )

    assert tuple(item.kind for item in monolithic) == tuple(item.kind for item in split)
    assert len({item.source_path for item in monolithic}) == 1
    assert len({item.source_path for item in split}) == 2


def test_generation_selector_chooses_the_requested_cached_pack_without_exact_paths(
    tmp_path: Path,
) -> None:
    root_23 = tmp_path / "hub" / "models--Lightricks--LTX-2.3" / "snapshots" / "revision-23"
    checkpoint_23 = root_23 / resources._LTX23_CHECKPOINT_FILENAMES[0]
    checkpoint_23.parent.mkdir(parents=True)
    checkpoint_23.touch()

    root_25 = tmp_path / "hub" / "models--Lightricks--LTX-2.5" / "snapshots" / "revision-25"
    expected_25 = {}
    for field, relative in resources._LTX25_COMPONENT_PATHS.items():
        path = root_25 / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        expected_25[field] = path

    infrastructure = Settings(hf_home=tmp_path)
    default = resources._discover_generation_selection(LTX2Settings(), infrastructure)
    selected = resources._discover_generation_selection(
        LTX2Settings(model_generation="2.5"),
        infrastructure,
    )

    assert default.weights_path == checkpoint_23
    assert default.transformer_path is None
    assert selected.weights_path is None
    assert {
        field: getattr(selected, field) for field in resources._LTX25_COMPONENT_PATHS
    } == expected_25


def test_generation_selector_preserves_explicit_component_overrides(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "hub" / "models--Lightricks--LTX-2.5" / "snapshots" / "revision"
    for field, relative in resources._LTX25_COMPONENT_PATHS.items():
        if field == "video_vae_path":
            continue
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    diffusion_vae = tmp_path / "community-diffusion-vae.safetensors"
    diffusion_vae.touch()

    selected = resources._discover_generation_selection(
        LTX2Settings(
            model_generation="2.5",
            video_vae="diffusion",
            video_vae_path=diffusion_vae,
        ),
        Settings(hf_home=tmp_path),
    )

    assert selected.video_vae_path == diffusion_vae
    assert (
        selected.transformer_path == snapshot / resources._LTX25_COMPONENT_PATHS["transformer_path"]
    )


def test_local_ltx25_pack_directory_resolves_official_tree_and_infers_generation(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "ltx25"
    expected = {}
    for field, relative in resources._LTX25_COMPONENT_PATHS.items():
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        expected[field] = path

    selected = resources._discover_generation_selection(
        LTX2Settings(weights_path=pack),
        Settings(hf_home=tmp_path / "hf"),
    )

    assert selected.model_generation == "2.5"
    assert selected.weights_path is None
    assert {
        field: getattr(selected, field) for field in resources._LTX25_COMPONENT_PATHS
    } == expected


def test_local_ltx25_pack_directory_accepts_flat_canonical_filenames(tmp_path: Path) -> None:
    pack = tmp_path / "ltx25-flat"
    pack.mkdir()
    expected = {}
    for field, relative in resources._LTX25_COMPONENT_PATHS.items():
        path = pack / Path(relative).name
        path.touch()
        expected[field] = path

    selected = resources._discover_generation_selection(
        LTX2Settings(model_generation="2.5", weights_path=pack),
        Settings(hf_home=tmp_path / "hf"),
    )

    assert {
        field: getattr(selected, field) for field in resources._LTX25_COMPONENT_PATHS
    } == expected


def test_local_ltx25_pack_directory_keeps_parts_while_overriding_transformer(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "ltx25"
    expected = {}
    for field, relative in resources._LTX25_COMPONENT_PATHS.items():
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        expected[field] = path
    dev_transformer = tmp_path / "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    dev_transformer.touch()

    selected = resources._discover_generation_selection(
        LTX2Settings(
            weights_path=pack,
            transformer_path=dev_transformer,
        ),
        Settings(hf_home=tmp_path / "hf"),
    )

    assert selected.model_generation == "2.5"
    assert selected.weights_path is None
    assert selected.transformer_path == dev_transformer
    for field, path in expected.items():
        if field != "transformer_path":
            assert getattr(selected, field) == path


def test_local_ltx25_pack_directory_accepts_every_component_override(tmp_path: Path) -> None:
    pack = tmp_path / "ltx25"
    for relative in resources._LTX25_COMPONENT_PATHS.values():
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    overrides = {}
    for field, relative in resources._LTX25_COMPONENT_PATHS.items():
        path = tmp_path / "overrides" / Path(relative).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        overrides[field] = path

    selected = resources._discover_generation_selection(
        LTX2Settings(weights_path=pack, **overrides),
        Settings(hf_home=tmp_path / "hf"),
    )

    for field, path in overrides.items():
        assert getattr(selected, field) == path


def test_generation_selection_preserves_monolithic_baseline_with_transformer_override(
    tmp_path: Path,
) -> None:
    monolith = tmp_path / "ltx-2.3.safetensors"
    transformer = tmp_path / "transformer.safetensors"
    monolith.touch()
    transformer.touch()

    selected = resources._discover_generation_selection(
        LTX2Settings(
            weights_path=monolith,
            transformer_path=transformer,
        ),
        Settings(hf_home=tmp_path / "hf"),
    )

    assert selected.weights_path == monolith
    assert selected.transformer_path == transformer


def test_local_ltx25_pack_directory_reports_missing_required_component(tmp_path: Path) -> None:
    pack = tmp_path / "incomplete-ltx25"
    pack.mkdir()
    transformer = pack / Path(resources._LTX25_COMPONENT_PATHS["transformer_path"]).name
    transformer.touch()

    with pytest.raises(ValueError, match="missing required components") as error:
        resources._discover_generation_selection(
            LTX2Settings(weights_path=pack),
            Settings(hf_home=tmp_path / "hf"),
        )

    assert resources._LTX25_COMPONENT_PATHS["text_encoder_path"] in str(error.value)
    assert resources._LTX25_VIDEO_VAE_PATHS["conv"] in str(error.value)


def test_local_ltx25_pack_directory_rejects_explicit_ltx23_selection(tmp_path: Path) -> None:
    pack = tmp_path / "ltx25"
    pack.mkdir()

    with pytest.raises(ValueError, match="cannot be used with model_generation=2.3"):
        resources._discover_generation_selection(
            LTX2Settings(model_generation="2.3", weights_path=pack),
            Settings(hf_home=tmp_path / "hf"),
        )


def test_generation_selector_discovers_requested_video_vae_variant(tmp_path: Path) -> None:
    snapshot = tmp_path / "hub" / "models--Lightricks--LTX-2.5" / "snapshots" / "revision"
    for relative in resources._LTX25_COMPONENT_PATHS.values():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    diffusion_vae = snapshot / resources._LTX25_VIDEO_VAE_PATHS["diffusion"]
    diffusion_vae.touch()

    conv = resources._discover_generation_selection(
        LTX2Settings(model_generation="2.5"),
        Settings(hf_home=tmp_path),
    )
    diffusion = resources._discover_generation_selection(
        LTX2Settings(model_generation="2.5", video_vae="diffusion"),
        Settings(hf_home=tmp_path),
    )

    assert conv.video_vae_path == snapshot / resources._LTX25_VIDEO_VAE_PATHS["conv"]
    assert diffusion.video_vae_path == diffusion_vae


def test_ltx23_selector_discovers_the_shared_ltx25_diffusion_vae(tmp_path: Path) -> None:
    root_23 = tmp_path / "hub" / "models--Lightricks--LTX-2.3" / "snapshots" / "revision-23"
    checkpoint_23 = root_23 / resources._LTX23_CHECKPOINT_FILENAMES[0]
    checkpoint_23.parent.mkdir(parents=True)
    checkpoint_23.touch()
    root_25 = tmp_path / "hub" / "models--Lightricks--LTX-2.5" / "snapshots" / "revision-25"
    diffusion_vae = root_25 / resources._LTX25_VIDEO_VAE_PATHS["diffusion"]
    diffusion_vae.parent.mkdir(parents=True)
    diffusion_vae.touch()

    discovered = resources._discover_generation_selection(
        LTX2Settings(model_generation="2.3", video_vae="diffusion"),
        Settings(hf_home=tmp_path),
    )
    explicit = resources._discover_generation_selection(
        LTX2Settings(
            model_generation="2.3",
            weights_path=checkpoint_23,
            video_vae="diffusion",
        ),
        Settings(hf_home=tmp_path),
    )

    assert discovered.weights_path == checkpoint_23
    assert discovered.video_vae_path == diffusion_vae
    assert explicit.weights_path == checkpoint_23
    assert explicit.video_vae_path == diffusion_vae


@pytest.mark.parametrize(
    ("generation", "primary_field", "expected_layout"),
    [
        ("2.3", "transformer_path", CheckpointLayout.SPLIT),
        ("2.5", "weights_path", CheckpointLayout.MONOLITHIC),
    ],
)
def test_generation_does_not_select_packaging_layout(
    generation: str,
    primary_field: str,
    expected_layout: CheckpointLayout,
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _split_paths(
        tmp_path,
        primary_field,
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
    )
    _stub_compatibility(
        monkeypatch,
        generation=generation,
        embedded_video_vae=False,
    )

    plan = resources.prepare_resources(
        LTX2Settings(model_generation=generation, **paths),
        infrastructure=Settings(cache_dir=tmp_path / "cache", hf_home=tmp_path / "hf"),
    )

    assert plan.checkpoint.model_generation == generation
    assert plan.checkpoint.layout is expected_layout


def test_ltx23_monolithic_baseline_accepts_every_component_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline = tmp_path / "ltx-2.3-22b-distilled.safetensors"
    baseline.touch()
    paths = _split_paths(
        tmp_path,
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "spatial_upscaler_path",
        "temporal_latent_upscaler_path",
        "duration_head_path",
    )
    calls = _stub_compatibility(
        monkeypatch,
        generation="2.3",
        embedded_video_vae=True,
    )

    plan = resources.prepare_resources(
        LTX2Settings(
            model_generation="2.3",
            weights_path=baseline,
            **paths,
        ),
        infrastructure=Settings(cache_dir=tmp_path / "cache", hf_home=tmp_path / "hf"),
    )

    assert calls == [
        ("config", baseline),
        ("config", paths["transformer_path"]),
        ("compatibility", paths["transformer_path"]),
    ]
    assert plan.checkpoint.layout is CheckpointLayout.MIXED
    assert plan.transformer_path == paths["transformer_path"].resolve()
    assert plan.require(ComponentKind.CONNECTOR).source_path == paths["transformer_path"].resolve()
    assert (
        plan.require(ComponentKind.TEXT_ENCODER).source_path == paths["text_encoder_path"].resolve()
    )
    assert plan.require(ComponentKind.VIDEO_VAE).source_path == paths["video_vae_path"].resolve()
    assert plan.require(ComponentKind.AUDIO_VAE).source_path == paths["audio_vae_path"].resolve()
    assert plan.require(ComponentKind.VOCODER).source_path == paths["audio_vae_path"].resolve()
    assert (
        plan.require(ComponentKind.SPATIAL_UPSCALER).source_path
        == paths["spatial_upscaler_path"].resolve()
    )
    assert (
        plan.require(ComponentKind.LATENT_TEMPORAL_UPSCALER).source_path
        == paths["temporal_latent_upscaler_path"].resolve()
    )
    assert (
        plan.require(ComponentKind.DURATION_HEAD).source_path
        == paths["duration_head_path"].resolve()
    )
    with pytest.raises(LookupError, match="mixed resources"):
        _ = plan.weights_path


def test_split_pack_prepares_transformer_cache_without_live_model_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _split_paths(
        tmp_path,
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "spatial_upscaler_path",
        "temporal_latent_upscaler_path",
        "duration_head_path",
    )
    compatibility_calls = _stub_compatibility(monkeypatch)

    plan = resources.prepare_resources(
        LTX2Settings(**paths),
        infrastructure=Settings(cache_dir=tmp_path / "cache"),
    )

    assert compatibility_calls == [
        ("config", paths["transformer_path"]),
        ("compatibility", paths["transformer_path"]),
    ]
    assert plan.checkpoint.layout is CheckpointLayout.SPLIT
    assert plan.checkpoint.model_generation == "2.5"
    assert plan.checkpoint.model_version == "2.5.0"
    assert plan.transformer_config == _transformer_25()
    assert {component.kind for component in plan.components} == set(ComponentKind)
    assert (
        plan.transformer_cache_path
        == (tmp_path / "cache" / "transformer-2.5.safetensors").resolve()
    )
    assert plan.require(ComponentKind.VIDEO_VAE).cache_path is not None
    assert plan.require(ComponentKind.AUDIO_VAE).cache_path is not None
    assert plan.require(ComponentKind.VOCODER).cache_path is not None
    assert all(
        component.cache_path is None
        for component in plan.components
        if component.kind
        not in {
            ComponentKind.TRANSFORMER,
            ComponentKind.VIDEO_VAE,
            ComponentKind.AUDIO_VAE,
            ComponentKind.VOCODER,
        }
    )
    assert (
        plan.require(ComponentKind.TRANSFORMER).source_path == paths["transformer_path"].resolve()
    )
    assert plan.require(ComponentKind.CONNECTOR).source_path == paths["transformer_path"].resolve()
    assert (
        plan.require(ComponentKind.TEXT_ENCODER).source_path == paths["text_encoder_path"].resolve()
    )
    assert (
        plan.require(ComponentKind.TEXT_PROJECTION).source_path
        == paths["text_encoder_path"].resolve()
    )
    assert plan.require(ComponentKind.AUDIO_VAE).source_path == paths["audio_vae_path"].resolve()
    assert plan.require(ComponentKind.VOCODER).source_path == paths["audio_vae_path"].resolve()
    assert plan.capabilities.recipe_families == ("distilled",)
    assert plan.capabilities.sampler_policy == "ancestral-stage1-deterministic-stage2"
    assert plan.capabilities.generates_audio
    assert plan.capabilities.duration_available
    assert plan.capabilities.temporal_latent_upscaler_available
    assert plan.transformer_path == paths["transformer_path"].resolve()
    assert plan.tokenizer_cache == _tokenizer_cache(tmp_path / "cache")
    with pytest.raises(LookupError, match="monolithic weights_path"):
        _ = plan.weights_path

    owned = tuple(_walk(plan))
    assert not any(isinstance(value, nn.Module) for value in owned)
    assert not any(isinstance(value, mx.array) for value in owned)
    assert not any(isinstance(value, Settings) for value in owned)
    assert not any(isinstance(value, LTX2Settings) for value in owned)
    assert not any(isinstance(value, (dict, list, set)) for value in owned)


def test_split_pack_keeps_optional_components_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _split_paths(
        tmp_path,
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
    )
    _stub_compatibility(monkeypatch)

    plan = resources.prepare_resources(LTX2Settings(**paths))

    assert plan.optional(ComponentKind.AUDIO_VAE) is None
    assert plan.optional(ComponentKind.VOCODER) is None
    assert plan.optional(ComponentKind.SPATIAL_UPSCALER) is None
    assert plan.optional(ComponentKind.LATENT_TEMPORAL_UPSCALER) is None
    assert plan.optional(ComponentKind.DURATION_HEAD) is None
    assert not plan.capabilities.generates_audio
    assert not plan.capabilities.duration_available
    assert not plan.capabilities.temporal_latent_upscaler_available


def test_componentized_pack_requires_an_external_or_embedded_video_vae(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _split_paths(tmp_path, "transformer_path", "text_encoder_path")

    parsed = _parsed("2.5", embedded_video_vae=False)
    monkeypatch.setattr(resources, "checkpoint_config", lambda path: parsed)

    def unexpected(*args, **kwargs):
        raise AssertionError("preflight continued after a missing required component")

    monkeypatch.setattr(resources, "inspect_ltx2_compatibility", unexpected)
    monkeypatch.setattr(resources, "ensure_weight_family_caches", unexpected)
    monkeypatch.setattr(resources, "ensure_transformer_cache", unexpected)

    with pytest.raises(ValueError, match="video_vae_path is required"):
        resources.prepare_resources(LTX2Settings(**paths))


def test_split_pack_rejects_cross_component_dimensions_before_cache_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _split_paths(
        tmp_path,
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
    )
    _stub_compatibility(
        monkeypatch,
        failure="text encoder hidden size does not match transformer caption channels",
    )

    with pytest.raises(ValueError, match="text encoder hidden size"):
        resources.prepare_resources(LTX2Settings(**paths))


def test_split_pack_accepts_unseen_same_schema_fingerprints_and_isolates_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_compatibility(monkeypatch)
    monkeypatch.setattr(
        resources,
        "_artifact_fingerprint",
        lambda path: f"sha256:{path.parent.name}:{path.name}",
    )

    def prepare(label: str):
        directory = tmp_path / label
        directory.mkdir()
        paths = {
            field: directory / f"renamed-{field}.safetensors"
            for field in ("transformer_path", "text_encoder_path", "video_vae_path")
        }
        for path in paths.values():
            path.touch()
        return resources.prepare_resources(
            LTX2Settings(**paths),
            infrastructure=Settings(cache_dir=tmp_path / "cache"),
        )

    first = prepare("community-a")
    second = prepare("community-b")

    assert first.transformer_config == second.transformer_config
    assert first.checkpoint.source_fingerprint != second.checkpoint.source_fingerprint
    assert (
        first.require(ComponentKind.TRANSFORMER).source_fingerprint
        != second.require(ComponentKind.TRANSFORMER).source_fingerprint
    )
    assert (
        first.require(ComponentKind.TRANSFORMER).metadata
        == second.require(ComponentKind.TRANSFORMER).metadata
    )
    assert "flavor" not in dict(first.require(ComponentKind.TRANSFORMER).metadata.entries)


def _fake_gemma_snapshot(hf_home: Path, repo_id: str, name: str) -> Path:
    snapshot = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / name
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").touch()
    return snapshot


def test_gemma_discovery_defaults_to_qat_variant(tmp_path: Path) -> None:
    qat = _fake_gemma_snapshot(
        tmp_path, "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized", "qat-snap"
    )
    _fake_gemma_snapshot(tmp_path, "google/gemma-3-12b-it", "plain-snap")

    resolved = resources._discover_gemma_path(
        LTX2Settings(),
        Settings(hf_home=tmp_path),
    )

    assert resolved == qat.resolve()


def test_gemma_discovery_plain_variant_switch(tmp_path: Path) -> None:
    _fake_gemma_snapshot(tmp_path, "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized", "qat-snap")
    plain = _fake_gemma_snapshot(tmp_path, "google/gemma-3-12b-it", "plain-snap")

    resolved = resources._discover_gemma_path(
        LTX2Settings(gemma_variant="plain"),
        Settings(hf_home=tmp_path),
    )

    assert resolved == plain.resolve()


def test_gemma_discovery_never_falls_back_across_variants(tmp_path: Path) -> None:
    _fake_gemma_snapshot(tmp_path, "google/gemma-3-12b-it", "plain-snap")

    resolved = resources._discover_gemma_path(LTX2Settings(), Settings(hf_home=tmp_path))

    assert resolved is None
