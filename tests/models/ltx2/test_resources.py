"""Immutable LTX-2 resource preparation and inventory contracts."""

from dataclasses import fields, is_dataclass
from pathlib import Path

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
)
from kinomlx.models.ltx2.resources import (
    CheckpointLayout,
    ComponentKind,
    ComponentLocator,
    ComponentMetadata,
)
from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.transformer_config import TransformerConstructorConfig
from kinomlx.models.ltx2.video_vae.config import LTX23_VIDEO_VAE_CONFIG
from kinomlx.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_huggingface_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


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
        "artifact_fingerprint",
        lambda path: f"sha256:{path.name}",
    )

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
    _stub_compatibility(
        monkeypatch,
        generation="2.3",
        embedded_video_vae=True,
    )

    settings = LTX2Settings(
        weights_path=checkpoint,
        text_encoder_path=gemma,
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
        infrastructure=Settings(),
    )

    assert plan.checkpoint.layout is CheckpointLayout.MONOLITHIC
    assert plan.checkpoint.model_version == "2.3.0"
    assert plan.dtype_policy.transformer == mx.float16
    assert plan.require(ComponentKind.TEXT_ENCODER).source_path == gemma.resolve()
    assert plan.require(ComponentKind.SPATIAL_UPSCALER).source_path == upscaler.resolve()

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
    {
        family: tmp_path / f"{family}.safetensors"
        for family in ("connector", "video_vae", "audio_vae", "vocoder")
    }
    _stub_compatibility(
        monkeypatch,
        generation="2.3",
        version="2.3.1",
        embedded_video_vae=True,
    )
    monkeypatch.setattr(resources, "cached_snapshots", lambda *args: ())

    plan = resources.prepare_resources(
        LTX2Settings(weights_path=checkpoint),
        infrastructure=Settings(hf_home=tmp_path / "hf"),
    )

    assert plan.optional(ComponentKind.TEXT_ENCODER) is None
    assert plan.optional(ComponentKind.SPATIAL_UPSCALER) is None


def test_huggingface_snapshot_symlink_keeps_logical_weights_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    blob = tmp_path / "b33b7fe4bbfe084f"
    checkpoint = tmp_path / "model.safetensors"
    blob.touch()
    checkpoint.symlink_to(blob)
    {
        family: tmp_path / f"{family}.safetensors"
        for family in ("connector", "video_vae", "audio_vae", "vocoder")
    }
    inspection_calls = _stub_compatibility(
        monkeypatch,
        generation="2.3",
        version="2.3.1",
        embedded_video_vae=True,
    )
    monkeypatch.setattr(resources, "cached_snapshots", lambda *args: ())

    plan = resources.prepare_resources(
        LTX2Settings(weights_path=checkpoint),
        infrastructure=Settings(hf_home=tmp_path / "hf"),
    )

    assert inspection_calls == [
        ("config", checkpoint),
        ("compatibility", checkpoint),
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


def test_generation_selector_chooses_the_requested_weightsd_pack_without_exact_paths(
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
        infrastructure=Settings(hf_home=tmp_path / "hf"),
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
        infrastructure=Settings(hf_home=tmp_path / "hf"),
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


def test_split_pack_prepares_transformer_weights_without_live_model_state(
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
        infrastructure=Settings(),
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
    assert plan.capabilities.stage_samplers == ("ancestral", "ancestral")
    assert plan.capabilities.generates_audio
    assert plan.capabilities.duration_available
    assert plan.capabilities.temporal_latent_upscaler_available
    assert plan.transformer_path == paths["transformer_path"].resolve()
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

    plan = resources.prepare_resources(
        LTX2Settings(**paths),
        infrastructure=Settings(hf_home=tmp_path / "hf"),
    )

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

    with pytest.raises(ValueError, match="video_vae_path is required"):
        resources.prepare_resources(
            LTX2Settings(**paths),
            infrastructure=Settings(hf_home=tmp_path / "hf"),
        )


def test_split_pack_rejects_cross_component_dimensions_before_weights_work(
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
        resources.prepare_resources(
            LTX2Settings(**paths),
            infrastructure=Settings(hf_home=tmp_path / "hf"),
        )


def test_split_pack_accepts_unseen_same_schema_fingerprints_and_isolates_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_compatibility(monkeypatch)
    monkeypatch.setattr(
        resources,
        "artifact_fingerprint",
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
            infrastructure=Settings(),
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


@pytest.mark.parametrize("explicit", [False, True])
def test_ltx23_text_encoder_is_discovered_unless_text_encoder_path_names_one(
    tmp_path: Path, monkeypatch, explicit: bool
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    upscaler = tmp_path / "upscaler.safetensors"
    checkpoint.touch()
    upscaler.touch()
    cached = _fake_gemma_snapshot(
        tmp_path, "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized", "qat-snap"
    )
    local = tmp_path / "local-gemma"
    local.mkdir()
    (local / "config.json").touch()
    _stub_compatibility(monkeypatch, generation="2.3", embedded_video_vae=True)

    plan = resources.prepare_resources(
        LTX2Settings(
            weights_path=checkpoint,
            spatial_upscaler_path=upscaler,
            text_encoder_path=local if explicit else None,
        ),
        infrastructure=Settings(hf_home=tmp_path),
    )

    selected = plan.require(ComponentKind.TEXT_ENCODER).source_path
    assert selected == (local if explicit else cached).resolve()


def test_gemma_discovery_never_falls_back_across_variants(tmp_path: Path) -> None:
    _fake_gemma_snapshot(tmp_path, "google/gemma-3-12b-it", "plain-snap")

    resolved = resources._discover_gemma_path(LTX2Settings(), Settings(hf_home=tmp_path))

    assert resolved is None


def test_prepare_resources_describes_every_component_by_source_only(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    upscaler = tmp_path / "upscaler.safetensors"
    gemma = tmp_path / "gemma"
    checkpoint.touch()
    upscaler.touch()
    gemma.mkdir()
    (gemma / "config.json").touch()
    _stub_compatibility(monkeypatch, generation="2.3", embedded_video_vae=True)

    settings = LTX2Settings(
        weights_path=checkpoint,
        text_encoder_path=gemma,
        spatial_upscaler_path=upscaler,
        transformer_dtype="float16",
    )

    plan = resources.prepare_resources(settings, infrastructure=Settings())

    assert plan.checkpoint.layout is CheckpointLayout.MONOLITHIC
    assert {component.kind for component in plan.components} >= {
        ComponentKind.TRANSFORMER,
        ComponentKind.CONNECTOR,
        ComponentKind.VIDEO_VAE,
        ComponentKind.TEXT_ENCODER,
        ComponentKind.TEXT_PROJECTION,
        ComponentKind.AUDIO_VAE,
        ComponentKind.VOCODER,
        ComponentKind.SPATIAL_UPSCALER,
    }
    assert plan.require(ComponentKind.VIDEO_VAE).source_path == checkpoint.resolve()
    assert plan.require(ComponentKind.TEXT_ENCODER).source_path == gemma.resolve()
    owned = tuple(_walk(plan))
    assert not any(isinstance(value, nn.Module) for value in owned)
    assert not any(isinstance(value, mx.array) for value in owned)
    assert not any(isinstance(value, (dict, list, set)) for value in owned)


@pytest.mark.parametrize("cache", [0.0, 1.0, 4.0, 8.0, None])
def test_streaming_keeps_the_configured_allocator_cache_limit(
    tmp_path: Path, monkeypatch, cache: float | None
) -> None:
    """Resource preparation must preserve explicit streaming memory controls."""
    checkpoint = tmp_path / "model.safetensors"
    upscaler = tmp_path / "upscaler.safetensors"
    gemma = tmp_path / "gemma"
    checkpoint.touch()
    upscaler.touch()
    gemma.mkdir()
    (gemma / "config.json").touch()
    _stub_compatibility(monkeypatch, generation="2.3", embedded_video_vae=True)
    settings = LTX2Settings(
        weights_path=checkpoint,
        text_encoder_path=gemma,
        spatial_upscaler_path=upscaler,
        stream_transformer=True,
    )
    plan = resources.prepare_resources(
        settings,
        infrastructure=Settings(mlx_cache_limit_gb=cache, mlx_wired_limit_gb=0.0),
    )
    assert plan.memory_policy.cache_limit_gb == cache
    assert plan.memory_policy.wired_limit_gb == 0.0


def test_weight_policy_carries_the_resolved_adapter_accumulation() -> None:
    from kinomlx.models.ltx2.resources import _weight_policy
    from kinomlx.models.ltx2.settings import LTX2Settings

    streamed_half = LTX2Settings(stream_transformer=True, transformer_dtype="float16")
    assert _weight_policy(streamed_half.resolve_presets()).adapter_accumulation == "float16"
    resident_half = LTX2Settings(transformer_dtype="float16")
    assert _weight_policy(resident_half).adapter_accumulation == "float32"
    streamed_wide = LTX2Settings(stream_transformer=True, transformer_dtype="bfloat16")
    assert _weight_policy(streamed_wide.resolve_presets()).adapter_accumulation == "float32"


@pytest.mark.parametrize("field", tuple(resources._LTX25_COMPONENT_PATHS))
def test_component_override_autodiscovers_every_other_ltx25_component(tmp_path, field):
    snapshot = tmp_path / "hub/models--Lightricks--LTX-2.5/snapshots/revision"
    expected = {}
    for name, relative in resources._LTX25_COMPONENT_PATHS.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name != field:
            path.touch()
        expected[name] = path
    override = tmp_path / "replacement.safetensors"
    override.touch()
    expected[field] = override
    settings = LTX2Settings(model_generation="2.5", **{field: override})
    selected = resources._discover_generation_selection(settings, Settings(hf_home=tmp_path))
    assert {name: getattr(selected, name) for name in expected} == expected


def test_transformer_override_prefers_its_local_pack_without_hub(tmp_path):
    pack = tmp_path / "local-pack"
    for relative in resources._LTX25_COMPONENT_PATHS.values():
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    dev = pack / "diffusion_models/dev.safetensors"
    dev.touch()
    selected = resources._discover_generation_selection(
        LTX2Settings(model_generation="2.5", transformer_path=dev),
        Settings(hf_home=tmp_path / "empty-hub"),
    )
    assert selected.transformer_path == dev
    assert (
        selected.text_encoder_path == pack / resources._LTX25_COMPONENT_PATHS["text_encoder_path"]
    )


def test_dry_run_report_names_every_resolved_source_but_no_resolved_policy(
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
    _stub_compatibility(monkeypatch, generation="2.3", embedded_video_vae=True)

    report = resources.prepare_resources(
        LTX2Settings(
            weights_path=checkpoint,
            text_encoder_path=gemma,
            spatial_upscaler_path=upscaler,
            transformer_dtype="float16",
        ),
        infrastructure=Settings(),
    ).describe()

    assert report["model"] == "ltx2"
    assert report["model_generation"] == "2.3"
    assert report["model_version"] == "2.3.0"
    assert report["layout"] == "monolithic"
    assert report["source"] == str(checkpoint.absolute())
    assert report["source_identity"] == "sha256:model.safetensors"
    dtypes = report["dtypes"]
    assert isinstance(dtypes, dict)
    assert dtypes["transformer"] == "float16"
    assert dtypes["video_vae"] == "bfloat16"
    assert report["check_adapter_weights"] is True
    # The transformer weight, execution and capability policies are resolved
    # decisions rather than sources, and naming them is a separate design call.
    assert not {"capabilities", "weight_policy", "execution_policy"} & set(report)
    components = report["components"]
    assert isinstance(components, list)
    transformer = next(entry for entry in components if entry["kind"] == "transformer")
    assert transformer["source"] == str(checkpoint.absolute())
    assert transformer["graph_identity"]
    assert "config_digest" not in transformer
    assert transformer["family"] == "ltx2-av-transformer"
    assert transformer["declared_model_version"] == "2.3.0"
    text_encoder = next(entry for entry in components if entry["kind"] == "text_encoder")
    assert text_encoder["source"] == str(gemma.resolve())
    # Only the transformer carries an inferred-field annotation, so the report
    # states each component's own annotations instead of padding a fixed shape.
    assert "inferred_constructor_fields" not in text_encoder
