"""Ownership and source-loading contracts for public component leases."""

import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.components as components
from kinomlx.adapters.lora import LoRAConfig
from kinomlx.components import ComponentLease
from kinomlx.models.ltx2.resources import (
    CheckpointIdentity,
    CheckpointLayout,
    ComponentKind,
    ComponentMetadata,
    ComponentResource,
    LTX2Capabilities,
    LTX2Resources,
    TransformerExecutionPolicy,
    TransformerWeightPolicy,
)
from kinomlx.models.ltx2.transformer_config import TransformerConstructorConfig
from kinomlx.models.ltx2.video_vae.config import LTX23_VIDEO_VAE_CONFIG
from kinomlx.models.ltx2.video_vae.kinds import VideoVAEKind


@pytest.fixture(autouse=True)
def synthetic_sources(monkeypatch):
    monkeypatch.setattr(
        components,
        "transformer_source_loader",
        lambda *a, **k: SimpleNamespace(
            load=lambda _index: {}, prepared_targets={}, streamed_targets=frozenset()
        ),
    )
    monkeypatch.setattr(
        components,
        "family_source_weights",
        lambda resources, kind, **kwargs: resources.require(kind).source_path,
    )


def _transformer_config(generation: str) -> TransformerConstructorConfig:
    is_25 = generation == "2.5"
    return TransformerConstructorConfig(
        model_generation=generation,
        declared_model_version=f"{generation}.0",
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
        config_digest=f"transformer-{generation}",
    )


def _resources(
    tmp_path: Path,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    generation: str = "2.3",
) -> LTX2Resources:
    checkpoint = tmp_path / "model.safetensors"
    entries = []
    for kind in (
        ComponentKind.TRANSFORMER,
        ComponentKind.CONNECTOR,
        ComponentKind.VIDEO_VAE,
        ComponentKind.AUDIO_VAE,
        ComponentKind.VOCODER,
    ):
        entries.append(
            ComponentResource(
                kind=kind,
                source_path=checkpoint,
                source_fingerprint="checkpoint-fingerprint",
                metadata=ComponentMetadata(),
            )
        )
    entries.append(
        ComponentResource(
            kind=ComponentKind.SPATIAL_UPSCALER,
            source_path=tmp_path / "upscaler.safetensors",
            source_fingerprint="upscaler-fingerprint",
            metadata=ComponentMetadata(),
        )
    )
    if generation == "2.5":
        for kind, filename in (
            (ComponentKind.LATENT_TEMPORAL_UPSCALER, "temporal.safetensors"),
            (ComponentKind.DURATION_HEAD, "duration.safetensors"),
        ):
            entries.append(
                ComponentResource(
                    kind=kind,
                    source_path=tmp_path / filename,
                    source_fingerprint=f"{kind.value}-fingerprint",
                    metadata=ComponentMetadata(),
                )
            )
    from kinomlx.models.ltx2.precision import LTX2DTypePolicy

    return LTX2Resources(
        checkpoint=CheckpointIdentity(
            source_path=checkpoint,
            source_fingerprint="checkpoint-fingerprint",
            model_generation=generation,
            model_version=f"{generation}.0",
            layout=CheckpointLayout.MONOLITHIC,
        ),
        components=tuple(entries),
        capabilities=LTX2Capabilities(
            model_generation=generation,
            recipe_families=("distilled",),
            condition_families=("text",),
            video_compression=LTX23_VIDEO_VAE_CONFIG.encoder_scale,
            video_vae_kind=VideoVAEKind.CONV,
            text_encoder_family="gemma-3-12b-it",
            native_hdr=False,
            generates_audio=True,
        ),
        dtype_policy=LTX2DTypePolicy.reference(transformer=dtype),
        weight_policy=TransformerWeightPolicy(
            include_audio=True,
            video_ff_layout_specs=(),
            video_ff_layout_layers=(),
            video_attn_layout_specs=(),
            video_attn_layout_layers=(),
            audio_ff_layout_specs=None,
            audio_ff_layout_layers=None,
            audio_attn_layout_specs=None,
            audio_attn_layout_layers=None,
            adaln_pretranspose=False,
            transformer_quantize="off",
            video_ff_quantize_specs=(),
            video_ff_quantize_layers=(),
            video_ff_quantize_group_size=None,
            video_ff_quantize_bits=None,
            resident_blocks=None,
        ),
        execution_policy=TransformerExecutionPolicy(
            compile_attention=False,
            fast_mode=True,
            compile_block_group_size=None,
            transformer_compile_group_size=None,
        ),
        video_vae_config=LTX23_VIDEO_VAE_CONFIG,
        transformer_config=_transformer_config(generation),
    )


def test_component_lease_is_idempotent_and_rejects_use_after_close() -> None:
    events = []

    class _Component:
        label = "live"

        def __call__(self, value):
            return value + 1

    holder = {}
    lease = ComponentLease(
        _Component(),
        close_component=lambda component: events.append(("close", component.label)),
        cleanup=lambda: events.append(("cleanup", holder["lease"].closed)),
    )
    holder["lease"] = lease

    assert lease.label == "live"
    assert lease(2) == 3
    lease.close()
    lease.close()

    assert events == [("close", "live"), ("cleanup", True)]
    with pytest.raises(RuntimeError, match="closed"):
        _ = lease.label
    with pytest.raises(RuntimeError, match="closed"):
        lease(2)
    with pytest.raises(RuntimeError, match="closed"):
        lease.__enter__()


def test_component_lease_context_exposes_the_typed_component() -> None:
    component = object()
    lease = ComponentLease(component)

    with lease as entered:
        assert entered is component

    assert lease.closed


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_transformer_lease_releases_callback_capture_before_cleanup(
    tmp_path: Path,
    monkeypatch,
    generation: str,
) -> None:
    plan = _resources(tmp_path, generation=generation)
    model_refs: list[weakref.ReferenceType[object]] = []
    cleanup_observations: list[bool] = []

    class _Model:
        transformer_block_streamer = None

        def enable_mpp_weight_conversion(self) -> None:
            pass

        def close_streamer(self) -> None:
            pass

    class _ModelFactory:
        @staticmethod
        def from_config(*_args, **_kwargs):
            model = _Model()
            model_refs.append(weakref.ref(model))
            return model

    class _X0:
        def __init__(self, model) -> None:
            self.velocity_model = model

    monkeypatch.setattr(components, "LTXAVModel", _ModelFactory)
    monkeypatch.setattr(components, "X0Model", _X0)
    monkeypatch.setattr(components, "bind_transformer", lambda *_args, **_kwargs: None)

    def observe_cleanup() -> None:
        cleanup_observations.append(model_refs[0]() is None)

    monkeypatch.setattr(components, "_cleanup_mlx", observe_cleanup)

    lease = components.load_transformer(plan)
    assert model_refs[0]() is not None
    lease.close()

    assert cleanup_observations == [True]
    assert model_refs[0]() is None


@pytest.mark.parametrize("streamed", [False, True])
def test_transformer_loader_binds_prepared_weights_and_passes_fp16_owner_explicitly(
    tmp_path: Path,
    monkeypatch,
    streamed,
) -> None:
    plan = _resources(tmp_path, dtype=mx.float16)
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    events = []

    class _Model:
        def enable_mpp_weight_conversion(self) -> None:
            events.append("convert mpp weights")

        def close_streamer(self) -> None:
            events.append("close streamer")

    class _X0:
        def __init__(self, model) -> None:
            self.velocity_model = model

        def __call__(self, *args, **kwargs):
            return args, kwargs

    model = _Model()
    model.transformer_block_streamer = SimpleNamespace(on_evaluated=None) if streamed else None
    monkeypatch.setattr(
        components,
        "transformer_source_loader",
        lambda *args, **kwargs: SimpleNamespace(
            load=lambda _index: {},
            prepared_targets={},
            streamed_targets=frozenset({"transformer_blocks.0.ff.project_out.weight"})
            if streamed
            else frozenset(),
        ),
    )

    class _ModelFactory:
        @staticmethod
        def from_config(config, **kwargs):
            events.append(("construct", config, kwargs))
            return model

    monkeypatch.setattr(components, "LTXAVModel", _ModelFactory)
    monkeypatch.setattr(
        components,
        "bind_transformer",
        lambda received, **kwargs: events.append(("bind", received, kwargs)),
    )
    monkeypatch.setattr(
        components,
        "fuse_community_loras_into_model",
        lambda received, profile, **kwargs: events.append(
            ("fuse", received, tuple(profile), kwargs)
        ),
    )
    monkeypatch.setattr(components, "X0Model", _X0)
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))

    lease = components.load_transformer(
        plan,
        (LoRAConfig(adapter, strength=0.5),),
    )

    bind_event = next(event for event in events if event[0] == "bind")
    assert callable(bind_event[2]["weight_loader"])
    assert bind_event[2]["include_audio"] is True
    fuse_event = next(event for event in events if event[0] == "fuse")
    assert fuse_event[3]["pending"].enabled is True
    assert fuse_event[3]["model_generation"] == "2.3"
    if streamed:
        assert callable(model.transformer_block_streamer.on_evaluated)
    # Converting MPP weights in place would add to a streamer's memory, not replace it.
    assert ("convert mpp weights" in events) is not streamed
    lease.close()
    lease.close()
    assert events[-2:] == ["close streamer", "cleanup"]


def test_ltx25_transformer_fuses_nonempty_lora_after_weights_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path, generation="2.5")
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    events = []

    class _Model:
        transformer_block_streamer = None

        def enable_mpp_weight_conversion(self) -> None:
            pass

        def close_streamer(self) -> None:
            events.append("close")

    class _Wrapped:
        def __init__(self, model) -> None:
            self.velocity_model = model

    monkeypatch.setattr(components.LTXAVModel, "from_config", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(
        components,
        "bind_transformer",
        lambda *_args, **_kwargs: events.append("bind"),
    )
    monkeypatch.setattr(
        components,
        "fuse_community_loras_into_model",
        lambda *_args, **kwargs: events.append(("fuse", kwargs["model_generation"])) or (),
    )
    monkeypatch.setattr(components, "X0Model", _Wrapped)
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))

    lease = components.load_transformer(plan, (LoRAConfig(adapter),))
    assert events[:2] == ["bind", ("fuse", "2.5")]
    lease.close()


def test_spatial_upscaler_loads_only_video_vae_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path)
    statistics = object()
    events = []

    class _Upscaler:
        def __call__(self, latent, *, reporter=None):
            return latent

    monkeypatch.setattr(
        components,
        "_load_spatial_upscaler_model",
        lambda path, **kwargs: events.append(("upscaler", path)) or _Upscaler(),
    )
    monkeypatch.setattr(
        components,
        "load_vae_encoder_statistics",
        lambda path, **kwargs: (
            events.append(("statistics", path))
            or SimpleNamespace(per_channel_statistics=statistics)
        ),
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: None)

    lease = components.load_spatial_upscaler(plan)

    assert lease.per_channel_statistics is statistics
    assert events == [
        ("upscaler", plan.spatial_upscaler_path),
        ("statistics", plan.require(ComponentKind.VIDEO_VAE).source_path),
    ]
    lease.close()


def test_temporal_upscaler_loads_only_video_vae_statistics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path, generation="2.5")
    statistics = object()
    events = []

    class _Upscaler:
        def __call__(self, latent, *, reporter=None):
            return latent

    monkeypatch.setattr(
        components,
        "_load_temporal_upscaler_model",
        lambda path, **kwargs: events.append(("upscaler", path)) or _Upscaler(),
    )
    monkeypatch.setattr(
        components,
        "load_vae_encoder_statistics",
        lambda path, **kwargs: (
            events.append(("statistics", path))
            or SimpleNamespace(per_channel_statistics=statistics)
        ),
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: None)

    lease = components.load_temporal_upscaler(plan)

    assert lease.per_channel_statistics is statistics
    assert events == [
        ("upscaler", plan.temporal_upscaler_path),
        ("statistics", plan.require(ComponentKind.VIDEO_VAE).source_path),
    ]
    lease.close()


def test_duration_predictor_uses_optional_component_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path, generation="2.5")
    model = SimpleNamespace(predict_num_frames=lambda *_args, **_kwargs: 121)
    events = []
    monkeypatch.setattr(
        components,
        "_load_duration_head_model",
        lambda path, **kwargs: events.append((path, kwargs["compute_dtype"])) or model,
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: events.append("cleanup"))

    lease = components.load_duration_predictor(plan)

    assert lease.value is model
    assert events == [(plan.duration_head_path, mx.bfloat16)]
    lease.close()
    assert events[-1] == "cleanup"


def test_split_audio_leases_construct_from_resolved_logical_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path, generation="2.5")
    audio_source = tmp_path / "audio-component.safetensors"
    audio_source.touch()
    plan = replace(
        plan,
        checkpoint=replace(plan.checkpoint, layout=CheckpointLayout.SPLIT),
        components=tuple(
            replace(component, source_path=audio_source)
            if component.kind in {ComponentKind.AUDIO_VAE, ComponentKind.VOCODER}
            else component
            for component in plan.components
        ),
    )
    events: list[tuple[str, Path]] = []

    class _Model:
        output_sample_rate = 48_000

        def parameters(self):
            return ()

    monkeypatch.setattr(
        components,
        "create_audio_decoder_from_checkpoint",
        lambda path, **kwargs: events.append(("decoder constructor", Path(path))) or _Model(),
    )
    monkeypatch.setattr(
        components,
        "load_audio_decoder_weights",
        lambda model, path, **kwargs: events.append(("decoder weights", Path(path))) or 54,
    )
    monkeypatch.setattr(
        components,
        "create_vocoder_from_checkpoint",
        lambda path: events.append(("vocoder constructor", Path(path))) or _Model(),
    )
    monkeypatch.setattr(
        components,
        "load_vocoder_weights",
        lambda model, path, **kwargs: events.append(("vocoder weights", Path(path))) or 1227,
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: None)

    with components.load_audio_decoder(plan):
        pass
    with components.load_vocoder(plan):
        pass

    assert events == [
        ("decoder constructor", audio_source),
        ("decoder weights", plan.require(ComponentKind.AUDIO_VAE).source_path),
        ("vocoder constructor", audio_source),
        ("vocoder weights", plan.require(ComponentKind.VOCODER).source_path),
    ]


def test_component_parity_component_leases_each_close_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path, generation="2.5")
    cleanups: list[str] = []

    class _Model:
        output_sample_rate = 48_000

        def parameters(self):
            return ()

    class _Upscaler(_Model):
        def __call__(self, latent, *, reporter=None):
            return latent

    monkeypatch.setattr(components, "Conv3dVideoEncoder", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(components, "Conv3dVideoDecoder", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(components, "load_vae_encoder_weights", lambda *args, **kwargs: 86)
    monkeypatch.setattr(components, "load_vae_decoder_weights", lambda *args, **kwargs: 86)
    monkeypatch.setattr(
        components,
        "create_audio_decoder_from_checkpoint",
        lambda *args, **kwargs: _Model(),
    )
    monkeypatch.setattr(components, "load_audio_decoder_weights", lambda *args, **kwargs: 58)
    monkeypatch.setattr(
        components,
        "create_vocoder_from_checkpoint",
        lambda *args, **kwargs: _Model(),
    )
    monkeypatch.setattr(components, "load_vocoder_weights", lambda *args, **kwargs: 1227)
    monkeypatch.setattr(
        components, "_load_spatial_upscaler_model", lambda *args, **kwargs: _Upscaler()
    )
    monkeypatch.setattr(
        components,
        "load_vae_encoder_statistics",
        lambda *args, **kwargs: SimpleNamespace(per_channel_statistics=object()),
    )
    monkeypatch.setattr(components.mx, "eval", lambda *args, **kwargs: None)
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: cleanups.append("cleanup"))
    monkeypatch.setattr(components, "_cleanup_video_decoder", lambda: cleanups.append("decoder"))

    for loader in (
        components.load_video_encoder,
        components.load_video_decoder,
        components.load_audio_decoder,
        components.load_vocoder,
        components.load_spatial_upscaler,
    ):
        lease = loader(plan)
        lease.close()
        lease.close()

    assert cleanups == ["cleanup", "decoder", "cleanup", "cleanup", "cleanup"]


def test_video_decoder_close_does_not_clear_allocator_cache(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(components.mx, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(components.mx, "clear_cache", lambda: events.append("clear"))

    components._cleanup_video_decoder()

    assert events == ["synchronize"]


def test_family_components_bind_prepared_source_tensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _resources(tmp_path)
    prepared = {kind: {f"{kind.value}.weight": mx.zeros((1,))} for kind in ComponentKind}
    events: list[object] = []

    def fake_family_weights(resources, kind, **kwargs):
        events.append(("select", kind))
        return prepared[kind]

    class _Model:
        output_sample_rate = 48_000

        def parameters(self):
            return ()

    monkeypatch.setattr(components, "family_source_weights", fake_family_weights)
    monkeypatch.setattr(components, "Conv3dVideoEncoder", lambda *a, **k: _Model())
    monkeypatch.setattr(components, "Conv3dVideoDecoder", lambda *a, **k: _Model())
    monkeypatch.setattr(
        components,
        "load_vae_encoder_weights",
        lambda model, source: events.append(("encoder", source)) or 1,
    )
    monkeypatch.setattr(
        components,
        "load_vae_decoder_weights",
        lambda model, source: events.append(("decoder", source)) or 1,
    )
    monkeypatch.setattr(
        components,
        "load_vae_encoder_statistics",
        lambda source, **kwargs: (
            events.append(("statistics", source)) or SimpleNamespace(per_channel_statistics=None)
        ),
    )
    monkeypatch.setattr(components, "_load_spatial_upscaler_model", lambda path, **k: _Model())
    monkeypatch.setattr(components, "create_audio_decoder_from_checkpoint", lambda p, **k: _Model())
    monkeypatch.setattr(
        components,
        "load_audio_decoder_weights",
        lambda model, source, **kwargs: events.append(("audio", source)) or 1,
    )
    monkeypatch.setattr(components, "create_vocoder_from_checkpoint", lambda p: _Model())
    monkeypatch.setattr(
        components,
        "load_vocoder_weights",
        lambda model, source, **kwargs: events.append(("vocoder", source)) or 1,
    )
    monkeypatch.setattr(components, "_cleanup_mlx", lambda: None)

    for loader in (
        components.load_video_encoder,
        components.load_video_decoder,
        components.load_spatial_upscaler,
        components.load_audio_decoder,
        components.load_vocoder,
    ):
        with loader(plan):
            pass

    video = prepared[ComponentKind.VIDEO_VAE]
    assert events == [
        ("select", ComponentKind.VIDEO_VAE),
        ("encoder", video),
        ("select", ComponentKind.VIDEO_VAE),
        ("decoder", video),
        ("select", ComponentKind.VIDEO_VAE),
        ("statistics", video),
        ("select", ComponentKind.AUDIO_VAE),
        ("audio", prepared[ComponentKind.AUDIO_VAE]),
        ("select", ComponentKind.VOCODER),
        ("vocoder", prepared[ComponentKind.VOCODER]),
    ]
