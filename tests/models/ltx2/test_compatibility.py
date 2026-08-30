"""Central, generation-labeled LTX-2 compatibility orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

import kinomlx.models.ltx2.compatibility as compatibility
from kinomlx.models.ltx2.audio_vae.config import AudioVAEConfig
from kinomlx.models.ltx2.compatibility import LTX2ComponentSources
from kinomlx.models.ltx2.metadata import (
    ConnectorHeaderConfig,
    DurationHeadConfig,
    LatentUpscalerConfig,
    LTX2CheckpointConfig,
    TextEncoderHeaderConfig,
    TextProjectionHeaderConfig,
    TransformerConstructorConfig,
)
from kinomlx.models.ltx2.video_vae.config import LTX23_VIDEO_VAE_CONFIG


def _transformer(generation: str, *, declared: str | None = None) -> TransformerConstructorConfig:
    is_25 = generation == "2.5"
    return TransformerConstructorConfig(
        model_generation=generation,
        declared_model_version=declared or f"{generation}.0",
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


def _parsed(generation: str, *, declared: str | None = None) -> LTX2CheckpointConfig:
    return LTX2CheckpointConfig(
        transformer=_transformer(generation, declared=declared),
        video_vae=None,
    )


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
@pytest.mark.parametrize("monolithic", [False, True])
def test_both_generations_and_both_packaging_layouts_use_the_same_checks(
    generation: str,
    monolithic: bool,
    tmp_path: Path,
    monkeypatch,
) -> None:
    names = (
        "transformer",
        "text",
        "video",
        "audio",
        "spatial",
        "temporal",
        "duration",
    )
    paths = {name: tmp_path / f"{name}.safetensors" for name in names}
    for path in paths.values():
        path.touch()
    if monolithic:
        paths = dict.fromkeys(names, paths["transformer"])

    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        compatibility,
        "validate_transformer_header",
        lambda path, config: events.append(("transformer", config.model_generation)),
    )

    def inspect_text(path: Path, *, model_generation: str) -> TextEncoderHeaderConfig:
        events.append(("text", model_generation))
        return TextEncoderHeaderConfig(
            family=f"text-{model_generation}",
            hidden_size=3840,
            num_hidden_layers=48,
            video_projection_dim=4096,
            audio_projection_dim=2048,
            tokenizer_json_bytes=1,
            config_digest="text",
        )

    def inspect_projection(
        path: Path,
        *,
        model_generation: str,
        hidden_size: int,
        num_hidden_layers: int,
    ) -> TextProjectionHeaderConfig:
        events.append(("projection", model_generation))
        assert (hidden_size, num_hidden_layers) == (3840, 48)
        return TextProjectionHeaderConfig(4096, 2048, "projection")

    monkeypatch.setattr(compatibility, "inspect_text_encoder", inspect_text)
    monkeypatch.setattr(compatibility, "inspect_text_projection", inspect_projection)
    monkeypatch.setattr(
        compatibility,
        "inspect_connectors",
        lambda path, *, config: (
            events.append(("connectors", config.model_generation))
            or ConnectorHeaderConfig(4096, 2048, "connectors")
        ),
    )
    monkeypatch.setattr(
        compatibility,
        "inspect_video_vae",
        lambda path, *, model_generation: (
            events.append(("video", model_generation)) or LTX23_VIDEO_VAE_CONFIG
        ),
    )
    monkeypatch.setattr(
        compatibility,
        "inspect_audio_vae",
        lambda path, *, model_generation: (
            events.append(("audio", model_generation)) or AudioVAEConfig()
        ),
    )
    monkeypatch.setattr(
        compatibility,
        "inspect_latent_upscaler",
        lambda path, *, expected_kind, model_generation: (
            events.append((expected_kind, model_generation))
            or LatentUpscalerConfig(
                kind=expected_kind,
                in_channels=128,
                mid_channels=1024,
                num_blocks_per_stage=4,
                scale=2.0 if expected_kind == "spatial" else 1.0,
                rational_resampler=True,
                config_digest=expected_kind,
            )
        ),
    )
    monkeypatch.setattr(
        compatibility,
        "inspect_duration_head",
        lambda path, *, model_generation: (
            events.append(("duration", model_generation))
            or DurationHeadConfig(4096, 2048, "duration")
        ),
    )

    sources = LTX2ComponentSources(
        transformer=paths["transformer"],
        text_encoder=paths["text"],
        video_vae=paths["video"],
        audio_vae=paths["audio"],
        spatial_upscaler=paths["spatial"],
        temporal_upscaler=paths["temporal"],
        duration_head=paths["duration"],
        text_projection_candidates=(paths["text"], paths["transformer"]),
        connector_candidates=(paths["transformer"], paths["text"]),
    )
    report = compatibility.inspect_ltx2_compatibility(
        sources,
        parsed_checkpoint=_parsed(generation),
        expected_generation=generation,
    )

    assert report.label == f"LTX-{generation}"
    assert [name for name, _generation in events] == [
        "transformer",
        "text",
        "projection",
        "connectors",
        "video",
        "audio",
        "spatial",
        "temporal",
        "duration",
    ]
    assert {checked_generation for _name, checked_generation in events} == {generation}


def test_requested_generation_mismatch_is_central_and_explicit(tmp_path: Path) -> None:
    transformer = tmp_path / "transformer.safetensors"
    transformer.touch()
    with pytest.raises(
        ValueError,
        match=r"LTX-2\.5 compatibility was requested.*selects LTX-2\.3",
    ):
        compatibility.inspect_ltx2_compatibility(
            LTX2ComponentSources(
                transformer=transformer,
                text_encoder=None,
                video_vae=None,
            ),
            parsed_checkpoint=_parsed("2.3"),
            expected_generation="2.5",
        )


def test_central_inspection_labels_nested_component_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    transformer = tmp_path / "transformer.safetensors"
    video = tmp_path / "video.safetensors"
    transformer.touch()
    video.touch()
    monkeypatch.setattr(compatibility, "validate_transformer_header", lambda path, config: None)
    monkeypatch.setattr(
        compatibility,
        "inspect_video_vae",
        lambda path, *, model_generation: (_ for _ in ()).throw(
            ValueError("community VAE config is incomplete")
        ),
    )

    with pytest.raises(
        ValueError,
        match="LTX-2.5 compatibility: community VAE config is incomplete",
    ):
        compatibility.inspect_ltx2_compatibility(
            LTX2ComponentSources(
                transformer=transformer,
                text_encoder=None,
                video_vae=video,
            ),
            parsed_checkpoint=_parsed("2.5"),
        )


def test_stale_declared_version_is_recorded_but_does_not_select_the_binder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    transformer = tmp_path / "transformer.safetensors"
    transformer.touch()
    monkeypatch.setattr(compatibility, "validate_transformer_header", lambda path, config: None)

    report = compatibility.inspect_ltx2_compatibility(
        LTX2ComponentSources(
            transformer=transformer,
            text_encoder=None,
            video_vae=None,
        ),
        parsed_checkpoint=_parsed("2.5", declared="2.3.0"),
    )

    assert report.model_generation == "2.5"
    assert report.declared_generation == "2.3"
    assert report.metadata_notes == (
        "declared model_version identifies LTX-2.3, but consumed graph fields select LTX-2.5",
    )
