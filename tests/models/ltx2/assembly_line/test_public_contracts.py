"""Public assembly-line resource, recipe, signal, and example contracts."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.pipelines.distilled as distilled
from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.artifacts import TEXT_CONDITIONING
from kinomlx.models.ltx2.types import DistilledRequest
from kinomlx.reporting import NullReporter, RecordingReporter, Reporter
from tests.models.ltx2.test_distilled_pipeline import (
    _patch_operations,
    _RecordingComponents,
    _RecordingTextConditioner,
    _Resources,
)

_REPO = Path(__file__).resolve().parents[4]


def test_existing_reporter_default_satisfies_the_host_protocol() -> None:
    assert isinstance(NullReporter(), Reporter)


def test_public_resource_plan_is_frozen_and_has_no_live_component_fields() -> None:
    resources = importlib.import_module("kinomlx.models.ltx2.resources")
    resource_type = resources.LTX2Resources

    assert is_dataclass(resource_type)
    assert resource_type.__dataclass_params__.frozen
    names = {field.name for field in fields(resource_type)}
    assert names.isdisjoint(
        {
            "transformer",
            "velocity_model",
            "video_encoder",
            "audio_decoder",
            "vocoder",
            "video_decoder",
        }
    )


def test_public_component_module_exposes_the_structural_provider_surface() -> None:
    components = importlib.import_module("kinomlx.models.ltx2.components")
    required = {
        "TransformerPort",
        "TransformerProvider",
        "VideoEncoderProvider",
        "SpatialUpscalerProvider",
        "AudioDecoderProvider",
        "VocoderProvider",
        "VideoDecoderProvider",
        "DistilledComponents",
        "NativeLTX2Components",
        "load_transformer",
        "load_video_encoder",
        "load_spatial_upscaler",
        "load_audio_decoder",
        "load_vocoder",
        "load_video_decoder",
    }
    assert required.issubset(vars(components))


def test_public_distilled_recipe_has_human_first_injection_signature() -> None:
    recipe = importlib.import_module("kinomlx.models.ltx2.pipelines.distilled")
    signature = inspect.signature(recipe.generate_distilled)
    parameters = signature.parameters

    assert callable(recipe.DistilledRequest)
    assert list(parameters)[:2] == ["request", "resources"]
    assert parameters["components"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["components"].default is None
    assert parameters["reporter"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["reporter"].default is None


def test_public_signal_modules_split_media_types_from_ltx2_constants() -> None:
    media_signals = importlib.import_module("kinomlx.media.signals")
    ltx2_signals = importlib.import_module("kinomlx.models.ltx2.signals")
    media_required = {
        "VideoSignalSpec",
        "VideoValueDomain",
        "EncodedVideoDeliverySpec",
        "OutputColorPlan",
        "BT2020_HLG_DELIVERY",
    }
    ltx2_required = {
        "LTX23_SDR_SIGNAL",
        "ACESCCT_WORKING_SIGNAL",
        "SCENE_LINEAR_HDR_SIGNAL",
    }
    assert media_required.issubset(vars(media_signals))
    assert ltx2_required.issubset(vars(ltx2_signals))


def test_external_distilled_example_imports_and_exposes_composition() -> None:
    example_path = _REPO / "examples" / "ltx2_distilled.py"
    spec = importlib.util.spec_from_file_location("kinomlx_example_ltx2_distilled", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.compose_distilled)
    assert callable(module.main)


def test_external_distilled_example_executes_with_synthetic_components(monkeypatch) -> None:
    example_path = _REPO / "examples" / "ltx2_distilled.py"
    spec = importlib.util.spec_from_file_location("kinomlx_example_ltx2_synthetic", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    components = _RecordingComponents()

    def decode_frames(_latent, decoder_provider, *, spec, frame_count, **_kwargs):
        def produce():
            with decoder_provider():
                for _index in range(frame_count):
                    yield mx.zeros((spec.height, spec.width, 3), dtype=mx.float16)

        return VideoFrameStream(produce, spec=spec, frame_count=frame_count)

    monkeypatch.setattr(module, "decode_ltx23_sdr_frames", decode_frames)
    monkeypatch.setattr(module, "release_stage_temporaries", lambda: None)

    output = module.compose_distilled(
        DistilledRequest(prompt="test", width=64, height=64, frames=9),
        _Resources(),
        components=components,
        text_conditioner=_RecordingTextConditioner(),
    )
    with output:
        assert len(list(output.frames)) == 9
    assert components.active == set()
    assert components.close_counts == {
        "transformer": 1,
        "spatial_upscaler": 1,
        "video_decoder": 1,
    }


@pytest.mark.parametrize("generation", ["2.3", "2.5"])
def test_external_example_and_product_recipe_have_the_same_synthetic_call_trace(
    generation: str,
    monkeypatch,
) -> None:
    example_path = _REPO / "examples" / "ltx2_distilled.py"
    spec = importlib.util.spec_from_file_location("kinomlx_example_ltx2_trace", example_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _patch_operations(monkeypatch)
    monkeypatch.setattr(module, "release_stage_temporaries", lambda: None)

    class _Artifacts:
        def __init__(self) -> None:
            self.events = []

        def save(self, artifact) -> None:
            tensors = dict(artifact.tensors)
            metadata = dict(artifact.metadata)
            if artifact.name == TEXT_CONDITIONING:
                self.events.append(
                    ("text", metadata["prompt"], tuple(tensors["video_encoding"].shape))
                )
                return
            audio_latent = tensors.get("audio_latent")
            self.events.append(
                (
                    "latents",
                    int(metadata["stage"]),
                    metadata["final"] == "true",
                    tuple(tensors["video_latent"].shape),
                    None if audio_latent is None else tuple(audio_latent.shape),
                )
            )

    request = DistilledRequest(prompt="trace", width=64, height=64, frames=9)
    product_components = _RecordingComponents()
    product_text = _RecordingTextConditioner()
    product_reporter = RecordingReporter()
    product_artifacts = _Artifacts()
    product = distilled.generate_distilled(
        request,
        _Resources(generation=generation),
        components=product_components,
        text_conditioner=product_text,
        reporter=product_reporter,
        artifact_sink=product_artifacts,
    )
    list(product.frames)

    example_components = _RecordingComponents()
    example_text = _RecordingTextConditioner()
    example_reporter = RecordingReporter()
    example_artifacts = _Artifacts()
    external = module.compose_distilled(
        request,
        _Resources(generation=generation),
        components=example_components,
        text_conditioner=example_text,
        reporter=example_reporter,
        artifact_sink=example_artifacts,
    )
    list(external.frames)

    assert example_components.events == product_components.events
    assert len(example_text.calls) == len(product_text.calls) == 1
    assert set(example_text.calls[0]) == set(product_text.calls[0])
    assert example_text.calls[0]["request"] == product_text.calls[0]["request"]
    assert example_reporter.events == product_reporter.events
    assert example_artifacts.events == product_artifacts.events
    assert external.metadata == product.metadata
