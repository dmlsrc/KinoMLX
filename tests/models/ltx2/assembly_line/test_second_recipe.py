"""Synthetic second-recipe composition proof."""

from __future__ import annotations

from dataclasses import dataclass, fields

import mlx.core as mx

from kinomlx.media.frames import VideoFrameStream
from kinomlx.models.ltx2.pipelines.distilled import prepare_stage, run_stage
from kinomlx.models.ltx2.runner import GenerationOutput, LTX2Runner
from kinomlx.models.ltx2.signals import ltx23_sdr_signal
from kinomlx.models.ltx2.types import DistilledRequest as ProductDistilledRequest
from kinomlx.types import VideoPixelShape
from tests.models.ltx2.test_distilled_pipeline import (
    _RecordingComponents,
    _RecordingTextConditioner,
    _Resources,
)

from ._contracts import (
    AdapterSpec,
    ContractResources,
    OneStageRequest,
    RawVideoCondition,
    RecordingComponents,
    assert_balanced,
    effective_profile,
    run_one_stage_contract,
)


@dataclass(frozen=True)
class SyntheticOneStageRequest:
    """Test-only request proving that the host does not encode recipe branches."""

    prompt: str
    width: int = 64
    height: int = 64
    frames: int = 9
    seed: int = 11


def generate_synthetic_one_stage(
    request: SyntheticOneStageRequest,
    resources,
    *,
    components=None,
    text_conditioner=None,
    reporter=None,
    artifact_sink=None,
) -> GenerationOutput:
    """Compose public text, state, denoise, signal, and output products once."""
    del artifact_sink
    if components is None or text_conditioner is None:
        raise ValueError("the synthetic proof requires injected public ports")
    distilled_request = ProductDistilledRequest(
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        frames=request.frames,
        seed=request.seed,
    )
    text = text_conditioner(distilled_request, resources, reporter=reporter)
    geometry = VideoPixelShape(
        batch=1,
        frames=request.frames,
        height=request.height,
        width=request.width,
    )
    stage = prepare_stage(
        distilled_request,
        geometry,
        dtype_policy=resources.dtype_policy,
    )
    with components.transformer(resources, ()) as transformer:
        run_stage(
            stage,
            transformer,
            text,
            (1.0, 0.0),
            seed=request.seed,
            reporter=reporter,
            phase="synthetic one-stage denoise",
        )
    signal = ltx23_sdr_signal(
        width=request.width,
        height=request.height,
        fps=distilled_request.fps,
    )
    frames = VideoFrameStream(
        lambda: (
            mx.zeros((request.height, request.width, 3), dtype=mx.float16)
            for _index in range(request.frames)
        ),
        spec=signal,
        frame_count=request.frames,
    )
    return GenerationOutput(frames=frames, metadata={"recipe": "synthetic-one-stage"})


def test_raw_condition_owns_no_encoder_or_model() -> None:
    condition = RawVideoCondition(kind="video", source_id="synthetic-source")

    assert {field.name for field in fields(condition)} == {"kind", "source_id"}


def test_conditioned_one_stage_recipe_uses_only_its_narrow_component_set() -> None:
    components = RecordingComponents()
    output = run_one_stage_contract(
        OneStageRequest(
            condition=RawVideoCondition(kind="video", source_id="synthetic-source"),
            profile=effective_profile(AdapterSpec("adapter-a", 0.75)),
            frame_count=2,
        ),
        ContractResources(),
        components=components,
    )
    with output:
        assert list(output.frames) == ["frame-0", "frame-1"]

    loads = [event.component for event in components.events if event.action == "load"]
    assert loads == ["video_encoder", "transformer", "video_decoder"]
    assert "spatial_upscaler" not in loads
    assert "audio_decoder" not in loads
    assert "vocoder" not in loads
    assert_balanced(components)


def test_condition_preparation_closes_encoder_before_transformer_load() -> None:
    components = RecordingComponents()
    output = run_one_stage_contract(
        OneStageRequest(
            condition=RawVideoCondition(kind="keyframe", source_id="synthetic-keyframe")
        ),
        ContractResources(),
        components=components,
    )
    output.close()

    transformer_load = next(
        event
        for event in components.events
        if event.action == "load" and event.component == "transformer"
    )
    assert transformer_load.active == frozenset({"transformer"})
    assert_balanced(components)


def test_generic_runner_executes_a_second_public_station_composition() -> None:
    components = _RecordingComponents()
    runner = LTX2Runner(
        resources=_Resources(),
        components=components,
        text_conditioner=_RecordingTextConditioner(),
    )

    output = runner.run(
        generate_synthetic_one_stage,
        SyntheticOneStageRequest(prompt="synthetic"),
    )

    assert output.metadata == {"recipe": "synthetic-one-stage"}
    assert len(list(output.frames)) == 9
    loads = [event[1] for event in components.events if event[0] == "load"]
    assert loads == ["transformer"]
    assert components.close_counts == {"transformer": 1}
    assert components.active == set()
