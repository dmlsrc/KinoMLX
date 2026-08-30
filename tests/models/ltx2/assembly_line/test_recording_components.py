"""Recording provider and default/injected call-graph contracts."""

from __future__ import annotations

from ._contracts import (
    AdapterSpec,
    ContractResources,
    DistilledRequest,
    RecordingComponents,
    RecordingProgressReporter,
    RecordingTextConditioner,
    assert_balanced,
    effective_profile,
    event_signature,
    run_distilled_contract,
)


def test_injected_factory_records_load_use_close_and_active_sets() -> None:
    resources = ContractResources()
    components = RecordingComponents()

    with components.video_encoder(resources) as encoder:
        encoder.use("condition preparation")

    assert [event.action for event in components.events] == ["load", "use", "close"]
    assert components.events[0].active == frozenset({"video_encoder"})
    assert components.events[1].active == frozenset({"video_encoder"})
    assert components.events[2].active == frozenset()
    assert_balanced(components)


def test_text_conditioner_returns_a_product_without_loading_a_component() -> None:
    station = RecordingTextConditioner()
    components = RecordingComponents()

    output = run_distilled_contract(
        DistilledRequest(image_conditioned=False, generate_audio=False),
        ContractResources(),
        components=components,
        text_conditioner=station,
    )

    assert output.text_conditioning.value == "encoded prompt"
    assert station.events == [("start", "prompt"), ("return", "encoded prompt")]
    assert all(event.component != "text_encoder" for event in components.events)
    output.close()
    assert_balanced(components)


def test_default_and_injected_components_follow_the_same_recipe_call_graph() -> None:
    request = DistilledRequest(
        stage_1_profile=effective_profile(AdapterSpec("adapter-a", 0.5)),
        stage_2_profile=effective_profile(AdapterSpec("adapter-b", 1.0)),
    )
    resources = ContractResources()

    default_output = run_distilled_contract(request, resources)
    assert default_output.recording is not None
    with default_output:
        list(default_output.frames)

    injected = RecordingComponents()
    injected_output = run_distilled_contract(request, resources, components=injected)
    with injected_output:
        list(injected_output.frames)

    assert [event_signature(event) for event in default_output.recording.events] == [
        event_signature(event) for event in injected.events
    ]
    assert default_output.text_conditioning == injected_output.text_conditioning
    assert_balanced(default_output.recording)
    assert_balanced(injected)


def test_recipe_reports_the_same_station_boundaries_as_the_component_trace() -> None:
    reporter = RecordingProgressReporter()
    output = run_distilled_contract(
        DistilledRequest(),
        ContractResources(),
        reporter=reporter,
    )
    output.close()

    starts = [phase for action, phase in reporter.events if action == "start"]
    ends = [phase for action, phase in reporter.events if action == "end"]
    assert starts == [
        "prompt",
        "stage 1 condition",
        "stage 1",
        "upscale",
        "stage 2 condition",
        "stage 2",
        "audio decode",
        "vocoder",
    ]
    assert ends == starts
