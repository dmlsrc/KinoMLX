"""Component lifetime, failure cleanup, and lazy-frame ownership contracts."""

from __future__ import annotations

from ._contracts import (
    AdapterSpec,
    ContractResources,
    DistilledRequest,
    RecordingComponents,
    assert_balanced,
    effective_profile,
    run_distilled_contract,
)


def _load_event(components: RecordingComponents, component: str):
    return next(
        event
        for event in components.events
        if event.action == "load" and event.component == component
    )


def test_distinct_profiles_release_transformer_before_upscaler_load() -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(
            stage_1_profile=effective_profile(AdapterSpec("adapter-a", 0.5)),
            stage_2_profile=effective_profile(AdapterSpec("adapter-b", 1.0)),
        ),
        ContractResources(),
        components=components,
    )
    with output:
        list(output.frames)

    upscaler_load = _load_event(components, "spatial_upscaler")
    assert upscaler_load.active == frozenset({"spatial_upscaler"})
    assert _load_event(components, "audio_decoder").active == frozenset({"audio_decoder"})
    assert _load_event(components, "vocoder").active == frozenset({"vocoder"})
    assert _load_event(components, "video_decoder").active == frozenset({"video_decoder"})
    assert_balanced(components)


def test_identical_profiles_may_retain_transformer_across_upscale() -> None:
    profile = effective_profile(AdapterSpec("adapter-a", 0.5))
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(stage_1_profile=profile, stage_2_profile=profile),
        ContractResources(),
        components=components,
    )
    output.close()

    upscaler_load = _load_event(components, "spatial_upscaler")
    assert upscaler_load.active == frozenset({"transformer", "spatial_upscaler"})
    assert_balanced(components)


def test_close_early_releases_video_decoder_once() -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(frame_count=4),
        ContractResources(),
        components=components,
    )

    assert next(output.frames) == "frame-0"
    output.close()
    output.close()

    decoder_events = [
        event.action for event in components.events if event.component == "video_decoder"
    ]
    assert decoder_events == ["load", "use", "close"]
    assert_balanced(components)


def test_exhaustion_releases_video_decoder() -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(frame_count=2),
        ContractResources(),
        components=components,
    )

    assert list(output.frames) == ["frame-0", "frame-1"]
    assert_balanced(components)
