"""The five canonical two-stage LoRA transition-table cases."""

from __future__ import annotations

import pytest

from ._contracts import (
    EMPTY_PROFILE,
    AdapterSpec,
    ContractResources,
    DistilledRequest,
    LoraProfile,
    RecordingComponents,
    assert_balanced,
    effective_profile,
    run_distilled_contract,
)

PROFILE_A = effective_profile(AdapterSpec("adapter-a", 0.5, ("audio",)))
PROFILE_B = effective_profile(AdapterSpec("adapter-b", 1.0, ()))


@pytest.mark.parametrize(
    ("stage_1", "stage_2", "expected_load_profiles"),
    [
        pytest.param(EMPTY_PROFILE, EMPTY_PROFILE, [EMPTY_PROFILE], id="empty-empty"),
        pytest.param(PROFILE_A, PROFILE_A, [PROFILE_A], id="a-identical-a"),
        pytest.param(EMPTY_PROFILE, PROFILE_B, [EMPTY_PROFILE, PROFILE_B], id="empty-b"),
        pytest.param(PROFILE_A, EMPTY_PROFILE, [PROFILE_A, EMPTY_PROFILE], id="a-empty"),
        pytest.param(PROFILE_A, PROFILE_B, [PROFILE_A, PROFILE_B], id="a-b"),
    ],
)
def test_lora_profile_transition_table(
    stage_1: LoraProfile,
    stage_2: LoraProfile,
    expected_load_profiles: list[LoraProfile],
) -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(
            stage_1_profile=stage_1,
            stage_2_profile=stage_2,
            generate_audio=False,
        ),
        ContractResources(),
        components=components,
    )
    output.close()

    transformer_loads = [
        event
        for event in components.events
        if event.action == "load" and event.component == "transformer"
    ]
    transformer_closes = [
        event
        for event in components.events
        if event.action == "close" and event.component == "transformer"
    ]
    assert [event.profile for event in transformer_loads] == expected_load_profiles
    assert len(transformer_closes) == len(transformer_loads)

    upscaler_load = next(
        event
        for event in components.events
        if event.action == "load" and event.component == "spatial_upscaler"
    )
    if stage_1 == stage_2:
        assert "transformer" in upscaler_load.active
    else:
        assert "transformer" not in upscaler_load.active
        first_transformer_close = next(
            index
            for index, event in enumerate(components.events)
            if event.action == "close" and event.component == "transformer"
        )
        upscaler_index = components.events.index(upscaler_load)
        assert first_transformer_close < upscaler_index
    assert_balanced(components)


def test_effective_profile_omits_zero_strength_without_reordering() -> None:
    profile = effective_profile(
        AdapterSpec("adapter-a", 0.5),
        AdapterSpec("disabled", 0.0),
        AdapterSpec("adapter-b", 1.0),
    )

    assert profile == (
        AdapterSpec("adapter-a", 0.5),
        AdapterSpec("adapter-b", 1.0),
    )
