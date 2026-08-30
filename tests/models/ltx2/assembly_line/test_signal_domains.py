"""Signal-domain separation and current SDR-terminal rejection contracts."""

from __future__ import annotations

import pytest

from kinomlx.errors import KinoMLXError
from kinomlx.media.signals import (
    BT709_SDR_DELIVERY,
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    OutputColorPlan,
    UnsupportedSignalError,
    VideoValueDomain,
    validate_sdr_output_plan,
)
from kinomlx.models.ltx2.signals import (
    ACESCCT_WORKING_SIGNAL,
    LTX23_SDR_SIGNAL,
    SCENE_LINEAR_HDR_SIGNAL,
)

from ._contracts import (
    ContractResources,
    DistilledRequest,
    RecordingComponents,
    assert_balanced,
    run_distilled_contract,
)


def test_unsupported_signal_is_a_typed_operational_failure() -> None:
    assert issubclass(UnsupportedSignalError, KinoMLXError)


def test_signal_types_keep_model_domain_and_delivery_encoding_separate() -> None:
    assert LTX23_SDR_SIGNAL.value_domain is VideoValueDomain.NORMALIZED_SDR
    assert ACESCCT_WORKING_SIGNAL.value_domain is VideoValueDomain.ACESCCT_WORKING_CODES
    assert SCENE_LINEAR_HDR_SIGNAL.value_domain is VideoValueDomain.SCENE_LINEAR
    assert BT2020_HLG_DELIVERY.primaries is ColorPrimaries.BT2020
    assert BT2020_HLG_DELIVERY.transfer is ColorTransfer.HLG

    plan = OutputColorPlan(
        source=SCENE_LINEAR_HDR_SIGNAL,
        deliveries=(BT2020_HLG_DELIVERY,),
    )
    assert plan.source.transfer is ColorTransfer.LINEAR
    assert plan.deliveries[0].transfer is ColorTransfer.HLG


def test_current_sdr_sink_accepts_ltx23_sdr() -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(frame_count=2),
        ContractResources(),
        components=components,
    )
    plan = OutputColorPlan(
        source=LTX23_SDR_SIGNAL,
        deliveries=(BT709_SDR_DELIVERY,),
    )
    validate_sdr_output_plan(plan)
    frames = tuple(output.frames)

    assert frames == ("frame-0", "frame-1")
    assert_balanced(components)


@pytest.mark.parametrize(
    "source",
    [ACESCCT_WORKING_SIGNAL, SCENE_LINEAR_HDR_SIGNAL],
)
def test_current_sdr_sink_rejects_hdr_source_before_consuming_a_frame(source) -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(frame_count=2),
        ContractResources(),
        components=components,
    )

    with pytest.raises(UnsupportedSignalError, match="SDR terminal cannot consume"):
        validate_sdr_output_plan(OutputColorPlan(source=source, deliveries=(BT709_SDR_DELIVERY,)))

    assert not any(event.component == "video_decoder" for event in components.events)
    output.close()
    assert_balanced(components)


def test_current_sdr_sink_rejects_hlg_delivery_before_consuming_a_frame() -> None:
    components = RecordingComponents()
    output = run_distilled_contract(
        DistilledRequest(frame_count=2),
        ContractResources(),
        components=components,
    )

    with pytest.raises(UnsupportedSignalError, match="cannot produce hlg delivery"):
        validate_sdr_output_plan(
            OutputColorPlan(
                source=LTX23_SDR_SIGNAL,
                deliveries=(BT2020_HLG_DELIVERY,),
            )
        )

    assert not any(event.component == "video_decoder" for event in components.events)
    output.close()
    assert_balanced(components)
