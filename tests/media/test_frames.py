"""Ownership and validation tests for model-neutral frame streams."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from kinomlx.media.frames import VideoFrameStream
from kinomlx.media.signals import (
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    ExrDeliverySpec,
    ExrSampleType,
    OutputColorPlan,
)
from kinomlx.models.ltx2.signals import (
    ACESCCT_WORKING_SIGNAL,
    SCENE_LINEAR_HDR_SIGNAL,
    ltx23_sdr_signal,
)


def _frame() -> mx.array:
    return mx.zeros((32, 64, 3), dtype=mx.float16)


def _stream(factory, *, count: int = 2) -> VideoFrameStream:
    return VideoFrameStream(
        factory,
        spec=ltx23_sdr_signal(width=64, height=32, fps=24.0),
        frame_count=count,
    )


def test_close_before_first_pull_discards_factory_without_opening_it() -> None:
    opened = []
    stream = _stream(lambda: opened.append(True) or iter((_frame(), _frame())))

    stream.close()

    assert opened == []
    assert stream.closed
    assert list(stream) == []


def test_exhaustion_validates_exact_count_and_closes_producer() -> None:
    events = []

    def produce():
        events.append("open")
        try:
            yield _frame()
            yield _frame()
        finally:
            events.append("close")

    stream = _stream(produce)
    assert len(list(stream)) == 2
    assert events == ["open", "close"]
    assert stream.closed
    assert stream.consumed == 2


def test_early_close_runs_producer_finally_once() -> None:
    events = []

    def produce():
        try:
            yield _frame()
            yield _frame()
        finally:
            events.append("close")

    stream = _stream(produce)
    next(stream)
    stream.close()
    stream.close()

    assert events == ["close"]


def test_short_stream_fails_loudly_after_closing() -> None:
    stream = _stream(lambda: iter((_frame(),)))

    with pytest.raises(RuntimeError, match="frame count 1 does not match expected 2"):
        list(stream)

    assert stream.closed


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (mx.zeros((31, 64, 3), dtype=mx.float16), "frame shape"),
        (mx.zeros((32, 64, 3), dtype=mx.float32), "frame dtype"),
    ],
)
def test_frame_contract_mismatch_closes_before_exposing_value(frame, message) -> None:
    stream = _stream(lambda: iter((frame, _frame())), count=2)

    with pytest.raises(RuntimeError, match=message):
        next(stream)

    assert stream.closed
    assert stream.consumed == 0


def test_synthetic_float32_hdr_transform_fans_out_one_decode_in_frame_order() -> None:
    """Prove the public seam without claiming an implemented HDR transform."""
    events = []
    working_spec = replace(ACESCCT_WORKING_SIGNAL, width=2, height=1)
    scene_spec = replace(SCENE_LINEAR_HDR_SIGNAL, width=2, height=1)
    exr = ExrDeliverySpec(
        primaries=ColorPrimaries.ACESCG,
        transfer=ColorTransfer.LINEAR,
        sample_type=ExrSampleType.FLOAT16,
        color_space_tag="ACEScg",
    )
    plan = OutputColorPlan(
        source=scene_spec,
        deliveries=(exr, BT2020_HLG_DELIVERY),
    )

    def decode_once():
        try:
            for index, peak in enumerate((0.75, 1.0)):
                events.append(f"decode:{index}")
                yield mx.full((1, 2, 3), peak, dtype=mx.float32)
        finally:
            events.append("decode:close")

    source = VideoFrameStream(decode_once, spec=working_spec, frame_count=2)
    for index, working_codes in enumerate(source):
        # Deliberately synthetic inverse: its purpose is to establish that a
        # named float32 transform can preserve values above 1.0 at the seam.
        scene_linear = working_codes * 4.0
        mx.eval(scene_linear)
        assert scene_linear.dtype == mx.float32
        assert float(mx.max(scene_linear)) > 1.0
        for delivery in plan.deliveries:
            sink = "exr" if isinstance(delivery, ExrDeliverySpec) else "hlg"
            events.append(f"{sink}:{index}")

    assert events == [
        "decode:0",
        "exr:0",
        "hlg:0",
        "decode:1",
        "exr:1",
        "hlg:1",
        "decode:close",
    ]
    assert source.closed
