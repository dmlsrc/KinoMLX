"""LTX-family signal constants and checkpoint-resolved geometry helpers."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from kinomlx.media.signals import (
    ColorMatrix,
    ColorPrimaries,
    ColorRange,
    ColorTransfer,
    UnsupportedSignalError,
    VideoLayout,
    VideoSignalSpec,
    VideoValueDomain,
)

LTX23_SDR_SIGNAL = VideoSignalSpec(
    layout=VideoLayout.HWC_RGB,
    dtype="float16",
    value_domain=VideoValueDomain.NORMALIZED_SDR,
    primaries=ColorPrimaries.REC709,
    transfer=ColorTransfer.SRGB,
    matrix=ColorMatrix.RGB,
    range=ColorRange.FULL,
    width=1024,
    height=576,
    cadence=Fraction(24, 1),
)

ACESCCT_WORKING_SIGNAL = VideoSignalSpec(
    layout=VideoLayout.HWC_RGB,
    dtype="float32",
    value_domain=VideoValueDomain.ACESCCT_WORKING_CODES,
    primaries=ColorPrimaries.ACESCG,
    transfer=ColorTransfer.ACESCCT,
    matrix=ColorMatrix.RGB,
    range=ColorRange.FULL,
    width=1024,
    height=576,
    cadence=Fraction(24, 1),
)

LOGC3_WORKING_SIGNAL = VideoSignalSpec(
    layout=VideoLayout.HWC_RGB,
    dtype="float32",
    value_domain=VideoValueDomain.LOGC3_WORKING_CODES,
    primaries=ColorPrimaries.REC709,
    transfer=ColorTransfer.LOGC3,
    matrix=ColorMatrix.RGB,
    range=ColorRange.FULL,
    width=1024,
    height=576,
    cadence=Fraction(24, 1),
)

SCENE_LINEAR_HDR_SIGNAL = VideoSignalSpec(
    layout=VideoLayout.HWC_RGB,
    dtype="float32",
    value_domain=VideoValueDomain.SCENE_LINEAR,
    primaries=ColorPrimaries.ACESCG,
    transfer=ColorTransfer.LINEAR,
    matrix=ColorMatrix.RGB,
    range=ColorRange.FULL,
    width=1024,
    height=576,
    cadence=Fraction(24, 1),
)


def ltx23_sdr_signal(*, width: int, height: int, fps: float) -> VideoSignalSpec:
    """Resolve the LTX-2.3 SDR contract for one generation request."""
    cadence = Fraction(str(fps)).limit_denominator(1_000_000)
    return replace(
        LTX23_SDR_SIGNAL,
        width=width,
        height=height,
        cadence=cadence,
    )


def ltx_hdr_working_signal(
    *,
    transfer: ColorTransfer,
    width: int,
    height: int,
    fps: float,
) -> VideoSignalSpec:
    """Resolve one LTX HDR working-code stream for a generation request."""
    if transfer is ColorTransfer.ACESCCT:
        template = ACESCCT_WORKING_SIGNAL
    elif transfer is ColorTransfer.LOGC3:
        template = LOGC3_WORKING_SIGNAL
    else:
        raise ValueError("LTX HDR working transfer must be ACEScct or LogC3")
    return replace(
        template,
        width=width,
        height=height,
        cadence=Fraction(str(fps)).limit_denominator(1_000_000),
    )


def validate_ltx23_sdr_signal(spec: VideoSignalSpec) -> None:
    """Reject a signal label that the LTX-2.3 SDR postprocess cannot produce."""
    expected = LTX23_SDR_SIGNAL
    if (
        spec.layout is not expected.layout
        or spec.dtype != expected.dtype
        or spec.value_domain is not expected.value_domain
        or spec.primaries is not expected.primaries
        or spec.transfer is not expected.transfer
        or spec.matrix is not expected.matrix
        or spec.range is not expected.range
    ):
        raise UnsupportedSignalError(
            "LTX-2.3 SDR decode cannot produce "
            f"{spec.value_domain.value} {spec.primaries.value}/{spec.transfer.value} frames"
        )


def validate_ltx_hdr_working_signal(spec: VideoSignalSpec) -> None:
    """Reject labels the bounded float32 HDR decode cannot produce."""
    expected_domain = {
        ColorTransfer.ACESCCT: VideoValueDomain.ACESCCT_WORKING_CODES,
        ColorTransfer.LOGC3: VideoValueDomain.LOGC3_WORKING_CODES,
    }.get(spec.transfer)
    if (
        spec.layout is not VideoLayout.HWC_RGB
        or spec.dtype != "float32"
        or expected_domain is None
        or spec.value_domain is not expected_domain
        or spec.primaries
        is not (
            ColorPrimaries.ACESCG
            if spec.transfer is ColorTransfer.ACESCCT
            else ColorPrimaries.REC709
        )
        or spec.matrix is not ColorMatrix.RGB
        or spec.range is not ColorRange.FULL
    ):
        raise UnsupportedSignalError(
            "LTX HDR decode cannot produce "
            f"{spec.value_domain.value} {spec.primaries.value}/{spec.transfer.value} frames"
        )


__all__ = [
    "ACESCCT_WORKING_SIGNAL",
    "LOGC3_WORKING_SIGNAL",
    "LTX23_SDR_SIGNAL",
    "SCENE_LINEAR_HDR_SIGNAL",
    "ltx_hdr_working_signal",
    "ltx23_sdr_signal",
    "validate_ltx_hdr_working_signal",
    "validate_ltx23_sdr_signal",
]
