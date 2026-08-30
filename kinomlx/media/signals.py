"""Typed video-signal and terminal-delivery vocabulary.

A signal describes the values owned by a frame stream. A delivery describes
what a terminal must encode. Keeping those facts separate prevents a sink
from relabelling scene-linear or working-space values as display video.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from kinomlx.errors import KinoMLXError


class VideoLayout(StrEnum):
    """Owned public frame layouts."""

    HWC_RGB = "hwc-rgb"


class VideoValueDomain(StrEnum):
    """Semantic domain represented by pixel values."""

    NORMALIZED_SDR = "normalized-sdr"
    ACESCCT_WORKING_CODES = "acescct-working-codes"
    LOGC3_WORKING_CODES = "logc3-working-codes"
    SCENE_LINEAR = "scene-linear"


class ColorPrimaries(StrEnum):
    REC709 = "rec709"
    ACESCG = "acescg"
    BT2020 = "bt2020"


class ColorTransfer(StrEnum):
    SRGB = "srgb"
    ACESCCT = "acescct"
    LOGC3 = "logc3"
    LINEAR = "linear"
    HLG = "hlg"


class ColorMatrix(StrEnum):
    RGB = "rgb"
    BT709 = "bt709"
    BT2020_NCL = "bt2020-ncl"


class ColorRange(StrEnum):
    FULL = "full"
    VIDEO = "video"


class VideoCodec(StrEnum):
    HEVC = "hevc"


class VideoCodecProfile(StrEnum):
    MAIN10 = "main10"
    MAIN42210 = "main42210"


class ChromaSubsampling(StrEnum):
    YUV420 = "4:2:0"
    YUV422 = "4:2:2"


class ExrSampleType(StrEnum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


@dataclass(frozen=True)
class VideoSignalSpec:
    """Immutable interpretation and geometry for one frame stream."""

    layout: VideoLayout
    dtype: str
    value_domain: VideoValueDomain
    primaries: ColorPrimaries
    transfer: ColorTransfer
    matrix: ColorMatrix
    range: ColorRange
    width: int
    height: int
    cadence: Fraction

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video signal dimensions must be positive")
        if self.cadence <= 0:
            raise ValueError("video signal cadence must be positive")


@dataclass(frozen=True)
class EncodedVideoDeliverySpec:
    """Codec, pixel encoding, and container color description for a video."""

    codec: VideoCodec
    profile: VideoCodecProfile
    primaries: ColorPrimaries
    transfer: ColorTransfer
    matrix: ColorMatrix
    range: ColorRange
    bit_depth: int
    chroma: ChromaSubsampling

    def __post_init__(self) -> None:
        if self.bit_depth <= 0:
            raise ValueError("encoded video bit depth must be positive")


@dataclass(frozen=True)
class ExrDeliverySpec:
    """HDR-safe still-sequence delivery: authoring space for an EXR master."""

    primaries: ColorPrimaries
    transfer: ColorTransfer
    sample_type: ExrSampleType
    color_space_tag: str


VideoDeliverySpec = EncodedVideoDeliverySpec | ExrDeliverySpec


@dataclass(frozen=True)
class OutputColorPlan:
    """A source signal and one or more independently resolved deliveries."""

    source: VideoSignalSpec
    deliveries: tuple[VideoDeliverySpec, ...]

    def __post_init__(self) -> None:
        if not self.deliveries:
            raise ValueError("output color plan must contain at least one delivery")


class UnsupportedSignalError(KinoMLXError, ValueError):
    """The selected terminal cannot preserve the requested signal contract."""


BT709_SDR_420_DELIVERY = EncodedVideoDeliverySpec(
    codec=VideoCodec.HEVC,
    profile=VideoCodecProfile.MAIN10,
    primaries=ColorPrimaries.REC709,
    transfer=ColorTransfer.SRGB,
    matrix=ColorMatrix.BT709,
    range=ColorRange.VIDEO,
    bit_depth=10,
    chroma=ChromaSubsampling.YUV420,
)

BT709_SDR_422_DELIVERY = EncodedVideoDeliverySpec(
    codec=VideoCodec.HEVC,
    profile=VideoCodecProfile.MAIN42210,
    primaries=ColorPrimaries.REC709,
    transfer=ColorTransfer.SRGB,
    matrix=ColorMatrix.BT709,
    range=ColorRange.VIDEO,
    bit_depth=10,
    chroma=ChromaSubsampling.YUV422,
)

# The current high-fidelity default. Hosts resolve 4:2:0 explicitly for the
# passthrough/low-latency path instead of mutating this value.
BT709_SDR_DELIVERY = BT709_SDR_422_DELIVERY

BT2020_HLG_DELIVERY = EncodedVideoDeliverySpec(
    codec=VideoCodec.HEVC,
    profile=VideoCodecProfile.MAIN10,
    primaries=ColorPrimaries.BT2020,
    transfer=ColorTransfer.HLG,
    matrix=ColorMatrix.BT2020_NCL,
    range=ColorRange.VIDEO,
    bit_depth=10,
    chroma=ChromaSubsampling.YUV420,
)


def validate_sdr_delivery(delivery: VideoDeliverySpec) -> EncodedVideoDeliverySpec:
    """Validate one delivery supported by the current BT.709 terminal."""
    if not isinstance(delivery, EncodedVideoDeliverySpec):
        raise UnsupportedSignalError("SDR terminal cannot produce EXR delivery")
    if (
        delivery.codec is not VideoCodec.HEVC
        or delivery.primaries is not ColorPrimaries.REC709
        or delivery.transfer is not ColorTransfer.SRGB
        or delivery.matrix is not ColorMatrix.BT709
        or delivery.range is not ColorRange.VIDEO
        or delivery.bit_depth != 10
        or delivery.chroma not in (ChromaSubsampling.YUV420, ChromaSubsampling.YUV422)
    ):
        raise UnsupportedSignalError(
            f"SDR terminal cannot produce {delivery.transfer.value} delivery"
        )
    expected_profile = (
        VideoCodecProfile.MAIN10
        if delivery.chroma is ChromaSubsampling.YUV420
        else VideoCodecProfile.MAIN42210
    )
    if delivery.profile is not expected_profile:
        raise UnsupportedSignalError(
            f"{delivery.chroma.value} delivery requires {expected_profile.value}"
        )
    return delivery


def validate_hlg_delivery(delivery: VideoDeliverySpec) -> EncodedVideoDeliverySpec:
    """Validate the bounded BT.2020/HLG terminal's encoded master."""
    if not isinstance(delivery, EncodedVideoDeliverySpec):
        raise UnsupportedSignalError("HLG terminal requires an encoded video delivery")
    if (
        delivery.codec is not VideoCodec.HEVC
        or delivery.profile is not VideoCodecProfile.MAIN10
        or delivery.primaries is not ColorPrimaries.BT2020
        or delivery.transfer is not ColorTransfer.HLG
        or delivery.matrix is not ColorMatrix.BT2020_NCL
        or delivery.range is not ColorRange.VIDEO
        or delivery.bit_depth != 10
        or delivery.chroma is not ChromaSubsampling.YUV420
    ):
        raise UnsupportedSignalError(
            "HLG terminal requires HEVC Main10 4:2:0 BT.2020/HLG video-range delivery"
        )
    return delivery


def validate_encoded_delivery(delivery: VideoDeliverySpec) -> EncodedVideoDeliverySpec:
    """Dispatch one encoded delivery to its exact SDR or HLG contract."""
    if isinstance(delivery, EncodedVideoDeliverySpec) and delivery.transfer is ColorTransfer.HLG:
        return validate_hlg_delivery(delivery)
    return validate_sdr_delivery(delivery)


def validate_sdr_output_plan(plan: OutputColorPlan) -> None:
    """Reject unsupported source/delivery semantics without pulling a frame."""
    source = plan.source
    if (
        source.layout is not VideoLayout.HWC_RGB
        or source.dtype != "float16"
        or source.value_domain is not VideoValueDomain.NORMALIZED_SDR
        or source.primaries is not ColorPrimaries.REC709
        or source.transfer is not ColorTransfer.SRGB
        or source.matrix is not ColorMatrix.RGB
        or source.range is not ColorRange.FULL
    ):
        raise UnsupportedSignalError(
            f"SDR terminal cannot consume {source.value_domain.value} "
            f"{source.primaries.value}/{source.transfer.value} frames"
        )

    for delivery in plan.deliveries:
        validate_sdr_delivery(delivery)


__all__ = [
    "BT2020_HLG_DELIVERY",
    "BT709_SDR_420_DELIVERY",
    "BT709_SDR_422_DELIVERY",
    "BT709_SDR_DELIVERY",
    "ChromaSubsampling",
    "ColorMatrix",
    "ColorPrimaries",
    "ColorRange",
    "ColorTransfer",
    "EncodedVideoDeliverySpec",
    "ExrDeliverySpec",
    "ExrSampleType",
    "OutputColorPlan",
    "UnsupportedSignalError",
    "VideoCodec",
    "VideoCodecProfile",
    "VideoDeliverySpec",
    "VideoLayout",
    "VideoSignalSpec",
    "VideoValueDomain",
    "validate_sdr_output_plan",
    "validate_sdr_delivery",
    "validate_encoded_delivery",
    "validate_hlg_delivery",
]
