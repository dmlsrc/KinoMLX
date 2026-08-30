"""MLX-native HDR working transfers and linear-primary conversion.

The working curves are model facts. They are deliberately separate from
display encoding: this module never produces HLG, YUV, or a container tag.
All operations retain float32 scene-linear values until a terminal chooses a
delivery representation.
"""

from __future__ import annotations

import mlx.core as mx

from .signals import ColorPrimaries, ColorTransfer, UnsupportedSignalError

# Published ACEScct constants from AMPAS S-2016-001.
_ACESCCT_LINEAR_SLOPE = 10.5402377416545
_ACESCCT_LINEAR_OFFSET = 0.0729055341958355
_ACESCCT_LINEAR_BREAK = 0.0078125
_ACESCCT_CODE_BREAK = 0.155251141552511
_ACESCCT_LOG_DIVISOR = 17.52
_ACESCCT_LOG_OFFSET = 9.72

# Published ARRI LogC3 EI 800 constants.
_LOGC3_A = 5.555556
_LOGC3_B = 0.052272
_LOGC3_C = 0.247190
_LOGC3_D = 0.385537
_LOGC3_E = 5.367655
_LOGC3_F = 0.092809
_LOGC3_LINEAR_BREAK = 0.010591
_LOGC3_CODE_BREAK = _LOGC3_E * _LOGC3_LINEAR_BREAK + _LOGC3_F

_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# D60-adapted ACEScg AP1 <-> D65 Rec.709/sRGB linear-light matrices.
_ACESCG_TO_REC709 = (
    (1.70505000, -0.62179000, -0.08326000),
    (-0.13026000, 1.14080000, -0.01055000),
    (-0.02400000, -0.12897000, 1.15297000),
)
_REC709_TO_ACESCG = (
    (0.61309848, 0.33952419, 0.04738073),
    (0.07019608, 0.91635904, 0.01345405),
    (0.02061420, 0.10957042, 0.86981648),
)

# D65 Rec.709 -> D65 BT.2020 and D60-adapted ACEScg -> D65 BT.2020.
_REC709_TO_BT2020 = (
    (0.62740389, 0.32928304, 0.04331307),
    (0.06909729, 0.91954040, 0.01136232),
    (0.01639144, 0.08801331, 0.89559525),
)
_ACESCG_TO_BT2020 = (
    (1.02582475, -0.02005319, -0.00577156),
    (-0.00223437, 1.00458650, -0.00235213),
    (-0.00501335, -0.02529007, 1.03030342),
)


def _float32(value: mx.array) -> mx.array:
    return value if value.dtype == mx.float32 else value.astype(mx.float32)


def _bounded_codes(value: mx.array) -> mx.array:
    """Return finite float32 working codes bounded to their defined interval."""
    codes = _float32(value)
    return mx.clip(mx.nan_to_num(codes, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def acescct_to_scene_linear(codes: mx.array) -> mx.array:
    """Decode bounded ACEScct working codes into float32 scene-linear light."""
    value = _bounded_codes(codes)
    log_branch = mx.power(2.0, value * _ACESCCT_LOG_DIVISOR - _ACESCCT_LOG_OFFSET)
    linear_branch = (value - _ACESCCT_LINEAR_OFFSET) / _ACESCCT_LINEAR_SLOPE
    return mx.where(value > _ACESCCT_CODE_BREAK, log_branch, linear_branch)


def scene_linear_to_acescct(linear: mx.array) -> mx.array:
    """Encode non-negative scene-linear light as bounded float32 ACEScct."""
    value = mx.maximum(_float32(linear), 0.0)
    safe = mx.maximum(value, mx.array(1e-30, dtype=mx.float32))
    log_branch = (mx.log2(safe) + _ACESCCT_LOG_OFFSET) / _ACESCCT_LOG_DIVISOR
    linear_branch = _ACESCCT_LINEAR_SLOPE * value + _ACESCCT_LINEAR_OFFSET
    return mx.clip(
        mx.where(value > _ACESCCT_LINEAR_BREAK, log_branch, linear_branch),
        0.0,
        1.0,
    )


def logc3_to_scene_linear(codes: mx.array) -> mx.array:
    """Decode bounded ARRI LogC3 EI 800 codes into float32 scene-linear light."""
    value = _bounded_codes(codes)
    log_branch = (mx.power(10.0, (value - _LOGC3_D) / _LOGC3_C) - _LOGC3_B) / _LOGC3_A
    linear_branch = (value - _LOGC3_F) / _LOGC3_E
    return mx.where(value >= _LOGC3_CODE_BREAK, log_branch, linear_branch)


def scene_linear_to_logc3(linear: mx.array) -> mx.array:
    """Encode non-negative scene-linear light as bounded LogC3 EI 800 codes."""
    value = mx.maximum(_float32(linear), 0.0)
    log_branch = _LOGC3_C * mx.log10(_LOGC3_A * value + _LOGC3_B) + _LOGC3_D
    linear_branch = _LOGC3_E * value + _LOGC3_F
    return mx.clip(
        mx.where(value >= _LOGC3_LINEAR_BREAK, log_branch, linear_branch),
        0.0,
        1.0,
    )


def decode_working_transfer(codes: mx.array, transfer: ColorTransfer) -> mx.array:
    """Decode one supported model working transfer to scene-linear light."""
    if transfer is ColorTransfer.ACESCCT:
        return acescct_to_scene_linear(codes)
    if transfer is ColorTransfer.LOGC3:
        return logc3_to_scene_linear(codes)
    if transfer is ColorTransfer.LINEAR:
        return _float32(codes)
    raise UnsupportedSignalError(f"no HDR working decoder for {transfer.value}")


def _matrix(
    source: ColorPrimaries,
    target: ColorPrimaries,
) -> tuple[tuple[float, float, float], ...]:
    if source is target:
        return _IDENTITY
    if source is ColorPrimaries.ACESCG and target is ColorPrimaries.REC709:
        return _ACESCG_TO_REC709
    if source is ColorPrimaries.REC709 and target is ColorPrimaries.ACESCG:
        return _REC709_TO_ACESCG
    if source is ColorPrimaries.REC709 and target is ColorPrimaries.BT2020:
        return _REC709_TO_BT2020
    if source is ColorPrimaries.ACESCG and target is ColorPrimaries.BT2020:
        return _ACESCG_TO_BT2020
    raise UnsupportedSignalError(
        f"no scene-linear primary conversion from {source.value} to {target.value}"
    )


def convert_scene_linear_primaries(
    rgb: mx.array,
    *,
    source: ColorPrimaries,
    target: ColorPrimaries,
) -> mx.array:
    """Convert HWC or batched HWC float RGB between supported linear primaries."""
    if rgb.ndim < 3 or rgb.shape[-1] != 3:
        raise ValueError(f"scene-linear RGB must end in 3 channels, got {tuple(rgb.shape)}")
    value = _float32(rgb)
    matrix = mx.array(_matrix(source, target), dtype=mx.float32)
    return value @ matrix.T


__all__ = [
    "acescct_to_scene_linear",
    "convert_scene_linear_primaries",
    "decode_working_transfer",
    "logc3_to_scene_linear",
    "scene_linear_to_acescct",
    "scene_linear_to_logc3",
]
