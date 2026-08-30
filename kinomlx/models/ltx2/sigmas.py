"""LTX-2.3 distilled sigma schedules.

The distilled two-stage pipeline uses *literal* sigma schedules - no
dynamic scheduler.  The values match the official LTX-2 ComfyUI
distilled schedule (9 values driving 8 denoising steps in stage 1;
4 values driving 3 steps in stage 2).

If a non-distilled path is ever added that needs dynamic
scheduling, the generic scheduler module lands in
``kinomlx/samplers/`` at that time; it is deferred until then.
"""

from __future__ import annotations

import struct

import mlx.core as mx


def _float32(value: float) -> float:
    """Round one host scalar exactly as the official float32 sigma tensor."""
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


# Stage-1 denoise: 8 steps (9 sigma values, ``sigmas[i]`` -> ``sigmas[i+1]``).
DISTILLED_STAGE_1_SIGMAS: tuple[float, ...] = tuple(
    _float32(value)
    for value in (
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    )
)

# Stage-2 denoise: 3 steps after the spatial upscaler.  The first
# three values mirror the tail of stage 1; the final ``0.0`` ends
# the denoise. Keep this independent literal pin: deriving it from stage 1
# would let an accidental change to both schedules validate itself.
DISTILLED_STAGE_2_START_INDEX = 5
DISTILLED_STAGE_2_SIGMAS: tuple[float, ...] = tuple(
    _float32(value)
    for value in (
        0.909375,
        0.725,
        0.421875,
        0.0,
    )
)


def distilled_stage_1_sigmas() -> mx.array:
    """Stage-1 sigma schedule as an ``mx.array`` (float32)."""
    return mx.array(DISTILLED_STAGE_1_SIGMAS, dtype=mx.float32)


def distilled_stage_2_sigmas() -> mx.array:
    """Stage-2 sigma schedule as an ``mx.array`` (float32)."""
    return mx.array(DISTILLED_STAGE_2_SIGMAS, dtype=mx.float32)
