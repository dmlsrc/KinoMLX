"""LTX-2.3 distilled sigma schedules.

The distilled two-stage pipeline uses *literal* sigma schedules - no
dynamic scheduler.  The values match the official LTX-2 ComfyUI
distilled schedule (9 values driving 8 denoising steps in stage 1;
4 values driving 3 steps in stage 2).

If a non-distilled path is ever added that needs dynamic
scheduling, the generic scheduler module lands in
``kinomlx/samplers/`` at that time; it is deferred until then.
"""

import struct
from collections.abc import Sequence

from kinomlx.samplers.sigmas import resolved_sigmas


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
DISTILLED_STAGE_2_SIGMAS: tuple[float, ...] = tuple(
    _float32(value)
    for value in (
        0.909375,
        0.725,
        0.421875,
        0.0,
    )
)


def stage_sigmas(
    stage_1: Sequence[float] = (), stage_2: Sequence[float] = ()
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Resolve both stage schedules, preferring supplied values over the pins.

    LTX-2 denoises in two stages with independent schedules, so each is overridden
    on its own. Stage 2 legitimately starts below one: it continues from stage 1's
    tail after the spatial upscaler.
    """
    return (
        resolved_sigmas(
            stage_1, lambda: DISTILLED_STAGE_1_SIGMAS, label="LTX-2 stage 1 schedule"
        ).values,
        resolved_sigmas(
            stage_2, lambda: DISTILLED_STAGE_2_SIGMAS, label="LTX-2 stage 2 schedule"
        ).values,
    )
