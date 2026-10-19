"""Context sharpening for the LTX-2.5 diffusion video decoder, an option outside the reference.

The diffusion decoder's final stage paints pixels from a context: one feature vector for each
``patch_size`` by ``patch_size`` block of output pixels, which every final-stage block adds to its
residual stream through a linear projection. An unsharp mask on that context,
``context + amount * (context - blur(context))``, makes the stage draw sharper edges. To first
order it equals guiding the stage with a second pass on a blurred context and adding ``amount``
times the difference, but it costs one pass over the context instead of a second pass of the
stage. ``blur`` is a per-frame spatial Gaussian whose standard deviation ``radius`` is given in
output pixels, so a radius of 2 blurs half a context cell when a cell is 4 pixels wide.
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from kinomlx.models.ltx2.kernels.context_sharpening import spatial_unsharp_mask

DEFAULT_RADIUS = 2.0
MAX_AMOUNT = 10.0
MAX_RADIUS = 8.0


@dataclass(frozen=True, kw_only=True)
class ContextSharpening:
    """Sharpen the final-stage context by ``amount`` with a Gaussian of ``radius`` output pixels."""

    amount: float
    radius: float = DEFAULT_RADIUS

    def __post_init__(self) -> None:
        for name in ("amount", "radius"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"context sharpening {name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if not 0 < self.amount <= MAX_AMOUNT:
            raise ValueError(
                f"context sharpening amount must be greater than 0 and at most {MAX_AMOUNT:g}"
            )
        if not 0 < self.radius <= MAX_RADIUS:
            raise ValueError(
                f"context sharpening radius must be greater than 0 and at most {MAX_RADIUS:g} pixels"
            )

    def apply(self, context: mx.array, *, patch_size: int) -> mx.array:
        """Sharpen a ``(batch, frames, height, width, channels)`` context of ``patch_size`` cells."""
        return spatial_unsharp_mask(context, sigma=self.radius / patch_size, amount=self.amount)

    def to_dict(self) -> dict[str, float]:
        return {"amount": self.amount, "radius_pixels": self.radius}


__all__ = ["DEFAULT_RADIUS", "MAX_AMOUNT", "MAX_RADIUS", "ContextSharpening"]
