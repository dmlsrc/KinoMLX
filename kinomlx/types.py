"""Model-agnostic shape and state primitives.

Per-model constants (sigma schedules, VAE compression ratios, native
frame rates, audio encoder geometry) live under
``kinomlx/models/<name>/`` - keeping this file generic lets future
models reuse the shape primitives without inheriting another model's
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    import mlx.core as mx


NoiseBackend = Literal["mlx", "torch-mps"]
NOISE_BACKEND_CHOICES = ("mlx", "torch-mps")
DEFAULT_NOISE_BACKEND: NoiseBackend = "mlx"


class VideoPixelShape(NamedTuple):
    """Dimensions of a pixel-space video tensor.

    Carries only spatial and temporal dims.  Pixel value semantics
    (channel order, dtype, normalization range) and metadata such as
    fps are per-model and per-call - keep them out of "shape."
    """

    batch: int
    frames: int
    height: int
    width: int


class VideoLatentShape(NamedTuple):
    """Dimensions of a latent-space video tensor.

    The mapping between pixel and latent shapes depends on the
    model's VAE compression ratios; the conversion lives per-model
    (see e.g. :mod:`kinomlx.models.ltx2.types`).
    """

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_tuple(self) -> tuple[int, int, int, int, int]:
        """Plain-tuple view (a NamedTuple already is one; callers want plain)."""
        return (self.batch, self.channels, self.frames, self.height, self.width)

    @classmethod
    def from_tuple(cls, shape: tuple[int, int, int, int, int]) -> VideoLatentShape:
        return cls(*shape)

    def with_channels(self, channels: int) -> VideoLatentShape:
        """Same shape with the channel dim swapped.

        ``with_channels(1)`` is the canonical way to get a mask shape.
        """
        return self._replace(channels=channels)


class SpatioTemporalScaleFactors(NamedTuple):
    """VAE downscaling ratios between pixel and latent space.

    No defaults - every model's VAE has its own ratios; concrete
    instances live per-model.
    """

    time: int
    height: int
    width: int


@dataclass(frozen=True)
class LatentState:
    """State of a latent tensor during diffusion denoising.

    Fields:
        latent: The currently-noisy latent being denoised.
        denoise_mask: Per-token denoising strength (1 = full, 0 = none).
        positions: Positional indices used for positional embeddings.
        clean_latent: Initial latent before denoising; may include
            conditioning latents (image, keyframe).
        uniform_mask: Python-side flag - True iff every entry of
            ``denoise_mask`` is 1.  Not derived from mask values in
            the hot path (that would force an ``mx.eval``).  Models
            may key a fastpath on this flag to skip per-token
            timestep work when the mask is uniform.  LTX-2 A/V
            cross-attention also relies on the distinction: scale/shift
            follows own per-token timesteps, not the other modality's
            scalar sigma.

    **uniform_mask footgun.**  When constructing a new state via
    ``dataclasses.replace(state, denoise_mask=new_mask)``, you MUST
    also pass ``uniform_mask=False`` if the new mask is mixed.  The
    flag does not auto-update; silent staleness would fire the
    fastpath against a non-uniform mask and break downstream
    conditioning::

        # correct
        state = dataclasses.replace(state, denoise_mask=m, uniform_mask=False)

        # WRONG - fastpath stays active
        state = dataclasses.replace(state, denoise_mask=m)
    """

    latent: mx.array
    denoise_mask: mx.array
    positions: mx.array
    clean_latent: mx.array
    uniform_mask: bool = True
