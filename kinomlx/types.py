"""Model-agnostic shape and policy primitives.

Per-model constants (sigma schedules, VAE compression ratios, native
frame rates, audio encoder geometry) live under
``kinomlx/models/<name>/`` - keeping this file generic lets future
models reuse the shape primitives without inheriting another model's
defaults.
"""

from enum import StrEnum
from typing import NamedTuple


class NoiseBackend(StrEnum):
    """Which normal-noise stream a request reproduces."""

    MLX = "mlx"
    TORCH_MPS = "torch-mps"


NOISE_BACKEND_CHOICES = tuple(item.value for item in NoiseBackend)
DEFAULT_NOISE_BACKEND: NoiseBackend = NoiseBackend.MLX


class VAETilingMode(StrEnum):
    """How a VAE decode is split to fit its memory budget."""

    AUTO = "auto"
    SINGLE = "single"
    CUSTOM = "custom"


VAE_TILING_MODE_CHOICES = tuple(item.value for item in VAETilingMode)


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
