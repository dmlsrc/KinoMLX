"""The video VAE decoder vocabulary shared by requests and checkpoint facts."""

from enum import StrEnum


class VideoVAEKind(StrEnum):
    """Which decoder a video VAE provides.

    ``CONV`` is the three-dimensional convolutional decoder. ``DIFFUSION`` is the
    LTX-2.5 diffusion decoder, which attends over bounded neighborhoods rather
    than the whole grid. Nothing branches on that difference beyond selecting the
    decoder and its tiling planner, so the architecture stays documented here
    rather than spelled into the value.
    """

    CONV = "conv"
    DIFFUSION = "diffusion"


VIDEO_VAE_KIND_CHOICES = tuple(item.value for item in VideoVAEKind)

__all__ = ["VIDEO_VAE_KIND_CHOICES", "VideoVAEKind"]
