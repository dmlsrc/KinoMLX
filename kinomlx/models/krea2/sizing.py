"""Output sizing from the first source image's aspect ratio for Krea 2 editing."""

from dataclasses import dataclass

from kinomlx.config_records import derive_config
from kinomlx.io.geometry import derive_output_size, pinned_axis
from kinomlx.io.image import image_size

from .types import Krea2Request

# VAE spatial compression 8 times transformer patch 2. resolve_geometry derives the
# same alignment from the checkpoint configs once resources are prepared.
KREA2_OUTPUT_ALIGNMENT = 16
_AXES = ("width", "height")


@dataclass(frozen=True)
class SourceSizing:
    """One output size derived from a source image and the pinned dimension."""

    source_width: int
    source_height: int
    mode: str
    axis: str
    pixels: int
    width: int
    height: int

    def describe(self) -> str:
        pinned = self.mode if self.mode in _AXES else f"{self.mode} edge ({self.axis})"
        return (
            f"{self.width}x{self.height} matched to source "
            f"{self.source_width}x{self.source_height}, {pinned} {self.pixels}"
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "mode": self.mode,
            "axis": self.axis,
            "pixels": self.pixels,
            "width": self.width,
            "height": self.height,
        }


def pinned_pixels(request: Krea2Request, mode: str, cli_sizes: frozenset[str]) -> int:
    """Take the one size given on the command line, else the configured size.

    For width and height the number must come from that same option; for long and
    short either --width or --height supplies the number for the pinned edge.
    """
    unknown = cli_sizes - set(_AXES)
    if unknown:
        raise ValueError(f"unknown size options: {', '.join(sorted(unknown))}")
    if len(cli_sizes) == 2:
        raise ValueError(
            "match_source_aspect pins one dimension; pass --width or --height, not both"
        )
    if mode in _AXES:
        other = "height" if mode == "width" else "width"
        if other in cli_sizes:
            raise ValueError(f"match_source_aspect {mode} keeps --{mode}; drop --{other}")
        return int(getattr(request, mode))
    if cli_sizes:
        return int(getattr(request, next(iter(cli_sizes))))
    return (
        max(request.width, request.height) if mode == "long" else min(request.width, request.height)
    )


def resolve_source_sizing(
    request: Krea2Request,
    *,
    alignment: int = KREA2_OUTPUT_ALIGNMENT,
    source_size: tuple[int, int] | None = None,
    cli_sizes: frozenset[str] = frozenset(),
) -> tuple[Krea2Request, SourceSizing | None]:
    """Return the request with width/height matched to the first source image.

    Without ``match_source_aspect`` the request is returned unchanged. The result is
    idempotent: the pinned dimension is already concrete, so resolving again derives
    the same other dimension.
    """
    mode = request.match_source_aspect
    if mode is None:
        return request, None
    if not request.source_images:
        raise ValueError("match_source_aspect requires a source image")
    if source_size is None:
        source_size = image_size(request.source_images[0])
    axis = pinned_axis(mode, source_size)
    pixels = pinned_pixels(request, mode, cli_sizes)
    width, height = derive_output_size(source_size, axis=axis, pixels=pixels, alignment=alignment)
    sizing = SourceSizing(source_size[0], source_size[1], mode, axis, pixels, width, height)
    return derive_config(request, width=width, height=height), sizing


__all__ = [
    "KREA2_OUTPUT_ALIGNMENT",
    "SourceSizing",
    "derive_output_size",
    "pinned_axis",
    "pinned_pixels",
    "resolve_source_sizing",
]
