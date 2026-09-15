"""Pure coupled LTX request policies owned by participating fields."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from kinomlx.config_rules import RecordRule, RuleInputs

from .keyframe_positions import generated_keyframe_indices


def custom_temporal_tile_frames(configured: int | None) -> int:
    """Resolve the custom decoder plan's default temporal extent once."""
    return 256 if configured is None else configured


def _vae_tiling(values: Mapping[str, object]) -> None:
    temporal = cast(int | None, values["temporal_tile_frames"])
    spatial = cast(int | None, values["spatial_tile_pixels"])
    if values["mode"] == "custom":
        temporal = custom_temporal_tile_frames(temporal)
    if temporal not in (None, 0) and cast(int, values["temporal_overlap_frames"]) >= temporal:
        raise ValueError("VAE temporal overlap must be smaller than its tile size")
    if spatial not in (None, 0) and cast(int, values["spatial_overlap_pixels"]) >= spatial:
        raise ValueError("VAE spatial overlap must be smaller than its tile size")
    if values["mode"] == "custom" and temporal == 0 and spatial in (None, 0):
        raise ValueError("custom VAE tiling needs a temporal or spatial tile size")


VAE_TILING = RecordRule(
    "vae_tiling",
    frozenset(
        {
            "mode",
            "temporal_tile_frames",
            "temporal_overlap_frames",
            "spatial_tile_pixels",
            "spatial_overlap_pixels",
        }
    ),
    _vae_tiling,
)


class _FrameIndexed(Protocol):
    @property
    def frame_index(self) -> int: ...


def _image_frame_bounds(values: Mapping[str, object]) -> None:
    image = values["image"]
    frames = cast(int | None, values["frames"])
    if image is not None and frames is not None:
        frame_index = cast(_FrameIndexed, image).frame_index
        if frame_index >= frames:
            raise ValueError(f"image frame_index {frame_index} is outside {frames} frames")


IMAGE_FRAME_BOUNDS = RecordRule(
    "image_frame_bounds", frozenset({"image", "frames"}), _image_frame_bounds
)


class _PathSource(Protocol):
    @property
    def path(self) -> Path: ...


def _hdr_sources(values: Mapping[str, object]) -> None:
    image = cast(_PathSource | None, values["image"])
    if image is not None and image.path.suffix.lower() == ".exr" and values["hdr"] is None:
        raise ValueError("EXR conditioning requires an explicit --hdr signal interpretation")
    if values["hdr_reference"] is not None and values["hdr"] is None:
        raise ValueError("hdr_reference requires an explicit HDR output mode")


def _keyframe_positions(values: Mapping[str, object]) -> None:
    frames = cast(int | None, values["frames"])
    count = cast(int, values["generated_keyframes"])
    if frames is not None and count:
        generated_keyframe_indices(frames, count)


HDR_SOURCES = RecordRule("hdr_sources", frozenset({"image", "hdr", "hdr_reference"}), _hdr_sources)
KEYFRAME_POSITIONS = RecordRule(
    "keyframe_positions", frozenset({"frames", "generated_keyframes"}), _keyframe_positions
)


def _capabilities(values: RuleInputs) -> None:
    observed = values.context
    generation = observed["model_generation"]
    if values["vae_decode_dtype"] == "float16" and observed["video_vae_kind"] != "diffusion-na":
        raise ValueError("experimental float16 VAE decode requires --video-vae diffusion")
    if "distilled" not in cast(tuple[str, ...], observed["recipe_families"]):
        raise ValueError("checkpoint does not support the distilled recipe")
    if values["frames"] is None and not observed["duration_available"]:
        raise ValueError(f"LTX-{generation} resources do not provide an auto-duration head")
    if values["generated_keyframes"] and not observed["generated_keyframes_available"]:
        raise ValueError(f"LTX-{generation} does not provide generated-keyframe positions")
    if values["generate_audio"] and not observed["generates_audio"]:
        raise ValueError("checkpoint does not support joint audio generation")
    image = cast(_PathSource | None, values["image"])
    if image is not None and image.path.suffix.lower() == ".exr" and generation != "2.5":
        raise ValueError("EXR conditioning is currently supported only by LTX-2.5 native HDR")
    if values["hdr_reference"] is not None and generation != "2.3":
        raise ValueError("hdr_reference is supported only by the LTX-2.3 HDR IC-LoRA recipe")
    for family in cast(tuple[str, ...], observed["requested_condition_families"]):
        if family not in cast(tuple[str, ...], observed["condition_families"]):
            raise ValueError(f"checkpoint does not support {family} conditioning")


DISTILLED_CAPABILITIES = RecordRule(
    "distilled_capabilities",
    frozenset(
        {
            "vae_decode_dtype",
            "frames",
            "generated_keyframes",
            "generate_audio",
            "image",
            "hdr_reference",
        }
    ),
    _capabilities,
    phase="capability",
    context_reads=frozenset(
        {
            "model_generation",
            "video_vae_kind",
            "recipe_families",
            "duration_available",
            "generated_keyframes_available",
            "generates_audio",
            "condition_families",
            "requested_condition_families",
        }
    ),
)
