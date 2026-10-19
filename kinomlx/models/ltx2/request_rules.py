"""Pure coupled LTX request policies owned by participating fields."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from kinomlx.config_rules import RecordRule, RuleInputs

from .frame_position import (
    FIRST,
    parse_frame_position,
    repeated_position_message,
    resolve_frame_positions,
)
from .keyframe_positions import generated_keyframe_indices
from .sigmas import stage_sigmas
from .sparse_attention import parse_step_schedule
from .video_vae.kinds import VideoVAEKind


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
    def frame_index(self) -> str: ...


def _image_frame_bounds(values: Mapping[str, object]) -> None:
    images = cast(tuple[_FrameIndexed, ...], values["images"])
    frames = cast(int | None, values["frames"])
    positions = tuple(image.frame_index for image in images)
    if frames is not None:
        resolve_frame_positions(positions, frames)
        return
    # Without a frame count, ``last`` has no absolute index yet. The duplicates
    # that need no count are still refusable: two spellings of frame 0, or the
    # same name twice.
    conditioned: dict[object, str] = {}
    for position in positions:
        parsed = parse_frame_position(position)
        frame: object = 0 if parsed == FIRST else parsed
        earlier = conditioned.get(frame)
        if earlier is not None:
            raise ValueError(repeated_position_message(frame, earlier, position))
        conditioned[frame] = position


IMAGE_FRAME_BOUNDS = RecordRule(
    "image_frame_bounds", frozenset({"images", "frames"}), _image_frame_bounds
)


class _PathSource(Protocol):
    @property
    def path(self) -> Path: ...


def _has_exr(values: Mapping[str, object]) -> bool:
    images = cast(tuple[_PathSource, ...], values["images"])
    return any(image.path.suffix.lower() == ".exr" for image in images)


def _hdr_sources(values: Mapping[str, object]) -> None:
    if _has_exr(values) and values["hdr"] is None:
        raise ValueError("EXR conditioning requires an explicit --hdr signal interpretation")
    if values["hdr_reference"] is not None and values["hdr"] is None:
        raise ValueError("hdr_reference requires an explicit HDR output mode")


def _keyframe_positions(values: Mapping[str, object]) -> None:
    frames = cast(int | None, values["frames"])
    count = cast(int, values["generated_keyframes"])
    if frames is not None and count:
        generated_keyframe_indices(frames, count)


HDR_SOURCES = RecordRule("hdr_sources", frozenset({"images", "hdr", "hdr_reference"}), _hdr_sources)
KEYFRAME_POSITIONS = RecordRule(
    "keyframe_positions", frozenset({"frames", "generated_keyframes"}), _keyframe_positions
)


def _sparse_attention_steps(values: Mapping[str, object]) -> None:
    schedules = stage_sigmas(
        cast(tuple[float, ...], values["stage_1_sigmas"]),
        cast(tuple[float, ...], values["stage_2_sigmas"]),
    )
    for stage, sigmas in enumerate(schedules, start=1):
        name = f"stage_{stage}_sparse_attention"
        parse_step_schedule(cast(str, values[name])).resolve(
            len(sigmas) - 1, label=name, stage=stage
        )


SPARSE_ATTENTION_STEPS = RecordRule(
    "sparse_attention_steps",
    frozenset(
        {
            "stage_1_sparse_attention",
            "stage_2_sparse_attention",
            "stage_1_sigmas",
            "stage_2_sigmas",
        }
    ),
    _sparse_attention_steps,
)


def _capabilities(values: RuleInputs) -> None:
    observed = values.context
    generation = observed["model_generation"]
    if "distilled" not in cast(tuple[str, ...], observed["recipe_families"]):
        raise ValueError("checkpoint does not support the distilled recipe")
    if values["frames"] is None and not observed["duration_available"]:
        raise ValueError(f"LTX-{generation} resources do not provide an auto-duration head")
    if values["generated_keyframes"] and not observed["generated_keyframes_available"]:
        raise ValueError(f"LTX-{generation} does not provide generated-keyframe positions")
    if values["generate_audio"] and not observed["generates_audio"]:
        raise ValueError("checkpoint does not support joint audio generation")
    if values["vae_sharpen"] and observed["video_vae_kind"] != VideoVAEKind.DIFFUSION:
        raise ValueError("vae_sharpen applies only to the diffusion video VAE")
    if _has_exr(values) and generation != "2.5":
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
            "vae_sharpen",
            "frames",
            "generated_keyframes",
            "generate_audio",
            "images",
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
