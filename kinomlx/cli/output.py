"""Machine output and VideoToolbox writing for resolved generations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from kinomlx.audio.trim_mode import parse_trim_mode
from kinomlx.media.signals import (
    BT709_SDR_420_DELIVERY,
    BT709_SDR_422_DELIVERY,
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    EncodedVideoDeliverySpec,
    ExrDeliverySpec,
    ExrSampleType,
    OutputColorPlan,
)
from kinomlx.output import (
    Generation,
    GenerationSink,
    HDRGenerationSink,
    OutputError,
    VideoToolboxEncoder,
    VideoToolboxGenerationSink,
    default_hdr_heic_directory,
    default_vae_frame_directory,
)
from kinomlx.reporting import NullReporter, Reporter

if TYPE_CHECKING:
    from .config import OutputConfig


def _resolved_sdr_delivery(
    config: OutputConfig,
    *,
    fps: float,
) -> EncodedVideoDeliverySpec:
    temporal = config.target_fps is not None and abs(config.target_fps - fps) > 1e-6
    if config.vsr_spatial_mode in {"balanced", "image"} or temporal:
        return BT709_SDR_422_DELIVERY
    return BT709_SDR_420_DELIVERY


def emit_json(
    payload: dict[str, object],
    *,
    stream: TextIO | None = None,
    logger_name: str = "kinomlx.cli.json_output",
) -> None:
    """Emit one compact JSON record on an isolated stdout logger."""
    from kinomlx.ui import configure_machine_output

    logger = configure_machine_output(logger_name, stream=stream)
    logger.info("%s", json.dumps(payload, sort_keys=True, default=str))


def write_generation(
    generation: Generation,
    config: OutputConfig,
    *,
    fps: float,
    hdr_authoring: str | None = None,
    reporter: Reporter | None = None,
    encoder: VideoToolboxEncoder | None = None,
    native_verbose: bool = False,
) -> Path:
    """Resolve CLI output settings and delegate to the typed public sink."""
    if config.path is None:
        generation.close()
        raise OutputError("output path is required")
    try:
        onset_mode, onset_ms = parse_trim_mode(config.audio_onset_trim)
        sink: GenerationSink
        if hdr_authoring is None:
            delivery = _resolved_sdr_delivery(config, fps=fps)
            plan = OutputColorPlan(source=generation.signal, deliveries=(delivery,))
            sink = VideoToolboxGenerationSink(
                path=config.path,
                fps=fps,
                reporter=reporter if reporter is not None else NullReporter(),
                encoder=encoder,
                save_audio_sidecar=config.save_audio_sidecar,
                vsr_spatial_mode=(
                    None if config.vsr_spatial_mode == "off" else config.vsr_spatial_mode
                ),
                target_fps=config.target_fps,
                vsr_temporal_mode=config.vsr_temporal_mode,
                cut_detect_mode=config.cut_detect_mode,
                cut_detect_threshold=config.cut_detect_threshold,
                vsr_save_original=config.vsr_save_original,
                encode_quality=config.encode_quality,
                audio_codec=config.audio_codec,
                audio_onset_trim_mode=onset_mode,
                audio_onset_trim_ms=onset_ms,
                native_verbose=native_verbose,
                vae_frame_directory=(
                    default_vae_frame_directory(config.path) if config.save_vae_frames else None
                ),
            )
        else:
            exr_delivery = _resolved_hdr_exr_delivery(hdr_authoring)
            plan = OutputColorPlan(
                source=generation.signal,
                deliveries=(exr_delivery, BT2020_HLG_DELIVERY),
            )
            sink = HDRGenerationSink(
                path=config.path,
                fps=fps,
                reporter=reporter if reporter is not None else NullReporter(),
                heic_directory=(
                    default_hdr_heic_directory(config.path) if config.save_hdr_heic_frames else None
                ),
                save_audio_sidecar=config.save_audio_sidecar,
                encode_quality=config.encode_quality,
                audio_codec=config.audio_codec,
                audio_onset_trim_mode=onset_mode,
                audio_onset_trim_ms=onset_ms,
                native_verbose=native_verbose,
            )
    except BaseException:
        generation.close()
        raise
    return sink.write(generation, plan).video


def _resolved_hdr_exr_delivery(authoring: str) -> ExrDeliverySpec:
    """Map public HDR authoring intent to one exact EXR signal contract."""
    if authoring == "SRGB_LINEAR":
        return ExrDeliverySpec(
            primaries=ColorPrimaries.REC709,
            transfer=ColorTransfer.LINEAR,
            sample_type=ExrSampleType.FLOAT16,
            color_space_tag="Linear sRGB",
        )
    if authoring == "ACESCG":
        return ExrDeliverySpec(
            primaries=ColorPrimaries.ACESCG,
            transfer=ColorTransfer.LINEAR,
            sample_type=ExrSampleType.FLOAT16,
            color_space_tag="ACEScg",
        )
    if authoring == "ACESCCT":
        return ExrDeliverySpec(
            primaries=ColorPrimaries.ACESCG,
            transfer=ColorTransfer.ACESCCT,
            sample_type=ExrSampleType.FLOAT16,
            color_space_tag="ACEScct",
        )
    raise OutputError(f"unsupported HDR EXR authoring mode {authoring!r}")


__all__ = ["emit_json", "write_generation"]
