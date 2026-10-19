"""LTX-2 generation-to-VideoToolbox output glue."""

from pathlib import Path

from kinomlx.audio.trim_mode import parse_trim_mode
from kinomlx.config_rules import validate_capabilities
from kinomlx.media.signals import (
    BT709_SDR_420_DELIVERY,
    BT709_SDR_422_DELIVERY,
    BT2020_HLG_DELIVERY,
    ColorPrimaries,
    ColorTransfer,
    EncodedVideoDeliverySpec,
    ExrDeliverySpec,
    OutputColorPlan,
    SampleDType,
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

from .output_config import OutputConfig, sidecar_selection


def _resolved_sdr_delivery(
    config: OutputConfig,
    *,
    fps: float,
) -> EncodedVideoDeliverySpec:
    temporal = config.target_fps is not None and abs(config.target_fps - fps) > 1e-6
    if config.vsr_spatial_mode in {"balanced", "image"} or temporal:
        return BT709_SDR_422_DELIVERY
    return BT709_SDR_420_DELIVERY


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
        validate_capabilities(
            config, {"hdr": hdr_authoring is not None, "fps": fps}, location="[output]"
        )
        onset_mode, onset_ms = parse_trim_mode(config.audio_onset_trim)
        sidecars = sidecar_selection(config)
        sink: GenerationSink
        if hdr_authoring is None:
            delivery = _resolved_sdr_delivery(config, fps=fps)
            plan = OutputColorPlan(source=generation.signal, deliveries=(delivery,))
            sink = VideoToolboxGenerationSink(
                path=config.path,
                fps=fps,
                reporter=reporter if reporter is not None else NullReporter(),
                encoder=encoder,
                save_audio_sidecar=sidecars.audio_sidecar,
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
                    default_vae_frame_directory(config.path) if sidecars.vae_frames else None
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
                    default_hdr_heic_directory(config.path) if sidecars.hdr_heic_frames else None
                ),
                save_audio_sidecar=sidecars.audio_sidecar,
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
            sample_type=SampleDType.FLOAT16,
            color_space_tag="Linear sRGB",
        )
    if authoring == "ACESCG":
        return ExrDeliverySpec(
            primaries=ColorPrimaries.ACESCG,
            transfer=ColorTransfer.LINEAR,
            sample_type=SampleDType.FLOAT16,
            color_space_tag="ACEScg",
        )
    if authoring == "ACESCCT":
        return ExrDeliverySpec(
            primaries=ColorPrimaries.ACESCG,
            transfer=ColorTransfer.ACESCCT,
            sample_type=SampleDType.FLOAT16,
            color_space_tag="ACEScct",
        )
    raise OutputError(f"unsupported HDR EXR authoring mode {authoring!r}")


__all__ = ["write_generation"]
