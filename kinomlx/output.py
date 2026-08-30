"""Typed generation-to-terminal boundary with a native VideoToolbox sink."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from kinomlx.errors import KinoMLXError
from kinomlx.media.frames import CloseableVideoFrameStream
from kinomlx.media.signals import (
    ColorPrimaries,
    ColorTransfer,
    EncodedVideoDeliverySpec,
    ExrDeliverySpec,
    OutputColorPlan,
    UnsupportedSignalError,
    VideoLayout,
    VideoSignalSpec,
    VideoValueDomain,
    validate_hlg_delivery,
    validate_sdr_output_plan,
)
from kinomlx.reporting import NullReporter, Reporter

if TYPE_CHECKING:
    import mlx.core as mx

    from kinomlx.videotoolbox.frame_dump import PNGFrameDumpStream
    from kinomlx.videotoolbox.writer import AVWriter


class OutputError(KinoMLXError, RuntimeError):
    """A typed failure while materializing a generated output."""


class Generation(Protocol):
    """Typed products consumed by a terminal generation sink."""

    @property
    def frames(self) -> CloseableVideoFrameStream: ...

    @property
    def audio_waveform(self) -> mx.array | None: ...

    @property
    def audio_sample_rate(self) -> int | None: ...

    @property
    def signal(self) -> VideoSignalSpec: ...

    @property
    def frame_count(self) -> int: ...

    def close(self) -> None: ...


def default_vae_frame_directory(path: Path | str) -> Path:
    """Return the output-adjacent directory for an opt-in VAE frame dump."""
    video = Path(path)
    return video.parent / f"{video.stem}_vae_frames"


def default_hdr_exr_directory(path: Path | str) -> Path:
    """Return the output-adjacent directory for an HDR EXR master."""
    video = Path(path)
    return video.parent / f"{video.stem}_exr"


def default_hdr_heic_directory(path: Path | str) -> Path:
    """Return the output-adjacent directory for opt-in HDR HEIC previews."""
    video = Path(path)
    return video.parent / f"{video.stem}_heic"


@dataclass(frozen=True)
class ArtifactSet:
    """Artifacts materialized by one terminal sink invocation."""

    video: Path
    vae_frames: Path | None = None
    exr_frames: Path | None = None
    heic_frames: Path | None = None


class GenerationSink(Protocol):
    """Host-injected terminal consuming one explicit color plan."""

    def write(self, generation: Generation, plan: OutputColorPlan) -> ArtifactSet: ...


class VideoToolboxEncoder(Protocol):
    """Callable surface used by the native terminal sink."""

    def __call__(
        self,
        frames: CloseableVideoFrameStream,
        output_path: Path,
        *,
        fps: float,
        source_signal: VideoSignalSpec,
        delivery: EncodedVideoDeliverySpec,
        n_source_frames: int,
        reporter: Reporter,
        audio_waveform: mx.array | None,
        audio_sample_rate: int | None,
        save_audio_sidecar: bool,
        vsr_spatial_mode: str | None,
        target_fps: float | None,
        vsr_temporal_mode: str,
        cut_detect_mode: str,
        cut_detect_threshold: float | None,
        vsr_save_original: bool,
        encode_quality: float,
        audio_codec: str,
        audio_onset_trim_mode: str,
        audio_onset_trim_ms: float | None,
        native_verbose: bool,
    ) -> Path: ...


@dataclass(frozen=True)
class VideoToolboxGenerationSink:
    """Write one SDR generation through the installed native media path."""

    path: Path
    fps: float
    reporter: Reporter | None = None
    encoder: VideoToolboxEncoder | None = None
    save_audio_sidecar: bool = False
    vsr_spatial_mode: str | None = None
    target_fps: float | None = None
    vsr_temporal_mode: str = "normal"
    cut_detect_mode: str = "simple"
    cut_detect_threshold: float | None = None
    vsr_save_original: bool = False
    encode_quality: float = 0.65
    audio_codec: str = "alac"
    audio_onset_trim_mode: str = "auto"
    audio_onset_trim_ms: float | None = None
    native_verbose: bool = False
    vae_frame_directory: Path | None = None

    def write(self, generation: Generation, plan: OutputColorPlan) -> ArtifactSet:
        """Validate the source/delivery pair before opening the native encoder."""
        dump_stream: PNGFrameDumpStream | None = None
        try:
            if plan.source != generation.signal:
                raise UnsupportedSignalError(
                    "output color plan source does not match the generation signal"
                )
            if len(plan.deliveries) != 1 or not isinstance(
                plan.deliveries[0],
                EncodedVideoDeliverySpec,
            ):
                raise UnsupportedSignalError(
                    "VideoToolbox generation sink requires one encoded video delivery"
                )
            validate_sdr_output_plan(plan)
            delivery = plan.deliveries[0]
            encoder = self.encoder
            if encoder is None:
                from kinomlx.videotoolbox import encode_video_videotoolbox

                encoder = encode_video_videotoolbox
            reporter = self.reporter if self.reporter is not None else NullReporter()
            frames: CloseableVideoFrameStream = generation.frames
            if self.vae_frame_directory is not None:
                from kinomlx.videotoolbox.frame_dump import (
                    FrameDumpError,
                    PNGFrameDumpStream,
                )

                try:
                    dump_stream = PNGFrameDumpStream(frames, self.vae_frame_directory)
                except FrameDumpError as exc:
                    raise OutputError(str(exc)) from exc
                frames = dump_stream
            video = encoder(
                frames,
                self.path,
                fps=self.fps,
                source_signal=plan.source,
                delivery=delivery,
                n_source_frames=generation.frame_count,
                reporter=reporter,
                audio_waveform=generation.audio_waveform,
                audio_sample_rate=generation.audio_sample_rate,
                save_audio_sidecar=self.save_audio_sidecar,
                vsr_spatial_mode=self.vsr_spatial_mode,
                target_fps=self.target_fps,
                vsr_temporal_mode=self.vsr_temporal_mode,
                cut_detect_mode=self.cut_detect_mode,
                cut_detect_threshold=self.cut_detect_threshold,
                vsr_save_original=self.vsr_save_original,
                encode_quality=self.encode_quality,
                audio_codec=self.audio_codec,
                audio_onset_trim_mode=self.audio_onset_trim_mode,
                audio_onset_trim_ms=self.audio_onset_trim_ms,
                native_verbose=self.native_verbose,
            )
            if dump_stream is not None:
                dump_stream.commit()
            return ArtifactSet(
                video=video,
                vae_frames=self.vae_frame_directory,
            )
        except OutputError, UnsupportedSignalError:
            raise
        except Exception as exc:
            if dump_stream is not None:
                from kinomlx.videotoolbox.frame_dump import FrameDumpError

                if isinstance(exc, FrameDumpError):
                    raise OutputError(str(exc)) from exc
            from kinomlx.videotoolbox.errors import is_video_toolbox_operation_error

            if not is_video_toolbox_operation_error(exc):
                raise
            raise OutputError(f"cannot write {self.path}: {exc}") from exc
        finally:
            if dump_stream is not None:
                dump_stream.close()
            generation.close()


@dataclass(frozen=True)
class HDRGenerationSink:
    """Transactionally write one HDR stream to EXR, HLG, and optional HEIC."""

    path: Path
    fps: float
    reporter: Reporter | None = None
    exr_directory: Path | None = None
    heic_directory: Path | None = None
    save_audio_sidecar: bool = False
    encode_quality: float = 0.65
    audio_codec: str = "alac"
    audio_onset_trim_mode: str = "auto"
    audio_onset_trim_ms: float | None = None
    native_verbose: bool = False

    def _deliveries(
        self,
        plan: OutputColorPlan,
    ) -> tuple[ExrDeliverySpec, EncodedVideoDeliverySpec]:
        exr = tuple(item for item in plan.deliveries if isinstance(item, ExrDeliverySpec))
        encoded = tuple(
            item for item in plan.deliveries if isinstance(item, EncodedVideoDeliverySpec)
        )
        if len(exr) != 1 or len(encoded) != 1 or len(plan.deliveries) != 2:
            raise UnsupportedSignalError(
                "HDR generation sink requires one EXR and one encoded HLG delivery"
            )
        from kinomlx.videotoolbox.exr import validate_exr_delivery

        validate_exr_delivery(exr[0])
        validate_hlg_delivery(encoded[0])
        return exr[0], encoded[0]

    @staticmethod
    def _validate_source(source: VideoSignalSpec) -> None:
        domains = {
            ColorTransfer.ACESCCT: VideoValueDomain.ACESCCT_WORKING_CODES,
            ColorTransfer.LOGC3: VideoValueDomain.LOGC3_WORKING_CODES,
            ColorTransfer.LINEAR: VideoValueDomain.SCENE_LINEAR,
        }
        if (
            source.layout is not VideoLayout.HWC_RGB
            or source.dtype != "float32"
            or source.value_domain is not domains.get(source.transfer)
            or source.primaries not in {ColorPrimaries.REC709, ColorPrimaries.ACESCG}
        ):
            raise UnsupportedSignalError(
                "HDR terminal requires float32 ACEScct, LogC3, or scene-linear RGB frames"
            )

    @staticmethod
    def _exr_frame(
        working: mx.array,
        linear: mx.array,
        *,
        source: VideoSignalSpec,
        delivery: ExrDeliverySpec,
    ) -> mx.array:
        from kinomlx.media.hdr import (
            convert_scene_linear_primaries,
            scene_linear_to_acescct,
            scene_linear_to_logc3,
        )

        if delivery.transfer is source.transfer and delivery.primaries is source.primaries:
            return working
        converted = convert_scene_linear_primaries(
            linear,
            source=source.primaries,
            target=delivery.primaries,
        )
        if delivery.transfer is ColorTransfer.LINEAR:
            return converted
        if delivery.transfer is ColorTransfer.ACESCCT:
            return scene_linear_to_acescct(converted)
        if delivery.transfer is ColorTransfer.LOGC3:
            return scene_linear_to_logc3(converted)
        raise UnsupportedSignalError(f"cannot author {delivery.transfer.value} EXR frames")

    def write(self, generation: Generation, plan: OutputColorPlan) -> ArtifactSet:
        """Consume one frame at a time and publish every HDR output together."""
        transaction: Path | None = None
        writer: AVWriter | None = None
        final_exr = self.exr_directory or default_hdr_exr_directory(self.path)
        final_heic = self.heic_directory
        final_audio = (
            self.path.with_suffix(".wav")
            if self.save_audio_sidecar and generation.audio_waveform is not None
            else None
        )
        sink = self.reporter if self.reporter is not None else NullReporter()
        try:
            if plan.source != generation.signal:
                raise UnsupportedSignalError(
                    "output color plan source does not match the generation signal"
                )
            self._validate_source(plan.source)
            exr_delivery, hlg_delivery = self._deliveries(plan)
            if final_exr.exists():
                raise OutputError(f"EXR output already exists: {final_exr}")
            if final_heic is not None and final_heic.exists():
                raise OutputError(f"HEIC output already exists: {final_heic}")
            if self.path.exists() and self.path.stat().st_size:
                raise OutputError(f"output already exists: {self.path}")
            if final_audio is not None and final_audio.exists():
                raise OutputError(f"audio sidecar already exists: {final_audio}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            transaction = Path(
                tempfile.mkdtemp(prefix=f".{self.path.stem}-hdr-", dir=self.path.parent)
            )
            temp_video = transaction / "master.mp4"
            temp_exr = transaction / "exr"
            temp_exr.mkdir()
            temp_heic = None if final_heic is None else transaction / "heic"
            if temp_heic is not None:
                temp_heic.mkdir()

            import mlx.core as mx

            from kinomlx.media.hdr import decode_working_transfer
            from kinomlx.videotoolbox.encode import prepare_audio_track
            from kinomlx.videotoolbox.exr import save_exr_frame, write_exr_manifest
            from kinomlx.videotoolbox.hlg import PIX_P010_VIDEO, make_hlg_pixel_buffer
            from kinomlx.videotoolbox.writer import AVWriter

            if temp_heic is not None:
                from kinomlx.videotoolbox.heic import (
                    save_pq_heic_frame,
                    write_heic_manifest,
                )

            audio_track = prepare_audio_track(
                generation.audio_waveform,
                generation.audio_sample_rate,
                onset_trim_mode=self.audio_onset_trim_mode,
                onset_trim_ms=self.audio_onset_trim_ms,
                verbose=self.native_verbose,
            )
            writer = AVWriter(
                temp_video,
                width=plan.source.width,
                height=plan.source.height,
                fps=self.fps,
                source_pixel_format=PIX_P010_VIDEO,
                delivery=hlg_delivery,
                quality=self.encode_quality,
                label="hdr",
                audio_track=audio_track,
                audio_codec=self.audio_codec,
            )
            phase = (
                "HDR EXR + PQ HEIC + HLG encode"
                if temp_heic is not None
                else "HDR EXR + HLG encode"
            )
            sink.phase_start(phase, total=generation.frame_count, unit="frame")
            count = 0
            try:
                for count, working in enumerate(generation.frames, start=1):
                    linear = decode_working_transfer(working, plan.source.transfer)
                    mx.eval(linear)
                    exr_frame = self._exr_frame(
                        working,
                        linear,
                        source=plan.source,
                        delivery=exr_delivery,
                    )
                    mx.eval(exr_frame)
                    save_exr_frame(
                        exr_frame,
                        temp_exr / f"frame_{count - 1:05d}.exr",
                        delivery=exr_delivery,
                    )
                    if temp_heic is not None:
                        save_pq_heic_frame(
                            linear,
                            temp_heic / f"frame_{count - 1:05d}.heic",
                            primaries=plan.source.primaries,
                        )
                    pixel_buffer = make_hlg_pixel_buffer(
                        linear,
                        writer.adaptor,
                        primaries=plan.source.primaries,
                    )
                    writer.append(pixel_buffer)
                    sink.phase_advance(phase)
                    del working, linear, exr_frame, pixel_buffer
                    mx.clear_cache()
            finally:
                sink.phase_end(phase)
            if count != generation.frame_count:
                raise OutputError(
                    f"HDR terminal consumed {count} frames, expected {generation.frame_count}"
                )
            writer.finish()
            writer = None
            write_exr_manifest(
                temp_exr,
                delivery=exr_delivery,
                frame_count=count,
                width=plan.source.width,
                height=plan.source.height,
            )
            if temp_heic is not None:
                write_heic_manifest(
                    temp_heic,
                    source_primaries=plan.source.primaries,
                    frame_count=count,
                    width=plan.source.width,
                    height=plan.source.height,
                )
            temp_audio: Path | None = None
            if audio_track is not None and self.save_audio_sidecar:
                temp_audio = transaction / "audio.wav"
                audio_track.save_wav(temp_audio)

            # A generated output path is held by an empty exclusive-create
            # reservation. Move it into the transaction while publishing so a
            # later rename failure can restore the exact pre-publication state.
            reservation = transaction / "video-reservation"
            had_reservation = self.path.exists()
            if had_reservation:
                self.path.rename(reservation)
            published_exr = False
            published_heic = False
            published_video = False
            published_audio = False
            try:
                temp_exr.rename(final_exr)
                published_exr = True
                if temp_heic is not None:
                    assert final_heic is not None
                    temp_heic.rename(final_heic)
                    published_heic = True
                temp_video.rename(self.path)
                published_video = True
                if temp_audio is not None:
                    assert final_audio is not None
                    temp_audio.rename(final_audio)
                    published_audio = True
            except BaseException as exc:
                rollback_errors: list[str] = []
                for published, source, target in (
                    (published_audio, final_audio, temp_audio),
                    (published_video, self.path, temp_video),
                    (published_heic, final_heic, temp_heic),
                    (published_exr, final_exr, temp_exr),
                ):
                    if not published or source is None or target is None:
                        continue
                    try:
                        source.rename(target)
                    except OSError as rollback_exc:
                        rollback_errors.append(f"{source}: {rollback_exc}")
                if had_reservation and reservation.exists():
                    try:
                        reservation.rename(self.path)
                    except OSError as rollback_exc:
                        rollback_errors.append(f"{self.path}: {rollback_exc}")
                if rollback_errors:
                    exc.add_note("HDR publication rollback failures: " + "; ".join(rollback_errors))
                raise
            reservation.unlink(missing_ok=True)
            transaction.rmdir()
            transaction = None
            return ArtifactSet(
                video=self.path,
                exr_frames=final_exr,
                heic_frames=final_heic,
            )
        except OutputError, UnsupportedSignalError:
            raise
        except Exception as exc:
            raise OutputError(f"cannot write HDR output {self.path}: {exc}") from exc
        finally:
            if writer is not None:
                cancel = getattr(writer, "cancel", None)
                if callable(cancel):
                    cancel()
            if transaction is not None:
                shutil.rmtree(transaction, ignore_errors=True)
            generation.close()


__all__ = [
    "ArtifactSet",
    "default_hdr_exr_directory",
    "default_hdr_heic_directory",
    "default_vae_frame_directory",
    "Generation",
    "GenerationSink",
    "HDRGenerationSink",
    "OutputError",
    "VideoToolboxEncoder",
    "VideoToolboxGenerationSink",
]
