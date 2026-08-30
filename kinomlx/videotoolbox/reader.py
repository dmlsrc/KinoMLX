"""Native SDR video decode for bounded model conditioning inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import AVFoundation as av
import CoreMedia
import Foundation
import mlx.core as mx
import objc
import Quartz

from . import pixel_buffers as _pb


class _Asset(Protocol):
    def tracksWithMediaType_(self, media_type: object) -> list[object]: ...


class _VideoTrack(Protocol):
    def formatDescriptions(self) -> list[object] | tuple[object, ...] | None: ...


class _ReaderOutput(Protocol):
    def setVideoComposition_(self, composition: object) -> None: ...

    def setAlwaysCopiesSampleData_(self, always_copies: bool) -> None: ...

    def copyNextSampleBuffer(self) -> object | None: ...


def _first_video_track(asset: _Asset) -> _VideoTrack:
    tracks = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    if not tracks:
        raise ValueError("reference input contains no video track")
    return cast(_VideoTrack, tracks[0])


def _reject_explicit_hdr_track(track: _VideoTrack, path: Path) -> None:
    descriptions = track.formatDescriptions() or ()
    if not descriptions:
        return
    extensions = CoreMedia.CMFormatDescriptionGetExtensions(descriptions[0]) or {}
    primaries = extensions.get(Quartz.kCVImageBufferColorPrimariesKey)
    transfer = extensions.get(Quartz.kCVImageBufferTransferFunctionKey)
    matrix = extensions.get(Quartz.kCVImageBufferYCbCrMatrixKey)
    if (
        primaries == Quartz.kCVImageBufferColorPrimaries_ITU_R_2020
        or transfer
        in {
            Quartz.kCVImageBufferTransferFunction_ITU_R_2100_HLG,
            Quartz.kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
        }
        or matrix == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020
    ):
        raise ValueError(f"HDR reference input must be ordinary SDR Rec.709: {path}")


def _decoded_output(asset: object, track: object) -> _ReaderOutput:
    settings = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: _pb.PIX_BGRA,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    composition = av.AVVideoComposition.videoCompositionWithPropertiesOfAsset_(asset)
    output = av.AVAssetReaderVideoCompositionOutput.alloc().initWithVideoTracks_videoSettings_(
        [track],
        settings,
    )
    if output is None or composition is None:
        raise RuntimeError("AVFoundation cannot construct a color-managed reference decoder")
    output.setVideoComposition_(composition)
    output.setAlwaysCopiesSampleData_(False)
    return cast(_ReaderOutput, output)


def read_sdr_video_frames(
    path: Path | str,
    *,
    max_frames: int,
) -> mx.array:
    """Decode up to ``max_frames`` oriented SDR frames as float32 FHWC sRGB."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"reference video does not exist: {source_path}")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("reference max_frames must be a positive integer")

    url = Foundation.NSURL.fileURLWithPath_(str(source_path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    _reject_explicit_hdr_track(track, source_path)
    reader, error = av.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"AVAssetReader init failed: {error}")
    output = _decoded_output(asset, track)
    if not reader.canAddOutput_(output):
        raise RuntimeError("AVAssetReader cannot expose color-managed SDR frames")
    reader.addOutput_(output)
    if not reader.startReading():
        raise RuntimeError(f"AVAssetReader.startReading failed: {reader.error()}")

    frames: list[mx.array] = []
    while len(frames) < max_frames:
        with objc.autorelease_pool():
            sample = output.copyNextSampleBuffer()
            if sample is None:
                break
            pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample)
            if pixel_buffer is None:
                continue
            rgb = _pb.read_pixel_buffer_rgb(pixel_buffer)
            mx.eval(rgb)
            frames.append(rgb)
    if len(frames) == max_frames:
        reader.cancelReading()
    elif reader.status() == av.AVAssetReaderStatusFailed:
        raise RuntimeError(f"AVAssetReader failed: {reader.error()}")
    if not frames:
        raise ValueError(f"reference video contains no decodable frames: {source_path}")
    video = mx.stack(frames, axis=0).astype(mx.float32) / 255.0
    mx.eval(video)
    return video


__all__ = ["read_sdr_video_frames"]
