"""SDR reference-video preprocessing for the HDR IC-LoRA recipe."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from kinomlx.reporting import Reporter
from kinomlx.videotoolbox.reader import read_sdr_video_frames

from .components import _VideoEncoderCallablePort


def _resize_bilinear(video: mx.array, width: int, height: int) -> mx.array:
    """Resize FHWC float video with align-corners-false bilinear sampling."""
    source_height, source_width = int(video.shape[1]), int(video.shape[2])
    if (source_width, source_height) == (width, height):
        return video
    y = (mx.arange(height, dtype=mx.float32) + 0.5) * (source_height / height) - 0.5
    y = mx.clip(y, 0.0, float(source_height - 1))
    y0 = mx.floor(y).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, source_height - 1)
    y_weight = (y - y0.astype(mx.float32))[None, :, None, None]
    resized = mx.take(video, y0, axis=1) * (1.0 - y_weight) + mx.take(video, y1, axis=1) * y_weight

    x = (mx.arange(width, dtype=mx.float32) + 0.5) * (source_width / width) - 0.5
    x = mx.clip(x, 0.0, float(source_width - 1))
    x0 = mx.floor(x).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, source_width - 1)
    x_weight = (x - x0.astype(mx.float32))[None, None, :, None]
    return mx.take(resized, x0, axis=2) * (1.0 - x_weight) + mx.take(resized, x1, axis=2) * x_weight


def _tail_indices(source_size: int, target_size: int, *, reflect: bool) -> mx.array:
    indices = mx.arange(target_size, dtype=mx.int32)
    if reflect and source_size > 1:
        return mx.where(indices < source_size, indices, 2 * source_size - 2 - indices)
    return mx.minimum(indices, source_size - 1)


def resize_and_reflect_pad_sdr(
    video: mx.array,
    *,
    width: int,
    height: int,
) -> mx.array:
    """Match the Apache HDR reference path's resize and bottom/right padding."""
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"SDR reference video must be FHWC RGB, got {tuple(video.shape)}")
    source_height, source_width = int(video.shape[1]), int(video.shape[2])
    if height >= source_height and width >= source_width:
        new_height, new_width = source_height, source_width
        resized = video
    else:
        scale = min(height / source_height, width / source_width)
        new_height = max(1, round(source_height * scale))
        new_width = max(1, round(source_width * scale))
        resized = _resize_bilinear(video, new_width, new_height)

    pad_bottom = height - new_height
    pad_right = width - new_width
    reflect = pad_bottom < new_height and pad_right < new_width
    y = _tail_indices(new_height, height, reflect=reflect)
    x = _tail_indices(new_width, width, reflect=reflect)
    padded = mx.take(mx.take(resized, y, axis=1), x, axis=2)
    return mx.contiguous(padded)


def encode_reference_video(
    path: Path | str,
    video_encoder: _VideoEncoderCallablePort,
    *,
    width: int,
    height: int,
    frames: int,
    compute_dtype: mx.Dtype,
    reporter: Reporter | None = None,
) -> mx.array:
    """Decode, reflect-pad, and deterministically VAE-encode one SDR reference.

    The reference must cover every requested frame: the HDR IC-LoRA converts
    video to video, and generation frames without a matching reference-track
    frame run the fused adapter off-distribution.
    """
    pixels = read_sdr_video_frames(path, max_frames=frames)
    decoded = int(pixels.shape[0])
    if decoded < frames:
        raise ValueError(
            f"SDR reference {path} covers {decoded} of the {frames} requested frames; "
            "the HDR IC-LoRA is a video-to-video converter and runs off-distribution "
            "on frames without a reference track - shorten the generation or use a "
            "longer reference video"
        )
    pixels = resize_and_reflect_pad_sdr(pixels, width=width, height=height)
    video = (pixels * 2.0 - 1.0).transpose(3, 0, 1, 2)[None]
    latent = video_encoder(video.astype(compute_dtype), reporter=reporter)
    mx.eval(latent)
    return latent


__all__ = ["encode_reference_video", "resize_and_reflect_pad_sdr"]
