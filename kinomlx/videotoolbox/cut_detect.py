"""Scene-cut detector for native stages that retain adjacent-frame history.

VSR's ``balanced`` mode propagates previous source and output frames into the
upscaler for frame-to-frame coherence, while VTFRC interpolates adjacent source
frames. Across a hard cut that context is wrong: it can ghost or synthesize an
in-between frame that never existed. Generated video can contain cuts, so the
installed terminal enables lightweight detection whenever either history-
sensitive stage is active.

Two algorithms:
  simple  Downsampled-pixel mean absolute difference. ~1ms/frame. Catches
          hard cuts cleanly; doesn't flag dissolves (which don't ghost in
          VSR anyway because the frame-to-frame coherence kinda matches).
  hist    Per-channel 32-bin histogram chi-squared distance. ~3ms/frame.
          More robust to fast motion than simple-pixel diff.

False positives are cheap (one frame of "no prior-frame context", visually
invisible). False negatives let a cut ghost - tune the threshold down.
"""

from __future__ import annotations

import math
from typing import cast

import mlx.core as mx


def _to_uint8_rgb(frame: mx.array) -> mx.array:
    """Coerce any frame format (uint8 RGB, uint8 RGBA, fp16 RGBA, fp32 RGBA)
    to a uint8 RGB mlx array for histogram / thumbnail work.
    """
    f = frame
    if str(f.dtype).split(".")[-1] in ("float16", "float32"):
        return mx.clip(f[..., :3] * 255.0, 0, 255).astype(mx.uint8)
    if f.shape[-1] == 4:
        return f[..., :3]
    return f


def _frame_thumbnail(frame: mx.array, target_size: int = 32) -> mx.array:
    rgb = _to_uint8_rgb(frame)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    step_h = max(1, h // target_size)
    step_w = max(1, w // target_size)
    return mx.contiguous(rgb[::step_h, ::step_w])


def _channel_histogram(channel: mx.array, bins: int) -> mx.array:
    """Per-channel uint8 histogram, `bins` equal-width bins over [0, 256).

    mx has no bincount, so count via a one-hot sum: value v lands in bin
    v * bins // 256, which is np.histogram's binning for range=(0, 256).
    """
    idx = (channel.astype(mx.int32) * bins) // 256  # (H, W) in [0, bins)
    onehot = mx.equal(idx.reshape(-1, 1), mx.arange(bins).reshape(1, -1))
    return mx.sum(onehot, axis=0)  # (bins,) counts


def _frame_histogram(frame: mx.array, bins: int = 32) -> mx.array:
    rgb = _to_uint8_rgb(frame)
    hists = [_channel_histogram(rgb[..., c], bins) for c in range(3)]
    return mx.concatenate(hists).astype(mx.float32)


class CutDetector:
    """Detects hard cuts between consecutive frames.

    Modes:
        "off"     no-op
        "simple"  downsampled-pixel MAD. threshold ~0.2-0.35 typical.
        "hist"    per-channel histogram chi-squared. threshold ~0.4-0.8.

    Always returns False on the first frame (no previous to compare).
    """

    _DEFAULT_THRESHOLDS = {
        "off": 0.0,
        "simple": 0.25,
        "hist": 0.5,
    }

    def __init__(self, mode: str = "simple", threshold: float | None = None):
        if mode not in self._DEFAULT_THRESHOLDS:
            raise ValueError(f"unknown cut-detect mode: {mode!r}")
        resolved_threshold = (
            self._DEFAULT_THRESHOLDS[mode] if threshold is None else float(threshold)
        )
        if not math.isfinite(resolved_threshold) or resolved_threshold < 0.0:
            raise ValueError("cut-detect threshold must be finite and non-negative")
        self.mode = mode
        self.threshold = resolved_threshold
        self._prev: mx.array | None = None

    def is_cut(self, frame: mx.array) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "simple":
            curr = _frame_thumbnail(frame)
            mx.eval(curr)
            if self._prev is None:
                self._prev = curr
                return False
            diff = mx.abs(curr.astype(mx.int16) - self._prev.astype(mx.int16))
            mad = float(cast(int | float, (mx.mean(diff.astype(mx.float32)) / 255.0).item()))
            self._prev = curr
            return bool(mad > self.threshold)
        if self.mode == "hist":
            curr = _frame_histogram(_frame_thumbnail(frame))
            mx.eval(curr)
            if self._prev is None:
                self._prev = curr
                return False
            a, b = self._prev, curr
            eps = 1e-6
            chi2 = mx.sum((a - b) ** 2 / (a + b + eps))
            norm = float(cast(int | float, (chi2 / (mx.sum(a) + eps)).item()))
            self._prev = curr
            return bool(norm > self.threshold)
        raise AssertionError(f"unreachable cut-detect mode: {self.mode!r}")
