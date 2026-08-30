"""Explicit dtype boundaries for native LTX-2 pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .types import VideoVAEDecodeDType


def _dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


@dataclass(frozen=True)
class LTX2DTypePolicy:
    """Resolved component dtypes, separate from operator opmath policy.

    ``LTX2Settings.transformer_dtype`` is intentionally allowed to change only
    ``transformer``. LTX-2 carries its persistent latents and the sequential
    VAE/upscaler/audio components in BF16. Precision-sensitive arithmetic
    inside those boundaries belongs in :mod:`kinomlx.kernels`; widening an
    entire component to compensate for one operator would increase memory and
    obscure which operation actually required FP32 math.

    The individual fields remain constructible for focused tests, but only
    :meth:`reference` is used by the product runtime.
    """

    transformer: mx.Dtype
    latent: mx.Dtype
    video_vae: mx.Dtype
    spatial_upscaler: mx.Dtype
    audio_vae: mx.Dtype
    duration_head: mx.Dtype = mx.bfloat16
    temporal_upscaler: mx.Dtype = mx.bfloat16

    @classmethod
    def reference(cls, *, transformer: mx.Dtype) -> LTX2DTypePolicy:
        """Resolve the shared LTX-2 boundary contract around one transformer dtype."""
        return cls(
            transformer=transformer,
            latent=mx.bfloat16,
            video_vae=mx.bfloat16,
            spatial_upscaler=mx.bfloat16,
            audio_vae=mx.bfloat16,
            duration_head=mx.bfloat16,
            temporal_upscaler=mx.bfloat16,
        )

    def to_metadata(self) -> dict[str, str]:
        """Return stable user-facing names for generation receipts."""
        return {
            "transformer": _dtype_name(self.transformer),
            "latent": _dtype_name(self.latent),
            "video_vae": _dtype_name(self.video_vae),
            "spatial_upscaler": _dtype_name(self.spatial_upscaler),
            "audio_vae": _dtype_name(self.audio_vae),
            "duration_head": _dtype_name(self.duration_head),
            "temporal_upscaler": _dtype_name(self.temporal_upscaler),
        }


def resolve_video_vae_decode_dtype(
    value: VideoVAEDecodeDType,
    *,
    hdr: bool,
    default: mx.Dtype,
) -> mx.Dtype:
    """Resolve a request override against the recipe-aware decode default."""
    if value == "auto":
        return mx.float32 if hdr else default
    if value == "bfloat16":
        return mx.bfloat16
    if value == "float32":
        return mx.float32
    raise ValueError(f"unsupported video VAE decode dtype {value!r}")


__all__ = ["LTX2DTypePolicy", "resolve_video_vae_decode_dtype"]
