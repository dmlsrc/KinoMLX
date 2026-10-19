"""Explicit dtype boundaries for native LTX-2 pipelines."""

from dataclasses import dataclass

import mlx.core as mx

from kinomlx.modeling.dtypes import FloatDType, float_dtype_name, resolve_float_dtype

from .vocabularies import LTX2VideoVAEDecodeDType


@dataclass(frozen=True)
class LTX2DTypePolicy:
    """Resolved component dtypes, separate from operator opmath policy.

    ``LTX2Settings.transformer_dtype`` is intentionally allowed to change only
    ``transformer``. LTX-2 carries its persistent latents and reference component
    policy in BF16. The terminal video decoder selects FP16 separately (see
    :func:`resolve_video_vae_decode_precision`); precision-sensitive arithmetic
    inside components belongs in :mod:`kinomlx.kernels` rather than widening an
    entire component.

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
            "transformer": float_dtype_name(self.transformer),
            "latent": float_dtype_name(self.latent),
            "video_vae": float_dtype_name(self.video_vae),
            "spatial_upscaler": float_dtype_name(self.spatial_upscaler),
            "audio_vae": float_dtype_name(self.audio_vae),
            "duration_head": float_dtype_name(self.duration_head),
            "temporal_upscaler": float_dtype_name(self.temporal_upscaler),
        }


@dataclass(frozen=True)
class VideoVAEDecodePrecision:
    """The dtype a video decode computes in, and the one it decodes again in when its first
    decoded frames are not finite.

    ``auto`` decodes in FP16 with either decoder, SDR or HDR, and falls back to BF16, whose
    range matches FP32's at half its memory. A requested dtype has no fallback: a nonfinite
    decode stops the run.
    """

    dtype: mx.Dtype
    overflow_fallback: mx.Dtype | None


def resolve_video_vae_decode_precision(value: LTX2VideoVAEDecodeDType) -> VideoVAEDecodePrecision:
    """Resolve a request's video VAE decode dtype and its overflow fallback."""
    if value == LTX2VideoVAEDecodeDType.AUTO:
        return VideoVAEDecodePrecision(mx.float16, mx.bfloat16)
    dtype = resolve_float_dtype(FloatDType(value), owner="LTX-2 video VAE decode")
    return VideoVAEDecodePrecision(dtype, None)


__all__ = ["LTX2DTypePolicy", "VideoVAEDecodePrecision", "resolve_video_vae_decode_precision"]
