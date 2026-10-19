"""Transformer weight dtype names and normalization."""

import mlx.core as mx

from kinomlx.modeling.dtypes import FloatDType, resolve_float_dtype


def resolve_transformer_dtype(dtype: FloatDType | mx.Dtype | None) -> mx.Dtype | None:
    """Resolve an optional transformer weight override."""
    return None if dtype is None else resolve_float_dtype(dtype, owner="LTX transformer")


def normalize_transformer_dtypes(
    transformer_dtype: FloatDType | mx.Dtype | None,
    video_ff_dtype: FloatDType | mx.Dtype | None,
    audio_ff_dtype: FloatDType | mx.Dtype | None,
) -> tuple[mx.Dtype | None, mx.Dtype | None, mx.Dtype | None]:
    """Canonicalize global and targeted transformer weight dtype requests."""
    resolved_transformer = resolve_transformer_dtype(transformer_dtype)
    resolved_video = resolve_transformer_dtype(video_ff_dtype)
    resolved_audio = resolve_transformer_dtype(audio_ff_dtype)

    if resolved_transformer is None or resolved_transformer == mx.bfloat16:
        return (
            None,
            (
                None
                if resolved_video is not None and resolved_video == mx.bfloat16
                else resolved_video
            ),
            (
                None
                if resolved_audio is not None and resolved_audio == mx.bfloat16
                else resolved_audio
            ),
        )

    for label, override in (
        ("video FF", resolved_video),
        ("audio FF", resolved_audio),
    ):
        if override is not None and override != resolved_transformer:
            raise ValueError(
                f"{label} weight dtype {override} conflicts with transformer "
                f"dtype {resolved_transformer}"
            )
    return resolved_transformer, None, None


__all__ = [
    "normalize_transformer_dtypes",
    "resolve_transformer_dtype",
]
