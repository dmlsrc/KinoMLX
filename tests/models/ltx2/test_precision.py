"""Resolved dtype-boundary contracts for the native LTX-2 runtime."""

import mlx.core as mx

from kinomlx.models.ltx2.precision import (
    LTX2DTypePolicy,
    VideoVAEDecodePrecision,
    resolve_video_vae_decode_precision,
)


def test_reference_policy_keeps_non_transformer_boundaries_bfloat16() -> None:
    policy = LTX2DTypePolicy.reference(transformer=mx.float16)

    assert policy.transformer == mx.float16
    assert policy.latent == mx.bfloat16
    assert policy.video_vae == mx.bfloat16
    assert policy.spatial_upscaler == mx.bfloat16
    assert policy.audio_vae == mx.bfloat16
    assert policy.duration_head == mx.bfloat16
    assert policy.temporal_upscaler == mx.bfloat16


def test_default_reference_policy_preserves_the_all_bfloat16_contract() -> None:
    policy = LTX2DTypePolicy.reference(transformer=mx.bfloat16)

    assert policy.to_metadata() == {
        "transformer": "bfloat16",
        "latent": "bfloat16",
        "video_vae": "bfloat16",
        "spatial_upscaler": "bfloat16",
        "audio_vae": "bfloat16",
        "duration_head": "bfloat16",
        "temporal_upscaler": "bfloat16",
    }


def test_video_vae_decode_is_fp16_with_a_bfloat16_fallback_unless_requested() -> None:
    assert resolve_video_vae_decode_precision("auto") == VideoVAEDecodePrecision(
        mx.float16, mx.bfloat16
    )
    for requested, dtype in (
        ("float16", mx.float16),
        ("bfloat16", mx.bfloat16),
        ("float32", mx.float32),
    ):
        assert resolve_video_vae_decode_precision(requested) == VideoVAEDecodePrecision(dtype, None)
