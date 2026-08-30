"""Behavioral tests for ``kinomlx.models.ltx2.types``.

Pins the LTX-2.3 shape derivations and audio encoder geometry that
the generic ``kinomlx.types`` deliberately doesn't bake in.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from kinomlx.models.ltx2.settings import LTX2Settings
from kinomlx.models.ltx2.types import (
    AUDIO_HOP_LENGTH,
    AUDIO_LATENT_CHANNELS,
    AUDIO_LATENT_DOWNSAMPLE,
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_MEL_BINS,
    AUDIO_SAMPLE_RATE,
    NATIVE_FPS,
    VIDEO_LATENT_CHANNELS,
    VIDEO_PIXEL_CHANNELS,
    VIDEO_VAE_SCALE,
    AudioLatentShape,
    DistilledRequest,
    ImageConditioningConfig,
    VideoVAETilingConfig,
    video_latent_shape_from_pixel,
)
from kinomlx.types import VideoLatentShape, VideoPixelShape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_video_constants() -> None:
    assert NATIVE_FPS == 24.0
    assert VIDEO_LATENT_CHANNELS == 128
    assert VIDEO_PIXEL_CHANNELS == 3
    assert VIDEO_VAE_SCALE.time == 8
    assert VIDEO_VAE_SCALE.height == 32
    assert VIDEO_VAE_SCALE.width == 32


def test_distilled_request_defaults_match_the_production_shape() -> None:
    config = DistilledRequest(prompt="test")
    assert (config.width, config.height, config.frames, config.fps) == (
        1024,
        576,
        121,
        NATIVE_FPS,
    )
    assert config.vae_decode_dtype == "auto"
    assert config.reference_aligned_audio is False
    assert config.noise_backend == "mlx"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 1000},
        {"height": 500},
        {"frames": 120},
        {"fps": 0.0},
        {"fps": math.nan},
        {"fps": math.inf},
    ],
)
def test_distilled_request_rejects_invalid_geometry(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive|divisible|8\\*k|fps"):
        DistilledRequest(prompt="test", **kwargs)


@pytest.mark.parametrize("seed", [-1, -(2**63)])
def test_distilled_request_rejects_negative_seeds_at_the_public_boundary(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        DistilledRequest(prompt="test", seed=seed)


def test_distilled_request_validates_sampler_override() -> None:
    assert DistilledRequest(prompt="test").sampler == "auto"
    assert DistilledRequest(prompt="test", sampler="deterministic").sampler == "deterministic"
    with pytest.raises(ValueError, match="sampler must be one of"):
        DistilledRequest(prompt="test", sampler="future")  # type: ignore[arg-type]


def test_distilled_request_validates_noise_backend() -> None:
    assert DistilledRequest(prompt="test", noise_backend="torch-mps").noise_backend == "torch-mps"
    with pytest.raises(ValueError, match="noise_backend must be one of"):
        DistilledRequest(prompt="test", noise_backend="future")  # type: ignore[arg-type]


def test_distilled_request_validates_vae_decode_dtype_override() -> None:
    assert DistilledRequest(prompt="test", vae_decode_dtype="bfloat16").vae_decode_dtype == (
        "bfloat16"
    )
    assert DistilledRequest(prompt="test", vae_decode_dtype="float32").vae_decode_dtype == "float32"
    with pytest.raises(ValueError, match="vae_decode_dtype must be one of"):
        DistilledRequest(prompt="test", vae_decode_dtype="float16")  # type: ignore[arg-type]


def test_duration_rounds_up_to_a_valid_frame_count() -> None:
    assert DistilledRequest(prompt="test", duration=20.0).frames == 481
    assert DistilledRequest(prompt="test", duration=30.0).frames == 721
    assert DistilledRequest(prompt="test", duration=1.05, fps=24.0).frames == 33


def test_omitted_frame_count_is_reserved_for_auto_duration() -> None:
    assert DistilledRequest(prompt="test", frames=None).frames is None


@pytest.mark.parametrize("generated_keyframes", [-1, True, 1.5])
def test_generated_keyframe_count_must_be_a_nonnegative_integer(
    generated_keyframes: object,
) -> None:
    with pytest.raises(ValueError, match="generated_keyframes"):
        DistilledRequest(prompt="test", generated_keyframes=generated_keyframes)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan, math.inf])
def test_duration_must_be_finite_and_positive(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be finite and positive"):
        DistilledRequest(prompt="test", duration=duration)


def test_text_conditioning_can_replace_a_prompt(tmp_path: Path) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    DistilledRequest(text_conditioning=sidecar).validate_for_generation()
    with pytest.raises(ValueError, match="prompt or text_conditioning"):
        DistilledRequest().validate_for_generation()


def test_vae_tiling_resolves_single_and_custom_plans() -> None:
    single = VideoVAETilingConfig(mode="single").to_runtime_config()
    assert single is not None
    assert single.temporal_config is None
    assert single.spatial_config is None

    custom = VideoVAETilingConfig(
        mode="custom",
        temporal_tile_frames=64,
        temporal_overlap_frames=8,
        spatial_tile_pixels=512,
        spatial_overlap_pixels=64,
    ).to_runtime_config()
    assert custom is not None
    assert custom.temporal_config is not None
    assert custom.temporal_config.chunk_size_in_frames == 64
    assert custom.spatial_config is not None
    assert custom.spatial_config.tile_size_in_pixels == 512


def test_custom_vae_tiling_defaults_temporal_tiles_and_rejects_empty_plan() -> None:
    default_custom = VideoVAETilingConfig(mode="custom").to_runtime_config()
    assert default_custom is not None
    assert default_custom.temporal_config is not None
    assert default_custom.temporal_config.chunk_size_in_frames == 256
    with pytest.raises(ValueError, match="needs a temporal or spatial"):
        VideoVAETilingConfig(
            mode="custom",
            temporal_tile_frames=0,
            spatial_tile_pixels=0,
        )
    with pytest.raises(ValueError, match="overlap must be smaller"):
        VideoVAETilingConfig(
            mode="custom",
            temporal_overlap_frames=256,
        )


def test_image_config_validates_strength_and_frame_bounds() -> None:
    image = ImageConditioningConfig(Path("frame.png"), frame_index=8, strength=0.9)
    assert DistilledRequest(prompt="test", image=image).image == image
    with pytest.raises(ValueError, match="between 0 and 1"):
        ImageConditioningConfig(Path("frame.png"), strength=1.1)
    with pytest.raises(ValueError, match="outside"):
        DistilledRequest(
            prompt="test",
            frames=9,
            image=ImageConditioningConfig(Path("frame.png"), frame_index=9),
        )


def test_lora_one_or_n_values_expand_per_adapter() -> None:
    config = DistilledRequest(
        prompt="test",
        lora_paths=(Path("style.safetensors"), Path("motion.safetensors")),
        lora_strengths=(0.8,),
        lora_stage1_strengths=(0.25, 0.5),
        lora_stage2_strengths=(1.0,),
        lora_exclusions=("audio,cross", "video"),
    )
    resolved = config.resolved_loras()
    assert [(item.path.name, item.strength) for item in resolved] == [
        ("style.safetensors", 0.8),
        ("motion.safetensors", 0.8),
    ]
    assert [item.stage_1_strength for item in resolved] == [0.25, 0.5]
    assert [item.stage_2_strength for item in resolved] == [1.0, 1.0]
    assert [item.exclude for item in resolved] == [
        ("audio", "cross"),
        ("video",),
    ]


def test_lora_value_count_and_unknown_exclusion_fail_before_loading() -> None:
    with pytest.raises(ValueError, match="one value or one per LoRA"):
        DistilledRequest(
            prompt="test",
            lora_paths=(Path("a"), Path("b")),
            lora_strengths=(0.1, 0.2, 0.3),
        )
    with pytest.raises(ValueError, match="unknown LoRA exclusion"):
        DistilledRequest(
            prompt="test",
            lora_paths=(Path("a"),),
            lora_exclusions=("sideways",),
        )


def test_lora_strengths_are_normalized_and_rejected_at_request_construction() -> None:
    request = DistilledRequest(
        prompt="test",
        lora_paths=(Path("adapter"),),
        lora_strengths=(1,),
    )
    assert type(request.resolved_loras()[0].strength) is float

    broad = DistilledRequest(
        prompt="test",
        lora_paths=(Path("adapter"),),
        lora_strengths=(3.0,),
        lora_stage1_strengths=(-3.0,),
    )
    assert broad.resolved_loras()[0].strength == 3.0
    assert broad.resolved_loras()[0].stage_1_strength == -3.0

    with pytest.raises(ValueError, match="could not convert string to float"):
        DistilledRequest(
            prompt="test",
            lora_paths=(Path("adapter"),),
            lora_strengths=("not-a-number",),  # type: ignore[arg-type]
        )


def test_lora_rejects_streaming_and_quantized_cache_modes() -> None:
    config = DistilledRequest(prompt="test", lora_paths=(Path("adapter"),))
    with pytest.raises(ValueError, match="block streaming"):
        config.validate_with_settings(LTX2Settings(transformer_resident_blocks=4))
    with pytest.raises(ValueError, match="block streaming"):
        config.validate_with_settings(LTX2Settings(stream_transformer=True))
    with pytest.raises(ValueError, match="unquantized"):
        config.validate_with_settings(LTX2Settings(transformer_cache_quantize="mxfp8-blocks"))


def test_audio_constants() -> None:
    assert AUDIO_SAMPLE_RATE == 16000
    assert AUDIO_HOP_LENGTH == 160
    assert AUDIO_LATENT_DOWNSAMPLE == 4
    assert AUDIO_LATENT_CHANNELS == 8
    assert AUDIO_MEL_BINS == 16
    # Derived: 16000 / 160 / 4 = 25.
    assert AUDIO_LATENTS_PER_SECOND == 25.0


# ---------------------------------------------------------------------------
# video_latent_shape_from_pixel - VAE arithmetic
# ---------------------------------------------------------------------------


def test_video_latent_from_pixel_standard_production_shape() -> None:
    """1024x576x121 - the canonical production benchmark shape."""
    pixel = VideoPixelShape(batch=1, frames=121, height=576, width=1024)
    latent = video_latent_shape_from_pixel(pixel)
    assert latent == VideoLatentShape(
        batch=1,
        channels=VIDEO_LATENT_CHANNELS,
        # (121 - 1) // 8 + 1 = 16
        frames=16,
        # 576 // 32, 1024 // 32
        height=18,
        width=32,
    )


def test_video_latent_from_pixel_preserves_batch() -> None:
    pixel = VideoPixelShape(batch=4, frames=25, height=256, width=256)
    latent = video_latent_shape_from_pixel(pixel)
    assert latent.batch == 4
    assert latent.channels == VIDEO_LATENT_CHANNELS


# ---------------------------------------------------------------------------
# AudioLatentShape
# ---------------------------------------------------------------------------


def test_audio_latent_to_and_from_tuple() -> None:
    s = AudioLatentShape(batch=1, channels=8, frames=126, mel_bins=16)
    assert s.to_tuple() == (1, 8, 126, 16)
    assert AudioLatentShape.from_tuple(s.to_tuple()) == s


def test_audio_latent_from_duration_one_second() -> None:
    s = AudioLatentShape.from_duration(batch=1, duration_seconds=1.0)
    assert s.batch == 1
    assert s.channels == AUDIO_LATENT_CHANNELS
    assert s.frames == 25  # 1.0 s x 25 latents/s
    assert s.mel_bins == AUDIO_MEL_BINS


def test_audio_latent_from_duration_covers_causal_decode_boundary() -> None:
    # 12 latents decode to 450 ms; 13 decode to 490 ms, which is the closer
    # coverage for a requested 500 ms clip.
    s = AudioLatentShape.from_duration(batch=1, duration_seconds=0.5)
    assert s.frames == 13


def test_audio_latent_from_video_uses_native_fps_by_default() -> None:
    """A 121-frame video at NATIVE_FPS gets audio sized to that duration."""
    pixel = VideoPixelShape(batch=1, frames=121, height=576, width=1024)
    audio = AudioLatentShape.from_video(pixel)
    expected_frames = math.ceil((121 / NATIVE_FPS) * AUDIO_LATENTS_PER_SECOND)
    assert audio.batch == 1
    assert audio.frames == expected_frames


def test_audio_latent_from_video_can_match_reference_rounding() -> None:
    pixel = VideoPixelShape(batch=1, frames=121, height=576, width=1024)
    coverage = AudioLatentShape.from_video(pixel, fps=24.0)
    reference = AudioLatentShape.from_video(pixel, fps=24.0, reference_aligned=True)

    assert coverage.frames == 127
    assert reference.frames == 126


def test_audio_latent_from_video_honors_explicit_fps() -> None:
    """Same pixel shape at 30 fps -> different audio frame count."""
    pixel = VideoPixelShape(batch=1, frames=121, height=576, width=1024)
    at_24 = AudioLatentShape.from_video(pixel, fps=24.0)
    at_30 = AudioLatentShape.from_video(pixel, fps=30.0)
    # Higher fps -> shorter clip -> fewer audio latents.
    assert at_30.frames < at_24.frames
