"""Behavioral tests for ``kinomlx.models.ltx2.types``.

Pins the LTX-2.3 shape derivations and audio encoder geometry that
the generic ``kinomlx.types`` deliberately doesn't bake in.
"""

import dataclasses
import math
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.models.ltx2.adapters.inputs import LTX2AdapterConfig
from kinomlx.models.ltx2.types import (
    AUDIO_HOP_LENGTH,
    AUDIO_LATENT_CHANNELS,
    AUDIO_LATENT_DOWNSAMPLE,
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_MEL_BINS,
    AUDIO_SAMPLE_RATE,
    NATIVE_FPS,
    VIDEO_LATENT_CHANNELS,
    VIDEO_VAE_SCALE,
    AudioLatentShape,
    LatentState,
    LTX2ImageConditioningConfig,
    LTX2Request,
    LTX2VideoVAETilingConfig,
    video_latent_shape_from_pixel,
)
from kinomlx.models.ltx2.video_vae.sharpening import ContextSharpening
from kinomlx.types import VideoLatentShape, VideoPixelShape


def _dummy_state(uniform_mask: bool = True) -> LatentState:
    return LatentState(
        latent=mx.zeros((1, 128, 16, 18, 32)),
        denoise_mask=mx.ones((1, 1, 16, 18, 32)),
        positions=mx.zeros((1, 3, 16, 18, 32)),
        clean_latent=mx.zeros((1, 128, 16, 18, 32)),
        uniform_mask=uniform_mask,
    )


def test_latent_state_uniform_mask_defaults_true() -> None:
    state = LatentState(
        latent=mx.zeros((1,)),
        denoise_mask=mx.zeros((1,)),
        positions=mx.zeros((1,)),
        clean_latent=mx.zeros((1,)),
    )
    assert state.uniform_mask is True


def test_latent_state_is_frozen() -> None:
    state = _dummy_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.latent = mx.zeros((1,))  # type: ignore[misc]


def test_latent_state_replace_preserves_uniform_mask_flag() -> None:
    state = _dummy_state(uniform_mask=True)
    new_mask = mx.zeros_like(state.denoise_mask)
    unchanged_flag = dataclasses.replace(state, denoise_mask=new_mask)
    assert unchanged_flag.uniform_mask is True
    mixed = dataclasses.replace(state, denoise_mask=new_mask, uniform_mask=False)
    assert mixed.uniform_mask is False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_video_constants() -> None:
    assert NATIVE_FPS == 24.0
    assert VIDEO_LATENT_CHANNELS == 128
    assert VIDEO_VAE_SCALE.time == 8
    assert VIDEO_VAE_SCALE.height == 32
    assert VIDEO_VAE_SCALE.width == 32


def test_distilled_request_defaults_match_the_production_shape() -> None:
    config = LTX2Request(prompt="test")
    assert (config.width, config.height, config.frames, config.fps) == (
        1024,
        576,
        121,
        NATIVE_FPS,
    )
    assert config.vae_decode_dtype == "auto"
    assert config.vae_cache_limit_gb is None
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
        LTX2Request(prompt="test", **kwargs)


@pytest.mark.parametrize("seed", [-1, -(2**63)])
def test_distilled_request_rejects_negative_seeds_at_the_public_boundary(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        LTX2Request(prompt="test", seed=seed)


@pytest.mark.parametrize("field", ["stage_1_sampler", "stage_2_sampler"])
def test_distilled_request_validates_each_stage_sampler(field: str) -> None:
    assert getattr(LTX2Request(prompt="test"), field) == "auto"
    request = LTX2Request(prompt="test", **{field: "deterministic"})
    assert getattr(request, field) == "deterministic"
    with pytest.raises(ValueError, match=f"{field}: expected one of"):
        LTX2Request(prompt="test", **{field: "future"})


def test_distilled_request_validates_noise_backend() -> None:
    assert LTX2Request(prompt="test", noise_backend="torch-mps").noise_backend == "torch-mps"
    with pytest.raises(ValueError, match="noise_backend: expected one of"):
        LTX2Request(prompt="test", noise_backend="future")  # type: ignore[arg-type]


def test_distilled_request_validates_vae_decode_dtype_override() -> None:
    assert LTX2Request(prompt="test", vae_decode_dtype="bfloat16").vae_decode_dtype == ("bfloat16")
    assert LTX2Request(prompt="test", vae_decode_dtype="float32").vae_decode_dtype == "float32"
    assert LTX2Request(prompt="test", vae_decode_dtype="float16").vae_decode_dtype == "float16"
    with pytest.raises(ValueError, match="vae_decode_dtype: expected one of"):
        LTX2Request(prompt="test", vae_decode_dtype="float64")  # type: ignore[arg-type]


def test_distilled_request_validates_vae_cache_override() -> None:
    assert LTX2Request(prompt="test", vae_cache_limit_gb=16.0).vae_cache_limit_gb == 16.0
    with pytest.raises(ValueError, match="vae_cache_limit_gb"):
        LTX2Request(prompt="test", vae_cache_limit_gb=-1.0)


def test_vae_sharpening_defaults_off_and_validates_its_range() -> None:
    request = LTX2Request(prompt="test")
    assert request.vae_sharpen == 0.0
    assert request.vae_sharpen_radius == 2.0
    assert request.context_sharpening() is None
    sharpened = LTX2Request(prompt="test", vae_sharpen=2.5, vae_sharpen_radius=3.0)
    assert sharpened.context_sharpening() == ContextSharpening(amount=2.5, radius=3.0)
    for name, value in (
        ("vae_sharpen", -1.0),
        ("vae_sharpen", 10.5),
        ("vae_sharpen_radius", 0.0),
        ("vae_sharpen_radius", 8.5),
    ):
        with pytest.raises(ValueError, match=name):
            LTX2Request(prompt="test", **{name: value})


def test_duration_rounds_up_to_a_valid_frame_count() -> None:
    assert LTX2Request(prompt="test", duration=20.0).frames == 481
    assert LTX2Request(prompt="test", duration=30.0).frames == 721
    assert LTX2Request(prompt="test", duration=1.05, fps=24.0).frames == 33


def test_omitted_frame_count_is_reserved_for_auto_duration() -> None:
    assert LTX2Request(prompt="test", frames=None).frames is None


@pytest.mark.parametrize("generated_keyframes", [-1, True, 1.5])
def test_generated_keyframe_count_must_be_a_nonnegative_integer(
    generated_keyframes: object,
) -> None:
    with pytest.raises(ValueError, match="generated_keyframes"):
        LTX2Request(prompt="test", generated_keyframes=generated_keyframes)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan, math.inf])
def test_duration_must_be_finite_and_positive(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        LTX2Request(prompt="test", duration=duration)


def test_text_conditioning_can_replace_a_prompt(tmp_path: Path) -> None:
    sidecar = tmp_path / "conditioning.safetensors"
    sidecar.touch()
    LTX2Request(text_conditioning=sidecar).validate_for_generation()
    with pytest.raises(ValueError, match="prompt or text_conditioning"):
        LTX2Request().validate_for_generation()


def test_vae_tiling_resolves_single_and_custom_plans() -> None:
    single = LTX2VideoVAETilingConfig(mode="single").to_runtime_config()
    assert single is not None
    assert single.temporal_config is None
    assert single.spatial_config is None

    custom = LTX2VideoVAETilingConfig(
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
    default_custom = LTX2VideoVAETilingConfig(mode="custom").to_runtime_config()
    assert default_custom is not None
    assert default_custom.temporal_config is not None
    assert default_custom.temporal_config.chunk_size_in_frames == 256
    with pytest.raises(ValueError, match="needs a temporal or spatial"):
        LTX2VideoVAETilingConfig(
            mode="custom",
            temporal_tile_frames=0,
            spatial_tile_pixels=0,
        ).to_runtime_config()
    with pytest.raises(ValueError, match="overlap must be smaller"):
        LTX2VideoVAETilingConfig(
            mode="custom",
            temporal_overlap_frames=256,
        ).to_runtime_config()


def test_image_config_validates_strength_and_frame_bounds() -> None:
    image = LTX2ImageConditioningConfig(Path("frame.png"), frame_index=8, strength=0.9)
    assert LTX2Request(prompt="test", images=(image,)).images == (image,)
    with pytest.raises(ValueError, match="between 0 and 1"):
        LTX2ImageConditioningConfig(Path("frame.png"), strength=1.1)
    with pytest.raises(ValueError, match="outside"):
        LTX2Request(
            prompt="test",
            frames=9,
            images=(LTX2ImageConditioningConfig(Path("frame.png"), frame_index=9),),
        ).validate_for_generation()


def test_first_frame_mode_defaults_to_anchor_and_owns_its_vocabulary() -> None:
    from kinomlx.config import ConfigError, dataclass_from_mapping
    from kinomlx.config_records import replace_config

    assert LTX2Request(prompt="test").first_frame_mode == "anchor"
    assert LTX2Request(prompt="test", first_frame_mode="guide").first_frame_mode == "guide"
    with pytest.raises(ValueError, match="first_frame_mode: expected one of"):
        LTX2Request(prompt="test", first_frame_mode="hold")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="first_frame_mode: expected one of"):
        replace_config(LTX2Request(prompt="test"), first_frame_mode="hold")
    with pytest.raises(ConfigError, match="first_frame_mode: expected one of"):
        dataclass_from_mapping(
            LTX2Request,
            {"prompt": "test", "first_frame_mode": "hold"},
            table="generate",
        )


def test_first_frame_mode_does_not_depend_on_carrying_a_frame_zero_condition() -> None:
    """The mode selects how frame 0 would be applied; it never refuses a run."""
    LTX2Request(prompt="test", frames=17, first_frame_mode="guide").validate_for_generation()
    last_only = LTX2Request(
        prompt="test",
        frames=17,
        first_frame_mode="guide",
        images=(LTX2ImageConditioningConfig(Path("last.png"), frame_index=16),),
    )
    assert last_only.first_frame_mode == "guide"
    assert [image.frame_index for image in last_only.images] == ["16"]


def test_image_stage_strengths_are_optional_overrides_of_the_common_strength() -> None:
    inherited = LTX2ImageConditioningConfig(Path("frame.png"), strength=0.8)
    assert (inherited.stage1_strength, inherited.stage2_strength) == (None, None)
    assert [inherited.strength_for_stage(stage) for stage in (1, 2)] == [0.8, 0.8]

    split = LTX2ImageConditioningConfig(Path("frame.png"), strength=0.8, stage2_strength=0.5)
    assert [split.strength_for_stage(stage) for stage in (1, 2)] == [0.8, 0.5]
    assert LTX2ImageConditioningConfig(
        Path("frame.png"), stage1_strength=0.0, stage2_strength=1.0
    ).strength_for_stage(1) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        LTX2ImageConditioningConfig(Path("frame.png"), stage1_strength=2.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        LTX2ImageConditioningConfig(Path("frame.png"), stage2_strength=-0.5)
    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        split.strength_for_stage(3)


def test_image_stage_profiles_project_what_each_stage_applies() -> None:
    request = LTX2Request(
        prompt="test",
        frames=17,
        images=(
            LTX2ImageConditioningConfig(Path("first.png"), strength=1.0, stage2_strength=0.6),
            LTX2ImageConditioningConfig(Path("last.png"), frame_index=16, strength=0.9),
        ),
    )
    assert request.image_stage1_profile == (
        (Path("first.png"), 0, 1.0),
        (Path("last.png"), 16, 0.9),
    )
    assert request.image_stage_profiles == (
        request.image_stage1_profile,
        ((Path("first.png"), 0, 0.6), (Path("last.png"), 16, 0.9)),
    )
    assert LTX2Request(prompt="test").image_stage_profiles == ((), ())


def test_image_conditions_keep_author_order_and_refuse_a_repeated_frame() -> None:
    first = LTX2ImageConditioningConfig(Path("last.png"), frame_index=16)
    second = LTX2ImageConditioningConfig(Path("first.png"))
    request = LTX2Request(prompt="test", frames=17, images=(first, second))
    assert request.images == (first, second)
    with pytest.raises(ValueError, match="image frame_index 0 is conditioned more than once"):
        LTX2Request(
            prompt="test",
            frames=17,
            images=(
                LTX2ImageConditioningConfig(Path("a.png")),
                LTX2ImageConditioningConfig(Path("b.png"), frame_index=0),
            ),
        ).validate_for_generation()


def test_adapter_records_keep_independent_stage_values() -> None:
    config = LTX2Request(
        prompt="test",
        adapters=(
            LTX2AdapterConfig(
                path=Path("style.safetensors"),
                strength=0.8,
                stage1_strength=0.25,
                stage2_strength=1.0,
                exclude=("audio", "cross"),
            ),
            LTX2AdapterConfig(
                path=Path("motion.safetensors"),
                strength=0.8,
                stage1_strength=0.5,
                stage2_strength=1.0,
                exclude=("video",),
            ),
        ),
    )
    resolved = config.resolved_adapters()
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


def test_retired_adapter_arrays_and_unknown_exclusion_fail_before_loading() -> None:
    with pytest.raises(TypeError, match="adapter_paths"):
        LTX2Request(prompt="test", adapter_paths=(Path("a"),))
    with pytest.raises(ValueError, match="exclude"):
        LTX2Request(prompt="test", adapters=({"path": "a", "exclude": ["sideways"]},))


def test_adapter_strengths_are_normalized_and_rejected_at_request_construction() -> None:
    request = LTX2Request(
        prompt="test",
        adapters=(LTX2AdapterConfig(path=Path("adapter"), strength=1),),
    )
    assert type(request.resolved_adapters()[0].strength) is float

    broad = LTX2Request(
        prompt="test",
        adapters=(LTX2AdapterConfig(path=Path("adapter"), strength=3.0, stage1_strength=-3.0),),
    )
    assert broad.resolved_adapters()[0].strength == 3.0
    assert broad.resolved_adapters()[0].stage_1_strength == -3.0

    with pytest.raises(ValueError, match="strength.*expected float"):
        LTX2Request(
            prompt="test",
            adapters=(LTX2AdapterConfig(path=Path("adapter"), strength="not-a-number"),),
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_adapter_stage_strengths_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="stage1_strength.*finite"):
        LTX2Request(
            prompt="test",
            adapters=(LTX2AdapterConfig(path=Path("adapter"), stage1_strength=value),),
        ).resolved_adapters()


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
