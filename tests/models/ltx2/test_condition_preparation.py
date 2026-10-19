"""Raw-to-encoded condition product and ownership contracts."""

import gc
import weakref
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.ltx2.conditioning.preparation as preparation
from kinomlx.models.ltx2.conditioning import (
    EncodedCondition,
    HDRReferenceConditionSource,
    ImageConditionSource,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoConditionByReferenceLatent,
    prepare_conditions,
)
from kinomlx.models.ltx2.state import (
    apply_encoded_conditions,
    create_video_latent_tools,
    init_video_latent_state,
)
from kinomlx.models.ltx2.vocabularies import LTX2FirstFrameMode
from kinomlx.types import VideoLatentShape, VideoPixelShape


class _Encoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, video, *, reporter=None):
        del video, reporter
        self.calls += 1
        return mx.ones((1, 128, 1, 2, 2))


def _capabilities(*families: str) -> SimpleNamespace:
    return SimpleNamespace(condition_families=families)


def test_raw_image_source_is_frozen_and_owns_no_model_or_tensor() -> None:
    source = ImageConditionSource(Path("image.png"), frame_index=4, strength=0.75)
    assert is_dataclass(source)
    assert type(source).__dataclass_params__.frozen
    assert [field.name for field in fields(source)] == [
        "path",
        "frame_index",
        "strength",
        "stage1_strength",
        "stage2_strength",
        "hdr_authoring",
    ]
    assert source.family == "keyframe"
    assert not any(isinstance(value, mx.array) for value in vars(source).values())


def test_raw_image_source_holds_one_resolved_absolute_frame() -> None:
    """Named positions resolve in the request layer; a source carries an integer."""
    assert ImageConditionSource(Path("image.png")).frame_index == 0
    assert ImageConditionSource(Path("image.png"), frame_index=16).frame_index == 16
    with pytest.raises(ValueError, match="condition frame_index must be non-negative"):
        ImageConditionSource(Path("image.png"), frame_index=-1)


def test_raw_image_source_resolves_and_bounds_its_stage_strengths() -> None:
    inherited = ImageConditionSource(Path("image.png"), strength=0.8)
    assert [inherited.strength_for_stage(stage) for stage in (1, 2)] == [0.8, 0.8]
    split = ImageConditionSource(Path("image.png"), strength=0.8, stage2_strength=0.5)
    assert [split.strength_for_stage(stage) for stage in (1, 2)] == [0.8, 0.5]
    with pytest.raises(ValueError, match="stage 1 strength must be between 0 and 1"):
        ImageConditionSource(Path("image.png"), stage1_strength=1.5)
    with pytest.raises(ValueError, match="stage 2 strength must be between 0 and 1"):
        ImageConditionSource(Path("image.png"), stage2_strength=-0.1)
    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        split.strength_for_stage(0)


def test_raw_hdr_reference_source_is_frozen_and_owns_no_model_or_tensor() -> None:
    source = HDRReferenceConditionSource(Path("reference.mp4"), strength=0.8)
    assert is_dataclass(source)
    assert type(source).__dataclass_params__.frozen
    assert [field.name for field in fields(source)] == ["path", "strength"]
    assert source.family == "hdr-reference"
    assert not any(isinstance(value, mx.array) for value in vars(source).values())


@pytest.mark.parametrize(
    ("frame_index", "expected_type"),
    [(0, VideoConditionByLatentIndex), (4, VideoConditionByKeyframeIndex)],
)
def test_prepare_conditions_returns_only_encoded_products_and_drops_encoder(
    frame_index,
    expected_type,
    monkeypatch,
) -> None:
    def encode_image(_path, encoder, *, width, height, compute_dtype, reporter, hdr_authoring):
        assert hdr_authoring is None
        del width, height, compute_dtype, reporter
        return encoder(mx.zeros((1, 3, 1, 64, 64)))

    monkeypatch.setattr(preparation, "encode_image", encode_image)
    encoder = _Encoder()
    encoder_ref = weakref.ref(encoder)
    products = prepare_conditions(
        (ImageConditionSource(Path("image.png"), frame_index=frame_index),),
        VideoPixelShape(1, 9, 64, 64),
        encoder,
        _capabilities("image", "keyframe"),
        stage=1,
        compute_dtype=mx.bfloat16,
    )
    assert len(products) == 1
    assert isinstance(products[0], expected_type)
    assert encoder.calls == 1
    del encoder
    gc.collect()
    assert encoder_ref() is None


def test_first_and_last_frame_conditions_append_one_keyframe_at_its_own_time(
    monkeypatch,
) -> None:
    def encode_image(_path, encoder, *, width, height, compute_dtype, reporter, hdr_authoring):
        del width, height, compute_dtype, reporter, hdr_authoring
        return encoder(mx.zeros((1, 3, 1, 64, 64)))

    monkeypatch.setattr(preparation, "encode_image", encode_image)
    fps = 24.0
    products = prepare_conditions(
        (
            ImageConditionSource(Path("first.png"), frame_index=0),
            ImageConditionSource(Path("last.png"), frame_index=8),
        ),
        VideoPixelShape(1, 9, 64, 64),
        _Encoder(),
        _capabilities("image", "keyframe"),
        stage=1,
        compute_dtype=mx.float32,
    )
    assert [type(product) for product in products] == [
        VideoConditionByLatentIndex,
        VideoConditionByKeyframeIndex,
    ]

    target = VideoLatentShape(1, 128, 2, 2, 2)
    tools = create_video_latent_tools(target, fps=fps)
    state = init_video_latent_state(tools, dtype=mx.float32)
    conditioned = apply_encoded_conditions(state, products, tools)

    keyframe_tokens = target.height * target.width
    assert state.latent.shape[1] == target.frames * keyframe_tokens
    assert conditioned.latent.shape[1] == state.latent.shape[1] + keyframe_tokens
    appended = conditioned.positions[:, 0, -keyframe_tokens:, :]
    assert mx.allclose(appended[..., 0], mx.full(appended[..., 0].shape, 8.0 / fps)).item()
    assert mx.allclose(appended[..., 1], mx.full(appended[..., 1].shape, 9.0 / fps)).item()


def test_condition_sources_keep_author_order_when_the_last_frame_is_listed_first(
    monkeypatch,
) -> None:
    def encode_image(path, encoder, *, width, height, compute_dtype, reporter, hdr_authoring):
        del width, height, compute_dtype, reporter, hdr_authoring
        encoded.append(Path(path))
        return encoder(mx.zeros((1, 3, 1, 64, 64)))

    encoded: list[Path] = []
    monkeypatch.setattr(preparation, "encode_image", encode_image)
    products = prepare_conditions(
        (
            ImageConditionSource(Path("last.png"), frame_index=8),
            ImageConditionSource(Path("first.png"), frame_index=0),
        ),
        VideoPixelShape(1, 9, 64, 64),
        _Encoder(),
        _capabilities("image", "keyframe"),
        stage=1,
        compute_dtype=mx.float32,
    )
    assert encoded == [Path("last.png"), Path("first.png")]
    assert [type(product) for product in products] == [
        VideoConditionByKeyframeIndex,
        VideoConditionByLatentIndex,
    ]


def test_each_stage_builds_the_same_conditions_at_its_own_strength(monkeypatch) -> None:
    def encode_image(_path, encoder, *, width, height, compute_dtype, reporter, hdr_authoring):
        del width, height, compute_dtype, reporter, hdr_authoring
        return encoder(mx.zeros((1, 3, 1, 64, 64)))

    monkeypatch.setattr(preparation, "encode_image", encode_image)
    sources = (
        # An override on each stage, and a common strength stage 1 inherits.
        ImageConditionSource(Path("first.png"), strength=1.0, stage2_strength=0.6),
        ImageConditionSource(Path("last.png"), frame_index=8, strength=0.9),
    )

    def strengths(stage: int) -> list[float]:
        products = prepare_conditions(
            sources,
            VideoPixelShape(1, 9, 64, 64),
            _Encoder(),
            _capabilities("image", "keyframe"),
            stage=stage,
            compute_dtype=mx.float32,
        )
        assert [type(product) for product in products] == [
            VideoConditionByLatentIndex,
            VideoConditionByKeyframeIndex,
        ]
        return [product.strength for product in products]

    assert strengths(1) == [1.0, 0.9]
    assert strengths(2) == [0.6, 0.9]


def test_the_frame_zero_mode_selects_between_replacing_and_appending(monkeypatch) -> None:
    """Anchor holds latent frame 0 clean; guide leaves it denoised and appends tokens."""

    def encode_image(_path, encoder, *, width, height, compute_dtype, reporter, hdr_authoring):
        del width, height, compute_dtype, reporter, hdr_authoring
        return encoder(mx.zeros((1, 3, 1, 64, 64)))

    monkeypatch.setattr(preparation, "encode_image", encode_image)
    fps = 24.0
    strength = 0.75
    target = VideoLatentShape(1, 128, 2, 2, 2)
    slot_tokens = target.height * target.width
    canvas_tokens = target.frames * slot_tokens

    def conditioned(mode: LTX2FirstFrameMode):
        products = prepare_conditions(
            (ImageConditionSource(Path("first.png"), strength=strength),),
            VideoPixelShape(1, 9, 64, 64),
            _Encoder(),
            _capabilities("image", "keyframe"),
            stage=1,
            compute_dtype=mx.float32,
            first_frame_mode=mode,
        )
        tools = create_video_latent_tools(target, fps=fps)
        state = init_video_latent_state(tools, dtype=mx.float32)
        assert state.latent.shape[1] == canvas_tokens
        return products, apply_encoded_conditions(state, products, tools)

    anchor_products, anchored = conditioned(LTX2FirstFrameMode.ANCHOR)
    assert [type(product) for product in anchor_products] == [VideoConditionByLatentIndex]
    assert anchored.latent.shape[1] == canvas_tokens
    assert not anchored.uniform_mask
    anchor_mask = anchored.denoise_mask[0, :, 0]
    assert mx.allclose(anchor_mask[:slot_tokens], mx.full((slot_tokens,), 1.0 - strength)).item()
    assert mx.allclose(anchor_mask[slot_tokens:], mx.ones((canvas_tokens - slot_tokens,))).item()

    guide_products, guided = conditioned(LTX2FirstFrameMode.GUIDE)
    assert [type(product) for product in guide_products] == [VideoConditionByKeyframeIndex]
    assert guided.latent.shape[1] == canvas_tokens + slot_tokens
    assert not guided.uniform_mask
    guide_mask = guided.denoise_mask[0, :, 0]
    # The canvas keeps denoising, including the first latent frame.
    assert mx.allclose(guide_mask[:canvas_tokens], mx.ones((canvas_tokens,))).item()
    assert mx.allclose(guide_mask[canvas_tokens:], mx.full((slot_tokens,), 1.0 - strength)).item()
    # A frame-0 keyframe keeps the causal fix, so it starts at pixel frame 0 and
    # spans the one frame latent 0 stands for.
    appended = guided.positions[:, 0, canvas_tokens:, :]
    assert mx.allclose(appended[..., 0], mx.zeros(appended[..., 0].shape)).item()
    assert mx.allclose(appended[..., 1], mx.full(appended[..., 1].shape, 1.0 / fps)).item()


def test_prepare_conditions_refuses_a_stage_the_recipe_does_not_have(monkeypatch) -> None:
    monkeypatch.setattr(
        preparation,
        "encode_image",
        lambda *_args, **_kwargs: pytest.fail("an unknown stage must not encode"),
    )
    with pytest.raises(ValueError, match="stage must be 1 or 2, got 3"):
        prepare_conditions(
            (ImageConditionSource(Path("image.png")),),
            VideoPixelShape(1, 9, 64, 64),
            _Encoder(),
            _capabilities("image", "keyframe"),
            stage=3,
            compute_dtype=mx.float32,
        )


def test_encoded_conditions_apply_in_declared_order() -> None:
    order = []

    class _Condition:
        def __init__(self, label: str) -> None:
            self.label = label

        def apply_to(self, state, tools):
            del tools
            order.append(self.label)
            return state

    first: EncodedCondition = _Condition("first")
    second: EncodedCondition = _Condition("second")
    tools = create_video_latent_tools(VideoLatentShape(1, 2, 2, 2, 2), fps=24.0)
    state = init_video_latent_state(tools, dtype=mx.float16)

    result = apply_encoded_conditions(state, (first, second), tools)

    assert result is state
    assert order == ["first", "second"]


def test_prepare_conditions_rejects_unsupported_family_before_encoding(monkeypatch) -> None:
    monkeypatch.setattr(
        preparation,
        "encode_image",
        lambda *_args, **_kwargs: pytest.fail("unsupported conditions must not encode"),
    )
    with pytest.raises(ValueError, match="keyframe conditioning"):
        prepare_conditions(
            (ImageConditionSource(Path("image.png"), frame_index=4),),
            VideoPixelShape(1, 9, 64, 64),
            _Encoder(),
            _capabilities("image"),
            stage=1,
            compute_dtype=mx.float32,
        )


def test_prepare_conditions_encodes_typed_hdr_reference_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Path, int, int, int, mx.Dtype]] = []

    def encode_reference_video(
        path,
        _encoder,
        *,
        width,
        height,
        frames,
        compute_dtype,
        reporter,
    ):
        del reporter
        seen.append((Path(path), width, height, frames, compute_dtype))
        return mx.ones((1, 128, 3, 2, 2), dtype=compute_dtype)

    monkeypatch.setattr(preparation, "encode_reference_video", encode_reference_video)
    products = prepare_conditions(
        (HDRReferenceConditionSource(Path("reference.mp4"), strength=0.7),),
        VideoPixelShape(1, 9, 64, 64),
        _Encoder(),
        SimpleNamespace(model_generation="2.3", condition_families=("text",)),
        stage=2,
        compute_dtype=mx.bfloat16,
    )

    assert len(products) == 1
    assert isinstance(products[0], VideoConditionByReferenceLatent)
    assert products[0].strength == 0.7
    assert seen == [(Path("reference.mp4"), 64, 64, 9, mx.bfloat16)]
