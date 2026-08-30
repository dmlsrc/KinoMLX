"""Raw-to-encoded condition product and ownership contracts."""

from __future__ import annotations

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
        "hdr_authoring",
    ]
    assert source.family == "keyframe"
    assert not any(isinstance(value, mx.array) for value in vars(source).values())


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
    def encode_image(_path, encoder, *, width, height, compute_dtype, reporter):
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
        compute_dtype=mx.bfloat16,
    )
    assert len(products) == 1
    assert isinstance(products[0], expected_type)
    assert encoder.calls == 1
    del encoder
    gc.collect()
    assert encoder_ref() is None


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
        compute_dtype=mx.bfloat16,
    )

    assert len(products) == 1
    assert isinstance(products[0], VideoConditionByReferenceLatent)
    assert products[0].strength == 0.7
    assert seen == [(Path("reference.mp4"), 64, 64, 9, mx.bfloat16)]
