"""Vision grid, source admission, feature geometry, and multimodal contracts."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinomlx.models.krea2.grounding as grounding
from kinomlx.models.krea2.image_processing import grounding_patches
from kinomlx.models.krea2.text_encoder.model import Qwen3VLTextModel
from kinomlx.models.krea2.text_encoder.multimodal import Qwen3VLImageSpan, multimodal_positions
from kinomlx.models.krea2.text_encoder.vision import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionModel,
    Qwen3VLVisualFeatures,
    vision_grid_positions,
)
from kinomlx.models.krea2.text_encoder.vision_loading import (
    load_qwen3_vl_vision_weights,
    vision_weight_targets,
)

from .test_text_encoder import _config, _populate


def _features(grid: tuple[int, int], width: int = 32) -> Qwen3VLVisualFeatures:
    values = mx.ones((grid[0] * grid[1], width))
    return Qwen3VLVisualFeatures(values, (values, values, values), grid)


def test_spatial_merge_grid_keeps_each_two_by_two_patch_group_contiguous() -> None:
    expected = mx.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]])
    assert mx.array_equal(vision_grid_positions(2, 4, 2), expected)


def test_two_image_positions_resume_text_after_largest_spatial_axis() -> None:
    spans = (Qwen3VLImageSpan(2, _features((2, 2))), Qwen3VLImageSpan(7, _features((1, 4))))
    expected = mx.array(
        [
            [0, 1, 2, 2, 2, 2, 4, 5, 5, 5, 5, 9, 10],
            [0, 1, 2, 2, 3, 3, 4, 5, 5, 5, 5, 9, 10],
            [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]
    )[:, None]
    assert mx.array_equal(multimodal_positions(13, spans), expected)


def test_overlapping_image_spans_are_rejected() -> None:
    spans = (Qwen3VLImageSpan(1, _features((2, 2))), Qwen3VLImageSpan(3, _features((2, 2))))
    with pytest.raises(ValueError, match="disjoint"):
        multimodal_positions(10, spans)


def test_visual_features_replace_placeholder_embeddings() -> None:
    model = Qwen3VLTextModel(_config(), (1, 3))
    _populate(model)
    spans = (Qwen3VLImageSpan(1, _features((2, 2))),)
    positions = multimodal_positions(6, spans)
    first = mx.array([[0, 1, 2, 3, 4, 5]])
    second = mx.array([[0, 20, 21, 22, 23, 5]])
    outputs = [
        model(
            ids,
            attention_mask=mx.ones((1, 6), dtype=mx.bool_),
            position_ids=positions,
            image_spans=spans,
        ).hidden_states
        for ids in (first, second)
    ]
    assert mx.array_equal(*outputs)


def test_tiny_vision_source_binding_and_output_shapes(tmp_path: Path) -> None:
    config = Qwen3VLVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        depth=2,
        num_heads=2,
        out_hidden_size=8,
        patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=(0, 1),
    )
    model = Qwen3VLVisionModel(config)
    targets = vision_weight_targets(model)
    weights = {key: mx.zeros(target.shape) for key, target in targets.items()}
    source = tmp_path / "vision.safetensors"
    mx.save_safetensors(str(source), weights)
    assert load_qwen3_vl_vision_weights(model, source, dtype=mx.float32) == len(targets)
    result = model(mx.zeros((24, 24)), grid=(4, 6))
    assert result.grid == (2, 3)
    assert result.embeddings.shape == (6, 8)
    assert all(value.shape == (6, 8) for value in result.deepstack)
    assert mx.array_equal(result.embeddings, mx.zeros((6, 8)))


def test_production_patches_keep_bf16_weight_vision_computation_in_fp32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Qwen3VLVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        depth=2,
        num_heads=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        deepstack_visual_indexes=(0, 1),
    )
    model = Qwen3VLVisionModel(config)
    targets = vision_weight_targets(model)
    mx.random.seed(20260905)
    weights = {
        key: mx.random.normal(target.shape).astype(mx.bfloat16) * 0.03
        for key, target in targets.items()
    }
    source = tmp_path / "vision.safetensors"
    mx.save_safetensors(str(source), weights)
    load_qwen3_vl_vision_weights(model, source)
    assert model.patch_projection.weight.dtype == mx.bfloat16
    pixels = mx.random.uniform(shape=(1, 64, 96, 3))
    # This is the grounding boundary: do not insert a cast to the weight dtype.
    patches, grid = grounding_patches(pixels)
    assert patches.dtype == mx.float32
    actual = model(patches, grid=grid)
    assert actual.embeddings.dtype == mx.float32
    assert all(value.dtype == mx.float32 for value in actual.deepstack)
    widened = Qwen3VLVisionModel(config)
    load_qwen3_vl_vision_weights(widened, source, dtype=mx.float32)
    expected = widened(patches, grid=grid)
    assert mx.array_equal(actual.embeddings, expected.embeddings)
    for a, b in zip(actual.deepstack, expected.deepstack, strict=True):
        assert mx.array_equal(a, b)

    # Also exercise the actual caller so a new input cast there fails this contract.
    captured = []

    class RecordingVision(Qwen3VLVisionModel):
        def __call__(self, patches, *, grid):
            assert patches.dtype == mx.float32
            result = super().__call__(patches, grid=grid)
            captured.append(result)
            return result

    class ReachedLanguage(Exception):
        pass

    def stop_before_language(_path):
        raise ReachedLanguage

    monkeypatch.setattr(grounding, "Qwen3VLVisionModel", lambda _config: RecordingVision(config))
    monkeypatch.setattr(grounding, "load_component_tokenizer", stop_before_language)
    resources = SimpleNamespace(
        text_encoder=SimpleNamespace(bindings=(SimpleNamespace(source_file=source),)),
        tokenizer=SimpleNamespace(source_path=tmp_path / "unused-tokenizer"),
    )
    with pytest.raises(ReachedLanguage):
        grounding.GroundedKrea2TextConditioner((pixels,))("test", resources)
    assert len(captured) == 1
    assert mx.array_equal(captured[0].embeddings, expected.embeddings)
    for a, b in zip(captured[0].deepstack, expected.deepstack, strict=True):
        assert mx.array_equal(a, b)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_fp32_vision_features_cast_at_language_injection_boundary(dtype: mx.Dtype) -> None:
    model = Qwen3VLTextModel(_config(), (1, 3))
    _populate(model)
    model.apply(lambda value: value.astype(dtype))
    mx.random.seed(20260905)
    feature = Qwen3VLVisualFeatures(
        mx.random.normal((4, 32)),
        tuple(mx.random.normal((4, 32)) for _ in range(3)),
        (2, 2),
    )
    rounded = Qwen3VLVisualFeatures(
        feature.embeddings.astype(dtype),
        tuple(value.astype(dtype) for value in feature.deepstack),
        feature.grid,
    )
    outputs = []
    for item in (feature, rounded):
        spans = (Qwen3VLImageSpan(1, item),)
        outputs.append(
            model(
                mx.array([[0, 1, 2, 3, 4, 5]]),
                attention_mask=mx.ones((1, 6), dtype=mx.bool_),
                position_ids=multimodal_positions(6, spans),
                image_spans=spans,
            ).hidden_states
        )
    assert outputs[0].dtype == dtype
    assert mx.array_equal(*outputs)


@pytest.mark.parametrize("grid", [(1, 4), (4, 3), (0, 4)])
def test_vision_rejects_unmergeable_grid(grid: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="align"):
        vision_grid_positions(*grid, 2)


def test_grounded_native_causal_mask_matches_explicit_triangle(monkeypatch) -> None:
    model = Qwen3VLTextModel(_config(), (1, 3))
    _populate(model)
    spans = (Qwen3VLImageSpan(1, _features((2, 2))),)
    positions = multimodal_positions(6, spans)
    ids = mx.array([[0, 1, 2, 3, 4, 5]])
    mask = mx.ones((1, 6), dtype=mx.bool_)
    native = model(ids, attention_mask=mask, position_ids=positions, image_spans=spans)
    mx.eval(native.hidden_states)
    original = mx.fast.scaled_dot_product_attention
    seen = []

    def explicit(q, k, v, *, scale, mask=None, **kwargs):
        assert mask == "causal"
        seen.append(mask)
        triangle = mx.tril(mx.ones((q.shape[2], k.shape[2]), dtype=mx.bool_))
        return original(q, k, v, scale=scale, mask=triangle, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", explicit)
    dense = model(ids, attention_mask=mask, position_ids=positions, image_spans=spans)
    assert seen
    assert mx.allclose(native.hidden_states, dense.hidden_states, atol=1e-6).item()


@pytest.mark.parametrize("system", ["", "   ", "Focus on facial identity."])
def test_grounded_system_template_tracks_prefix_and_image_spans(system) -> None:
    from kinomlx.models.krea2.grounding import grounded_tokens
    from kinomlx.models.krea2.tokenizer.template import KREA_PREFIX

    class Tokenizer:
        def encode(self, text):
            return tuple(text.encode())

    ids, spans, prefix = grounded_tokens(
        Tokenizer(), "Edit.", (_features((1, 2)), _features((1, 1))), system_prompt=system
    )
    expected = (
        ("<|im_start|>system\n" + system.strip() + "<|im_end|>\n<|im_start|>user\n")
        if system.strip()
        else KREA_PREFIX
    )
    assert ids[0, :prefix].tolist() == list(expected.encode())
    assert spans[0].start == prefix + len("<|vision_start|>")
    assert ids[0, spans[0].start : spans[0].start + 2].tolist() == [151655, 151655]
    assert len(spans) == 2
