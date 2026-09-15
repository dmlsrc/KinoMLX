"""Synthetic Krea MMDiT graph, mask, padding, and precision contracts."""

import math
from dataclasses import replace

import mlx.core as mx
import pytest

import kinomlx.models.krea2.transformer.loading as loading_module
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.models.krea2.transformer.model import (
    Krea2Attention,
    Krea2RMSNorm,
    Krea2RotaryEmbedding,
    Krea2Transformer,
    apply_rotary_embedding,
    split_six_way_modulation,
)

from ._transformer_fixtures import populate_transformer, tiny_transformer_config


def _inputs(dtype=mx.float16) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(42)
    image = mx.random.uniform(
        low=-0.2,
        high=0.2,
        shape=(1, 2, 4),
    ).astype(dtype)
    context = mx.random.uniform(
        low=-0.2,
        high=0.2,
        shape=(1, 3, 3, 16),
    ).astype(dtype)
    timestep = mx.array((0.75,), dtype=dtype)
    positions = mx.array(
        (
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        dtype=mx.float32,
    )
    mask = mx.array(((True, False, True, True, True),), dtype=mx.bool_)
    return image, context, timestep, positions, mask


@pytest.mark.requires_metal
def test_transformer_compacts_text_without_padding_and_preserves_hot_dtypes(monkeypatch) -> None:
    config = tiny_transformer_config()
    model = Krea2Transformer(config)
    populate_transformer(model, loading_module.transformer_weight_targets(model))
    image, context, timestep, positions, mask = _inputs()
    observed: dict[str, mx.array] = {}
    masks = []
    original = Krea2Attention.__call__

    def attend(self, hidden_states, *, attention_mask=None, rotary=None, reference_attention=None):
        masks.append(attention_mask)
        return original(
            self,
            hidden_states,
            attention_mask=attention_mask,
            rotary=rotary,
            reference_attention=reference_attention,
        )

    monkeypatch.setattr(Krea2Attention, "__call__", attend)

    result = model(
        image,
        encoder_hidden_states=context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
        observer=lambda name, value: observed.__setitem__(name, value),
    )
    mx.eval(result.sample, *observed.values())

    assert result.sample.shape == image.shape
    assert result.sample.dtype == mx.float16
    assert result.combined_sequence_length == 4
    assert result.padding_tokens == 0
    assert result.padded_sequence_length == 4
    assert mx.all(mx.isfinite(result.sample)).item()
    assert set(observed) == {
        "time_embed",
        "time_mod",
        "text_fusion",
        "text_projection",
        "image_projection",
        "combined_input",
        "rotary_cos",
        "rotary_sin",
        "block_00",
        "block_01",
        "output",
    }
    for name, value in observed.items():
        expected = mx.float32 if name.startswith("rotary_") else mx.float16
        assert value.dtype == expected, name
    assert observed["combined_input"].shape == (1, 4, config.hidden_dim)
    assert observed["block_01"].shape == (1, 4, config.hidden_dim)
    assert masks
    assert all(mask is None for mask in masks)


@pytest.mark.requires_metal
def test_reference_mode_preserves_padding_mask_and_original_weight_layout() -> None:
    config = tiny_transformer_config()
    model = Krea2Transformer(config, reference_denoise=True)
    populate_transformer(model, loading_module.transformer_weight_targets(model), dtype=mx.bfloat16)
    image, context, timestep, positions, mask = _inputs(mx.bfloat16)
    observed = {}
    output = model(
        image,
        encoder_hidden_states=context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
        observer=lambda name, value: observed.__setitem__(name, value),
    )
    mx.eval(output.sample)
    assert output.combined_sequence_length == 5
    assert output.padding_tokens == 251
    assert output.padded_sequence_length == 256
    assert observed["combined_input"].shape == (1, 256, config.hidden_dim)
    assert observed["text_fusion"].shape[1] == 3
    assert all(
        not isinstance(block.ff.down, PretransposedLinear) for block in model.transformer_blocks
    )
    assert mx.all(mx.isfinite(output.sample)).item()


def test_reference_model_rejects_fp16_before_using_weights() -> None:
    model = Krea2Transformer(tiny_transformer_config(), reference_denoise=True)
    image, context, timestep, positions, mask = _inputs()
    with pytest.raises(ValueError, match="padding can cause FP16 overflow"):
        model(
            image,
            encoder_hidden_states=context,
            timestep=timestep,
            position_ids=positions,
            attention_mask=mask,
        )


@pytest.mark.requires_metal
def test_mixed_length_batch_uses_unmasked_groups_and_restores_original_order(monkeypatch) -> None:
    model = Krea2Transformer(tiny_transformer_config())
    populate_transformer(model, loading_module.transformer_weight_targets(model), dtype=mx.float32)
    image, context, timestep, _, _ = _inputs(mx.float32)
    image = mx.concatenate([image, image + 0.1, image - 0.1])
    context = mx.concatenate([context, context + 0.1, context - 0.1])
    timestep = mx.broadcast_to(timestep, (3,))
    positions = mx.arange(45, dtype=mx.float32).reshape(3, 5, 3) / 10
    mask = mx.array(
        [
            [True, False, True, True, True],
            [True, True, True, True, True],
            [False, True, True, True, True],
        ]
    )
    shapes = []
    original = model._forward

    def forward(tokens, **kwargs):
        assert kwargs["attention_mask"] is None
        shapes.append((tokens.shape[0], kwargs["encoder_hidden_states"].shape[1]))
        return original(tokens, **kwargs)

    monkeypatch.setattr(model, "_forward", forward)
    batch = model(
        image,
        encoder_hidden_states=context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
    )
    mx.eval(batch.sample)
    assert shapes == [(2, 2), (1, 3)]
    assert batch.padding_tokens == 0
    for row in range(3):
        individual = model(
            image[row : row + 1],
            encoder_hidden_states=context[row : row + 1],
            timestep=timestep[row : row + 1],
            position_ids=positions[row : row + 1],
            attention_mask=mask[row : row + 1],
        )
        assert mx.allclose(
            batch.sample[row : row + 1], individual.sample, atol=1e-6, rtol=1e-5
        ).item()


@pytest.mark.requires_metal
def test_masked_text_content_cannot_change_valid_image_output() -> None:
    model = Krea2Transformer(tiny_transformer_config())
    populate_transformer(model, loading_module.transformer_weight_targets(model))
    image, context, timestep, positions, mask = _inputs()
    masked_row = mx.array((False, True, False), dtype=mx.bool_)[None, :, None, None]
    changed_context = mx.where(masked_row, context + 5.0, context)

    baseline = model(
        image,
        encoder_hidden_states=context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
    ).sample
    changed = model(
        image,
        encoder_hidden_states=changed_context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
    ).sample
    mx.eval(baseline, changed)

    assert mx.array_equal(baseline, changed).item()


@pytest.mark.requires_metal
def test_exact_multiple_of_256_has_no_shape_padding() -> None:
    config = tiny_transformer_config()
    model = Krea2Transformer(config)
    populate_transformer(model, loading_module.transformer_weight_targets(model))
    image, context, timestep, _positions, _mask = _inputs()
    image = mx.broadcast_to(image[:, :1], (1, 253, config.in_channels))
    positions = mx.zeros((1, 256, 3), dtype=mx.float32)
    mask = mx.ones((1, 256), dtype=mx.bool_)

    result = model(
        image,
        encoder_hidden_states=context,
        timestep=timestep,
        position_ids=positions,
        attention_mask=mask,
    )
    mx.eval(result.sample)

    assert result.sample.shape == (1, 253, config.in_channels)
    assert result.combined_sequence_length == 256
    assert result.padding_tokens == 0
    assert result.padded_sequence_length == 256


@pytest.mark.requires_metal
@pytest.mark.parametrize("all_masked", [False, True])
def test_attention_explicitly_zeros_fully_masked_query_rows(all_masked) -> None:
    attention = Krea2Attention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        eps=1e-5,
    )
    identity = mx.eye(8, dtype=mx.float16)
    attention.to_q.weight = identity
    attention.to_k.weight = identity[:4]
    attention.to_v.weight = identity[:4]
    attention.to_gate.weight = mx.zeros((8, 8), dtype=mx.float16)
    attention.to_out.weight = identity
    attention.norm_q.weight = mx.zeros((4,), dtype=mx.float16)
    attention.norm_k.weight = mx.zeros((4,), dtype=mx.float16)
    value = mx.arange(24, dtype=mx.float16).reshape(1, 3, 8) / 16
    valid = mx.array(((not all_masked, False, not all_masked),), dtype=mx.bool_)
    if all_masked:
        # No valid input is consumed. Invalid masked values must not survive a
        # zero-times-NaN product at the query mask boundary.
        value = mx.full(value.shape, float("nan"), dtype=value.dtype)

    output = attention(value, attention_mask=valid)
    mx.eval(output)

    assert mx.array_equal(output[:, 1], mx.zeros((1, 8), dtype=mx.float16)).item()
    if all_masked:
        assert mx.array_equal(output, mx.zeros_like(output)).item()
    else:
        assert mx.any(output[:, 0] != 0).item()
        assert mx.any(output[:, 2] != 0).item()


def test_zero_centered_rms_norm_uses_fp32_opmath_and_returns_source_dtype() -> None:
    norm = Krea2RMSNorm(4, 1e-5)
    norm.weight = mx.array((0.0, 0.5, -0.25, 1.0), dtype=mx.float16)
    value = mx.array(((1.0, -2.0, 3.0, -4.0),), dtype=mx.float16)

    output = norm(value)
    widened = value.astype(mx.float32)
    expected = widened * mx.rsqrt(mx.mean(widened * widened, axis=-1, keepdims=True) + 1e-5)
    expected = expected * (norm.weight.astype(mx.float32) + 1.0)
    expected = expected.astype(mx.float16)

    assert output.dtype == mx.float16
    assert mx.array_equal(output, expected).item()


def test_rotary_embedding_uses_declared_axis_order_and_adjacent_pairs() -> None:
    rotary = Krea2RotaryEmbedding(theta=1000.0, axes_dims=(2, 2, 4))
    positions = mx.array((((1.0, 2.0, 3.0),),), dtype=mx.float32)

    cos, sin = rotary(positions)
    third_slow_angle = 3.0 / math.sqrt(1000.0)
    angles = (1.0, 1.0, 2.0, 2.0, 3.0, 3.0, third_slow_angle, third_slow_angle)
    expected_cos = mx.array([[tuple(math.cos(angle) for angle in angles)]], dtype=mx.float32)
    expected_sin = mx.array([[tuple(math.sin(angle) for angle in angles)]], dtype=mx.float32)

    assert mx.allclose(cos, expected_cos, atol=1e-6, rtol=0.0).item()
    assert mx.allclose(sin, expected_sin, atol=1e-6, rtol=0.0).item()

    query = mx.array(((((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),),),))
    key = query * 2
    rotated_query, rotated_key = apply_rotary_embedding(query, key, cos, sin)
    first_angle = 1.0
    assert rotated_query[0, 0, 0, 0].item() == pytest.approx(
        math.cos(first_angle) - 2 * math.sin(first_angle),
        abs=1e-6,
    )
    assert rotated_query[0, 0, 0, 1].item() == pytest.approx(
        math.sin(first_angle) + 2 * math.cos(first_angle),
        abs=1e-6,
    )
    assert mx.allclose(rotated_key, rotated_query * 2, atol=1e-6, rtol=0.0).item()


def test_six_way_modulation_chunk_order_is_stable() -> None:
    projected = mx.arange(24, dtype=mx.float32).reshape(1, 1, 24)
    table = mx.zeros((6, 4), dtype=mx.float32)

    chunks = split_six_way_modulation(projected, table)

    assert len(chunks) == 6
    for index, chunk in enumerate(chunks):
        assert mx.array_equal(chunk, projected[..., index * 4 : (index + 1) * 4]).item()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: replace(config, axes_dims_rope=(2, 2, 2)), "sum"),
        (lambda config: replace(config, axes_dims_rope=(1, 3, 4)), "even"),
        (lambda config: replace(config, num_key_value_heads=3), "query heads"),
        (lambda config: replace(config, text_hidden_dim=15), "text hidden"),
        (lambda config: replace(config, timestep_embed_dim=7), "timestep"),
    ],
)
def test_transformer_rejects_invalid_constructor_dimensions(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Krea2Transformer(mutate(tiny_transformer_config()))


def test_transformer_rejects_nonboolean_mask_and_masked_image_tokens() -> None:
    model = Krea2Transformer(tiny_transformer_config())
    image, context, timestep, positions, mask = _inputs()

    with pytest.raises(ValueError, match="boolean"):
        model(
            image,
            encoder_hidden_states=context,
            timestep=timestep,
            position_ids=positions,
            attention_mask=mask.astype(mx.int32),
        )

    invalid_image = mx.array(((True, False, True, True, False),), dtype=mx.bool_)
    with pytest.raises(ValueError, match="image token"):
        model(
            image,
            encoder_hidden_states=context,
            timestep=timestep,
            position_ids=positions,
            attention_mask=invalid_image,
        )
