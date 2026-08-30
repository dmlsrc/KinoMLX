from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from kinomlx.models.ltx2.transformer import (
    apply_rotary_emb,
    precompute_freqs_cis,
)
from kinomlx.models.ltx2.transformer.attention import scaled_dot_product_attention

from ._synthetic import (
    build_shaped_adaln,
    build_shaped_attention,
    build_shaped_feed_forward,
)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_split_rope_preserves_flat_and_headed_input_dtype(dtype: mx.Dtype) -> None:
    positions = mx.array([[[[0.0, 1.0], [1.0, 2.0]]]])
    frequencies = precompute_freqs_cis(
        positions,
        dim=8,
        out_dtype=mx.float32,
        max_pos=(8,),
        use_middle_indices_grid=True,
        num_attention_heads=2,
        use_double_precision=True,
    )
    flat = mx.arange(16).reshape(1, 2, 8).astype(dtype)
    headed = flat.reshape(1, 2, 2, 4).transpose(0, 2, 1, 3)
    flat_output = apply_rotary_emb(flat, frequencies)
    headed_output = apply_rotary_emb(headed, frequencies)
    mx.eval(flat_output, headed_output)
    assert flat_output.dtype == dtype
    assert headed_output.dtype == dtype
    assert mx.allclose(
        flat_output,
        headed_output.transpose(0, 2, 1, 3).reshape(1, 2, 8),
    ).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_stock_sdpa_receives_selected_transformer_dtype(
    dtype: mx.Dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = mx.fast.scaled_dot_product_attention
    observed: list[mx.Dtype] = []

    def recording_sdpa(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        *,
        scale: float,
        mask: mx.array | None = None,
    ) -> mx.array:
        observed.append(query.dtype)
        return original(query, key, value, scale=scale, mask=mask)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", recording_sdpa)
    value = mx.arange(16).reshape(1, 2, 8).astype(dtype)
    output = scaled_dot_product_attention(
        value,
        value,
        value,
        heads=2,
        dim_head=4,
        compile_attention=False,
    )
    mx.eval(output)
    assert observed == [dtype]
    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_stock_compiled_attention_matches_original_uncompiled_path(dtype: mx.Dtype) -> None:
    mx.random.seed(19)
    query = mx.random.normal((1, 7, 16)).astype(dtype)
    key = mx.random.normal((1, 9, 16)).astype(dtype)
    value = mx.random.normal((1, 9, 16)).astype(dtype)
    original = scaled_dot_product_attention(
        query,
        key,
        value,
        heads=2,
        dim_head=8,
        use_steel_attention=False,
        compile_attention=False,
    )
    compiled = scaled_dot_product_attention(
        query,
        key,
        value,
        heads=2,
        dim_head=8,
        use_steel_attention=False,
        compile_attention=True,
    )
    mx.eval(original, compiled)
    assert compiled.dtype == dtype
    assert mx.array_equal(compiled, original).item()


def test_double_precision_rope_grid_matches_float64_reference() -> None:
    positions = mx.array([[[[0.0, 1.0]]]])
    cosine, sine = precompute_freqs_cis(
        positions,
        dim=8,
        out_dtype=mx.float32,
        theta=10000.0,
        max_pos=(20,),
        use_middle_indices_grid=True,
        num_attention_heads=2,
        use_double_precision=True,
    )
    grid = np.power(10000.0, np.linspace(0.0, 1.0, 4, dtype=np.float64))
    angles = (grid * np.pi / 2).astype(np.float32) * np.float32(2 * 0.5 / 20 - 1)
    expected_cos = mx.array(np.cos(angles).astype(np.float32)).reshape(1, 2, 1, 2)
    expected_sin = mx.array(np.sin(angles).astype(np.float32)).reshape(1, 2, 1, 2)
    assert mx.allclose(cosine, expected_cos, atol=1e-6).item()
    assert mx.allclose(sine, expected_sin, atol=1e-6).item()


def test_boolean_attention_mask_matches_hard_additive_mask() -> None:
    query = mx.arange(16).reshape(1, 2, 8).astype(mx.float32) / 16
    key = mx.flip(query, axis=1)
    value = query + 0.25
    boolean = mx.array([[True, False], [True, True]])
    additive = mx.where(boolean, 0.0, -3.0e38)
    bool_output = scaled_dot_product_attention(
        query,
        key,
        value,
        heads=2,
        dim_head=4,
        mask=boolean,
    )
    additive_output = scaled_dot_product_attention(
        query,
        key,
        value,
        heads=2,
        dim_head=4,
        mask=additive,
    )
    assert mx.allclose(bool_output, additive_output).item()


def test_attention_pretranspose_releases_sources_without_changing_output() -> None:
    attention = build_shaped_attention(
        8,
        heads=2,
        dim_head=4,
        apply_gated_attention=True,
    )
    value = mx.random.normal((1, 3, 8))
    expected = attention(value)
    specs = tuple(
        (target, "pretranspose") for target in ("to_q", "to_k", "to_v", "to_out", "to_gate_logits")
    )
    attention.apply_layouts(specs)
    attention.drop_layout_sources(specs)
    actual = attention(value)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert "weight" not in attention.to_out
    assert attention._to_out_weight_t is not None


def test_adaln_pretranspose_releases_source_without_changing_output() -> None:
    adaln = build_shaped_adaln(8, 2)
    timestep = mx.array([500.0])
    expected, expected_embedded = adaln(timestep, mx.float32)
    adaln.apply_layout()
    adaln.drop_layout_source()
    actual, actual_embedded = adaln(timestep, mx.float32)
    mx.eval(expected, expected_embedded, actual, actual_embedded)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert mx.array_equal(actual_embedded, expected_embedded).item()
    assert "weight" not in adaln.linear
    assert adaln._linear_weight_t is not None


def test_feed_forward_pretranspose_releases_sources_without_changing_output() -> None:
    feed_forward = build_shaped_feed_forward(8)
    value = mx.random.normal((1, 3, 8))
    expected = feed_forward(value)
    specs = (("project_in", "pretranspose"), ("project_out", "pretranspose"))
    feed_forward.apply_layouts(specs)
    feed_forward.drop_layout_sources(specs)
    actual = feed_forward(value)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert "weight" not in feed_forward.project_in.proj
    assert "weight" not in feed_forward.project_out
    assert feed_forward._project_in_weight_t is not None
    assert feed_forward._project_out_weight_t is not None


def test_feed_forward_fp16_boundary_casts_back_to_residual_dtype() -> None:
    feed_forward = build_shaped_feed_forward(8)
    value = mx.random.normal((1, 3, 8)).astype(mx.bfloat16)
    specs = (("project_in", "pretranspose"), ("project_out", "pretranspose"))
    feed_forward.apply_layouts(specs)
    feed_forward._project_in_weight_t = feed_forward._project_in_weight_t.astype(mx.float16)
    feed_forward._project_out_weight_t = feed_forward._project_out_weight_t.astype(mx.float16)
    actual = feed_forward(value)
    mx.eval(actual)
    assert actual.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(actual)).item()
