from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import kinomlx._mlx_nn as nn
from kinomlx.kernels import mpp_linear
from kinomlx.models.ltx2.transformer import (
    apply_rotary_emb,
    precompute_freqs_cis,
)
from kinomlx.models.ltx2.transformer.attention import (
    _mpp_projection,
    scaled_dot_product_attention,
)
from kinomlx.models.ltx2.weights.layout import (
    install_block_layout_weight,
    install_top_level_layout_weight,
)

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
        compile_attention=False,
    )
    compiled = scaled_dot_product_attention(
        query,
        key,
        value,
        heads=2,
        dim_head=8,
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
    block = SimpleNamespace(attn1=attention)
    for target in ("to_q", "to_k", "to_v", "to_out", "to_gate_logits"):
        install_block_layout_weight(
            block, f"attn1.{target}.weight_t", mx.contiguous(getattr(attention, target).weight.T)
        )
    actual = attention(value)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert "weight" not in attention.to_out
    assert attention._to_out_weight_t is not None


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_attention_projections_through_mpp_keep_mlx_bits(
    dtype: mx.Dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not all(
        mpp_linear.available(transposed=transposed, bias=True, dtype=dtype)
        for transposed in (True, False)
    ):
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")
    attention = build_shaped_attention(1024, heads=8, dim_head=128, apply_gated_attention=True)
    for seed, target in enumerate(("to_q", "to_k", "to_v", "to_out")):
        layer = getattr(attention, target)
        layer.bias = mx.random.normal(layer.bias.shape, key=mx.random.key(900 + seed)) * 0.1
    attention.set_dtype(dtype)
    value = (mx.random.normal((1, 1056, 1024), key=mx.random.key(910)) * 0.5).astype(dtype)
    assert mpp_linear.applies(value, attention.to_q.weight, attention.to_q.bias, transposed=True)

    def both_paths() -> tuple[mx.array, mx.array]:
        monkeypatch.setattr(mpp_linear, "_enabled", True)
        through_mpp = attention(value)
        monkeypatch.setattr(mpp_linear, "_enabled", False)
        through_mlx = attention(value)
        mx.eval(through_mpp, through_mlx)
        return through_mpp, through_mlx

    through_mpp, through_mlx = both_paths()
    assert mx.array_equal(through_mpp, through_mlx).item()
    block = SimpleNamespace(attn1=attention)
    for target in ("to_q", "to_k", "to_v", "to_out", "to_gate_logits"):
        install_block_layout_weight(
            block, f"attn1.{target}.weight_t", mx.contiguous(getattr(attention, target).weight.T)
        )
    through_mpp, through_mlx = both_paths()
    assert mx.array_equal(through_mpp, through_mlx).item()


def test_bf16_attention_converts_its_mpp_weights_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not mpp_linear.available(transposed=True, bias=True, dtype=mx.bfloat16):
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")
    attention = build_shaped_attention(1024, heads=8, dim_head=128, apply_gated_attention=True)
    for seed, target in enumerate(("to_q", "to_k", "to_v", "to_out")):
        layer = getattr(attention, target)
        layer.bias = mx.random.normal(layer.bias.shape, key=mx.random.key(950 + seed)) * 0.1
    attention.set_dtype(mx.bfloat16)
    value = (mx.random.normal((1, 1056, 1024), key=mx.random.key(960)) * 0.5).astype(mx.bfloat16)
    few_rows = value[:, :512]
    monkeypatch.setattr(mpp_linear, "_enabled", False)
    expected, expected_few = attention(value), attention(few_rows)
    mx.eval(expected, expected_few)
    source = attention.to_q.weight

    monkeypatch.setattr(mpp_linear, "_enabled", True)
    attention.convert_mpp_weights = True
    first = attention(value)
    mx.eval(first)
    converted = attention.to_q.weight
    assert converted is not source
    assert converted.dtype == mx.float16
    second = attention(value)
    mx.eval(second)
    assert attention.to_q.weight is converted
    # Calls MPP does not take restore the BF16 weight for MLX's product.
    below_floor = attention(few_rows)
    monkeypatch.setattr(mpp_linear, "_enabled", False)
    switched_off = attention(value)
    mx.eval(below_floor, switched_off)
    for actual, wanted in (
        (first, expected),
        (second, expected),
        (below_floor, expected_few),
        (switched_off, expected),
    ):
        assert mx.array_equal(actual, wanted).item()

    # A weight bound again (a fresh load) is converted again, never read through a stale copy.
    monkeypatch.setattr(mpp_linear, "_enabled", True)
    attention.to_q.weight = source * 2
    rebound = attention(value)
    attention.convert_mpp_weights = False
    monkeypatch.setattr(mpp_linear, "_enabled", False)
    attention.to_q.weight = source * 2
    rebound_expected = attention(value)
    mx.eval(rebound, rebound_expected)
    assert mx.array_equal(rebound, rebound_expected).item()


def test_feed_forward_converts_its_up_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    if not mpp_linear.available(transposed=True, bias=True, dtype=mx.bfloat16):
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")
    feed_forward = build_shaped_feed_forward(1024)
    for seed, layer in enumerate((feed_forward.project_in.proj, feed_forward.project_out)):
        layer.bias = mx.random.normal(layer.bias.shape, key=mx.random.key(970 + seed)) * 0.1
    feed_forward.set_dtype(mx.bfloat16)
    value = (mx.random.normal((1, 1056, 1024), key=mx.random.key(980)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setattr(mpp_linear, "_enabled", False)
    expected = feed_forward(value)
    mx.eval(expected)
    monkeypatch.setattr(mpp_linear, "_enabled", True)
    feed_forward.convert_mpp_weights = True
    actual = feed_forward(value)
    mx.eval(actual)
    assert feed_forward.project_in.proj.weight.dtype == mx.float16
    assert mx.array_equal(actual, expected).item()


def test_mpp_projection_leaves_linear_subclasses_to_their_own_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Shifted(nn.Linear):
        def __call__(self, value: mx.array) -> mx.array:
            return super().__call__(value) + 1.0

    layer = Shifted(1024, 1024)
    layer.set_dtype(mx.bfloat16)
    value = mx.random.normal((1, 1056, 1024), key=mx.random.key(940)).astype(mx.bfloat16)

    def refuse(*args: object, **kwargs: object) -> mx.array:
        raise AssertionError("MPP took a Linear subclass")

    monkeypatch.setattr(mpp_linear, "linear", refuse)
    actual = _mpp_projection(layer, value, None)
    expected = layer(value)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_feed_forward_through_mpp_keeps_mlx_bits(
    dtype: mx.Dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not all(
        mpp_linear.available(transposed=transposed, bias=True, dtype=dtype)
        for transposed in (True, False)
    ):
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")
    feed_forward = build_shaped_feed_forward(1024)
    for seed, layer in enumerate((feed_forward.project_in.proj, feed_forward.project_out)):
        layer.bias = mx.random.normal(layer.bias.shape, key=mx.random.key(920 + seed)) * 0.1
    feed_forward.set_dtype(dtype)
    value = (mx.random.normal((1, 1056, 1024), key=mx.random.key(930)) * 0.5).astype(dtype)
    project_in = feed_forward.project_in.proj
    assert mpp_linear.applies(value, project_in.weight, project_in.bias, transposed=True)

    def both_paths() -> tuple[mx.array, mx.array]:
        monkeypatch.setattr(mpp_linear, "_enabled", True)
        through_mpp = feed_forward(value)
        monkeypatch.setattr(mpp_linear, "_enabled", False)
        through_mlx = feed_forward(value)
        mx.eval(through_mpp, through_mlx)
        return through_mpp, through_mlx

    through_mpp, through_mlx = both_paths()
    assert mx.array_equal(through_mpp, through_mlx).item()
    block = SimpleNamespace(ff=feed_forward)
    install_block_layout_weight(
        block, "ff.project_in.proj.weight_t", mx.contiguous(project_in.weight.T)
    )
    install_block_layout_weight(
        block, "ff.project_out.weight_t", mx.contiguous(feed_forward.project_out.weight.T)
    )
    through_mpp, through_mlx = both_paths()
    assert mx.array_equal(through_mpp, through_mlx).item()


def test_adaln_pretranspose_releases_source_without_changing_output() -> None:
    adaln = build_shaped_adaln(8, 2)
    timestep = mx.array([500.0])
    expected, expected_embedded = adaln(timestep, mx.float32)
    install_top_level_layout_weight(
        SimpleNamespace(adaln_single=adaln),
        "adaln_single.linear.weight_t",
        mx.contiguous(adaln.linear.weight.T),
    )
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
    block = SimpleNamespace(ff=feed_forward)
    install_block_layout_weight(
        block, "ff.project_in.proj.weight_t", mx.contiguous(feed_forward.project_in.proj.weight.T)
    )
    install_block_layout_weight(
        block, "ff.project_out.weight_t", mx.contiguous(feed_forward.project_out.weight.T)
    )
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
    block = SimpleNamespace(ff=feed_forward)
    install_block_layout_weight(
        block, "ff.project_in.proj.weight_t", mx.contiguous(feed_forward.project_in.proj.weight.T)
    )
    install_block_layout_weight(
        block, "ff.project_out.weight_t", mx.contiguous(feed_forward.project_out.weight.T)
    )
    feed_forward._project_in_weight_t = feed_forward._project_in_weight_t.astype(mx.float16)
    feed_forward._project_out_weight_t = feed_forward._project_out_weight_t.astype(mx.float16)
    actual = feed_forward(value)
    mx.eval(actual)
    assert actual.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(actual)).item()
