from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from kinomlx.kernels import (
    fused_ops,
    gelu_approx,
    group_norm,
    pixel_norm,
    rms_norm,
    silu,
)


@pytest.mark.parametrize(
    ("dtype", "max_abs"),
    [
        (mx.bfloat16, 4.0e-6),
        (mx.float16, 1.3e-4),
    ],
)
@pytest.mark.requires_metal
def test_gelu_approx_matches_explicit_fp32_opmath(
    dtype: mx.Dtype,
    max_abs: float,
) -> None:
    value = mx.linspace(-8.0, 8.0, 65536).astype(dtype)
    expected = nn.gelu_approx(value.astype(mx.float32)).astype(dtype)
    actual = gelu_approx(value)
    mx.eval(expected, actual)
    delta = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert actual.dtype == dtype
    assert delta <= max_abs


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.requires_metal
def test_gelu_approx_saturates_without_nan_at_real_gemma_range(dtype: mx.Dtype) -> None:
    value = mx.linspace(-128.0, 128.0, 65536).astype(dtype)
    expected = nn.gelu_approx(value.astype(mx.float32)).astype(dtype)
    actual = gelu_approx(value)
    mx.eval(expected, actual)
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.allclose(actual, expected, rtol=1.0e-5, atol=1.3e-4).item()


@pytest.mark.parametrize(
    ("dtype", "max_abs"),
    [
        (mx.bfloat16, 0.0),
        (mx.float16, 1.3e-4),
    ],
)
@pytest.mark.requires_metal
def test_silu_matches_explicit_fp32_opmath(dtype: mx.Dtype, max_abs: float) -> None:
    value = mx.linspace(-8.0, 8.0, 65536).astype(dtype)
    expected = nn.silu(value.astype(mx.float32)).astype(dtype)
    actual = silu(value)
    mx.eval(expected, actual)
    assert actual.dtype == dtype
    delta = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert delta <= max_abs


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.requires_metal
def test_weighted_rms_norm_matches_explicit_fp32_opmath(dtype: mx.Dtype) -> None:
    mx.random.seed(20260816)
    value = mx.random.normal((2, 5, 256)).astype(dtype)
    weight = (mx.random.normal((256,)) * 0.1 + 1.0).astype(dtype)
    expected = mx.fast.rms_norm(
        value.astype(mx.float32),
        weight.astype(mx.float32),
        1.0e-6,
    ).astype(dtype)
    actual = rms_norm(value, weight, 1.0e-6)
    mx.eval(expected, actual)
    delta = actual.astype(mx.float32) - expected.astype(mx.float32)
    relative_l2 = (mx.linalg.norm(delta) / mx.linalg.norm(expected.astype(mx.float32))).item()
    assert actual.dtype == dtype
    assert relative_l2 <= 5.0e-5


@pytest.mark.requires_metal
def test_gemma_rms_norm_keeps_weight_offset_in_fp32() -> None:
    mx.random.seed(20260816)
    value = (mx.random.normal((2, 3, 3840)) * 4.0).astype(mx.bfloat16)
    raw_weight = (mx.random.normal((3840,)) * 0.02).astype(mx.bfloat16)
    expected = mx.fast.rms_norm(
        value.astype(mx.float32),
        1.0 + raw_weight.astype(mx.float32),
        1.0e-6,
    ).astype(mx.bfloat16)
    actual = rms_norm(value, raw_weight, 1.0e-6, weight_offset=1.0)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.requires_metal
def test_unweighted_rms_norm_retains_stock_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unweighted RMSNorm should stay on stock MLX")

    monkeypatch.setattr(fused_ops, "_metal_rms_norm", fail)
    value = mx.linspace(-3.0, 3.0, 1024).reshape(4, 256).astype(mx.bfloat16)
    expected = mx.fast.rms_norm(value, None, 1.0e-6).astype(mx.bfloat16)
    actual = rms_norm(value, eps=1.0e-6)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (gelu_approx, nn.gelu_approx),
        (silu, nn.silu),
    ],
)
def test_low_precision_cpu_fallback_keeps_fp32_opmath(
    operation: Callable[[mx.array], mx.array],
    reference: Callable[[mx.array], mx.array],
) -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        value = mx.linspace(-4.0, 4.0, 257).astype(mx.bfloat16)
        expected = reference(value.astype(mx.float32)).astype(mx.bfloat16)
        actual = operation(value)
        mx.eval(expected, actual)
        assert mx.array_equal(actual, expected).item()
    finally:
        mx.set_default_device(previous)


def test_float32_uses_stock_mlx_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("FP32 activation should not dispatch a custom kernel")

    monkeypatch.setattr(fused_ops, "_metal_elementwise", fail)
    value = mx.linspace(-3.0, 3.0, 129)
    assert mx.array_equal(gelu_approx(value), nn.gelu_approx(value)).item()
    assert mx.array_equal(silu(value), nn.silu(value)).item()

    norm = nn.GroupNorm(4, 16, pytorch_compatible=True)
    monkeypatch.setattr(fused_ops, "_metal_group_norm", fail)
    shaped = value[:128].reshape(2, 4, 16)
    assert mx.array_equal(group_norm(shaped, norm), norm(shaped)).item()

    monkeypatch.setattr(fused_ops, "_metal_rms_norm", fail)
    weight = mx.linspace(0.9, 1.1, 16)
    expected = mx.fast.rms_norm(shaped, weight + 1.0, 1.0e-6)
    actual = rms_norm(shaped, weight, 1.0e-6, weight_offset=1.0)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.requires_metal
def test_custom_kernel_accepts_noncontiguous_low_precision_input() -> None:
    value = mx.arange(128).reshape(8, 16).T.astype(mx.bfloat16) / 16
    expected = nn.gelu_approx(value.astype(mx.float32)).astype(mx.bfloat16)
    actual = gelu_approx(value)
    mx.eval(expected, actual)
    delta = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert tuple(actual.shape) == tuple(value.shape)
    assert delta <= 4.0e-6

    weight = mx.linspace(0.9, 1.1, value.shape[-1]).astype(mx.bfloat16)
    expected_norm = mx.fast.rms_norm(
        value.astype(mx.float32),
        weight.astype(mx.float32),
        1.0e-6,
    ).astype(mx.bfloat16)
    actual_norm = rms_norm(value, weight, 1.0e-6)
    mx.eval(expected_norm, actual_norm)
    assert mx.array_equal(actual_norm, expected_norm).item()


def test_pixel_norm_preserves_reference_step_boundaries() -> None:
    value = mx.linspace(-2.0, 2.0, 48).reshape(2, 3, 8).astype(mx.bfloat16)
    expected_rms = mx.sqrt(mx.mean(value * value, axis=1, keepdims=True) + 1.0e-6)
    expected = value / expected_rms
    actual = pixel_norm(value, axis=1, eps=1.0e-6)
    mx.eval(expected, actual)
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("dtype", "relative_tolerance"),
    [
        (mx.bfloat16, 5.0e-5),
        (mx.float16, 5.0e-5),
    ],
)
@pytest.mark.requires_metal
def test_group_norm_matches_explicit_fp32_reduction(
    dtype: mx.Dtype,
    relative_tolerance: float,
) -> None:
    mx.random.seed(20260816)
    value = mx.random.normal((2, 3, 4, 64)).astype(dtype)
    norm = nn.GroupNorm(8, 64, eps=1.0e-5, pytorch_compatible=True)
    norm.weight = mx.random.normal((64,)) * 0.1 + 1.0
    norm.bias = mx.random.normal((64,)) * 0.1
    expected = norm(value.astype(mx.float32)).astype(dtype)
    actual = group_norm(value, norm)
    mx.eval(expected, actual)
    delta = actual.astype(mx.float32) - expected.astype(mx.float32)
    relative_l2 = (mx.linalg.norm(delta) / mx.linalg.norm(expected.astype(mx.float32))).item()
    assert actual.dtype == dtype
    assert relative_l2 <= relative_tolerance


@pytest.mark.parametrize(
    "pytorch_compatible",
    [False, True],
)
def test_group_norm_cpu_and_unsupported_forms_use_explicit_fp32_fallback(
    pytorch_compatible: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        value = mx.linspace(-2.0, 2.0, 256).reshape(2, 2, 2, 32).astype(mx.bfloat16)
        norm = nn.GroupNorm(
            4,
            32,
            affine=pytorch_compatible,
            pytorch_compatible=pytorch_compatible,
        )

        def fail(*args, **kwargs):
            del args, kwargs
            raise AssertionError("fallback GroupNorm should not dispatch a Metal kernel")

        monkeypatch.setattr(fused_ops, "_metal_group_norm", fail)
        expected = norm(value.astype(mx.float32)).astype(mx.bfloat16)
        actual = group_norm(value, norm)
        mx.eval(expected, actual)
        assert mx.array_equal(actual, expected).item()
    finally:
        mx.set_default_device(previous)


def test_weighted_rms_norm_cpu_fallback_keeps_fp32_opmath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:

        def fail(*args, **kwargs):
            del args, kwargs
            raise AssertionError("CPU RMSNorm should not dispatch a Metal kernel")

        monkeypatch.setattr(fused_ops, "_metal_rms_norm", fail)
        value = mx.linspace(-3.0, 3.0, 1024).reshape(4, 256).astype(mx.bfloat16)
        raw_weight = mx.linspace(-0.1, 0.1, 256).astype(mx.bfloat16)
        expected = mx.fast.rms_norm(
            value.astype(mx.float32),
            1.0 + raw_weight.astype(mx.float32),
            1.0e-6,
        ).astype(mx.bfloat16)
        actual = rms_norm(value, raw_weight, 1.0e-6, weight_offset=1.0)
        mx.eval(expected, actual)
        assert mx.array_equal(actual, expected).item()
    finally:
        mx.set_default_device(previous)


def test_ltx_models_use_central_precision_operations() -> None:
    model_root = Path(__file__).parents[2] / "kinomlx" / "models" / "ltx2"
    forbidden = ("mx.fast.rms_norm(", "nn.gelu_approx(", "nn.silu(")
    offenders = [
        f"{path.relative_to(model_root)}: {call}"
        for path in sorted(model_root.rglob("*.py"))
        for call in forbidden
        if call in path.read_text()
    ]
    assert offenders == []
