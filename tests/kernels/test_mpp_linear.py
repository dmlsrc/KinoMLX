"""Bit identity and routing of the MPP linear products."""

import mlx.core as mx
import pytest

from kinomlx.kernels import mpp_linear

# Ragged against the 32x32 output tile in both rows and columns, so edge tiles run.
_ROWS, _K, _N = 1056, 1024, 1040


def _operands(
    *, transposed: bool, seed: int, dtype: mx.Dtype = mx.float16, weight_scale: float = 1.0
) -> tuple[mx.array, mx.array, mx.array]:
    keys = mx.random.split(mx.random.key(seed), 3)
    value = (mx.random.normal((1, _ROWS, _K), key=keys[0]) * 0.5).astype(dtype)
    weight = (mx.random.normal((_N, _K), key=keys[1]) * _K**-0.5 * weight_scale).astype(dtype)
    if not transposed:
        weight = mx.contiguous(weight.T)
    bias = (mx.random.normal((_N,), key=keys[2]) * 0.5).astype(dtype)
    return value, weight, bias


def _skip_unless_available(*, transposed: bool, bias: bool, dtype: mx.Dtype) -> None:
    if not mpp_linear.available(transposed=transposed, bias=bias, dtype=dtype):
        pytest.skip("MPP matmul2d does not build or does not match MLX on this GPU")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
def test_linear_equals_mlx_bit_for_bit(dtype: mx.Dtype, transposed: bool, with_bias: bool) -> None:
    _skip_unless_available(transposed=transposed, bias=with_bias, dtype=dtype)
    value, weight, bias = _operands(transposed=transposed, seed=7, dtype=dtype)
    offset = bias if with_bias else None
    assert mpp_linear.applies(value, weight, offset, transposed=transposed)
    right = weight.T if transposed else weight
    expected = value @ right if offset is None else mx.addmm(offset, value, right)
    actual = mpp_linear.linear(value, weight, offset, transposed=transposed)
    mx.eval(actual, expected)
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(("input_scale", "weight_scale"), [(1.0, 1e-4), (1e-6, 1.0)])
def test_bf16_operands_below_fp16_normal_range_keep_mlx_bits(
    input_scale: float, weight_scale: float
) -> None:
    # LTX's attention outputs reach the output projections mostly below FP16's normal range.
    _skip_unless_available(transposed=True, bias=True, dtype=mx.bfloat16)
    value, weight, bias = _operands(
        transposed=True, seed=13, dtype=mx.bfloat16, weight_scale=weight_scale
    )
    value = (value.astype(mx.float32) * input_scale).astype(mx.bfloat16)
    # A plain FP16 copy of the small operand loses bits; the scaled copy must not.
    small = value if input_scale < 1.0 else weight
    plain = small.astype(mx.float16).astype(mx.float32)
    assert (plain != small.astype(mx.float32)).sum().item() > 0
    actual = mpp_linear.linear(value, weight, bias, transposed=True)
    expected = mx.addmm(bias, value, weight.T)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_bf16_operands_beyond_fp16_range_keep_mlx_bits() -> None:
    _skip_unless_available(transposed=True, bias=True, dtype=mx.bfloat16)
    value, weight, bias = _operands(transposed=True, seed=19, dtype=mx.bfloat16)
    value = (value.astype(mx.float32) * 1e6).astype(mx.bfloat16)
    actual = mpp_linear.linear(value, weight, bias, transposed=True)
    expected = mx.addmm(bias, value, weight.T)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_bf16_non_finite_inputs_stay_non_finite(bad: float) -> None:
    # A saturating conversion would turn an overflowed activation into finite garbage.
    _skip_unless_available(transposed=True, bias=True, dtype=mx.bfloat16)
    value, weight, bias = _operands(transposed=True, seed=23, dtype=mx.bfloat16)
    first = mx.full((1, 1, _K), bad, dtype=mx.bfloat16)
    value = mx.concatenate([first, value[:, 1:]], axis=1)
    actual = mpp_linear.linear(value, weight, bias, transposed=True)
    mx.eval(actual)
    assert not mx.any(mx.isfinite(actual[0, 0])).item()


def test_linear_takes_a_shared_fp16_operand() -> None:
    _skip_unless_available(transposed=True, bias=True, dtype=mx.bfloat16)
    value, weight, bias = _operands(transposed=True, seed=17, dtype=mx.bfloat16)
    operand = mpp_linear.fp16_operand(value)
    assert operand.values.dtype == mx.float16
    assert operand.peak is not None
    shared = mpp_linear.linear(value, weight, bias, transposed=True, operand=operand)
    alone = mpp_linear.linear(value, weight, bias, transposed=True)
    mx.eval(shared, alone)
    assert mx.array_equal(shared, alone).item()
    with pytest.raises(ValueError, match="fp16_operand"):
        mpp_linear.linear(
            value,
            weight,
            bias,
            transposed=True,
            operand=mpp_linear.Operand(value.astype(mx.float16), None),
        )
    halves = value.astype(mx.float16)
    as_is = mpp_linear.fp16_operand(halves)
    assert as_is.values is halves
    assert as_is.peak is None


def test_prepared_bf16_weights_restore_exactly_and_multiply_as_their_source() -> None:
    _skip_unless_available(transposed=True, bias=True, dtype=mx.bfloat16)
    value, weight, bias = _operands(transposed=True, seed=29, dtype=mx.bfloat16, weight_scale=1e-4)
    prepared = mpp_linear.prepare(weight)
    assert prepared.values.dtype == mx.float16
    assert prepared.peak is not None
    restored = mpp_linear.restore(prepared)
    through_prepared = mpp_linear.linear(value, prepared, bias, transposed=True)
    expected = mx.addmm(bias, value, weight.T)
    mx.eval(restored, through_prepared, expected)
    assert restored.dtype == mx.bfloat16
    assert mx.array_equal(restored, weight).item()
    assert mx.array_equal(through_prepared, expected).item()
    assert mpp_linear.applies(value, prepared, bias, transposed=True)
    assert not mpp_linear.applies(value.astype(mx.float16), prepared, None, transposed=True)
    with pytest.raises(ValueError, match="dtype"):
        mpp_linear.linear(value.astype(mx.float16), prepared, None, transposed=True)
    halves = weight.astype(mx.float16)
    assert mpp_linear.prepare(halves).values is halves
    with pytest.raises(ValueError, match="BF16"):
        mpp_linear.restore(mpp_linear.prepare(halves))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_prepared_weights_keep_non_finite_values(bad: float) -> None:
    # The adapter check reads a converted weight; it must still see a non-finite fusion.
    _skip_unless_available(transposed=True, bias=True, dtype=mx.bfloat16)
    _value, weight, _bias = _operands(transposed=True, seed=31, dtype=mx.bfloat16)
    weight = mx.concatenate([mx.full((1, _K), bad, dtype=mx.bfloat16), weight[1:]], axis=0)
    prepared = mpp_linear.prepare(weight)
    restored = mpp_linear.restore(prepared)
    mx.eval(prepared.values, restored)
    assert not mx.all(mx.isfinite(prepared.values[0])).item()
    assert not mx.all(mx.isfinite(restored[0])).item()


def test_supports_only_measured_sizes_and_layouts() -> None:
    for dtype in (mx.float16, mx.bfloat16):
        assert mpp_linear.supports(1024, 4096, 4096, transposed=True, dtype=dtype)
        assert mpp_linear.supports(1024, 4096, 4096, transposed=False, dtype=dtype)
        assert not mpp_linear.supports(1023, 4096, 4096, transposed=True, dtype=dtype)
        assert not mpp_linear.supports(4096, 1000, 4096, transposed=True, dtype=dtype)
        assert not mpp_linear.supports(4096, 4104, 4096, transposed=True, dtype=dtype)
        assert not mpp_linear.supports(4096, 4096, 32, transposed=True, dtype=dtype)
        # The feed-forward up projection only on the checkpoint's (out, in) weight, and never
        # the 16,384-wide inner dimension of its down projection.
        assert mpp_linear.supports(4096, 4096, 16384, transposed=True, dtype=dtype)
        assert not mpp_linear.supports(4096, 4096, 16384, transposed=False, dtype=dtype)
        assert not mpp_linear.supports(4096, 16384, 4096, transposed=False, dtype=dtype)
        assert not mpp_linear.supports(4096, 16384, 4096, transposed=True, dtype=dtype)
    assert not mpp_linear.supports(1024, 4096, 4096, transposed=True, dtype=mx.float32)
    # The kernels count rows and 32x32 output tiles in int.
    assert not mpp_linear.supports(2**31 - 1, 4096, 4096, transposed=True, dtype=mx.float16)


def test_applies_requires_matching_operands_and_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, weight, bias = _operands(transposed=True, seed=11)
    assert not mpp_linear.applies(value.astype(mx.bfloat16), weight, bias, transposed=True)
    assert not mpp_linear.applies(
        value.astype(mx.float32), weight.astype(mx.float32), None, transposed=True
    )
    assert not mpp_linear.applies(value, weight, bias.astype(mx.bfloat16), transposed=True)
    assert not mpp_linear.applies(value, weight, bias, transposed=False)
    assert not mpp_linear.applies(value, weight, bias[:-1], transposed=True)
    monkeypatch.setattr(mpp_linear, "_enabled", False)
    assert not mpp_linear.applies(value, weight, bias, transposed=True)


# Narrow like the diffusion video decoder's products, ragged in rows and columns.
_ROWS_SHAPE = (4170, 256, 776)


def _row_operands(seed: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    rows, k, n = _ROWS_SHAPE
    keys = mx.random.split(mx.random.key(seed), 4)
    value = (mx.random.normal((rows, k), key=keys[0]) * 0.5).astype(mx.float16)
    weight = (mx.random.normal((n, k), key=keys[1]) * k**-0.5).astype(mx.float16)
    bias = (mx.random.normal((n,), key=keys[2]) * 0.5).astype(mx.float16)
    residual = (mx.random.normal((rows, n), key=keys[3]) * 0.5).astype(mx.float16)
    return value, weight, bias, residual


@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("with_residual", [True, False])
def test_linear_rows_equals_mlx_bit_for_bit(with_bias: bool, with_residual: bool) -> None:
    if not mpp_linear.rows_available(bias=with_bias, residual=with_residual):
        pytest.skip("MPP's run-time-row form does not build or does not match MLX on this GPU")
    value, weight, bias, residual = _row_operands(29)
    offset = bias if with_bias else None
    added = residual if with_residual else None
    assert mpp_linear.rows_applies(value, weight, offset, added)
    expected = value @ weight.T if offset is None else mx.addmm(offset, value, weight.T)
    if added is not None:
        expected = added + expected
    # Two row counts through one build: the row count is read at run time.
    for count in (value.shape[0], 2_112):
        actual = mpp_linear.linear_rows(
            value[:count], weight, offset, residual=None if added is None else added[:count]
        )
        mx.eval(actual, expected)
        assert actual.dtype == mx.float16
        assert mx.array_equal(actual, expected[:count]).item()


def test_rows_supported_keeps_mlx_summation_order() -> None:
    # More than 2,048 output tiles of 16x16 at any width: MLX never splits those along K.
    assert mpp_linear.rows_supported(2_049, 256, 256)
    assert not mpp_linear.rows_supported(2_048, 256, 256)
    assert mpp_linear.rows_supported(65_536, 256, 2_048)
    assert not mpp_linear.rows_supported(65_536, 250, 2_048)
    assert not mpp_linear.rows_supported(65_536, 16_384, 256)
    # 2**31 elements and more: the tiles are addressed from a 64-bit base per threadgroup.
    assert mpp_linear.rows_supported(1 << 23, 256, 256)
    assert mpp_linear.rows_supported(1 << 24, 256, 256)
    # The kernel counts its rows and 32x32 output tiles in int.
    assert mpp_linear.rows_supported(2**31 - 32, 256, 256)
    assert not mpp_linear.rows_supported(2**31 - 31, 256, 256)
    assert not mpp_linear.rows_supported(2**30, 256, 2**16)


def test_rows_applies_requires_fp16_matching_operands_and_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, weight, bias, residual = _row_operands(31)
    assert not mpp_linear.rows_applies(value.astype(mx.bfloat16), weight.astype(mx.bfloat16))
    assert not mpp_linear.rows_applies(value, weight[:, :-16])
    assert not mpp_linear.rows_applies(value, weight, bias[:-1])
    assert not mpp_linear.rows_applies(value, weight, bias, residual[:-1])
    assert not mpp_linear.rows_applies(value, weight, bias, residual.astype(mx.float32))
    monkeypatch.setattr(mpp_linear, "_enabled", False)
    assert not mpp_linear.rows_applies(value, weight, bias, residual)
    with pytest.raises(ValueError, match="run-time-row"):
        mpp_linear.linear_rows(value, weight, bias, residual=residual)
