"""Bit identity and admission of the single-pass low-rank fusion kernel."""

import mlx.core as mx
import pytest

from kinomlx.kernels import low_rank_fusion

pytestmark = pytest.mark.requires_metal

# Scales that are not powers of two, where a separately rounded multiply and add would differ
# from the FMA of MLX's addmm epilogue.
_SCALES = (0.3, 1.7, -0.8, 2.25)
_BITS = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}


def _chain(stored, terms, *, transposed):
    """The adapter fuser's FP32 addmm chain, cast back once and laid out as stored."""
    logical = stored.T if transposed else stored
    fused = logical.astype(mx.float32)
    for up, down, scale in terms:
        fused = mx.addmm(fused, up.astype(mx.float32), down.astype(mx.float32), alpha=scale)
    candidate = fused.astype(stored.dtype)
    return mx.contiguous(candidate.T) if transposed else candidate


def _weight(out_dim, in_dim, ranks, *, seed, base_dtype=mx.bfloat16, factor_dtype=mx.bfloat16):
    keys = mx.random.split(mx.random.key(seed), 1 + 2 * len(ranks))
    logical = (mx.random.normal((out_dim, in_dim), key=keys[0]) * 0.02).astype(base_dtype)
    terms = [
        (
            (mx.random.normal((out_dim, rank), key=keys[1 + 2 * i]) * 0.01).astype(factor_dtype),
            (mx.random.normal((rank, in_dim), key=keys[2 + 2 * i]) * 0.01).astype(factor_dtype),
            _SCALES[i % len(_SCALES)],
        )
        for i, rank in enumerate(ranks)
    ]
    return logical, terms


def _assert_fuses_like_the_chain(logical, terms, *, transposed):
    stored = mx.contiguous(logical.T) if transposed else logical
    factors = [(up, down) for up, down, _scale in terms]
    assert low_rank_fusion.applies(stored, factors, transposed=transposed)
    actual = low_rank_fusion.fuse(stored, terms, transposed=transposed)
    expected = _chain(stored, terms, transposed=transposed)
    mx.eval(actual, expected)
    assert actual.dtype == stored.dtype
    assert actual.shape == stored.shape
    # Compare bits: NaN payloads and signed zeros included.
    bits = _BITS[stored.dtype]
    assert mx.array_equal(actual.view(bits), expected.view(bits)).item()


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    ("out_dim", "in_dim", "ranks"),
    [
        # Ragged against the 64 x 32 tile in both directions, so edge tiles run.
        (70, 100, (32, 16)),
        (1000, 1040, (16, 384)),
        # The staged term need not come first, and several terms run in the epilogue.
        (200, 136, (16, 48, 32)),
        (130, 96, (64, 64)),
        (32, 4096, (16, 32)),
        (512, 256, (256,)),
    ],
)
def test_fusion_equals_the_fp32_chain_bit_for_bit(out_dim, in_dim, ranks, transposed):
    logical, terms = _weight(out_dim, in_dim, ranks, seed=out_dim + in_dim + len(ranks))
    _assert_fuses_like_the_chain(logical, terms, transposed=transposed)


@pytest.mark.parametrize("base_dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("factor_dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_every_base_and_factor_dtype_keeps_the_chain_bits(base_dtype, factor_dtype):
    logical, terms = _weight(
        200, 136, (16, 48), seed=5, base_dtype=base_dtype, factor_dtype=factor_dtype
    )
    _assert_fuses_like_the_chain(logical, terms, transposed=False)


def test_mixed_factor_dtypes_within_a_term():
    logical, terms = _weight(130, 160, (32,), seed=9)
    up, down, scale = terms[0]
    _assert_fuses_like_the_chain(logical, [(up, down.astype(mx.float32), scale)], transposed=True)


def _signed_zeros(logical, terms):
    # Zero-initialized up rows, negative-zero factors and base entries.
    up, down, scale = terms[0]
    rows = mx.arange(up.shape[0])[:, None]
    up = mx.where(rows % 3 == 0, mx.zeros_like(up), up)
    down = mx.where(mx.arange(down.shape[1])[None, :] % 5 == 0, -mx.zeros_like(down), down)
    columns = mx.arange(logical.shape[1])[None, :]
    logical = mx.where(columns % 4 == 0, -mx.zeros_like(logical), logical)
    logical = mx.where(columns % 4 == 1, mx.zeros_like(logical), logical)
    return logical, [(up, down, scale), *terms[1:]]


def _subnormal_products(logical, terms):
    return logical * 1e-30, [(up * 1e-20, down * 1e-20, scale) for up, down, scale in terms]


def _large(logical, terms):
    return logical * 1e3, [(up * 3e4, down * 3e4, scale) for up, down, scale in terms]


def _non_finite(logical, terms):
    up, down, scale = terms[0]
    up = mx.where((mx.arange(up.shape[0]) == 3)[:, None], mx.array(float("inf"), up.dtype), up)
    down = mx.where(
        (mx.arange(down.shape[1]) == 5)[None, :], mx.array(float("nan"), down.dtype), down
    )
    return logical, [(up, down, scale), *terms[1:]]


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("edit", [_signed_zeros, _subnormal_products, _large, _non_finite])
def test_edge_values_keep_the_chain_bits(edit, transposed):
    logical, terms = _weight(300, 200, (32, 16), seed=11)
    logical, terms = edit(logical, terms)
    _assert_fuses_like_the_chain(logical, terms, transposed=transposed)


def test_factor_views_at_odd_offsets_keep_the_chain_bits():
    # A row-contiguous view starting one element into its buffer reaches the kernel uncopied.
    logical, terms = _weight(96, 128, (32, 16), seed=13)
    viewed = []
    for up, down, scale in terms:
        up_flat = mx.concatenate([mx.zeros((1,), dtype=up.dtype), up.reshape(-1)])
        down_flat = mx.concatenate([mx.zeros((1,), dtype=down.dtype), down.reshape(-1)])
        viewed.append((up_flat[1:].reshape(up.shape), down_flat[1:].reshape(down.shape), scale))
    _assert_fuses_like_the_chain(logical, viewed, transposed=False)


def test_kernel_is_available_in_both_layouts():
    assert low_rank_fusion.available(transposed=False)
    assert low_rank_fusion.available(transposed=True)


def _factors(out_dim, in_dim, ranks, dtype=mx.bfloat16):
    return [(mx.zeros((out_dim, rank), dtype), mx.zeros((rank, in_dim), dtype)) for rank in ranks]


@pytest.mark.parametrize(
    ("ranks", "admitted"),
    [
        ((16,), True),
        ((384, 16, 64), True),
        # Ranks off the 16-element blocks MLX's GEMM stages would be summed differently.
        ((24,), False),
        ((16, 8), False),
        # Only the largest term runs through the staged loop.
        ((384, 128), False),
        ((), False),
        ((16,) * (low_rank_fusion.MAX_TERMS + 1), False),
    ],
)
def test_supports_admits_ranks_the_kernel_sums_like_mlx(ranks, admitted):
    base = mx.zeros((64, 96), dtype=mx.bfloat16)
    factors = _factors(64, 96, ranks)
    assert low_rank_fusion.supports(base, factors, transposed=False) is admitted


def test_supports_checks_shapes_layouts_and_dtypes():
    base = mx.zeros((64, 96), dtype=mx.bfloat16)
    assert low_rank_fusion.supports(base, _factors(64, 96, (16,)), transposed=False)
    # A stored transpose is (in, out).
    assert not low_rank_fusion.supports(base, _factors(64, 96, (16,)), transposed=True)
    assert low_rank_fusion.supports(base, _factors(96, 64, (16,)), transposed=True)
    up, down = _factors(64, 96, (16,))[0]
    assert not low_rank_fusion.supports(base, [(up, down[:, :95])], transposed=False)
    assert not low_rank_fusion.supports(base, [(up, down[:15])], transposed=False)
    assert not low_rank_fusion.supports(
        mx.zeros((64,), mx.bfloat16), [(up, down)], transposed=False
    )
    assert not low_rank_fusion.supports(base.astype(mx.int32), [(up, down)], transposed=False)
    assert not low_rank_fusion.supports(base, [(up.astype(mx.int32), down)], transposed=False)


def test_fuse_refuses_what_it_does_not_support():
    base = mx.zeros((64, 96), dtype=mx.bfloat16)
    up, down = _factors(64, 96, (24,))[0]
    with pytest.raises(ValueError, match="does not take"):
        low_rank_fusion.fuse(base, [(up, down, 1.0)], transposed=False)


def test_switch_keeps_the_chain(monkeypatch):
    base = mx.zeros((64, 96), dtype=mx.bfloat16)
    factors = _factors(64, 96, (16,))
    assert low_rank_fusion.applies(base, factors, transposed=False)
    monkeypatch.setattr(low_rank_fusion, "_enabled", False)
    assert not low_rank_fusion.enabled()
    assert not low_rank_fusion.applies(base, factors, transposed=False)
