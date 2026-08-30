from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.kernels import steel_attention
from kinomlx.models.ltx2.transformer.attention import scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _reset_probe_state():
    steel_attention.reset_steel_attention_stats()
    yield
    steel_attention.reset_steel_attention_stats()


def _fake_array(
    shape: tuple[int, ...],
    dtype: mx.Dtype = mx.bfloat16,
) -> SimpleNamespace:
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def test_selector_accepts_only_supported_hot_path_shapes() -> None:
    q = _fake_array((1, 32, 512, 128))
    assert steel_attention._select_config(q, q, q, 128**-0.5, None) == (128, "")
    assert steel_attention._select_config(q, q, q, 0.25, None) == (None, "scale")
    assert steel_attention._select_config(q, q, q, 128**-0.5, object()) == (
        None,
        "mask",
    )

    fp16 = _fake_array((1, 32, 512, 128), mx.float16)
    assert steel_attention._select_config(fp16, fp16, fp16, None, None) == (
        None,
        "dtype_bf16",
    )

    short = _fake_array((1, 32, 511, 128))
    assert steel_attention._select_config(short, short, short, None, None) == (
        None,
        "seq",
    )


@pytest.mark.parametrize(
    ("q", "k", "v", "kwargs", "reason"),
    [
        (
            _fake_array((32, 512, 128)),
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 512, 128)),
            {},
            "ndim",
        ),
        (
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 512, 128), mx.float16),
            _fake_array((1, 32, 512, 128)),
            {},
            "dtype_mismatch",
        ),
        (
            _fake_array((1, 32, 512, 128)),
            _fake_array((2, 32, 512, 128)),
            _fake_array((2, 32, 512, 128)),
            {},
            "batch",
        ),
        *[
            (
                _fake_array(shape),
                _fake_array(shape),
                _fake_array(shape),
                {},
                reason,
            )
            for shape, reason in (
                ((2, 32, 512, 128), "batch"),
                ((1, 31, 512, 128), "heads"),
                ((1, 32, 512, 32), "dim"),
            )
        ],
        (
            _fake_array((1, 32, 512, 64)),
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 512, 128)),
            {},
            "dim_mismatch",
        ),
        (
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 513, 128)),
            {},
            "kv_len",
        ),
        (
            _fake_array((1, 32, 512, 64)),
            _fake_array((1, 32, 512, 64)),
            _fake_array((1, 32, 512, 64)),
            {"enable_d64": False},
            "d64_disabled",
        ),
        (
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 512, 128)),
            _fake_array((1, 32, 512, 128)),
            {"inputs_last_axis_contiguous": False},
            "last_axis_stride",
        ),
    ],
)
def test_selector_reject_reasons_are_pinned(q, k, v, kwargs, reason) -> None:
    assert steel_attention._select_config(q, k, v, None, None, **kwargs) == (None, reason)


def test_d64_config_tracks_self_and_cross_attention_direction() -> None:
    assert steel_attention._d64_config(1024, 1024) == "bk32"
    assert steel_attention._d64_config(1024, 2048) == "bk24_q8k2_scalefold"
    assert steel_attention._d64_config(2048, 1024) == "bk32_q8k4"


def test_partial_loader_and_head_count_are_bound_into_kernel_template() -> None:
    template = dict(
        steel_attention._template(
            64,
            align_q=True,
            align_k=True,
            d64_config="bk24_q8k2_scalefold",
        )
    )
    assert template["H"] == steel_attention._SUPPORTED_HEADS == 32
    assert template["K_LOADS_ALL_ACTIVE"] is False
    assert template["V_LOADS_ALL_ACTIVE"] is False


def test_probe_summary_distinguishes_hits_and_fallbacks() -> None:
    sample = _fake_array((1, 32, 512, 128))
    steel_attention._probe_hit(128, "", enabled=True, compiled=True)
    steel_attention._probe_hit(64, "bk32", enabled=True, compiled=True)
    steel_attention._probe_fallback(
        "mask",
        sample,
        sample,
        sample,
        sample,
        enabled=True,
        compiled=True,
    )
    assert steel_attention.steel_attention_summary() == {
        "enabled": True,
        "scope": "process_compiled_traces",
        "counter_unit": "compiled_trace",
        "hit_d128": 1,
        "hit_d64": 1,
        "fallback": 1,
        "d64_configs": {"bk32": 1},
        "fallback_reasons": {"mask": 1},
        "fallback_samples": {
            "mask": (
                "q=(1, 32, 512, 128) k=(1, 32, 512, 128) v=(1, 32, 512, 128) mask=(1, 32, 512, 128)"
            )
        },
    }


def test_probe_summary_labels_eager_call_counts() -> None:
    steel_attention._probe_hit(128, "", enabled=True, compiled=False)
    summary = steel_attention.steel_attention_summary()
    assert summary["scope"] == "process_eager_calls"
    assert summary["counter_unit"] == "attention_call"
    assert summary["hit_d128"] == 1


@pytest.mark.requires_metal
def test_eager_attention_dispatch_reports_call_scoped_probe() -> None:
    value = mx.zeros((1, 3, 8), dtype=mx.bfloat16)
    output = scaled_dot_product_attention(
        value,
        value,
        value,
        heads=2,
        dim_head=4,
        compile_attention=False,
        steel_attention_probe=True,
    )
    mx.eval(output)

    summary = steel_attention.steel_attention_summary()
    assert summary["scope"] == "process_eager_calls"
    assert summary["counter_unit"] == "attention_call"
    assert summary["fallback"] == 1
    assert summary["fallback_reasons"] == {"heads": 1}


@pytest.mark.requires_metal
def test_compiled_attention_dispatch_reports_trace_scoped_probe() -> None:
    value = mx.zeros((1, 5, 8), dtype=mx.bfloat16)
    output = scaled_dot_product_attention(
        value,
        value,
        value,
        heads=2,
        dim_head=4,
        compile_attention=True,
        steel_attention_probe=True,
    )
    mx.eval(output)

    summary = steel_attention.steel_attention_summary()
    assert summary["scope"] == "process_compiled_traces"
    assert summary["counter_unit"] == "compiled_trace"
    assert summary["fallback"] == 1
    assert summary["fallback_reasons"] == {"heads": 1}


@pytest.mark.parametrize(
    ("dim_head", "query_tokens", "key_tokens", "expected_config", "max_abs"),
    [
        (64, 512, 512, "bk32", 0.0),
        (64, 640, 512, "bk32_q8k4", 0.0),
        (64, 512, 640, "bk24_q8k2_scalefold", 0.00048828125),
        (64, 513, 513, "bk32", 0.0),
        (128, 512, 640, "", 0.001),
    ],
)
@pytest.mark.requires_metal
def test_steel_attention_matches_stock_sdpa(
    dim_head: int,
    query_tokens: int,
    key_tokens: int,
    expected_config: str,
    max_abs: float,
) -> None:
    mx.random.seed(7)
    q = mx.random.normal((1, 32, query_tokens, dim_head)).astype(mx.bfloat16)
    k = mx.random.normal((1, 32, key_tokens, dim_head)).astype(mx.bfloat16)
    v = mx.random.normal((1, 32, key_tokens, dim_head)).astype(mx.bfloat16)
    scale = dim_head**-0.5
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    actual = steel_attention.maybe_steel_attention(
        q,
        k,
        v,
        scale=scale,
        inputs_last_axis_contiguous=True,
    )
    assert actual is not None
    if dim_head == 64:
        assert steel_attention._d64_config(query_tokens, key_tokens) == expected_config
    mx.eval(expected, actual)
    delta = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item()
    assert delta <= max_abs


def test_strided_last_axis_falls_back_before_kernel_dispatch() -> None:
    base = mx.random.normal((1, 32, 512, 256)).astype(mx.bfloat16)
    strided = base[..., ::2]
    assert steel_attention.maybe_steel_attention(strided, strided, strided) is None
