"""The runtime fuser's single-pass kernel path keeps the FP32 chain's bits."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.adapters.formats import AdapterOperation
from kinomlx.adapters.fusion import WeightedAdapterOperation, fuse_adapter_operations, fuse_weight
from kinomlx.adapters.validation import AdapterWeightCheck
from kinomlx.kernels import low_rank_fusion
from kinomlx.modeling.loading import WeightTarget

pytestmark = pytest.mark.requires_metal

_OUT, _IN = 96, 160


def _operations():
    keys = mx.random.split(mx.random.key(3), 4)
    first = AdapterOperation(
        "layer",
        "lora",
        "weight",
        {
            "down": (mx.random.normal((32, _IN), key=keys[0]) * 0.02).astype(mx.bfloat16),
            "up": (mx.random.normal((_OUT, 32), key=keys[1]) * 0.02).astype(mx.bfloat16),
        },
        (),
        alpha=24.0,
    )
    second = AdapterOperation(
        "layer",
        "lora",
        "weight",
        {
            "down": (mx.random.normal((16, _IN), key=keys[2]) * 0.02).astype(mx.bfloat16),
            "up": (mx.random.normal((_OUT, 16), key=keys[3]) * 0.02).astype(mx.bfloat16),
        },
        (),
    )
    return (
        WeightedAdapterOperation(first, 0.7, "a"),
        WeightedAdapterOperation(second, -1.3, "b"),
    )


def _fused(*, transposed: bool, kernel: bool, monkeypatch) -> tuple[mx.array, int]:
    logical = (mx.random.normal((_OUT, _IN), key=mx.random.key(1)) * 0.05).astype(mx.bfloat16)
    stored = mx.contiguous(logical.T) if transposed else logical
    model = SimpleNamespace(weight=stored)
    target = WeightTarget(model, "weight", (_OUT, _IN))
    calls = []
    fuse = low_rank_fusion.fuse

    def counting_fuse(*args, **kwargs):
        calls.append(kwargs)
        return fuse(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(low_rank_fusion, "_enabled", kernel)
        patch.setattr(low_rank_fusion, "fuse", counting_fuse)
        fuse_adapter_operations(
            {"layer.weight": target},
            {"layer.weight": _operations()},
            pending=AdapterWeightCheck(),
            transposed=frozenset({"layer.weight"}) if transposed else frozenset(),
        )
        mx.eval(model.weight)
    return model.weight, len(calls)


@pytest.mark.parametrize("transposed", [False, True])
def test_kernel_fusion_matches_the_chain_through_the_fuser(transposed, monkeypatch):
    if not low_rank_fusion.available(transposed=transposed):
        pytest.skip("the fusion kernel does not build or does not match MLX on this GPU")
    chain, chain_calls = _fused(transposed=transposed, kernel=False, monkeypatch=monkeypatch)
    kernel, kernel_calls = _fused(transposed=transposed, kernel=True, monkeypatch=monkeypatch)
    assert chain_calls == 0
    # One validation pass and one binding pass.
    assert kernel_calls == 2
    assert kernel.shape == chain.shape
    assert kernel.dtype == chain.dtype
    assert mx.array_equal(kernel.view(mx.uint16), chain.view(mx.uint16)).item()


def test_fuse_weight_takes_the_kernel_for_plain_low_rank_terms(monkeypatch):
    if not low_rank_fusion.available(transposed=False):
        pytest.skip("the fusion kernel does not build or does not match MLX on this GPU")
    value = (mx.random.normal((_OUT, _IN), key=mx.random.key(2)) * 0.05).astype(mx.bfloat16)
    with monkeypatch.context() as patch:
        patch.setattr(low_rank_fusion, "_enabled", False)
        expected = fuse_weight(value, _operations())
        mx.eval(expected)
    calls = []
    fuse = low_rank_fusion.fuse

    def counting_fuse(*args, **kwargs):
        calls.append(kwargs)
        return fuse(*args, **kwargs)

    monkeypatch.setattr(low_rank_fusion, "fuse", counting_fuse)
    actual = fuse_weight(value, _operations())
    mx.eval(actual)
    assert len(calls) == 1
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
