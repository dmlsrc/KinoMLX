"""Shared binder: preflighted binding stays lazy with or without a preparation."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.io.safetensors import save_weights
from kinomlx.modeling.loading import WeightTarget, bind_weight_targets


def _fixture(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_weights(
        path,
        {
            "a.weight": mx.arange(6).reshape(2, 3).astype(mx.bfloat16),
            "b.weight": mx.ones((3,), dtype=mx.bfloat16),
            "unused": mx.zeros((4,), dtype=mx.bfloat16),
        },
    )
    owner = SimpleNamespace(a=SimpleNamespace(weight=None), b=SimpleNamespace(weight=None))
    targets = {
        "a.weight": WeightTarget(owner.a, "weight", (2, 3)),
        "b.weight": WeightTarget(owner.b, "weight", (3,)),
    }
    return path, owner, targets


def test_binding_without_preparation_binds_stored_tensors_lazily(tmp_path, monkeypatch):
    path, owner, targets = _fixture(tmp_path)

    def eager(*args):
        raise AssertionError("binding must stay lazy")

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", eager)
        count = bind_weight_targets(
            (path,), targets, lambda key: (key,), component="probe", phase="bind"
        )
    assert count == 2
    assert owner.a.weight.dtype == mx.bfloat16
    assert owner.a.weight.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert owner.b.weight.tolist() == [1, 1, 1]


def test_binding_applies_one_lazy_preparation_per_consumed_tensor(tmp_path, monkeypatch):
    path, owner, targets = _fixture(tmp_path)
    seen = []

    def prepare(binding, value):
        seen.append((binding.logical_key, binding.source_key, value.dtype))
        if binding.logical_key == "a.weight":
            return mx.contiguous(value.astype(mx.float16).T)
        return value.astype(mx.float16)

    def eager(*args):
        raise AssertionError("prepared binding must stay lazy")

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", eager)
        count = bind_weight_targets(
            (path,),
            targets,
            lambda key: (key,),
            component="probe",
            phase="bind",
            expected_dtype=mx.bfloat16,
            prepare=prepare,
        )
    assert count == 2
    assert seen == [("a.weight", "a.weight", mx.bfloat16), ("b.weight", "b.weight", mx.bfloat16)]
    assert owner.a.weight.shape == (3, 2)
    assert owner.a.weight.dtype == mx.float16
    assert owner.a.weight.tolist() == [[0, 3], [1, 4], [2, 5]]
    assert owner.b.weight.dtype == mx.float16
    with pytest.raises(ValueError, match="dtype BF16"):
        bind_weight_targets(
            (path,),
            targets,
            lambda key: (key,),
            component="probe",
            phase="bind",
            expected_dtype=mx.float16,
            prepare=prepare,
        )


def test_missing_component_reports_the_requested_source(tmp_path):
    from kinomlx.modeling.loading import weight_files

    path = tmp_path / "missing.safetensors"
    with pytest.raises(FileNotFoundError, match="no transformer weights found at") as error:
        weight_files(path, component="transformer")
    assert str(path) in str(error.value)
