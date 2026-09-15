"""Live effective-target validation after the first model evaluation."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinomlx.adapters.formats import AdapterOperation
from kinomlx.adapters.fusion import WeightedAdapterOperation, fuse_adapter_operations
from kinomlx.adapters.validation import AdapterWeightCheck, AdapterWeightError
from kinomlx.modeling.loading import WeightTarget
from tests._reporting import RecordingReporter


def _fuse(*, value=1.0, strength=1.0, alpha=None, enabled=True):
    model = SimpleNamespace(weight=mx.zeros((2, 3), dtype=mx.float16))
    target = WeightTarget(model, "weight", (2, 3))
    op = AdapterOperation(
        "layer",
        "lora",
        "weight",
        {"down": mx.full((1, 3), value), "up": mx.ones((2, 1))},
        (),
        alpha=alpha,
    )
    pending = AdapterWeightCheck(enabled=enabled)
    selected = WeightedAdapterOperation(op, strength, "adapter")
    fuse_adapter_operations(
        {"layer.weight": target}, {"layer.weight": (selected, selected)}, pending=pending
    )
    return model, pending


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 100000.0])
def test_fused_nonfinite_weight_is_rejected_after_use(value):
    model, pending = _fuse(value=value)
    mx.eval(model.weight @ mx.ones((3, 1), dtype=mx.float16))
    reporter = RecordingReporter()
    with pytest.raises(AdapterWeightError, match=r"step 1.*layer.weight.*float16"):
        pending.run(context="model step 1", reporter=reporter)
    assert reporter.events[-1] == ("end", "check adapter weights", {})


def test_success_reads_live_target_once_and_deduplicates_stacked_operations():
    model, pending = _fuse()
    assert len(pending.targets) == 1
    # Tracking resolves the actual current physical weight, not an earlier expression.
    model.weight = mx.full((2, 3), float("nan"), dtype=mx.float16)
    with pytest.raises(AdapterWeightError):
        pending.run(context="first step", reporter=RecordingReporter())
    model.weight = mx.ones((2, 3), dtype=mx.float16)
    reporter = RecordingReporter()
    pending.run(context="first step", reporter=reporter)
    count = len(reporter.events)
    model.weight = mx.full((2, 3), float("nan"), dtype=mx.float16)
    pending.run(context="second step", reporter=reporter)
    assert len(reporter.events) == count
    assert pending.targets == {}


@pytest.mark.parametrize("kwargs", [{"strength": 0.0}])
def test_ineffective_operations_schedule_no_scan(kwargs):
    _, pending = _fuse(**kwargs)
    assert pending.targets == {}


def test_opt_out_does_not_read_targets():
    _, pending = _fuse(enabled=False)
    pending.targets["bad"] = lambda: pytest.fail("disabled scan read a weight")
    pending.run(context="first step", reporter=RecordingReporter())
    assert pending.targets == {}


def test_zero_alpha_does_not_suppress_dora_magnitude_or_saved_bias():
    from math import sqrt

    weights = SimpleNamespace(weight=mx.ones((2, 3)), bias=mx.zeros(2))
    operations = {
        "weight": AdapterOperation(
            "layer",
            "lora",
            "weight",
            {
                "down": mx.ones((1, 3)),
                "up": mx.ones((2, 1)),
                "magnitude": mx.full((2,), 2 * sqrt(3)),
            },
            (),
            alpha=0,
            magnitude_is_peft=True,
        ),
        "bias": AdapterOperation(
            "layer",
            "lora",
            "bias",
            {"down": mx.ones((1, 3)), "up_bias": mx.ones(2), "base_bias": mx.full((2,), 3)},
            (),
            alpha=0,
        ),
    }
    pending = AdapterWeightCheck()
    fuse_adapter_operations(
        {name: WeightTarget(weights, name, getattr(weights, name).shape) for name in operations},
        {name: (WeightedAdapterOperation(op, 1, "adapter"),) for name, op in operations.items()},
        pending=pending,
    )
    assert set(pending.targets) == {"weight", "bias"}
    assert mx.allclose(weights.weight, mx.full((2, 3), 2)).item()
    assert mx.array_equal(weights.bias, mx.full((2,), 4)).item()


def test_saved_zero_alpha_uses_unit_scale_and_remains_a_checked_target():
    model, pending = _fuse(alpha=0.0)
    assert set(pending.targets) == {"layer.weight"}
    assert mx.array_equal(model.weight, mx.full((2, 3), 2)).item()


def test_active_filter_defers_unread_targets_until_their_first_use():
    pending = AdapterWeightCheck()
    good = SimpleNamespace(weight=mx.ones((2, 2)))
    pending.add("video.weight", WeightTarget(good, "weight", (2, 2)))
    pending.targets["audio.weight"] = lambda: pytest.fail("inactive target was read")
    reporter = RecordingReporter()
    pending.run(
        context="video step", reporter=reporter, active=lambda name: name.startswith("video")
    )
    assert set(pending.targets) == {"audio.weight"}
    previous = len(reporter.events)
    pending.run(
        context="later video step", reporter=reporter, active=lambda name: name.startswith("video")
    )
    assert len(reporter.events) == previous
    bad = SimpleNamespace(weight=mx.full((2, 2), float("nan")))
    pending.add("audio.weight", WeightTarget(bad, "weight", (2, 2)))
    with pytest.raises(AdapterWeightError, match="audio.weight"):
        pending.run(context="audio enabled", reporter=reporter)
