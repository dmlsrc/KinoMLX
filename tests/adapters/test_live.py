"""Live activation appliers, checked against the dense reconstruction they replace.

Every applier exists only because it reproduces ``reconstruct_delta`` without
building the dense delta, so the dense path is the oracle for all of them.
"""

import mlx.core as mx
import pytest

import kinomlx._mlx_nn as nn
from kinomlx.adapters.algebra import reconstruct_delta
from kinomlx.adapters.formats import AdapterOperation, LinearAdapterOptions
from kinomlx.adapters.live import (
    LiveAdapterLinear,
    LiveAdapterPretransposedLinear,
    ScheduledAdapterOperation,
    install_live_adapters,
    live_adapters,
    live_applier,
    set_live_adapter_step,
)
from kinomlx.adapters.schedule import AdapterSchedule
from kinomlx.modeling.layers import PretransposedLinear
from kinomlx.modeling.loading import WeightTarget

OUT, INP = 12, 18
RANK = 4


def _operation(kind, factors, **extra):
    return AdapterOperation(
        prefix="block.0.attn.to_q",
        kind=kind,
        part="weight",
        factors=factors,
        source_keys=(),
        **extra,
    )


def _pair(out=OUT, inp=INP, rank=RANK):
    return mx.random.normal((rank, inp)), mx.random.normal((out, rank))


def _lora(**extra):
    down, up = _pair()
    return _operation("lora", {"down": down, "up": up}, **extra)


def _adalora():
    down, up = _pair()
    return _operation(
        "adalora",
        {"down": down, "up": up, "diagonal": mx.random.normal((RANK, 1))},
        alpha=4.0,
    )


def _loha(rank=2, out=OUT, inp=INP):
    # A Hadamard pair expands to rank*rank columns, so rank 2 on this target is
    # the schedulable side of the crossover; see the size-guard test below.
    factors = {}
    for index in (1, 2):
        down, up = _pair(out=out, inp=inp, rank=rank)
        factors[f"hada_w{index}_a"] = up
        factors[f"hada_w{index}_b"] = down
    return _operation("loha", factors, alpha=4.0)


def _lokr_dense():
    return _operation(
        "lokr", {"lokr_w1": mx.random.normal((3, 3)), "lokr_w2": mx.random.normal((4, 6))}
    )


def _lokr_factored():
    return _operation(
        "lokr",
        {
            "lokr_w1_a": mx.random.normal((3, 2)),
            "lokr_w1_b": mx.random.normal((2, 3)),
            "lokr_w2_a": mx.random.normal((4, 2)),
            "lokr_w2_b": mx.random.normal((2, 6)),
        },
        alpha=2.0,
    )


SCHEDULABLE = {
    "lora": _lora,
    "lora-alpha": lambda: _lora(alpha=8.0),
    "lora-scalar": lambda: _lora(alpha=2.0, scalar=0.5),
    "adalora": _adalora,
    "loha": _loha,
    "lokr-dense": _lokr_dense,
    "lokr-factored": _lokr_factored,
}


@pytest.mark.parametrize("name", sorted(SCHEDULABLE))
@pytest.mark.parametrize("strength", [1.0, 0.35, -0.8])
def test_live_applier_matches_the_dense_reconstruction(name, strength) -> None:
    mx.random.seed(5)
    operation = SCHEDULABLE[name]()
    base = mx.random.normal((OUT, INP))
    value = mx.random.normal((2, 5, INP))
    built = live_applier(operation, (OUT, INP), dtype=mx.float32)
    assert built is not None
    apply, retained = built
    assert retained > 0

    expected = value @ (reconstruct_delta(operation, base) * strength).T
    applied = apply(value, mx.zeros(expected.shape), strength)
    scale = max(mx.abs(expected).max().item(), 1e-9)
    assert (mx.abs(applied - expected).max() / scale).item() < 1e-5


@pytest.mark.parametrize("name", sorted(SCHEDULABLE))
def test_live_applier_adds_to_an_existing_result(name) -> None:
    mx.random.seed(6)
    operation = SCHEDULABLE[name]()
    base = mx.random.normal((OUT, INP))
    value = mx.random.normal((3, INP))
    existing = mx.random.normal((3, OUT))
    built = live_applier(operation, (OUT, INP), dtype=mx.float32)
    assert built is not None
    expected = existing + value @ reconstruct_delta(operation, base).T
    applied = built[0](value, existing, 1.0)
    scale = max(mx.abs(expected).max().item(), 1e-9)
    assert (mx.abs(applied - expected).max() / scale).item() < 1e-5


def _dora():
    down, up = _pair()
    return _operation(
        "lora", {"down": down, "up": up, "magnitude": mx.random.normal((OUT, 1))}, alpha=4.0
    )


def _hira():
    down, up = _pair()
    return _operation(
        "hira",
        {"A": down, "B": up},
        linear_options=LinearAdapterOptions(rank=RANK),
    )


def _tucker_lora():
    down, up = _pair(out=OUT, inp=INP)
    return _operation(
        "lora",
        {"down": down, "up": up, "mid": mx.random.normal((RANK, RANK))},
    )


@pytest.mark.parametrize(
    ("name", "build"),
    [("dora", _dora), ("hira", _hira), ("tucker-lora", _tucker_lora)],
)
def test_base_dependent_and_tucker_formats_decline_a_schedule(name, build) -> None:
    mx.random.seed(7)
    assert live_applier(build(), (OUT, INP), dtype=mx.float32) is None


def test_nonmatrix_targets_decline_a_schedule() -> None:
    mx.random.seed(8)
    assert live_applier(_lora(), (OUT,), dtype=mx.float32) is None


@pytest.mark.parametrize(("rank", "schedulable"), [(2, True), (8, False)])
def test_a_loha_declines_a_schedule_once_its_pair_outgrows_the_dense_delta(
    rank, schedulable
) -> None:
    # Rank 8 on a 64x64 target retains 64 columns each side, 8192 elements
    # against the dense delta's 4096, so the activation form costs more than the
    # object it exists to avoid.
    mx.random.seed(16)
    operation = _loha(rank=rank, out=64, inp=64)
    built = live_applier(operation, (64, 64), dtype=mx.float32)
    assert (built is not None) is schedulable


def test_a_declined_loha_is_refused_by_name_at_install() -> None:
    mx.random.seed(17)
    layer = nn.Linear(64, 64, bias=False)
    targets = {"block.0.attn.to_q.weight": WeightTarget(layer, "weight", (64, 64))}
    with pytest.raises(ValueError, match="a.safetensors.*fuse it at one constant strength"):
        install_live_adapters(
            targets,
            {
                "block.0.attn.to_q.weight": (
                    ScheduledAdapterOperation(
                        _loha(rank=8, out=64, inp=64),
                        AdapterSchedule((1.0, 0.0)),
                        "a.safetensors",
                    ),
                )
            },
            model="test",
        )


def _targets(layer, name="block.0.attn.to_q.weight"):
    return {name: WeightTarget(layer, "weight", (OUT, INP))}


def _install(layer, operation, schedule, name="block.0.attn.to_q.weight"):
    return install_live_adapters(
        _targets(layer, name),
        {name: (ScheduledAdapterOperation(operation, schedule, "a.safetensors"),)},
        model="test",
    )


@pytest.mark.parametrize(
    ("build_layer", "live_class"),
    [
        (lambda: nn.Linear(INP, OUT, bias=False), LiveAdapterLinear),
        (
            lambda: PretransposedLinear(mx.random.normal((INP, OUT))),
            LiveAdapterPretransposedLinear,
        ),
    ],
)
def test_install_wraps_both_linear_layouts_without_touching_the_weight(
    build_layer, live_class
) -> None:
    mx.random.seed(9)
    layer = build_layer()
    original = mx.array(layer.weight)
    controller = _install(layer, _lora(), AdapterSchedule((1.0, 0.0)))

    assert isinstance(layer, live_class)
    assert mx.array_equal(layer.weight, original).item()
    assert len(controller.terms) == 1
    assert controller.retained_bytes > 0


def test_retained_factors_stay_outside_mlx_parameter_traversal() -> None:
    mx.random.seed(10)
    layer = nn.Linear(INP, OUT, bias=False)
    _install(layer, _lora(), AdapterSchedule((1.0, 0.0)))

    assert sorted(layer.parameters()) == ["weight"]
    layer.set_dtype(mx.bfloat16)
    assert layer._live_terms[0].apply.a_t.dtype == mx.float32


def test_a_gated_off_step_reproduces_the_unadapted_layer_exactly() -> None:
    mx.random.seed(11)
    layer = nn.Linear(INP, OUT, bias=False)
    reference = nn.Linear(INP, OUT, bias=False)
    reference.weight = mx.array(layer.weight)
    value = mx.random.normal((2, 5, INP))

    controller = _install(layer, _lora(alpha=8.0), AdapterSchedule((1.0, 0.0)))
    controller.set_step(1)
    assert mx.array_equal(layer(value), reference(value)).item()

    controller.set_step(0)
    assert not mx.array_equal(layer(value), reference(value)).item()


def test_each_step_selects_its_own_strength() -> None:
    mx.random.seed(12)
    layer = nn.Linear(INP, OUT, bias=False)
    schedule = AdapterSchedule((1.0, 0.5, 0.0))
    controller = _install(layer, _lora(), schedule)
    for index, expected in enumerate(schedule.per_step):
        controller.set_step(index)
        assert layer._live_terms[0].strength == expected


def test_controller_is_reachable_from_the_model_and_advances_by_step() -> None:
    mx.random.seed(13)
    layer = nn.Linear(INP, OUT, bias=False)
    controller = _install(layer, _lora(), AdapterSchedule((0.25, 0.0)))

    class Port:
        pass

    port = Port()
    assert live_adapters(port) is None
    port.live_adapters = controller
    set_live_adapter_step(port, 0)
    assert layer._live_terms[0].strength == 0.25
    set_live_adapter_step(Port(), 0)


def test_install_refuses_a_format_with_no_activation_form() -> None:
    mx.random.seed(14)
    layer = nn.Linear(INP, OUT, bias=False)
    with pytest.raises(ValueError, match="no activation form"):
        _install(layer, _dora(), AdapterSchedule((1.0, 0.0)))


def test_install_refuses_a_target_that_is_not_a_wrappable_linear() -> None:
    mx.random.seed(15)
    layer = nn.Conv2d(3, 4, 1)
    with pytest.raises(ValueError, match="cannot carry a per-step schedule"):
        _install(layer, _lora(), AdapterSchedule((1.0, 0.0)))


def test_install_refuses_a_bias_target() -> None:
    mx.random.seed(16)
    layer = nn.Linear(INP, OUT)
    name = "block.0.attn.to_q.bias"
    with pytest.raises(ValueError, match="only Linear weights"):
        install_live_adapters(
            {name: WeightTarget(layer, "bias", (OUT,))},
            {name: (ScheduledAdapterOperation(_lora(), AdapterSchedule((1.0,)), "a"),)},
            model="test",
        )


def test_install_rejects_factors_that_lose_range_in_the_compute_dtype() -> None:
    layer = nn.Linear(INP, OUT, bias=False)
    layer.weight = layer.weight.astype(mx.float16)
    overflowing = _operation(
        "lora",
        {"down": mx.full((RANK, INP), 1.0e20), "up": mx.full((OUT, RANK), 1.0)},
    )
    with pytest.raises(ValueError, match="nonfinite in the compute dtype"):
        _install(layer, overflowing, AdapterSchedule((1.0, 0.0)))


def test_two_scheduled_adapters_on_one_target_compose_and_order_by_source() -> None:
    mx.random.seed(17)
    layer = nn.Linear(INP, OUT, bias=False)
    name = "block.0.attn.to_q.weight"
    first, second = _lora(), _lora()
    controller = install_live_adapters(
        _targets(layer, name),
        {
            name: (
                ScheduledAdapterOperation(second, AdapterSchedule((1.0, 0.0)), "b.safetensors"),
                ScheduledAdapterOperation(first, AdapterSchedule((0.0, 1.0)), "a.safetensors"),
            )
        },
        model="test",
    )
    assert [term.order_key for term in controller.terms] == ["a.safetensors", "b.safetensors"]

    base = mx.array(layer.weight)
    value = mx.random.normal((3, INP))
    for step, operation in ((0, second), (1, first)):
        controller.set_step(step)
        expected = value @ base.T + value @ reconstruct_delta(operation, base).T
        scale = max(mx.abs(expected).max().item(), 1e-9)
        assert (mx.abs(layer(value) - expected).max() / scale).item() < 1e-5


def test_controller_metadata_records_targets_terms_and_schedules() -> None:
    mx.random.seed(18)
    layer = nn.Linear(INP, OUT, bias=False)
    controller = _install(layer, _lora(), AdapterSchedule((1.0, 0.0)))
    metadata = controller.to_metadata()
    assert metadata["targets"] == 1
    assert metadata["terms"] == 1
    assert metadata["kinds"] == ["lora"]
    assert metadata["retained_factor_bytes"] > 0
    assert metadata["schedules"] == {
        "a.safetensors": {
            "selection": "steps",
            "resolved": {"per_step": [1.0, 0.0], "active_steps": [0]},
        }
    }


def test_a_size_decline_and_a_format_decline_state_different_reasons() -> None:
    """F-09 follow-through: both declines are true, and they are not the same fact."""
    mx.random.seed(18)
    layer = nn.Linear(64, 64, bias=False)
    targets = {"block.0.attn.to_q.weight": WeightTarget(layer, "weight", (64, 64))}

    def install(operation, shape=(64, 64)):
        targets["block.0.attn.to_q.weight"] = WeightTarget(layer, "weight", shape)
        install_live_adapters(
            targets,
            {
                "block.0.attn.to_q.weight": (
                    ScheduledAdapterOperation(
                        operation, AdapterSchedule((1.0, 0.0)), "a.safetensors"
                    ),
                )
            },
            model="test",
        )

    with pytest.raises(ValueError, match="fuse it at one constant strength") as oversized:
        install(_loha(rank=8, out=64, inp=64))
    assert "expands to a factor pair larger than the dense delta" in str(oversized.value)
    assert "has no activation form" not in str(oversized.value)

    with pytest.raises(ValueError, match="fuse it at one constant strength") as unsupported:
        install(_dora(), shape=(OUT, INP))
    assert "has no activation form" in str(unsupported.value)
    assert "expands to a factor pair larger" not in str(unsupported.value)
