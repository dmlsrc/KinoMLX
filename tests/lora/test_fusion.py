"""Behavioral tests for ``kinomlx.lora.fusion``."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import kinomlx.lora.fusion as fusion_module
from kinomlx.lora.fusion import compute_delta, fuse, fuse_many
from kinomlx.lora.loading import LoRAEntry

# ---------------------------------------------------------------------------
# compute_delta - the B @ A * scaling * strength math
# ---------------------------------------------------------------------------


def test_compute_delta_matches_explicit_matmul() -> None:
    a = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=mx.float32)
    b = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=mx.float32)
    entry = LoRAEntry(a=a, b=b)  # rank=2, no alpha -> scaling=1.0
    delta = compute_delta(entry, strength=1.0)
    expected = np.matmul(np.asarray(b), np.asarray(a))
    assert np.allclose(np.asarray(delta), expected)


def test_compute_delta_applies_user_strength() -> None:
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)
    entry = LoRAEntry(a=a, b=b)
    delta = compute_delta(entry, strength=0.5)
    # Each (B @ A) entry is rank*1*1 = 2; with strength 0.5 -> 1.0.
    assert np.allclose(np.asarray(delta), 1.0)


def test_compute_delta_applies_alpha_over_rank_scaling() -> None:
    """alpha=16, rank=4 -> scaling = 4.0.  strength=1 -> delta = 4*(B@A)."""
    a = mx.ones((4, 4), dtype=mx.float32)
    b = mx.ones((3, 4), dtype=mx.float32)
    entry = LoRAEntry(a=a, b=b, alpha=16.0)
    delta = compute_delta(entry, strength=1.0)
    # (B @ A) entries are rank=4; scaling 4 -> 16.
    assert np.allclose(np.asarray(delta), 16.0)


def test_compute_delta_produces_fp32_even_with_bf16_input() -> None:
    a = mx.ones((2, 4), dtype=mx.bfloat16)
    b = mx.ones((3, 2), dtype=mx.bfloat16)
    entry = LoRAEntry(a=a, b=b)
    delta = compute_delta(entry)
    assert delta.dtype == mx.float32


# ---------------------------------------------------------------------------
# fuse - orchestration over a base-weights dict
# ---------------------------------------------------------------------------


def test_fuse_applies_delta_to_matched_keys() -> None:
    base = mx.zeros((3, 4), dtype=mx.float32)
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)
    base_weights = {"attn.to_q.weight": base}
    lora_weights = {"attn.to_q.lora_A.weight": a, "attn.to_q.lora_B.weight": b}
    out = fuse(base_weights, lora_weights, strength=1.0)
    # delta entries are rank * 1 * 1 = 2.
    assert np.allclose(np.asarray(out["attn.to_q.weight"]), 2.0)


def test_fuse_passes_untouched_keys_through_unchanged() -> None:
    base_w = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    base_weights = {
        "attn.to_q.weight": base_w,
        "ff.bias": mx.array([1.0, 2.0, 3.0]),  # no LoRA - must pass through.
    }
    lora_weights = {
        "attn.to_q.lora_A.weight": mx.zeros((2, 4)),
        "attn.to_q.lora_B.weight": mx.zeros((3, 2)),
    }
    out = fuse(base_weights, lora_weights)
    assert np.allclose(np.asarray(out["ff.bias"]), np.asarray(base_weights["ff.bias"]))


def test_fuse_preserves_base_dtype_by_default() -> None:
    base = mx.zeros((3, 4), dtype=mx.bfloat16)
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)
    out = fuse(
        {"w": base},
        {"w.lora_A.weight": a, "w.lora_B.weight": b},
    )
    assert out["w"].dtype == mx.bfloat16


def test_fuse_honors_target_dtype_override() -> None:
    base = mx.zeros((3, 4), dtype=mx.float32)
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)
    out = fuse(
        {"w": base},
        {"w.lora_A.weight": a, "w.lora_B.weight": b},
        target_dtype=mx.bfloat16,
    )
    assert out["w"].dtype == mx.bfloat16


def test_fuse_warns_shape_mismatch_and_returns_noop(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    base = mx.zeros((5, 5), dtype=mx.float32)
    # delta would be (3, 4) - doesn't match base (5, 5); fuse skips.
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)
    with caplog.at_level(logging.WARNING, logger="kinomlx.lora.fusion"):
        out = fuse(
            {"w": base},
            {"w.lora_A.weight": a, "w.lora_B.weight": b},
        )
    assert out["w"] is base
    assert any(
        "shape mismatch for w" in rec.message and "(5, 5)" in rec.message for rec in caplog.records
    ), caplog.records


def test_fuse_shape_mismatch_needs_no_override() -> None:
    base = mx.zeros((5, 5), dtype=mx.float32)
    out = fuse(
        {"w": base},
        {
            "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
            "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
        },
    )
    assert out["w"] is base


def test_fuse_returns_new_dict_does_not_mutate_input() -> None:
    base = mx.zeros((3, 4), dtype=mx.float32)
    base_weights = {"w": base}
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)
    fuse(base_weights, {"w.lora_A.weight": a, "w.lora_B.weight": b})
    # Original dict unchanged - same reference, same value.
    assert base_weights["w"] is base
    assert np.allclose(np.asarray(base_weights["w"]), 0.0)


def test_fuse_chain_for_multiple_loras() -> None:
    """Chaining fuse calls applies multiple LoRAs sequentially."""
    base = mx.zeros((3, 4), dtype=mx.float32)
    lora1 = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }  # delta = 2.0
    lora2 = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32) * 0.5,
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }  # delta = 1.0
    out = fuse(fuse({"w": base}, lora1), lora2)
    # 0 + 2 + 1 = 3.
    assert np.allclose(np.asarray(out["w"]), 3.0)


def test_fuse_many_accumulates_multiple_loras_once() -> None:
    base = mx.zeros((3, 4), dtype=mx.float32)
    lora1 = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    lora2 = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32) * 0.5,
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    out = fuse_many({"w": base}, [(lora1, 1.0), (lora2, 1.0)])
    assert np.allclose(np.asarray(out["w"]), 3.0)


def test_fuse_accepts_transposed_model_layout() -> None:
    base = mx.zeros((4, 3), dtype=mx.float32)
    a = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    b = mx.arange(6, dtype=mx.float32).reshape(3, 2)
    out = fuse(
        {"w": base},
        {"w.lora_A.weight": a, "w.lora_B.weight": b},
    )
    expected = mx.matmul(b, a).T
    assert np.allclose(np.asarray(out["w"]), np.asarray(expected))


def test_fuse_uses_declared_weight_t_layout_for_square_weights() -> None:
    weight = mx.array([[1.0, 2.0], [3.0, 4.0]])
    a = mx.array([[1.0, 2.0]])
    b = mx.array([[3.0], [4.0]])
    key = "__layout__.attn.to_q.weight_t"

    out = fuse(
        {key: weight.T},
        {
            f"{key}.lora_A.weight": a,
            f"{key}.lora_B.weight": b,
        },
    )

    expected = (weight + mx.matmul(b, a)).T
    assert mx.array_equal(out[key], expected).item()


def test_fuse_wrong_model_adapter_is_noop() -> None:
    base = {"w": mx.zeros((3, 4), dtype=mx.float32)}
    lora = {
        "other.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "other.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    out = fuse(base, lora)
    assert out["w"] is base["w"]


def test_fuse_many_keeps_placeable_targets_when_another_adapter_is_zero_percent() -> None:
    base = {"w": mx.zeros((3, 4), dtype=mx.float32)}
    valid = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    wrong = {
        "other.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "other.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    out = fuse_many(base, [(valid, 1.0), (wrong, 1.0)])
    assert mx.array_equal(out["w"], mx.full((3, 4), 2.0)).item()


def test_fuse_in_place_mutates_and_returns_same_dictionary() -> None:
    base = {"w": mx.zeros((3, 4), dtype=mx.float32)}
    original = base["w"]
    lora = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    out = fuse(base, lora, in_place=True)
    assert out is base
    assert base["w"] is not original
    assert np.allclose(np.asarray(base["w"]), 2.0)


def test_in_place_fusion_leaves_wide_dtype_values_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_eval = mx.eval
    eval_calls: list[int] = []

    def recording_eval(*values: mx.array) -> None:
        eval_calls.append(len(values))
        real_eval(*values)

    monkeypatch.setattr(fusion_module.mx, "eval", recording_eval)
    base = {"w": mx.zeros((3, 4), dtype=mx.bfloat16)}
    lora = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.bfloat16),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.bfloat16),
    }

    out = fuse(
        base,
        lora,
        in_place=True,
    )

    assert eval_calls == []
    real_eval(out["w"])
    assert np.allclose(np.asarray(out["w"].astype(mx.float32)), 2.0)


def test_in_place_fp16_range_guard_is_one_batched_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_eval = mx.eval
    eval_calls: list[int] = []

    def recording_eval(*values: mx.array) -> None:
        eval_calls.append(len(values))
        real_eval(*values)

    monkeypatch.setattr(fusion_module.mx, "eval", recording_eval)
    base = {
        "a": mx.zeros((3, 4), dtype=mx.float16),
        "b": mx.zeros((3, 4), dtype=mx.float16),
    }
    lora = {
        "a.lora_A.weight": mx.ones((2, 4), dtype=mx.float16),
        "a.lora_B.weight": mx.ones((3, 2), dtype=mx.float16),
        "b.lora_A.weight": mx.ones((2, 4), dtype=mx.float16),
        "b.lora_B.weight": mx.ones((3, 2), dtype=mx.float16),
    }

    out = fuse(
        base,
        lora,
        in_place=True,
    )

    assert eval_calls == [1]
    real_eval(*out.values())
    assert all(np.allclose(np.asarray(value.astype(mx.float32)), 2.0) for value in out.values())


def test_fp16_range_validation_can_consume_independent_lazy_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_base_value = mx.zeros((3, 4), dtype=mx.float16)
    production_a = mx.ones((2, 4), dtype=mx.float16)
    production_b = mx.ones((3, 2), dtype=mx.float16)
    production_base = {"w": production_base_value}
    production_adapter = {
        "w.lora_A.weight": production_a,
        "w.lora_B.weight": production_b,
    }
    validation_base_value = mx.zeros((3, 4), dtype=mx.float16)
    validation_a = mx.ones((2, 4), dtype=mx.float16)
    validation_b = mx.ones((3, 2), dtype=mx.float16)
    validation_base = {"w": validation_base_value}
    validation_adapter = {
        "w.lora_A.weight": validation_a,
        "w.lora_B.weight": validation_b,
    }
    checked_inputs: list[tuple[mx.array, mx.array, mx.array]] = []
    real_bound = fusion_module._fused_abs_bound

    def recording_bound(base, records, *, base_peak=None):
        checked_inputs.append((base, records[0][0].a, records[0][0].b))
        return real_bound(base, records, base_peak=base_peak)

    monkeypatch.setattr(fusion_module, "_fused_abs_bound", recording_bound)

    fused = fuse_many(
        production_base,
        [(production_adapter, 1.0)],
        in_place=True,
        range_validation_inputs=(
            validation_base,
            [(validation_adapter, 1.0)],
            None,
        ),
    )

    assert checked_inputs == [(validation_base_value, validation_a, validation_b)]
    assert validation_base == {}
    assert validation_adapter == {}
    assert fused["w"] is not production_base_value
    assert production_adapter["w.lora_A.weight"] is production_a
    assert production_adapter["w.lora_B.weight"] is production_b
    assert np.allclose(np.asarray(fused["w"].astype(mx.float32)), 2.0)


def test_fp16_range_validation_requires_every_cached_base_peak() -> None:
    production = {"w": mx.zeros((3, 4), dtype=mx.float16)}
    production_adapter = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float16),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float16),
    }
    validation = {"w": mx.zeros((3, 4), dtype=mx.float16)}
    validation_adapter = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float16),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float16),
    }

    with pytest.raises(ValueError, match="missing LoRA target 'w'"):
        fuse_many(
            production,
            [(production_adapter, 1.0)],
            in_place=True,
            range_validation_inputs=(
                validation,
                [(validation_adapter, 1.0)],
                {},
            ),
        )

    assert validation == {}
    assert validation_adapter == {}
    assert production["w"].shape == (3, 4)


def test_in_place_zero_percent_noop_does_not_mutate_base() -> None:
    original = mx.zeros((3, 4), dtype=mx.float32)
    base = {"w": original}
    wrong = {
        "other.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "other.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    fuse(base, wrong, in_place=True)
    assert base["w"] is original


def test_in_place_fp16_overflow_raises_before_any_dict_mutation() -> None:
    first = mx.zeros((2, 2), dtype=mx.float16)
    second = mx.zeros((2, 2), dtype=mx.float16)
    base = {"first": first, "second": second}
    adapter = {
        "first.lora_A.weight": mx.ones((1, 2), dtype=mx.float32),
        "first.lora_B.weight": mx.ones((2, 1), dtype=mx.float32),
        "second.lora_A.weight": mx.full((1, 2), 1e4, dtype=mx.float32),
        "second.lora_B.weight": mx.full((2, 1), 1e4, dtype=mx.float32),
    }
    mutation_callbacks = 0

    def before_mutation() -> None:
        nonlocal mutation_callbacks
        mutation_callbacks += 1

    with pytest.raises(ValueError, match="second"):
        fuse_many(
            base,
            [(adapter, 1.0)],
            in_place=True,
            before_mutation=before_mutation,
        )

    assert set(base) == {"first", "second"}
    assert base["first"] is first
    assert base["second"] is second
    assert mutation_callbacks == 0


def test_fuse_rejects_incompatible_factor_ranks() -> None:
    base = {"w": mx.zeros((3, 4), dtype=mx.float32)}
    malformed = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "w.lora_B.weight": mx.ones((3, 3), dtype=mx.float32),
    }
    with pytest.raises(ValueError, match="incompatible ranks"):
        fuse(base, malformed)


@pytest.mark.parametrize("strength", [float("nan"), float("inf")])
def test_fuse_rejects_nonfinite_strength(strength) -> None:
    base = {"w": mx.zeros((3, 4), dtype=mx.float32)}
    lora = {
        "w.lora_A.weight": mx.ones((2, 4), dtype=mx.float32),
        "w.lora_B.weight": mx.ones((3, 2), dtype=mx.float32),
    }
    with pytest.raises(ValueError, match="finite"):
        fuse(base, lora, strength=strength)


# ---------------------------------------------------------------------------
# _guard_fused_range - narrow-dtype (fp16) overflow guard
# ---------------------------------------------------------------------------


def test_fuse_fp16_target_raises_on_overflow() -> None:
    """Fusing into fp16 raises when the fused |max| exceeds the 65504 ceiling."""
    base = mx.zeros((3, 4), dtype=mx.float32)
    # delta entries = (B @ A) = rank * 1e4 * 1e4 = 2e8, far over the fp16 ceiling.
    a = mx.full((2, 4), 1e4, dtype=mx.float32)
    b = mx.full((3, 2), 1e4, dtype=mx.float32)
    with pytest.raises(ValueError, match="overflows"):
        fuse(
            {"w": base},
            {"w.lora_A.weight": a, "w.lora_B.weight": b},
            target_dtype=mx.float16,
        )


def test_fuse_bf16_target_allows_wide_range() -> None:
    """bf16 shares fp32's exponent range, so the overflow guard is a no-op."""
    base = mx.zeros((3, 4), dtype=mx.float32)
    a = mx.full((2, 4), 1e4, dtype=mx.float32)
    b = mx.full((3, 2), 1e4, dtype=mx.float32)  # fused ~2e8, fine for bf16
    out = fuse(
        {"w": base},
        {"w.lora_A.weight": a, "w.lora_B.weight": b},
        target_dtype=mx.bfloat16,
    )
    assert out["w"].dtype == mx.bfloat16
    assert bool(mx.isfinite(out["w"]).all().item())


def test_fuse_fp16_target_within_range_ok() -> None:
    """Fusing into fp16 succeeds when the fused weight stays under the ceiling."""
    base = mx.zeros((3, 4), dtype=mx.float32)
    a = mx.ones((2, 4), dtype=mx.float32)
    b = mx.ones((3, 2), dtype=mx.float32)  # delta = 2.0, well under 65504
    out = fuse(
        {"w": base},
        {"w.lora_A.weight": a, "w.lora_B.weight": b},
        target_dtype=mx.float16,
    )
    assert out["w"].dtype == mx.float16
    assert np.allclose(np.asarray(out["w"].astype(mx.float32)), 2.0)


def test_fp16_bound_near_limit_falls_back_to_exact_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls = 0
    real_fused_weight = fusion_module._fused_weight_f32

    def recording_fused_weight(base, records):
        nonlocal exact_calls
        exact_calls += 1
        return real_fused_weight(base, records)

    monkeypatch.setattr(fusion_module, "_fused_weight_f32", recording_fused_weight)
    out = fuse(
        {"w": mx.zeros((1, 1), dtype=mx.float16)},
        {
            "w.lora_A.weight": mx.ones((1, 1), dtype=mx.float32),
            "w.lora_B.weight": mx.full((1, 1), 65000.0, dtype=mx.float32),
        },
        target_dtype=mx.float16,
    )

    # Once for the exact safety fallback and once for the returned lazy graph.
    assert exact_calls == 2
    assert float(out["w"].item()) == 64992.0


def test_nonmutating_fp16_range_checks_are_evaluated_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_eval = mx.eval
    eval_calls: list[int] = []

    def recording_eval(*values: mx.array) -> None:
        eval_calls.append(len(values))
        real_eval(*values)

    monkeypatch.setattr(fusion_module.mx, "eval", recording_eval)
    base = {
        "a": mx.zeros((2, 2), dtype=mx.float16),
        "b": mx.zeros((2, 2), dtype=mx.float16),
    }
    adapter = {
        "a.lora_A.weight": mx.ones((1, 2)),
        "a.lora_B.weight": mx.ones((2, 1)),
        "b.lora_A.weight": mx.ones((1, 2)),
        "b.lora_B.weight": mx.ones((2, 1)),
    }

    fused = fuse_many(base, [(adapter, 1.0)], in_place=False)

    assert eval_calls == [1]
    assert all(value.dtype == mx.float16 for value in fused.values())
