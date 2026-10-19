"""Behavioral tests for low-rank pairs and application controls."""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

from kinomlx.adapters.lora import (
    LoRAConfig,
    LoRAEntry,
    iter_lora_entries,
)

# ---------------------------------------------------------------------------
# LoRAConfig
# ---------------------------------------------------------------------------


def test_config_normalizes_str_to_path() -> None:
    cfg = LoRAConfig(path="/some/lora.safetensors", strength=0.8)
    assert isinstance(cfg.path, Path)
    assert cfg.path == Path("/some/lora.safetensors")


def test_config_accepts_finite_strengths_outside_the_typical_artistic_range() -> None:
    assert LoRAConfig(path=Path("/x"), strength=3.0).strength == 3.0
    assert LoRAConfig(path=Path("/x"), strength=-3.0).strength == -3.0


def test_config_default_strength_is_one() -> None:
    assert LoRAConfig(path=Path("/x")).strength == 1.0


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite_strengths(strength: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        LoRAConfig(path="/x", strength=strength)


def test_config_rejects_bad_exclusion() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        LoRAConfig(path="/x", exclude=("",))


def test_config_canonicalizes_exclusion_identity() -> None:
    config = LoRAConfig(path="/x", exclude=("video", "audio", "video"))
    assert config.exclude == ("audio", "video")


def test_config_keeps_the_selected_path_outside_its_identity(tmp_path: Path) -> None:
    blob = tmp_path / "0123abcd"
    blob.write_bytes(b"weights")
    link = tmp_path / "adapter_model.safetensors"
    link.symlink_to(blob)
    selected = LoRAConfig(path=link)
    assert selected.path == blob.resolve()
    assert selected.source == link
    assert selected == LoRAConfig(path=blob)
    assert replace(selected, strength=0.5).source == link


# ---------------------------------------------------------------------------
# LoRAEntry - rank + scaling
# ---------------------------------------------------------------------------


def test_entry_rank_is_first_dim_of_a() -> None:
    e = LoRAEntry(a=mx.zeros((4, 64)), b=mx.zeros((128, 4)))
    assert e.rank == 4


def test_entry_scaling_defaults_to_one() -> None:
    e = LoRAEntry(a=mx.zeros((4, 64)), b=mx.zeros((128, 4)))
    assert e.scaling == 1.0


def test_entry_scaling_is_alpha_over_rank() -> None:
    e = LoRAEntry(a=mx.zeros((8, 64)), b=mx.zeros((128, 8)), alpha=16.0)
    assert e.scaling == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# iter_lora_entries - naming conventions
# ---------------------------------------------------------------------------


def _ab_pair() -> tuple[mx.array, mx.array]:
    return mx.arange(8, dtype=mx.float32).reshape(2, 4), mx.arange(6, dtype=mx.float32).reshape(
        3, 2
    )


def test_finds_lora_a_b_with_weight_suffix() -> None:
    a, b = _ab_pair()
    weights = {"layer.attn.to_q.lora_A.weight": a, "layer.attn.to_q.lora_B.weight": b}
    entry = dict(iter_lora_entries(weights)).get("layer.attn.to_q")
    assert entry is not None
    assert entry.alpha is None


def test_finds_lora_down_up_with_weight_suffix() -> None:
    a, b = _ab_pair()
    weights = {"layer.attn.to_q.lora_down.weight": a, "layer.attn.to_q.lora_up.weight": b}
    entry = dict(iter_lora_entries(weights)).get("layer.attn.to_q")
    assert entry is not None


def test_finds_lora_a_b_without_weight_suffix() -> None:
    a, b = _ab_pair()
    weights = {"layer.attn.to_q.lora_A": a, "layer.attn.to_q.lora_B": b}
    entry = dict(iter_lora_entries(weights)).get("layer.attn.to_q")
    assert entry is not None


def test_reads_alpha_when_present() -> None:
    a, b = _ab_pair()
    weights = {
        "layer.attn.to_q.lora_A.weight": a,
        "layer.attn.to_q.lora_B.weight": b,
        "layer.attn.to_q.alpha": mx.array(8.0, dtype=mx.float32),
    }
    entry = dict(iter_lora_entries(weights)).get("layer.attn.to_q")
    assert entry is not None
    assert entry.alpha == 8.0


# ---------------------------------------------------------------------------
# Alpha garbage filtering - real community LoRAs ship corrupt alpha values
# surprisingly often.  These tests pin the safe fallback (scaling=1.0)
# rather than letting inf/nan/etc. propagate into a weight multiply.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_alpha",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -1.0,
        -16.0,
        1e7,  # above _MAX_REASONABLE_ALPHA
        1e12,
    ],
    ids=["inf", "-inf", "nan", "zero", "neg_one", "neg_sixteen", "1e7", "1e12"],
)
def test_alpha_garbage_falls_back_to_unit_scaling(bad_alpha: float) -> None:
    """Any non-finite, non-positive, or absurdly large alpha is dropped."""
    a, b = _ab_pair()
    weights = {
        "w.lora_A.weight": a,
        "w.lora_B.weight": b,
        "w.alpha": mx.array(bad_alpha, dtype=mx.float32),
    }
    entry = dict(iter_lora_entries(weights)).get("w")
    assert entry is not None
    assert entry.alpha is None  # corrupt value rejected
    assert entry.scaling == 1.0  # falls back to no scaling override


def test_alpha_at_max_reasonable_boundary_is_accepted() -> None:
    """The upper bound is inclusive - exactly _MAX_REASONABLE_ALPHA passes."""
    from kinomlx.adapters.lora import _MAX_REASONABLE_ALPHA

    a, b = _ab_pair()
    weights = {
        "w.lora_A.weight": a,
        "w.lora_B.weight": b,
        "w.alpha": mx.array(_MAX_REASONABLE_ALPHA, dtype=mx.float32),
    }
    entry = dict(iter_lora_entries(weights)).get("w")
    assert entry is not None
    assert entry.alpha == _MAX_REASONABLE_ALPHA


def test_alpha_tiny_positive_value_is_accepted() -> None:
    """Small-but-positive alphas are valid - only ``<= 0`` is rejected."""
    a, b = _ab_pair()
    weights = {
        "w.lora_A.weight": a,
        "w.lora_B.weight": b,
        "w.alpha": mx.array(1e-3, dtype=mx.float32),
    }
    entry = dict(iter_lora_entries(weights)).get("w")
    assert entry is not None
    assert entry.alpha == pytest.approx(1e-3)


def test_returns_none_when_no_pair_matches() -> None:
    weights = {"foo.bar": mx.zeros((2, 4))}
    assert dict(iter_lora_entries(weights)).get("layer.attn.to_q") is None


def test_does_not_apply_model_specific_prefix_transforms() -> None:
    """Generic matcher must not invent prefixes (e.g. ``diffusion_model.``)."""
    a, b = _ab_pair()
    # LoRA file has the prefix; base key doesn't.  Generic matcher should NOT
    # find this - that's a per-model concern (handled in models/<name>/adapters/).
    weights = {
        "diffusion_model.layer.attn.to_q.lora_A.weight": a,
        "diffusion_model.layer.attn.to_q.lora_B.weight": b,
    }
    assert dict(iter_lora_entries(weights)).get("layer.attn.to_q") is None


def test_iter_lora_entries_rejects_orphan_factors() -> None:
    weights = {
        "a.lora_A.weight": mx.ones((2, 4)),
        "a.lora_B.weight": mx.ones((3, 2)),
        "b.lora_down": mx.ones((1, 5)),
        "b.lora_up": mx.ones((6, 1)),
        "orphan.lora_A.weight": mx.ones((2, 4)),
    }
    with pytest.raises(ValueError, match="unmatched A/B factors.*orphan"):
        list(iter_lora_entries(weights))


def test_iter_lora_entries_yields_each_complete_target_once() -> None:
    weights = {
        "a.lora_A.weight": mx.ones((2, 4)),
        "a.lora_B.weight": mx.ones((3, 2)),
        "b.lora_down": mx.ones((1, 5)),
        "b.lora_up": mx.ones((6, 1)),
    }
    entries = list(iter_lora_entries(weights))
    assert [prefix for prefix, _entry in entries] == ["a", "b"]
    assert [entry.rank for _prefix, entry in entries] == [2, 1]
