from __future__ import annotations

import inspect
import logging

import mlx.core as mx

from kinomlx.lora.fusion import fuse, fuse_many
from kinomlx.models.ltx2.types import DistilledRequest


def test_partial_override_is_absent_from_public_and_generic_surfaces() -> None:
    assert "lora_allow_partial" not in DistilledRequest.__dataclass_fields__
    for callable_ in (fuse, fuse_many):
        parameters = inspect.signature(callable_).parameters
        assert "allow_partial" not in parameters
        assert "min_coverage" not in parameters


def test_twenty_percent_adapter_fuses_and_warns(caplog) -> None:
    base = {"block.0.weight": mx.zeros((2, 2), dtype=mx.float32)}
    adapter: dict[str, mx.array] = {}
    for index in range(5):
        adapter[f"block.{index}.lora_A.weight"] = mx.ones((1, 2))
        adapter[f"block.{index}.lora_B.weight"] = mx.ones((2, 1))

    with caplog.at_level(logging.WARNING, logger="kinomlx.lora.fusion"):
        result = fuse(base, adapter)

    assert mx.array_equal(result["block.0.weight"], mx.ones((2, 2))).item()
    assert any("coverage 20%" in record.getMessage() for record in caplog.records)


def test_zero_percent_adapter_is_a_warned_noop(caplog) -> None:
    base_weight = mx.zeros((2, 2), dtype=mx.float32)
    adapter = {
        "other.lora_A.weight": mx.ones((1, 2)),
        "other.lora_B.weight": mx.ones((2, 1)),
    }
    with caplog.at_level(logging.WARNING, logger="kinomlx.lora.fusion"):
        result = fuse({"target.weight": base_weight}, adapter)
    assert result["target.weight"] is base_weight
    assert any("coverage 0%" in record.getMessage() for record in caplog.records)
